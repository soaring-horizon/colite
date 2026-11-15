#include <QTemporaryDir>
#include <QThread>
#include <QSemaphore>
#include <QtTest>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>

#include <QImageReader>
#include <png.h>
#include "project/project.h"
#include "project/recovery_store.h"
#include "workflow/workflow_state.h"
#include "project/scene_export.h"
#include "workflow/task_runner.h"
#include "workflow/task_plan.h"
#include "workflow/parameter_schema.h"
#include "project/workspace.h"
#ifdef Q_OS_WIN
#define NOMINMAX
#include <windows.h>
#else
#include <signal.h>
#endif
using namespace colite::gui;
namespace {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4611) // No automatic objects with destructors cross longjmp.
#endif
bool writeLargePng(QFile* file, png_bytep row) {
  auto* png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png) return false;
  auto* info = png_create_info_struct(png);
  if (!info) { png_destroy_write_struct(&png, nullptr); return false; }
  if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); return false; }
  png_set_write_fn(png, file, [](png_structp p, png_bytep data, png_size_t size) {
    auto* output = static_cast<QFile*>(png_get_io_ptr(p));
    if (output->write(reinterpret_cast<char*>(data), qint64(size)) != qint64(size))
      png_error(p, "fixture write failed");
  }, [](png_structp) {});
  png_set_IHDR(png, info, 12000, 12000, 8, PNG_COLOR_TYPE_GRAY,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  for (int y = 0; y < 12000; ++y) png_write_row(png, row);
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  return true;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif
void write(const QString& path, const QByteArray& value) {
  QFile f(path);
  if (!f.open(QIODevice::WriteOnly)) qFatal("fixture write failed");
  f.write(value);
}
QByteArray feature() {
  QByteArray payload;
  QDataStream points(&payload, QIODevice::WriteOnly);
  points.setByteOrder(QDataStream::LittleEndian);
  points.setFloatingPointPrecision(QDataStream::SinglePrecision);
  points << 1.f << 2.f << 3.f << 4.f << 5.f;
  quint64 hash = 14695981039346656037ULL;
  for (unsigned char b : payload) {
    hash ^= b;
    hash *= 1099511628211ULL;
  }
  QByteArray file;
  QDataStream out(&file, QIODevice::WriteOnly);
  out.setByteOrder(QDataStream::LittleEndian);
  out << quint32(0x434c4654) << quint32(2) << quint32(1) << quint32(1)
      << quint32(1) << quint32(2) << quint64(1) << quint64(0) << quint64(128)
      << quint64(0) << hash;
  file += payload;
  return file;
}
bool processAlive(qint64 pid) {
#ifdef Q_OS_WIN
  HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, DWORD(pid));
  if (!process) return false;
  const bool alive = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
  CloseHandle(process);
  return alive;
#else
  QFile stat(QString("/proc/%1/stat").arg(pid));
  if (stat.open(QIODevice::ReadOnly) && stat.readAll().contains(") Z")) return false;
  return ::kill(pid_t(pid), 0) == 0;
#endif
}
}  // namespace
class GuiContracts : public QObject {
  Q_OBJECT
 private slots:
  void algorithmParametersRouteOnlyApplicableOptions() {
    QTemporaryDir dir;
    Project p; p.imageRoot = dir.filePath("photos"); p.assignOutputs(dir.filePath("work"));
    const auto defaults = effectiveParameters({});
    QCOMPARE(defaults.value("PoissonMeshing.depth").toString(), QString("9"));
    QCOMPARE(defaults.value("PatchMatchStereo.max_image_size").toString(), QString("50%"));
    p.parameters = {{"mapper_method", "global"}, {"Mapper.init_min_num_inliers", 999},
                    {"GlobalMapper.min_num_inliers", 20}};
    const auto mapping = buildTaskPlan(p, 3, false).first().arguments;
    QVERIFY(mapping.contains("global"));
    QVERIFY(mapping.contains("--GlobalMapper.min_num_inliers"));
    QVERIFY(!mapping.contains("--Mapper.init_min_num_inliers"));
    p.parameters = {{"FeatureExtraction.type", "aliked_n32"}, {"matching_method", "lightglue"},
                    {"FeatureMatching.type", "SIFT_BRUTEFORCE"}, {"FeatureMatching.use_gpu", true},
                    {"SiftMatching.max_ratio", 0.5}, {"device", "cuda"}, {"learned_matching_device", "cpu"}};
    const auto matching = buildTaskPlan(p, 2, false).first().arguments;
    QCOMPARE(matching.count("--FeatureMatching.type"), 1);
    QCOMPARE(matching.count("--FeatureMatching.use_gpu"), 1);
    QVERIFY(matching.contains("ALIKED_LIGHTGLUE"));
    QVERIFY(!matching.contains("--SiftMatching.max_ratio"));
    QVERIFY(!matching.contains("--FeatureMatching.guided_matching"));
    p.parameters = {{"surface_method", "delaunay"}, {"simplify_mesh", true}};
    p.mesh = dir.filePath("work/mesh-full.ply");
    const auto surface = buildTaskPlan(p, 5, false);
    QCOMPARE(surface.size(), 2);
    QCOMPARE(surface[0].arguments.first(), QString("delaunay_mesher"));
    QCOMPARE(surface[1].arguments.first(), QString("mesh_simplifier"));
    QVERIFY(!surface[0].arguments.contains("--PoissonMeshing.depth"));
    QCOMPARE(surface[1].arguments[surface[1].arguments.indexOf("--output_path") + 1], p.mesh);
    QVERIFY(surface[1].arguments[surface[1].arguments.indexOf("--input_path") + 1] != p.mesh);
  }
  void externalMasksInvalidateFeaturesButInactiveOptionsDoNot() {
    QTemporaryDir dir;
    Project p; p.imageRoot = dir.filePath("photos"); QDir().mkpath(p.imageRoot);
    const auto mask = dir.filePath("mask.png"); write(mask, "mask-one");
    p.parameters["ImageReader.camera_mask_path"] = mask;
    const auto stamp = stageInputStamp(p, 1);
    p.parameters["AlikedExtraction.min_score"] = 0.9;
    QCOMPARE(stageInputStamp(p, 1), stamp);
    write(mask, "replacement-mask");
    QVERIFY(stageInputStamp(p, 1) != stamp);
  }
  void sourceIdentityDoesNotDependOnImagePlugins() {
    QTemporaryDir dir;
    Project project; project.imageRoot = dir.path();
    write(dir.filePath("a.JpG"), "jpeg-source");
    write(dir.filePath("b.png"), "png-source");
    write(dir.filePath("ignored.txt"), "not-a-photo");
    struct RestorePaths {
      QStringList paths = QCoreApplication::libraryPaths();
      ~RestorePaths() { QCoreApplication::setLibraryPaths(paths); }
    } restore;
    QCoreApplication::setLibraryPaths({dir.path()});
    const auto original = stageInputStamp(project, 1);
    QCOMPARE(Workspace::read(project).images.size(), 2);
    QCOMPARE(stageInputStamp(project, 1), original);
    write(dir.filePath("a.JpG"), "changed-jpeg-source");
    QVERIFY(stageInputStamp(project, 1) != original);
    const auto changed = stageInputStamp(project, 1);
    QCoreApplication::setLibraryPaths(restore.paths);
    QCOMPARE(stageInputStamp(project, 1), changed);
  }
  void backendFailureExplainsCauseAndAllowsRetry() {
    QTemporaryDir dir;
    Project project; project.assignOutputs(dir.filePath("work"));
    project.backend = QCoreApplication::applicationFilePath();
    TaskRunner runner;
    runner.start(project, {{"write", {"--fixture-worker", "disk-full"}, {}}}, {});
    QTRY_COMPARE_WITH_TIMEOUT(runner.state(), TaskRunner::State::Failed, 5000);
    QVERIFY(runner.lastError().contains(QString::fromUtf8("工作磁盘空间不足")));
    QVERIFY(runner.lastError().contains("No space left on device"));
    QCOMPARE(runner.completed(), 0);
    runner.start(project, {{"retry", {"--fixture-worker", "ok"}, {}}}, {});
    QTRY_COMPARE_WITH_TIMEOUT(runner.state(), TaskRunner::State::Succeeded, 5000);
    QVERIFY(runner.lastError().isEmpty());
  }
  void progressProtocolAndElapsedHistory() {
    QTemporaryDir dir;
    Project project; project.assignOutputs(dir.filePath("work"));
    project.backend = QCoreApplication::applicationFilePath();
    TaskRunner runner;
    QVector<qint64> counts;
    connect(&runner, &TaskRunner::progressChanged, this, [&] {
      if (runner.progressTotal()) counts << runner.progressDone();
    });
    runner.start(project, {{"progress", {"--fixture-worker", "progress"}, {}}}, {});
    QTRY_COMPARE_WITH_TIMEOUT(runner.state(), TaskRunner::State::Succeeded, 5000);
    QCOMPARE(counts, QVector<qint64>({1, 2}));
    QCOMPARE(runner.progressTotal(), qint64(2));
    QVERIFY(runner.elapsedMilliseconds() >= 100);
    TaskRunner reopened;
    QVERIFY(reopened.restoreHistory(project));
    QCOMPARE(reopened.elapsedMilliseconds(), runner.elapsedMilliseconds());
  }
  void selectedSparseModelRoutesDenseProcessing() {
    QTemporaryDir dir;
    Project project; project.imageRoot = dir.filePath("photos");
    project.assignOutputs(dir.filePath("work"));
    project.parameters["sparse_model"] = 2;
    const auto plan = buildTaskPlan(project, 4, false);
    QVERIFY(!plan.isEmpty());
    const auto args = plan.first().arguments;
    QCOMPARE(args.value(args.indexOf("--input_path") + 1), QDir(project.sparse).filePath("models/2"));
    const auto stamp = stageInputStamp(project, 4);
    project.parameters["sparse_model"] = 1;
    QVERIFY(stamp != stageInputStamp(project, 4));
  }
  void exportKeepsTextureAndDoesNotOverwrite() {
    QTemporaryDir dir;
    const auto mesh = dir.filePath("source.ply");
    QDir().mkpath(dir.filePath("assets"));
    QImage image(8, 8, QImage::Format_RGB32); image.fill(Qt::green);
    QVERIFY(image.save(dir.filePath("assets/original.png")));
    write(mesh, "ply\nformat ascii 1.0\ncomment TextureFile assets/original.png\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n");
    const auto target = dir.filePath("export");
    exportScene(mesh, target);
    const auto result = validateScene(QDir(target).filePath("model.ply"));
    QCOMPARE(result.triangles, quint64(1));
    QCOMPARE(result.texture, QDir(target).filePath("texture.png"));
    QVERIFY_EXCEPTION_THROWN(exportScene(mesh, target), std::runtime_error);
    QVERIFY_EXCEPTION_THROWN(exportScene(mesh, dir.filePath("cancelled"), [] { return true; }), std::runtime_error);
    QVERIFY(!QFileInfo::exists(dir.filePath("cancelled")));
    QFile texture(dir.filePath("assets/original.png"));
    QVERIFY(texture.remove());
    QVERIFY_EXCEPTION_THROWN(exportScene(mesh, dir.filePath("missing-texture")), std::runtime_error);
    QVERIFY(!QFileInfo::exists(dir.filePath("missing-texture")));
    QVERIFY(QFileInfo::exists(QDir(target).filePath("texture.png")));
  }
  void stageValidityTracksInputsParametersAndOutputs() {
    QTemporaryDir dir;
    Project project; project.name = "Validity";
    project.imageRoot = dir.filePath("images");
    QDir().mkpath(project.imageRoot);
    project.assignOutputs(dir.filePath("work"));
    QDir().mkpath(QDir(project.workspace).filePath("features"));
    QDir().mkpath(QDir(project.workspace).filePath("matches"));
    write(QDir(project.imageRoot).filePath("a.png"), "image-a");
    write(QDir(project.imageRoot).filePath("b.png"), "image-b");
    write(QDir(project.workspace).filePath("images.tsv"), "1\t1\ta.png\tunused\n2\t1\tb.png\tunused\n");
    for (int i : {1, 2}) write(QDir(project.workspace).filePath(QString("features/%1.bin").arg(i)), feature());
    write(QDir(project.workspace).filePath("matches/raw_matches.tsv"), "# raw\n");
    write(QDir(project.workspace).filePath("matches/verified_matches.tsv"), "# verified\n");
    QCOMPARE(outputState(project, 1), OutputState::Unverified);
    recordStage(project, 1); recordStage(project, 2);
    QCOMPARE(outputState(project, 2), OutputState::Current);
    project.parameters["PoissonMeshing.depth"] = 11;
    project.parameters["PatchMatchStereo.max_image_size"] = -1;
    QCOMPARE(outputState(project, 1), OutputState::Current);
    QCOMPARE(outputState(project, 2), OutputState::Current);
    project.parameters["SiftExtraction.peak_threshold"] = 0.01;
    QCOMPARE(outputState(project, 1), OutputState::Stale);
    QCOMPARE(outputState(project, 2), OutputState::Stale);
    QVERIFY_EXCEPTION_THROWN(checkStageInputs(project, 3), std::invalid_argument);
    project.parameters.remove("SiftExtraction.peak_threshold");
    checkStageInputs(project, 3);
    const auto inputBefore = stageInputStamp(project, 1);
    write(QDir(project.imageRoot).filePath("a.png"), "changed-image");
    QVERIFY_EXCEPTION_THROWN(recordStage(project, 1, inputBefore), std::runtime_error);
    QCOMPARE(outputState(project, 1), OutputState::Stale);
    recordStage(project, 1);
    invalidateStages(project, 2);
    QCOMPARE(outputState(project, 1), OutputState::Current);
    QCOMPARE(outputState(project, 2), OutputState::Stale);
    project.parameters["PoissonMeshing.depth"] = 1.5;
    QVERIFY_EXCEPTION_THROWN(validateParameters(project), std::invalid_argument);
  }
  void photoPreparationCanCancelWithoutChangingProject() {
    QTemporaryDir dir;
    Project project; project.name = "Input";
    project.assignOutputs(dir.filePath("work"));
    write(dir.filePath("photo.png"), QByteArray(3 * 1024 * 1024, 'x'));
    project.photoFiles.insert("photo.png", dir.filePath("photo.png"));
    bool cancel = false;
    QVERIFY_EXCEPTION_THROWN(project.prepareInput([&](qint64 done, qint64 total, const QString&) {
      QVERIFY(done > 0); QCOMPARE(total, qint64(3 * 1024 * 1024)); cancel = true;
    }, [&] { return cancel; }), std::runtime_error);
    QVERIFY(project.imageRoot.isEmpty());
    QVERIFY(QDir(project.workDirectory).entryList({"photos-*"}, QDir::Dirs | QDir::NoDotAndDotDot).isEmpty());
    project.prepareInput();
    QCOMPARE(QFileInfo(QDir(project.imageRoot).filePath("photo.png")).size(), qint64(3 * 1024 * 1024));
  }
  void recoveryPreservesIdentityAndHonorsLiveSessions() {
    QTemporaryDir dir;
    Project project;
    project.name = QString::fromUtf8("未命名工程");
    project.photoFiles.insert("image.png", dir.filePath("image.png"));
    project.assignOutputs(dir.filePath("work"));
    project.parameters["Mapper.ba_refine_focal_length"] = false;
    RecoveryStore next(dir.filePath("recovery"));
    {
      RecoveryStore active(dir.filePath("recovery"));
      active.save(project);
      QVERIFY(next.available().isEmpty());
    } // Represents process exit without a successful save/discard.
    QCOMPARE(next.available().size(), 1);
    const auto restored = next.restore(next.available().first());
    QCOMPARE(restored.id, project.id);
    QVERIFY(restored.file.isEmpty());
    QCOMPARE(restored.photoFiles, project.photoFiles);
    QCOMPARE(restored.workspace, project.workspace);
    QCOMPARE(restored.parameters, project.parameters);
    QVERIFY(next.available().isEmpty());
    next.clear();
    QVERIFY(QDir(dir.filePath("recovery")).entryList({"*.json"}, QDir::Files).isEmpty());
    project.save(dir.filePath("saved.colite.json"));
    const auto original = Project::load(project.file);
    {
      RecoveryStore active(dir.filePath("recovery"));
      project.name = "Changed draft";
      active.save(project);
    }
    const auto savedDraft = next.restore(next.available().first());
    QCOMPARE(savedDraft.file, project.file);
    QCOMPARE(Project::load(project.file).name, original.name);
    next.clear();
  }
  void artifactValidationIgnoresDisplayBudget() {
    QTemporaryDir dir;
    const auto path = dir.filePath("mesh.ply");
    write(path, "ply\nformat ascii 1.0\ncomment TextureFile texture.png\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n");
    // A 144 MP atlas is downsampled while decoding instead of allocating a full-size image.
    QFile texture(dir.filePath("texture.png"));
    QVERIFY(texture.open(QIODevice::WriteOnly));
    QByteArray row(12000, '\x60');
    QVERIFY(writeLargePng(&texture, reinterpret_cast<png_bytep>(row.data())));
    texture.close();
    const auto result = validateScene(path);
    QCOMPARE(result.points, quint64(3));
    QCOMPARE(result.triangles, quint64(1));
    const auto preview = readScene(path).texturePreview;
    QCOMPARE(preview.size(), QSize(4096, 4096));
    QCOMPARE(preview.pixelColor(0, 0), QColor(96, 96, 96));
    QCOMPARE(preview.pixelColor(4095, 4095), QColor(96, 96, 96));
    QVERIFY(texture.open(QIODevice::ReadWrite));
    QVERIFY(texture.resize(texture.size() - 20));
    texture.close();
    QVERIFY_EXCEPTION_THROWN(validateScene(path), std::runtime_error);
  }
  void largeGeometryValidation() {
    if (qEnvironmentVariableIsEmpty("COLITE_GUI_LARGE_ARTIFACTS"))
      QSKIP("Set COLITE_GUI_LARGE_ARTIFACTS to stream a 17-million-point fixture");
    QTemporaryDir dir;
    QFile file(dir.filePath("large.ply"));
    QVERIFY(file.open(QIODevice::WriteOnly));
    const QByteArray header("ply\nformat binary_little_endian 1.0\nelement vertex 17000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
    QCOMPARE(file.write(header), qint64(header.size()));
    const QByteArray block(12 * 100000, '\0');
    for (int i = 0; i < 170; ++i) QCOMPARE(file.write(block), qint64(block.size()));
    file.close();
    const auto preview = readScene(file.fileName());
    QCOMPARE(preview.totalPoints, quint64(17000000));
    QVERIFY(!preview.points.isEmpty() && preview.points.size() <= 2000000);
    QCOMPARE(validateScene(file.fileName()).points, quint64(17000000));
    QVERIFY(file.open(QIODevice::ReadWrite));
    QVERIFY(file.resize(file.size() - 12));
    file.close();
    QVERIFY_EXCEPTION_THROWN(validateScene(file.fileName()), std::runtime_error);
  }
  void taskPlanRespectsConfiguredOutputs() {
    QTemporaryDir dir;
    Project project;
    project.imageRoot = dir.filePath("images");
    project.workspace = dir.filePath("workspace");
    project.sparse = dir.filePath("custom model");
    project.dense = dir.filePath("dense");
    project.mesh = dir.filePath("mesh/mesh.ply");
    project.parameters = {{"Undistortion.max_image_size", 1600},
                          {"StereoFusion.min_num_pixels", 3},
                          {"PatchMatchStereo.geom_consistency", true}};
    const auto steps = buildTaskPlan(project, 3, true);
    QCOMPARE(steps.size(), 7);
    QCOMPARE(steps[1].arguments[0], QString("model_converter"));
    const auto& exportArgs = steps[1].arguments;
    QCOMPARE(exportArgs[exportArgs.indexOf("--output_path") + 1], project.sparse);
    QCOMPARE(exportArgs[exportArgs.indexOf("--output_type") + 1], QString("both"));
    const auto& undistort = steps[2].arguments;
    QCOMPARE(undistort[undistort.indexOf("--input_path") + 1], project.sparse);
    QCOMPARE(undistort[undistort.indexOf("--max_image_size") + 1], QString("1600"));
    QVERIFY(!undistort.contains("--Undistortion.max_image_size"));
    const auto& stereo = steps[3].arguments;
    QCOMPARE(stereo[stereo.indexOf("--PatchMatchStereo.geom_consistency") + 1], QString("1"));
    QVERIFY(!stereo.contains("--PatchMatchStereo.max_image_size"));
    QVERIFY(stereo.contains("--Colite.resolution_level"));
    QCOMPARE(stereo.value(stereo.indexOf("--Colite.resolution_level") + 1), QString("1"));
    QVERIFY(!undistort.contains("--Colite.resolution_level"));
    const auto& fusion = steps[4].arguments;
    QCOMPARE(fusion[fusion.indexOf("--StereoFusion.min_num_pixels") + 1], QString("3"));
    QCOMPARE(fusion[fusion.indexOf("--input_type") + 1], QString("geometric"));
    QVERIFY(!fusion.contains("--Colite.resolution_level"));
    for (const int value : {-1, 1600}) {
      project.parameters["PatchMatchStereo.max_image_size"] = value;
      const auto explicitSize = buildTaskPlan(project, 4, false)[1].arguments;
      QVERIFY(!explicitSize.contains("--Colite.resolution_level"));
      QVERIFY(explicitSize.contains("--PatchMatchStereo.max_image_size"));
      QCOMPARE(explicitSize.value(explicitSize.indexOf("--PatchMatchStereo.max_image_size") + 1), QString::number(value));
    }
    project.parameters["PatchMatchStereo.geom_consistency"] = false;
    const auto photoFusion = buildTaskPlan(project, 4, false)[2].arguments;
    QCOMPARE(photoFusion[photoFusion.indexOf("--input_type") + 1], QString("photometric"));
    project.sparse = QDir(project.workspace).filePath("sparse");
    QCOMPARE(buildTaskPlan(project, 3, true).size(), 6);
    QVERIFY_EXCEPTION_THROWN(buildTaskPlan(project, 0, true), std::invalid_argument);
  }
  void desktopPipelineIntegration() {
    const auto path = qEnvironmentVariable("COLITE_GUI_PIPELINE_PROJECT");
    if (path.isEmpty()) QSKIP("Set COLITE_GUI_PIPELINE_PROJECT for the actual desktop task plan");
    const auto project = Project::load(path);
    auto inputs = project; inputs.workspace.clear();
    QVERIFY2(Workspace::read(inputs).images.size() >= 2, "Full pipeline needs source photos in this process");
    TaskRunner runner;
    const auto steps = buildTaskPlan(project, 1, true);
    // Optional benchmark output uses the same scheduler and validators as the
    // desktop pipeline, keeping algorithm wall time separate from validation.
    const auto timingPath = qEnvironmentVariable("COLITE_GUI_TIMING_REPORT");
    QElapsedTimer benchmarkClock;
    benchmarkClock.start();
    QVector<qint64> starts(steps.size(), -1), backendEnds(steps.size(), -1), ends(steps.size(), -1);
    auto saveTimings = [&] {
      if (timingPath.isEmpty()) return;
      QJsonArray records;
      for (int i = 0; i < steps.size(); ++i) {
        QJsonObject record{{"index", i}, {"stage", steps[i].stage}, {"title", steps[i].title},
            {"arguments", QJsonArray::fromStringList(steps[i].arguments)},
            {"start_ms", double(starts[i])}, {"backend_end_ms", double(backendEnds[i])},
            {"end_ms", double(ends[i])}};
        if (starts[i] >= 0 && backendEnds[i] >= 0) record["backend_ms"] = double(backendEnds[i] - starts[i]);
        if (backendEnds[i] >= 0 && ends[i] >= 0) record["validation_ms"] = double(ends[i] - backendEnds[i]);
        if (starts[i] >= 0 && ends[i] >= 0) record["total_ms"] = double(ends[i] - starts[i]);
        records.append(record);
      }
      QSaveFile file(timingPath);
      if (!file.open(QIODevice::WriteOnly)) qFatal("Cannot open timing report");
      const auto bytes = QJsonDocument(QJsonObject{{"project", project.file},
          {"elapsed_ms", double(runner.elapsedMilliseconds())}, {"completed", runner.completed()},
          {"success", runner.state() == TaskRunner::State::Succeeded},
          {"error", runner.lastError()}, {"steps", records}}).toJson();
      if (file.write(bytes) != bytes.size() || !file.commit()) qFatal("Cannot save timing report");
    };
    connect(&runner, &TaskRunner::output, this, [&](const QString& text) {
      const int i = runner.completed();
      if (i < steps.size() && text == QString::fromUtf8("开始：") + steps[i].title + "\n") {
        starts[i] = benchmarkClock.elapsed(); saveTimings();
      }
    });
    connect(&runner, &TaskRunner::changed, this, [&] {
      const int i = runner.completed();
      if (i > 0 && ends[i - 1] < 0) ends[i - 1] = benchmarkClock.elapsed();
      if (i < steps.size() && runner.validating() && backendEnds[i] < 0)
        backendEnds[i] = benchmarkClock.elapsed();
      saveTimings();
    });
    runner.start(project, steps, QDir(QCoreApplication::applicationDirPath()).filePath("worker_bridge.py"));
    const int configuredTimeout = qEnvironmentVariableIntValue("COLITE_GUI_PIPELINE_TIMEOUT_MS");
    const int timeout = configuredTimeout > 0 ? configuredTimeout : 7200000;
    QTRY_VERIFY_WITH_TIMEOUT(!runner.busy(), timeout);
    saveTimings();
    QCOMPARE(int(runner.state()), int(TaskRunner::State::Succeeded));
    QCOMPARE(runner.completed(), steps.size());
    for (int stage = 1; stage <= 6; ++stage) QCOMPARE(outputState(project, stage), OutputState::Current);
    const auto reopened = Project::load(path);
    for (int stage = 1; stage <= 6; ++stage) QCOMPARE(outputState(reopened, stage), OutputState::Current);
  }
  void completedProjectArtifacts() {
    const auto path = qEnvironmentVariable("COLITE_GUI_ARTIFACT_PROJECT");
    if (path.isEmpty())
      QSKIP("Set COLITE_GUI_ARTIFACT_PROJECT to validate completed real outputs");
    const auto project = Project::load(path);
    const auto workspace = Workspace::read(project);
    QVERIFY(!workspace.images.isEmpty());
    quint64 featureCount = 0;
    QMap<quint32, int> counts;
    for (const auto& image : workspace.images) {
      const auto features = workspace.features(image.id);
      QVERIFY2(!features.isEmpty(), qPrintable(image.name));
      counts[image.id] = features.size();
      featureCount += features.size();
    }
    QVERIFY(workspace.images.size() >= 2);
    const auto pair = workspace.pair(
        workspace.images[0].id, workspace.images[1].id,
        counts[workspace.images[0].id], counts[workspace.images[1].id]);
    QVERIFY(pair.verified);
    QVERIFY(pair.inliers > 0);
    const auto sparse = readScene(project.sparse);
    const auto dense = readScene(QDir(project.dense).filePath("fused.ply"));
    const auto mesh = readScene(project.mesh);
    const auto textured = readScene(
        QFileInfo(project.mesh).dir().filePath("textured/mesh.ply"));
    QVERIFY(!sparse.points.isEmpty());
    QCOMPARE(sparse.cameras.size() / 16, workspace.images.size());
    QVERIFY(!dense.points.isEmpty());
    QVERIFY(!mesh.triangles.isEmpty());
    QCOMPARE(mesh.triangles.size() % 3, 0);
    QVERIFY(!textured.triangles.isEmpty());
    QVERIFY(!textured.texture.isEmpty());
    const auto texture = QImageReader(textured.texture).read();
    QVERIFY(!texture.isNull());
    QVERIFY(texture.width() > 0 && texture.height() > 0);
    const auto reportPath = QDir(QFileInfo(project.file).absolutePath())
                                .filePath("artifact-validation.json");
    const QJsonObject report{
        {"project", project.file},
        {"image_count", workspace.images.size()},
        {"feature_count", double(featureCount)},
        {"registered_cameras", sparse.cameras.size() / 16},
        {"sparse_points", double(sparse.totalPoints)},
        {"dense_points", double(dense.totalPoints)},
        {"mesh_vertices", double(mesh.totalPoints)},
        {"mesh_triangles", mesh.triangles.size() / 3},
        {"textured_vertices", double(textured.totalPoints)},
        {"textured_triangles", textured.triangles.size() / 3},
        {"texture_width", texture.width()},
        {"texture_height", texture.height()},
        {"first_pair_matches", pair.matches.size()},
        {"first_pair_inliers", pair.inliers},
        {"status", "passed"}};
    write(reportPath, QJsonDocument(report).toJson());
  }
  void backendIntegration() {
    const auto path = qEnvironmentVariable("COLITE_GUI_INTEGRATION_PROJECT");
    if (path.isEmpty())
      QSKIP(
          "Set COLITE_GUI_INTEGRATION_PROJECT to exercise a configured "
          "backend");
    const auto project = Project::load(path);
    TaskRunner runner;
    const QStringList common = {"--Colite.workspace_path", project.workspace};
    QVector<TaskStep> steps = {
        {"extract",
         QStringList{"feature_extractor", "--image_path", project.imageRoot,
                     "--Colite.feature_extractor", "vlsift",
                     "--FeatureExtraction.max_image_size", "1000",
                     "--SiftExtraction.max_num_features", "2048",
                     "--FeatureExtraction.num_threads", "2"} +
             common,
         {}},
        {"match",
         QStringList{"exhaustive_matcher", "--FeatureMatching.num_threads",
                     "2"} +
             common,
         {}},
        {"mapper",
         QStringList{"mapper", "--image_path", project.imageRoot,
                     "--Mapper.num_threads", "2"} +
             common,
         {}}};
    if (qEnvironmentVariable("COLITE_GUI_INTEGRATION_FULL") == "1") {
      const auto dense = project.dense;
      const auto fused = QDir(dense).filePath("fused.ply");
      const auto mesh = project.mesh;
      const auto textured = QFileInfo(mesh).dir().filePath("textured");
      steps.push_back({"undistort",
                       {"image_undistorter", "--input_path", project.sparse,
                        "--image_path", project.imageRoot, "--output_path",
                        dense, "--max_image_size", "400", "--resume"},
                       {}});
      steps.push_back({"stereo",
                       {"patch_match_stereo", "--workspace_path", dense,
                        "--PatchMatchStereo.max_image_size", "400",
                        "--PatchMatchStereo.num_iterations", "2",
                        "--Colite.patch_match_backend", "cuda", "--resume"},
                       {}});
      steps.push_back(
          {"fusion",
           {"stereo_fusion", "--workspace_path", dense, "--output_path", fused,
            "--StereoFusion.min_num_pixels", "2", "--resume"},
           [fused] {
             if (readScene(fused).points.isEmpty())
               throw std::runtime_error("Empty fused cloud");
           }});
      steps.push_back({"surface",
                       {"poisson_mesher", "--input_path", fused,
                        "--output_path", mesh, "--PoissonMeshing.depth", "6",
                        "--PoissonMeshing.trim", "0", "--resume"},
                       [mesh] {
                         if (readScene(mesh).triangles.isEmpty())
                           throw std::runtime_error("Empty mesh");
                       }});
      steps.push_back(
          {"texture",
           {"mesh_texturer", "--workspace_path", dense, "--input_path", mesh,
            "--output_path", textured, "--resume"},
           [textured] {
             if (readScene(QDir(textured).filePath("mesh.ply"))
                     .texture.isEmpty())
               throw std::runtime_error("Missing texture metadata");
           }});
    }
    connect(&runner, &TaskRunner::output, this, [](const QString& s) {
      QFile out;
      out.open(stdout, QIODevice::WriteOnly);
      out.write(s.toUtf8());
    });
    runner.start(project, steps,
                 QDir(QCoreApplication::applicationDirPath())
                     .filePath("worker_bridge.py"));
    QTRY_VERIFY_WITH_TIMEOUT(!runner.busy(), 240000);
    QCOMPARE(int(runner.state()), int(TaskRunner::State::Succeeded));
    const auto workspace = Workspace::read(project);
    QVERIFY(workspace.images.size() >= 2);
    const auto a = workspace.features(workspace.images[0].id),
               b = workspace.features(workspace.images[1].id);
    QVERIFY(!a.isEmpty());
    QVERIFY(!b.isEmpty());
    const auto pair = workspace.pair(
        workspace.images[0].id, workspace.images[1].id, a.size(), b.size());
    QVERIFY(pair.verified);
    QVERIFY(pair.inliers > 0);
    QVERIFY(!readScene(project.sparse).points.isEmpty());
  }
  void projectRoundTrip() {
    QTemporaryDir dir;
    Project p;
    p.name = QString::fromUtf8("中文 项目");
    p.imageRoot = dir.filePath("原图 空格");
    p.workspace = dir.filePath("run/workspace");
    p.parameters = {{"SiftExtraction.max_num_features", "1024"}};
    const auto path = dir.filePath("工程.colite.json");
    p.save(path);
    const auto q = Project::load(path);
    QCOMPARE(q.name, p.name);
    QCOMPARE(q.workspace, p.workspace);
    QCOMPARE(q.imageRoot, p.imageRoot);
    QCOMPARE(q.parameters, p.parameters);
    QFile original(path);
    QVERIFY(original.open(QIODevice::ReadOnly));
    const auto saved = original.readAll();
    original.close();
    auto invalid = q;
    invalid.name.clear();
    QVERIFY_EXCEPTION_THROWN(invalid.save(path), std::runtime_error);
    QVERIFY(original.open(QIODevice::ReadOnly));
    QCOMPARE(original.readAll(), saved);
    original.close();
    write(path, "{\"format\":\"colite-project\",\"schema\":2}");
    QVERIFY_EXCEPTION_THROWN(Project::load(path), std::runtime_error);
    QVERIFY(QFile(path).exists());
  }
  void featureIntegrity() {
    QTemporaryDir dir;
    QDir().mkpath(dir.filePath("features"));
    Workspace w;
    w.root = dir.path();
    const auto path = dir.filePath("features/7.bin");
    auto bytes = feature();
    write(path, bytes);
    const auto points = w.features(7);
    QCOMPARE(points.size(), 1);
    QCOMPARE(points[0].orientation, 4.f);
    bytes[bytes.size() - 1] = char(bytes.at(bytes.size() - 1) ^ 1);
    write(path, bytes);
    QVERIFY_EXCEPTION_THROWN(w.features(7), std::runtime_error);
    write(path, feature().left(70));
    QVERIFY_EXCEPTION_THROWN(w.features(7), std::runtime_error);
  }
  void pairSemantics() {
    QTemporaryDir dir;
    QDir().mkpath(dir.filePath("matches"));
    Workspace w;
    w.root = dir.path();
    write(dir.filePath("matches/raw_matches.tsv"), "# ids\n2\t7\t0\t1\n");
    auto pair = w.pair(7, 2, 2, 1);
    QCOMPARE(pair.matches.size(), 1);
    QCOMPARE(pair.matches[0].a, quint32(1));
    QVERIFY(!pair.matches[0].verified);
    QVERIFY(!pair.matches[0].inlier);
    QVERIFY_EXCEPTION_THROWN(w.pair(7, 2, 1, 1), std::runtime_error);
    write(dir.filePath("matches/verified_matches.tsv"),
          "2\t7\t0\t0\t1\t0\t0\t0\t0\t0\t0\t0\t1\n");
    QVERIFY_EXCEPTION_THROWN(w.pair(7, 2, 2, 1), std::runtime_error);
  }
  void sceneParser() {
    QTemporaryDir dir;
    const auto path = dir.filePath("mesh.ply");
    write(path,
          "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty "
          "float y\nproperty float z\nelement face 1\nproperty list uchar int "
          "vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n");
    const auto scene = readScene(path);
    QCOMPARE(scene.points.size(), 3);
    QCOMPARE(scene.triangles.size(), 3);
    write(path,
          "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty "
          "float y\nproperty float z\nelement face 1\nproperty list uchar int "
          "vertex_indices\nend_header\n0 0 0\n3 0 1 2\n");
    QVERIFY_EXCEPTION_THROWN(readScene(path), std::runtime_error);
  }
  void sceneProgressCancellationAndPropertyOrder() {
    QTemporaryDir dir;
    const auto path = dir.filePath("ordered.ply");
    write(path, "ply\nformat ascii 1.0\nelement vertex 1\nproperty uchar blue\n"
                "property float y\nproperty float u\nproperty float z\nproperty uchar red\n"
                "property float x\nproperty float v\nproperty uchar green\nend_header\n"
                "64 2 0.25 3 255 1 0.75 128\n");
    QStringList phases;
    const auto scene = readScene(path, [&](const QString& phase, int percent) {
      QVERIFY(percent >= -1 && percent <= 100);
      phases << phase;
    });
    QCOMPARE(scene.points.size(), 1);
    const auto p = scene.points.first();
    QCOMPARE(p.x, 1.f); QCOMPARE(p.y, 2.f); QCOMPARE(p.z, 3.f);
    QCOMPARE(p.r, 1.f); QCOMPARE(p.b, 64 / 255.f);
    QCOMPARE(p.u, .25f); QCOMPARE(p.v, .75f);
    QVERIFY(phases.contains(QString::fromUtf8("读取顶点")));
    QVERIFY_EXCEPTION_THROWN(readScene(path, {}, [] { return true; }), std::runtime_error);
    bool cancel = false;
    QVERIFY_EXCEPTION_THROWN(readScene(path, [&](const QString& phase, int) {
      if (phase == QString::fromUtf8("读取顶点")) cancel = true;
    }, [&] { return cancel; }), std::runtime_error);
  }
  void realProcessLifecycle() {
    QTemporaryDir dir;
    Project project;
    project.file = dir.filePath("fixture.colite.json");
    project.backend = QCoreApplication::applicationFilePath();
    project.assignOutputs(dir.filePath("run"));
    TaskRunner runner;
    int validated = 0;
    QVector<TaskStep> steps = {
        {"first", {"--fixture-worker", "ok"}, [&] { ++validated; }, 4},
        {"second", {"--fixture-worker", "ok"}, [&] { ++validated; }, 5}};
    runner.start(project, steps, {});
    QCOMPARE(runner.currentStage(), 4); QVERIFY(runner.containsStage(5));
    QVERIFY(!runner.completedStage(4));
    runner.pauseOrResume();
    QTRY_VERIFY_WITH_TIMEOUT(runner.state() == TaskRunner::State::Paused, 5000);
    QCOMPARE(validated, 1);
    QCOMPARE(runner.completed(), 1);
    QVERIFY(runner.completedStage(4)); QCOMPARE(runner.currentStage(), 5);
    runner.pauseOrResume();
    QTRY_VERIFY_WITH_TIMEOUT(runner.state() == TaskRunner::State::Succeeded,
                             5000);
    QCOMPARE(validated, 2);
    QFile result(runner.resultPath());
    QVERIFY(result.open(QIODevice::ReadOnly));
    const auto summary = QJsonDocument::fromJson(result.readAll()).object();
    QCOMPARE(summary["state"].toString(), QString("succeeded"));
    QCOMPARE(summary["completed"].toInt(), 2);
    TaskRunner reopened;
    QVERIFY(reopened.restoreHistory(project));
    QCOMPARE(int(reopened.state()), int(TaskRunner::State::Succeeded));
    QCOMPARE(reopened.completed(), 2);
    QVERIFY(reopened.completedStage(5)); QCOMPARE(reopened.currentStage(), 0);
    auto interrupted = summary;
    interrupted["state"] = "running";
    interrupted["completed"] = 1;
    result.close();
    const auto interruptedBytes = QJsonDocument(interrupted).toJson();
    write(runner.resultPath(), interruptedBytes);
    QVERIFY(reopened.restoreHistory(project));
    QVERIFY(!reopened.busy());
    QCOMPARE(int(reopened.state()), int(TaskRunner::State::Failed));
    QCOMPARE(reopened.currentStage(), 5);
    QFile unchanged(runner.resultPath());
    QVERIFY(unchanged.open(QIODevice::ReadOnly));
    QCOMPARE(unchanged.readAll(), interruptedBytes);
    runner.start(project, {{"failure", {"--fixture-worker", "fail"}, {}}}, {});
    QTRY_VERIFY_WITH_TIMEOUT(runner.state() == TaskRunner::State::Failed, 5000);
    runner.start(project, {{"abort", {"--fixture-worker", "wait"}, {}}}, {});
    QTest::qWait(100);
    runner.abort();
    QTRY_VERIFY_WITH_TIMEOUT(runner.state() == TaskRunner::State::Aborted,
                             5000);
  }
  void abortOwnsDescendantsAndLocksOutputs() {
    QTemporaryDir dir;
    Project project;
    project.file = dir.filePath("fixture.colite.json");
    project.workspace = dir.filePath("shared workspace");
    project.backend = QCoreApplication::applicationFilePath();
    const auto pidPath = dir.filePath("child.pid");
    TaskRunner owner, contender;
    owner.start(project, {{"tree", {"--fixture-worker", "tree", pidPath}, {}}}, {});
    QTRY_VERIFY_WITH_TIMEOUT(QFileInfo(pidPath).size() > 0, 5000);
    QFile pidFile(pidPath);
    QVERIFY(pidFile.open(QIODevice::ReadOnly));
    const auto pid = pidFile.readAll().toLongLong();
    QVERIFY(pid > 0);
    QVERIFY(processAlive(pid));
    contender.start(project, {{"conflict", {"--fixture-worker", "ok"}, {}}}, {});
    QCOMPARE(int(contender.state()), int(TaskRunner::State::Failed));
    QVERIFY(owner.busy());
    owner.abort();
    QTRY_VERIFY_WITH_TIMEOUT(!owner.busy(), 5000);
    QTRY_VERIFY_WITH_TIMEOUT(!processAlive(pid), 5000);
    contender.start(project, {{"released", {"--fixture-worker", "ok"}, {}}}, {});
    QTRY_COMPARE_WITH_TIMEOUT(int(contender.state()), int(TaskRunner::State::Succeeded), 5000);
  }
  void abortDuringValidationKeepsOutputsOwned() {
    QTemporaryDir dir;
    Project project;
    project.file = dir.filePath("validation.colite.json");
    project.workspace = dir.filePath("shared outputs");
    project.backend = QCoreApplication::applicationFilePath();
    QSemaphore entered, release;
    TaskRunner owner, contender;
    owner.start(project, {{"validate", {"--fixture-worker", "ok"}, [&] {
      entered.release();
      release.acquire();
    }}, {"must not run", {"--fixture-worker", "fail"}, {}}}, {});
    QTRY_VERIFY_WITH_TIMEOUT(entered.available() > 0, 5000);
    owner.abort();
    const bool stillOwned = owner.busy();
    contender.start(project, {{"conflict", {"--fixture-worker", "ok"}, {}}}, {});
    const bool blocked = contender.state() == TaskRunner::State::Failed;
    release.release();
    QVERIFY(stillOwned);
    QVERIFY(blocked);
    QTRY_COMPARE_WITH_TIMEOUT(int(owner.state()), int(TaskRunner::State::Aborted), 5000);
    QCOMPARE(owner.completed(), 0);
  }
};
int main(int argc, char** argv) {
  QCoreApplication app(argc, argv);
  if (!qEnvironmentVariableIsEmpty("COLITE_GUI_PIPELINE_PROJECT") &&
      qEnvironmentVariableIsEmpty("QTEST_FUNCTION_TIMEOUT")) {
    const int configured = qEnvironmentVariableIntValue("COLITE_GUI_PIPELINE_TIMEOUT_MS");
    qputenv("QTEST_FUNCTION_TIMEOUT", QByteArray::number((configured > 0 ? configured : 7200000) + 60000));
  }
  const auto args = app.arguments();
  if (args.size() > 2 && args[1] == "--fixture-worker") {
    if (args[2] == "disk-full") {
      QFile out; out.open(stderr, QIODevice::WriteOnly);
      out.write("Cannot persist model: No space left on device\n"); out.flush();
      return 8;
    }
    if (args[2] == "progress") {
      QFile out; out.open(stdout, QIODevice::WriteOnly);
      out.write("@colite.progress {\"schema\":1,\"phase\":\"features\",\"done\":1,\"total\":2}\n");
      out.write("@colite.progress {\"schema\":1,\"phase\":\"features\",\"done\":0,\"total\":2}\n");
      out.write("@colite.progress {\"schema\":1,\"phase\":\"features\",\"done\":3,\"total\":2}\n");
      out.write("@colite.progress {\"schema\":1,\"phase\":\"features\",\"done\":2,\"total\":2}\n");
      out.flush();
    }
    if (args[2] == "tree" && args.size() == 4) {
      QProcess child;
      child.start(app.applicationFilePath(), {"--fixture-worker", "wait"});
      if (!child.waitForStarted()) return 10;
      write(args[3], QByteArray::number(child.processId()));
      child.waitForFinished(20000);
      return 0;
    }
    if (args[2] == "wait")
      QThread::msleep(20000);
    else
      QThread::msleep(150);
    return args[2] == "fail" ? 9 : 0;
  }
  GuiContracts tests;
  return QTest::qExec(&tests, argc, argv);
}
#include "gui_tests.moc"
