#include "task_plan.h"
#include "parameter_schema.h"

#include <QDir>
#include <QFileInfo>
#include <cmath>
#include <stdexcept>

#include "project/workspace.h"

namespace colite::gui {
namespace {
QString tr(const char* value) { return QString::fromUtf8(value); }
const QStringList stageNames = workflowNames();
void ensureFile(const QString& path) {
  if (!QFileInfo(path).isFile() || QFileInfo(path).size() == 0)
    throw std::runtime_error(("Missing output: " + path).toStdString());
}
}
Workflow Workflow::from(int firstStage, bool all) {
  if (firstStage < 1 || firstStage > 6)
    throw std::invalid_argument("Task stage must be in the range 1..6");
  Workflow workflow;
  for (int stage = firstStage; stage <= (all ? 6 : firstStage); ++stage)
    workflow.stages.append(static_cast<Stage>(stage));
  return workflow;
}
QString stageTitle(Stage stage) { return stageNames.at(static_cast<int>(stage)); }
QVector<TaskStep> buildTaskPlan(const Project& project, int firstStage, bool all) {
  return cliSteps(project, Workflow::from(firstStage, all));
}
QVector<TaskStep> cliSteps(const Project& project, const Workflow& workflow) {
  if (project.imageRoot.isEmpty() || project.workspace.isEmpty() ||
      project.sparse.isEmpty() || project.dense.isEmpty() || project.mesh.isEmpty())
    throw std::invalid_argument("Configure image, workspace, sparse, dense and mesh paths before running");
  auto identity = [](const QString& path) {
    const QFileInfo info(path);
    const auto resolved = info.exists() ? info.canonicalFilePath() : info.absoluteFilePath();
    const auto normalized = QDir::fromNativeSeparators(QDir::cleanPath(resolved));
#ifdef Q_OS_WIN
    return normalized.toCaseFolded();
#else
    return normalized;
#endif
  };
  const auto images = identity(project.imageRoot);
  for (const auto& path : {project.workspace, project.sparse, project.dense,
                           QFileInfo(project.mesh).absolutePath()}) {
    const auto output = identity(path);
    if (QDir(output).isRoot() || output == images ||
        output.startsWith(images + '/') || images.startsWith(output + '/'))
      throw std::invalid_argument("Output directories must be separate from the source image tree: " + path.toStdString());
  }
  QVector<TaskStep> steps;
  auto p = project;
  p.parameters = effectiveParameters(project.parameters);
  const auto cachedSparse = QDir(p.workspace).absoluteFilePath("sparse");
  const bool exportSparse = QDir::cleanPath(QFileInfo(p.sparse).absoluteFilePath())
      .compare(QDir::cleanPath(cachedSparse),
#ifdef Q_OS_WIN
               Qt::CaseInsensitive
#else
               Qt::CaseSensitive
#endif
               ) != 0;
  auto params = [&](const QString& prefix) {
    QStringList args;
    for (auto it = p.parameters.begin(); it != p.parameters.end(); ++it) {
      if (!it.key().startsWith(prefix)) continue;
      // These are derived once from the selected algorithm and device below.
      if (it.key() == "FeatureMatching.type" || it.key() == "FeatureMatching.use_gpu" ||
          it.key() == "FeatureExtraction.use_gpu") continue;
      QString value;
      if (it.value().isString())
        value = it.value().toString();
      else if (it.value().isBool())
        value = it.value().toBool() ? "1" : "0";
      else if (it.value().isDouble() && std::isfinite(it.value().toDouble()))
        value = QString::number(it.value().toDouble(), 'g', 17);
      else
        throw std::invalid_argument(("Invalid parameter: " + it.key()).toStdString());
      const auto flag = prefix == "Undistortion."
                            ? it.key().mid(prefix.size()) : it.key();
      args << "--" + flag << value;
    }
    return args;
  };
  int currentStage = 0;
  auto add = [&](const QString& title, QStringList args,
                 std::function<void()> validate) {
    steps.push_back({title, args, std::move(validate), currentStage});
  };
  QStringList cameraArgs{
      "--ImageReader.camera_model",
      p.parameters["camera_model"].toString("SIMPLE_RADIAL")};
  const auto cameraStrategy = p.parameters["camera_strategy"].toString("Auto");
  if (!QStringList{"Auto", "Single", "Per-folder", "Per-image"}.contains(cameraStrategy))
    throw std::invalid_argument("Unsupported camera strategy: " + cameraStrategy.toStdString());
  if (cameraStrategy == "Single")
    cameraArgs << "--ImageReader.single_camera" << "1";
  else if (cameraStrategy == "Per-folder")
    cameraArgs << "--ImageReader.single_camera_per_folder" << "1";
  else if (cameraStrategy == "Per-image")
    cameraArgs << "--ImageReader.single_camera_per_image" << "1";
  for (auto operation : workflow.stages) {
    const int stage = static_cast<int>(operation);
    currentStage = stage;
    if (stage == 1)
      add(stageNames[stage],
          QStringList{"feature_extractor", "--image_path", p.imageRoot,
                      "--Colite.workspace_path", p.workspace} +
              params("SiftExtraction.") + params("FeatureExtraction.") + params("AlikedExtraction.") +
              cameraArgs + params("ImageReader.") +
              (p.parameters["FeatureExtraction.type"].toString() == "sift"
                ? QStringList{"--descriptor_normalization", p.parameters["descriptor_normalization"].toString(),
                    "--Colite.feature_extractor", p.parameters["Colite.feature_extractor"].toString() != "auto"
                        ? p.parameters["Colite.feature_extractor"].toString()
                        : p.parameters["device"].toString() == "cpu" ? "vlsift"
                        : p.parameters["device"].toString() == "cuda" ? "cuda_vlfeat" : "auto"}
                : p.parameters["learned_extraction_device"].toString() == "auto" ? QStringList{} :
                    QStringList{"--FeatureExtraction.use_gpu", p.parameters["learned_extraction_device"].toString() == "cpu" ? "0" : "1"}),
          [p] {
            const auto w = Workspace::read(p);
            if (w.images.isEmpty())
              throw std::runtime_error("No extracted images");
            for (const auto& i : w.images) {
              ensureFile(
                  QDir(w.root).filePath(QString("features/%1.bin").arg(i.id)));
              w.features(i.id);
            }
          });
    if (stage == 2)
      add(stageNames[stage],
          QStringList{
              p.parameters["matching_strategy"].toString("exhaustive_matcher"),
              "--Colite.workspace_path", p.workspace} +
              params("SiftMatching.") + params("FeatureMatching.") + params("AlikedMatching.") + params("TwoViewGeometry.") +
              params("SequentialMatching.") + params("SpatialMatching.") +
              QStringList{"--FeatureMatching.type", (p.parameters["FeatureExtraction.type"].toString() == "sift" ? QString("SIFT_") : QString("ALIKED_")) +
                  (p.parameters["matching_method"].toString() == "lightglue" ? "LIGHTGLUE" : "BRUTEFORCE")} +
              ((p.parameters["FeatureExtraction.type"].toString() != "sift" && p.parameters["matching_method"].toString() == "bruteforce") ||
                  p.parameters[p.parameters["matching_method"].toString() == "lightglue" ? "learned_matching_device" : "device"].toString("auto") == "cpu"
                   ? QStringList{"--FeatureMatching.use_gpu", "0"}
                   : p.parameters[p.parameters["matching_method"].toString() == "lightglue" ? "learned_matching_device" : "device"].toString() == "cuda" ? QStringList{"--FeatureMatching.use_gpu", "1"} : QStringList{}),
          [p] {
            ensureFile(QDir(p.workspace).filePath("matches/raw_matches.tsv"));
            ensureFile(
                QDir(p.workspace).filePath("matches/verified_matches.tsv"));
          });
    if (stage == 3)
      add(stageNames[stage],
          QStringList{"mapper", "--Colite.workspace_path", p.workspace,
                      "--image_path", p.imageRoot} +
              params("Mapper.") + params("GlobalMapper.") + QStringList{"--Colite.mapper", p.parameters["mapper_method"].toString()},
          [p] {
            if (!validateScene(QDir(p.workspace).filePath("sparse")).points)
              throw std::runtime_error("Sparse model has no points");
          });
    if (stage == 3 && exportSparse)
      add(tr("导出稀疏主模型"),
          {"model_converter", "--input_path", cachedSparse,
           "--output_path", p.sparse, "--output_type", "both"},
          [p] {
            if (!validateScene(p.sparse).points)
              throw std::runtime_error("Exported sparse model has no points");
          });
    const QStringList resume = p.parameters["resume"].toBool(true)
                                   ? QStringList{"--resume"}
                                   : QStringList{};
    if (stage == 4) {
      auto stereoParams = params("PatchMatchStereo.");
      const int size = stereoParams.indexOf("--PatchMatchStereo.max_image_size");
      if (size >= 0 && stereoParams[size + 1] == "50%") {
        stereoParams.removeAt(size + 1);
        stereoParams.removeAt(size);
        stereoParams << "--Colite.resolution_level" << "1";
      }
      const auto consistency = stereoParams.indexOf("--PatchMatchStereo.geom_consistency");
      const bool geometric = consistency < 0 || stereoParams[consistency + 1] != "0";
      add(tr("图像去畸变"),
          QStringList{"image_undistorter", "--input_path", p.activeSparse(),
                      "--image_path", p.imageRoot, "--output_path", p.dense} +
              params("Undistortion.") +
              resume,
          [p] {
            ensureFile(QDir(p.dense).filePath("stereo/patch-match.cfg"));
          });
      add(tr("PatchMatch Stereo"),
          QStringList{"patch_match_stereo", "--workspace_path", p.dense} +
              stereoParams +
              QStringList{"--Colite.patch_match_backend",
                          p.parameters["device"].toString("auto")} +
              resume,
          [p] {
            ensureFile(QDir(p.dense).filePath(".colite/photometric_cache.txt"));
          });
      add(tr("点云融合"),
          QStringList{"stereo_fusion", "--workspace_path", p.dense,
                      "--output_path", QDir(p.dense).filePath("fused.ply"),
                      "--input_type", geometric ? "geometric" : "photometric"} +
              params("StereoFusion.") +
              resume,
          [p] {
            if (!validateScene(QDir(p.dense).filePath("fused.ply")).points)
              throw std::runtime_error("Fused cloud is empty");
          });
    }
    if (stage == 5) {
      const bool simplify = p.parameters["simplify_mesh"].toString() == "1";
      const auto rawMesh = simplify ? QFileInfo(p.mesh).dir().filePath(QFileInfo(p.mesh).completeBaseName() + "-full.ply") : p.mesh;
      const bool delaunay = p.parameters["surface_method"].toString() == "delaunay";
      add(stageNames[stage],
          (delaunay ? QStringList{"delaunay_mesher", "--input_path", p.dense, "--input_type", "dense", "--output_path", rawMesh}
                    : QStringList{"poisson_mesher", "--input_path", QDir(p.dense).filePath("fused.ply"), "--output_path", rawMesh}) +
              (delaunay ? params("DelaunayMeshing.") : params("PoissonMeshing.")) + resume,
          [rawMesh] {
            if (!validateScene(rawMesh).triangles)
              throw std::runtime_error("Mesh has no triangles");
          });
      if (simplify) add(tr("网格简化"), QStringList{"mesh_simplifier", "--input_path", rawMesh, "--output_path", p.mesh} + params("MeshSimplification.") + resume,
          [p] { if (!validateScene(p.mesh).triangles) throw std::runtime_error("Simplified mesh has no triangles"); });
    }
    if (stage == 6) {
      const auto output = QFileInfo(p.mesh).dir().filePath("textured");
      add(stageNames[stage],
          QStringList{"mesh_texturer", "--workspace_path", p.dense,
                      "--input_path", p.mesh, "--output_path", output} +
              params("MeshTextureMapping.") + params("Colite.texture_") + resume,
          [output] {
            const auto result = validateScene(QDir(output).filePath("mesh.ply"));
            if (!result.triangles || result.texture.isEmpty())
              throw std::runtime_error("Textured mesh lacks triangles or texture reference");
          });
    }
  }
  return steps;
}
}  // namespace colite::gui
