#include <QtTest>
#include <algorithm>

#include "widgets/feature_view.h"
#include "app/main_window.h"
#include "widgets/parameter_dialog.h"
#include "workflow/task_plan.h"
#include "workflow/workflow_state.h"
#include "widgets/ui_style.h"
using namespace colite::gui;
class InspectionTests : public QObject {
  Q_OBJECT
 private slots:
  void recentProjectsRetainFivePathsAcrossWindows() {
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QStringList paths;
    for (int i = 0; i < 6; ++i) {
      Project project;
      project.name = QStringLiteral("House %1").arg(i);
      const auto folder = dir.filePath(QString::number(i));
      QVERIFY(QDir().mkpath(folder));
      project.save(QDir(folder).filePath("House.colite.json"));
      paths << project.file;
    }
    {
      MainWindow window;
      for (const auto& path : paths) window.openProject(path);
    }
    MainWindow reopened;
    auto* menu = reopened.findChild<QMenu*>("recentProjects");
    QVERIFY(menu);
    QVERIFY(QMetaObject::invokeMethod(menu, "aboutToShow"));
    QCOMPARE(menu->actions().size(), 5);
    QCOMPARE(menu->actions().first()->data().toString(), paths[5]);
    QCOMPARE(menu->actions().last()->data().toString(), paths[1]);
    QVERIFY(menu->actions().first()->text().contains(QDir::toNativeSeparators(paths[5])));
    menu->actions().last()->trigger();
    QVERIFY(reopened.windowTitle().contains("House 1"));
    QVERIFY(QFile::remove(paths[2]));
    QVERIFY(QMetaObject::invokeMethod(menu, "aboutToShow"));
    QCOMPARE(menu->actions().first()->data().toString(), paths[1]);
    for (const auto* action : menu->actions())
      if (action->data().toString() == paths[2]) QVERIFY(!action->isEnabled());
    menu->actions().last()->trigger(); // Remove missing entries.
    QVERIFY(QMetaObject::invokeMethod(menu, "aboutToShow"));
    QCOMPARE(menu->actions().size(), 4);
  }
  void algorithmPagesSwitchFieldsAndKeepAdvancedCollapsed() {
    Project p;
    ParameterDialog dialog(p, 3, false, true); dialog.show();
    auto* method = dialog.findChild<QComboBox*>("mapper_method");
    auto* advanced = dialog.findChild<QToolButton*>("advancedParameters3");
    auto* initial = dialog.findChild<QLineEdit*>("Mapper.init_min_num_inliers");
    auto* global = dialog.findChild<QLineEdit*>("GlobalMapper.min_num_inliers");
    QVERIFY(!advanced->isChecked()); QVERIFY(!initial->isVisible());
    advanced->click(); QVERIFY(initial->isVisible()); QVERIFY(!global->isVisible());
    method->setCurrentIndex(method->findData("global"));
    QVERIFY(!initial->isVisible()); QVERIFY(global->isVisible());
    const auto capture = qEnvironmentVariable("COLITE_GUI_PARAMETER_CAPTURE");
    if (!capture.isEmpty()) {
      QFile theme(":/ui/theme.qss");
      QVERIFY(theme.open(QIODevice::ReadOnly));
      dialog.setStyleSheet(QString::fromUtf8(theme.readAll()));
      dialog.resize(880, 680);
      QVERIFY(QDir().mkpath(capture));
      QTest::qWait(50);
      QVERIFY(dialog.grab().save(QDir(capture).filePath("global-parameters.png")));
      dialog.findChild<QListWidget*>("parameterStages")->setCurrentRow(1);
      auto* feature = dialog.findChild<QComboBox*>("FeatureExtraction.type");
      feature->setCurrentIndex(feature->findData("aliked_n16rot"));
      QTest::qWait(50);
      QVERIFY(dialog.grab().save(QDir(capture).filePath("feature-parameters.png")));
      feature->setCurrentIndex(feature->findData("sift"));
      dialog.findChild<QListWidget*>("parameterStages")->setCurrentRow(3);
    }
    method->setCurrentIndex(method->findData("incremental"));
    QVERIFY(initial->isVisible()); QVERIFY(!global->isVisible());
    QCOMPARE(dialog.findChild<QLineEdit*>("PoissonMeshing.depth")->text(), QString("9"));
    QCOMPARE(dialog.findChild<QLineEdit*>("PatchMatchStereo.max_image_size")->text(), QString("50%"));
  }
  void parameterPagesPreserveUntouchedValuesAndCancel() {
    Project project;
    project.parameters = {{"PoissonMeshing.depth", 9}, {"Mapper.ba_refine_focal_length", true},
                          {"device", "cuda"}, {"custom_metadata", "keep"}};
    ParameterDialog unchanged(project, 1, false, true);
    unchanged.show();
    auto* navigation = unchanged.findChild<QListWidget*>("parameterStages");
    QCOMPARE(navigation->count(), 7);
    for (int page = 0; page < 7; ++page) navigation->setCurrentRow(page);
    unchanged.findChild<QPushButton*>("saveParameters")->click();
    QCOMPARE(unchanged.result(), int(QDialog::Accepted));
    QCOMPARE(unchanged.parameters(), project.parameters);
    QCOMPARE(unchanged.project().workDirectory, project.workDirectory);
    QCOMPARE(unchanged.project().workspace, project.workspace);
    QVERIFY(!unchanged.runRequested());

    ParameterDialog cancelled(project, 5, true, true);
    cancelled.findChild<QLineEdit*>("PoissonMeshing.depth")->setText("8");
    cancelled.reject();
    QCOMPARE(cancelled.parameters(), project.parameters);

    ParameterDialog edited(project, 5, true, true);
    edited.show();
    QVERIFY(edited.findChild<QListWidget*>("parameterStages")->isHidden());
    QVERIFY(edited.findChild<QCheckBox*>("followStages")->isHidden());
    auto* depth = edited.findChild<QLineEdit*>("PoissonMeshing.depth");
    auto* run = edited.findChild<QPushButton*>("runStage");
    QCOMPARE(run->text(), QString::fromUtf8("运行节点"));
    depth->setText("1.5"); run->click();
    QCOMPARE(edited.result(), int(QDialog::Rejected));
    QVERIFY(!edited.findChild<QLabel*>("parameterError")->isHidden());
    depth->setText("8"); run->click();
    QVERIFY(edited.runRequested()); QVERIFY(!edited.followStages());
    QCOMPARE(edited.selectedStage(), 5);
    auto expected = project.parameters; expected["PoissonMeshing.depth"] = "8";
    QCOMPARE(edited.parameters(), expected);
    project.parameters = edited.parameters();
    QTemporaryDir dir;
    project.imageRoot = dir.filePath("photos");
    project.assignOutputs(dir.filePath("results"));
    const auto plan = cliSteps(project, Workflow::from(edited.selectedStage(), edited.followStages()));
    QCOMPARE(plan.size(), 1); QCOMPARE(plan.first().stage, 5);
    QCOMPARE(plan.first().arguments.first(), QString("poisson_mesher"));
    const auto args = plan.first().arguments;
    QCOMPARE(args[args.indexOf("--PoissonMeshing.depth") + 1], QString("8"));
  }
  void globalWorkDirectoryIsTransactionalAndIndependentOfPages() {
    QTemporaryDir dir;
    Project project;
    project.imageRoot = dir.filePath("photos");
    QVERIFY(QDir().mkpath(project.imageRoot));
    project.assignOutputs(dir.filePath("old"));
    project.parameters["sparse_model"] = "2";
    QVERIFY(QDir().mkpath(QFileInfo(project.mesh).absolutePath()));
    QFile oldMesh(project.mesh); QVERIFY(oldMesh.open(QIODevice::WriteOnly));
    oldMesh.write("preserved"); oldMesh.close();
    ParameterDialog dialog(project, 5, false, true);
    dialog.show();
    auto* path = dialog.findChild<QLineEdit*>("parameterWorkDirectory");
    auto* pages = dialog.findChild<QListWidget*>("parameterStages");
    auto* run = dialog.findChild<QPushButton*>("runStage");
    for (int i = 0; i < 7; ++i) {
      pages->setCurrentRow(i); QVERIFY(path->isVisible());
    }
    path->setText(dir.filePath("new"));
    QVERIFY(!run->isEnabled());
    pages->setCurrentRow(1); QVERIFY(run->isEnabled());
    QCOMPARE(run->text(), QString::fromUtf8("运行节点"));
    auto* save = dialog.findChild<QPushButton*>("saveParameters");
    path->setText("relative-output"); save->click();
    QCOMPARE(dialog.result(), int(QDialog::Rejected));
    path->setText(project.imageRoot + "/output"); save->click();
    QCOMPARE(dialog.result(), int(QDialog::Rejected));
    path->setText(project.mesh + "/output"); save->click();
    QCOMPARE(dialog.result(), int(QDialog::Rejected));
    path->setText(dir.filePath("new")); save->click();
    QCOMPARE(dialog.result(), int(QDialog::Accepted));
    const auto changed = dialog.project();
    QCOMPARE(changed.workDirectory, dir.filePath("new"));
    QVERIFY(changed.workspace.startsWith(changed.workDirectory + "/runs/"));
    QVERIFY(changed.mesh != project.mesh);
    QVERIFY(!changed.parameters.contains("sparse_model"));
    QCOMPARE(changed.imageRoot, project.imageRoot);
    QVERIFY(QFileInfo::exists(project.mesh));
    QVERIFY(!QFileInfo::exists(changed.workDirectory)); // Saving paths does not create/move data.
    ParameterDialog cancel(project, 3, false, true);
    cancel.findChild<QLineEdit*>("parameterWorkDirectory")->setText(dir.filePath("cancelled"));
    cancel.reject();
    QCOMPARE(cancel.project().workspace, project.workspace);
    QCOMPARE(cancel.project().workDirectory, project.workDirectory);
    ParameterDialog same(project, 3, false, true);
    same.findChild<QLineEdit*>("parameterWorkDirectory")->setText(project.workDirectory + "/.");
    same.findChild<QPushButton*>("saveParameters")->click();
    QCOMPARE(same.project().workspace, project.workspace);
  }
  void workDirectoryChangesPersistThroughMainWindow() {
    QTemporaryDir dir;
    Project project; project.name = "Directory regression";
    project.imageRoot = dir.filePath("photos");
    QVERIFY(QDir().mkpath(project.imageRoot));
    QImage photo(32, 32, QImage::Format_RGB32); photo.fill(Qt::green);
    QVERIFY(photo.save(project.imageRoot + "/photo.png"));
    project.assignOutputs(dir.filePath("old"));
    project.save(dir.filePath("project.colite.json"));
    MainWindow window; window.openProject(project.file); window.show();
    QTRY_COMPARE(window.findChild<QListWidget*>("photos")->count(), 1);
    QTimer::singleShot(0, &window, [&] {
      auto* dialog = dynamic_cast<ParameterDialog*>(QApplication::activeModalWidget());
      QVERIFY(dialog);
      dialog->findChild<QLineEdit*>("parameterWorkDirectory")->setText(dir.filePath("new"));
      dialog->findChild<QPushButton*>("saveParameters")->click();
    });
    window.findChild<QPushButton*>("workflowParameters")->click();
    QVERIFY(window.windowTitle().contains('*'));
    QTRY_COMPARE(window.findChild<QListWidget*>("photos")->count(), 1);
    for (auto* action : window.findChildren<QAction*>())
      if (action->shortcut() == QKeySequence::Save) { action->trigger(); break; }
    const auto saved = Project::load(project.file);
    QCOMPARE(saved.workDirectory, dir.filePath("new"));
    QVERIFY(saved.workspace != project.workspace);
    QCOMPARE(saved.imageRoot, project.imageRoot);
    QVERIFY(!QFileInfo::exists(saved.workDirectory));
  }
  void workspacePanelsAndActiveStageSettings() {
    MainWindow window; window.resize(1280, 800); window.show();
    auto* right = window.findChild<QTabWidget*>("rightTabs");
    auto* log = window.findChild<QPlainTextEdit*>("taskLog");
    QVERIFY(right && log);
    QCOMPARE(right->tabText(0), QString::fromUtf8("照片"));
    QCOMPARE(right->tabText(1), QString::fromUtf8("日志"));
    right->setCurrentIndex(1);
    QTRY_VERIFY(log->height() > 300);
    QVERIFY(right->isAncestorOf(log));
    QVERIFY(!window.findChild<QTabBar*>("inspectorTabs"));
    const auto toolbars = window.findChildren<QToolBar*>();
    QStringList actions;
    for (auto* bar : toolbars) for (auto* item : bar->actions()) if (!item->isSeparator()) actions << item->text();
    QVERIFY(!actions.contains(QString::fromUtf8("加载模型")));
    QVERIFY(!actions.contains(QString::fromUtf8("加载 PLY")));
    QCOMPARE(actions.mid(actions.size() - 3), QStringList({QString::fromUtf8("3D视图"), QString::fromUtf8("2D视图"), QString::fromUtf8("适配视图")}));
    for (int stage = 0; stage <= 6; ++stage) {
      window.navigate(stage);
      QVERIFY(!window.findChild<QToolButton*>("configureStage" + QString::number(stage)));
      QVERIFY(window.findChild<QPushButton*>("stage" + QString::number(stage))->isChecked());
      QTimer::singleShot(0, &window, [stage] {
        auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
        QVERIFY(dialog);
        QCOMPARE(dialog->findChild<QListWidget*>("parameterStages")->currentRow(), stage);
        dialog->reject();
      });
      window.findChild<QAction*>("configureStage" + QString::number(stage))->trigger();
    }
  }
  void singleStageSurfaceTuningIntegration() {
    const auto source = qEnvironmentVariable("COLITE_GUI_SURFACE_TUNING_PROJECT");
    if (source.isEmpty()) QSKIP("Set COLITE_GUI_SURFACE_TUNING_PROJECT to run isolated House surface tuning");
    // Only read the completed dataset. All generated surfaces, receipts and
    // logs belong to this temporary project; never rerun the review project.
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const auto original = Project::load(source);
    Project project;
    project.name = "Surface tuning integration";
    project.imageRoot = original.imageRoot;
    project.assignOutputs(dir.filePath("outputs"));
    project.dense = original.dense; project.sparse = original.sparse;
    project.parameters = original.parameters;
    // Bound OpenMP parallelism for this regression; host-wide defaults can
    // oversubscribe WSL while other local validation is running.
    project.parameters["PoissonMeshing.num_threads"] = "8";
    project.parameters["MeshTextureMapping.num_threads"] = "8";
    project.save(dir.filePath("tuning.colite.json"));
    MainWindow window; window.openProject(project.file); window.show();
    QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QPushButton*>("primary")->isEnabled(), 30000);
    for (const int depth : {8, 7}) {
      window.navigate(5);
      QTimer::singleShot(0, &window, [depth] {
        auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget()); QVERIFY(dialog);
        QCOMPARE(dialog->findChild<QPushButton*>("runStage")->text(), depth == 8
            ? QString::fromUtf8("运行节点") : QString::fromUtf8("重新运行节点"));
        dialog->findChild<QLineEdit*>("PoissonMeshing.depth")->setText(QString::number(depth));
        dialog->findChild<QPushButton*>("runStage")->click();
      });
      window.findChild<QAction*>("configureStage5")->trigger();
      auto expected = project; expected.parameters["PoissonMeshing.depth"] = QString::number(depth);
      auto failed = [&] { return window.findChild<QLabel*>("statusState")->text().contains(QString::fromUtf8("未完成")); };
      QTRY_VERIFY_WITH_TIMEOUT(outputState(expected, 5) == OutputState::Current || failed(), 120000);
      QVERIFY2(!failed(), qPrintable(window.findChild<QPlainTextEdit*>("taskLog")->toPlainText()));
      QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QPushButton*>("workflowParameters")->isEnabled(), 30000);
      QCOMPARE(window.findChild<QTabWidget*>("rightTabs")->currentIndex(), 1);
      QVERIFY(window.findChild<QPlainTextEdit*>("taskLog")->toPlainText().contains("poisson_mesher"));
      const auto paths = QDir(project.historyRoot()).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
      QCOMPARE(paths.size(), depth == 8 ? 1 : 2);
      for (const auto& path : paths) {
        QFile request(QDir(project.historyRoot()).filePath(path + "/request.json"));
        QVERIFY(request.open(QIODevice::ReadOnly));
        const auto steps = QJsonDocument::fromJson(request.readAll()).object()["steps"].toArray();
        QCOMPARE(steps.size(), 1);
        QCOMPARE(steps[0].toObject()["arguments"].toArray()[0].toString(), QString("poisson_mesher"));
      }
      qInfo() << "Surface tuning depth" << depth << "passed; only selected stage executed";
    }
    // The primary action skips configuration and continues to texture, using
    // the most recently edited depth (7), not the original project value (9).
    window.navigate(5);
    bool unexpectedDialog = false;
    QTimer dismiss;
    connect(&dismiss, &QTimer::timeout, this, [&] {
      if (auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget())) {
        unexpectedDialog = true; dialog->reject();
      }
    });
    dismiss.start(20);
    window.findChild<QPushButton*>("primary")->click();
    QVERIFY(!unexpectedDialog);
    auto expected = project; expected.parameters["PoissonMeshing.depth"] = "7";
    auto failed = [&] { return window.findChild<QLabel*>("statusState")->text().contains(QString::fromUtf8("未完成")); };
    QTRY_VERIFY_WITH_TIMEOUT(outputState(expected, 6) == OutputState::Current || failed(), 120000);
    dismiss.stop(); QVERIFY(!unexpectedDialog);
    QVERIFY2(!failed(), qPrintable(window.findChild<QPlainTextEdit*>("taskLog")->toPlainText()));
    QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QPushButton*>("workflowParameters")->isEnabled(), 30000);
    bool fullPlan = false;
    for (const auto& path : QDir(project.historyRoot()).entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
      QFile request(QDir(project.historyRoot()).filePath(path + "/request.json"));
      QVERIFY(request.open(QIODevice::ReadOnly));
      const auto steps = QJsonDocument::fromJson(request.readAll()).object()["steps"].toArray();
      if (steps.size() == 2) {
        fullPlan = true;
        const auto args = steps[0].toObject()["arguments"].toArray().toVariantList();
        QCOMPARE(args[args.indexOf("--PoissonMeshing.depth") + 1].toString(), QString("7"));
        QCOMPARE(steps[1].toObject()["arguments"].toArray()[0].toString(), QString("mesh_texturer"));
      }
    }
    QVERIFY(fullPlan);
    QTRY_COMPARE(window.findChild<QPushButton*>("stage6")->property("statusText").toString(), QString::fromUtf8("已完成"));
    qInfo() << "Direct start continued surface to texture with updated parameters and no configuration dialog";
  }
  void featureDisplayDefaultsAndModeMemory() {
    FeatureView view;
    auto* canvas = static_cast<InspectionCanvas*>(view.findChild<QWidget*>("inspectionCanvas"));
    view.setMode(false); QCOMPARE(canvas->limit, 10000);
    view.setMode(true); QCOMPARE(canvas->limit, 2000);
    auto* slider = view.findChild<QSlider*>(); QVERIFY(slider);
    slider->setValue(900);
    view.setMode(false); QCOMPARE(canvas->limit, 10000);
    view.setMode(true); QCOMPARE(canvas->limit, 900);
  }
  void outputDefaultsPreserveExplicitAndExistingProjects() {
    Project fresh;
    fresh.ensureOutputs();
    QCOMPARE(fresh.workDirectory, Project::defaultWorkDirectory());
    QVERIFY(fresh.workspace.startsWith(Project::defaultWorkDirectory() + "/runs/"));
    const auto previous = fresh.workspace;
    fresh.ensureOutputs(); QCOMPARE(fresh.workspace, previous);
    QTemporaryDir dir;
    Project custom; custom.workDirectory = dir.filePath("custom"); custom.ensureOutputs();
    QVERIFY(custom.workspace.startsWith(custom.workDirectory + "/runs/"));
    Project another; another.workDirectory = custom.workDirectory; another.ensureOutputs();
    QVERIFY(another.workspace != custom.workspace);
    Project legacy; legacy.workspace = dir.filePath("legacy"); legacy.mesh = dir.filePath("old.ply");
    legacy.ensureOutputs(); QCOMPARE(legacy.workspace, dir.filePath("legacy"));
    QCOMPARE(legacy.mesh, dir.filePath("old.ply")); QVERIFY(legacy.workDirectory.isEmpty());
    Project overlapping; overlapping.imageRoot = dir.path(); overlapping.workDirectory = dir.filePath("inside");
    QVERIFY_EXCEPTION_THROWN(overlapping.ensureOutputs(), std::runtime_error);
  }
  void nativeSurfaceKeepsAppearanceDuringNavigation() {
    if (!qEnvironmentVariableIsSet("COLITE_GUI_NATIVE_FRAMES")) QSKIP("Native OpenGL context required");
    SceneView view;
    view.resize(600, 420); view.setGrid(false); view.setCameras(false); view.show();
    for (const bool textured : {false, true}) {
      SceneData data; data.source = "navigation-surface";
      // Cross the former 60,000-triangle cutoff. A textured PLY can contain
      // no vertex colours; only its UV-mapped image defines its appearance.
      constexpr int side = 180;
      for (int y = 0; y <= side; ++y)
        for (int x = 0; x <= side; ++x)
          data.points << SceneVertex{2.f * x / side - 1, 2.f * y / side - 1, 0,
                                     1, 0, 0, float(x) / side, float(y) / side};
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
          const int a = y * (side + 1) + x, b = a + side + 1;
          for (const int index : {a, a + 1, b, a + 1, b + 1, b})
            data.triangles << data.points[index];
        }
      if (textured) {
        data.texturePreview = QImage(8, 8, QImage::Format_RGB32);
        data.texturePreview.fill(Qt::green);
      }
      SceneView::prepare(data);
      QSignalSpy presented(&view, &SceneView::firstFramePresented);
      view.setScene(data);
      QTRY_VERIFY_WITH_TIMEOUT(!presented.isEmpty(), 5000);
      QVERIFY2(view.graphicsError().isEmpty(), qPrintable(view.graphicsError()));
      const auto center = view.rect().center();
      const auto colour = view.grabFramebuffer().pixelColor(center);
      QVERIFY(textured ? colour.green() > 200 && colour.red() < 30
                       : colour.red() > 200 && colour.green() < 30);
      for (const auto button : {Qt::LeftButton, Qt::RightButton, Qt::NoButton}) {
        QTest::mousePress(&view, Qt::LeftButton, Qt::NoModifier, center);
        if (button == Qt::NoButton) {
          QWheelEvent wheel(QPointF(center), QPointF(view.mapToGlobal(center)),
                            QPoint(), QPoint(0, 120), Qt::NoButton,
                            Qt::NoModifier, Qt::NoScrollPhase, false);
          QApplication::sendEvent(&view, &wheel);
        } else {
          QMouseEvent move(QEvent::MouseMove, center + QPoint(12, 6),
                           Qt::NoButton, button, Qt::NoModifier);
          QApplication::sendEvent(&view, &move);
        }
        const auto moving = view.grabFramebuffer();
        QTest::mouseRelease(&view, Qt::LeftButton, Qt::NoModifier, center + QPoint(12, 6));
        QTest::qWait(180);
        QCOMPARE(moving, view.grabFramebuffer());
      }
    }
  }
  void completedProjectKeepsValidStageIndicators() {
    const auto path = qEnvironmentVariable("COLITE_GUI_ARTIFACT_PROJECT");
    if (path.isEmpty()) QSKIP("Set COLITE_GUI_ARTIFACT_PROJECT for completed project reopen validation");
    MainWindow window;
    window.openProject(path);
    for (int stage = 1; stage <= 6; ++stage) {
      auto* button = window.findChild<QPushButton*>("stage" + QString::number(stage));
      QVERIFY(button);
      QTRY_VERIFY_WITH_TIMEOUT(button->property("hasOutput").toBool(), 30000);
      QVERIFY2(!button->property("outputStale").toBool(), qPrintable(button->toolTip()));
      QVERIFY2(!button->property("outputUnverified").toBool(), qPrintable(button->toolTip()));
      QCOMPARE(button->property("statusText").toString(), QString::fromUtf8("已完成"));
    }
  }
  void nativeGpuCacheTracksIdentityAndContext() {
    if (!qEnvironmentVariableIsSet("COLITE_GUI_NATIVE_FRAMES")) QSKIP("Native OpenGL context required");
    QWidget firstHost, secondHost;
    auto* firstLayout = new QVBoxLayout(&firstHost);
    auto* secondLayout = new QVBoxLayout(&secondHost);
    SceneView view;
    firstLayout->addWidget(&view);
    firstHost.resize(600, 420); firstHost.show();
    QSignalSpy frames(&view, &SceneView::firstFramePresented);
    SceneData a; a.source = "cache-a";
    a.points = {{-1, -1, 0, 1, 0, 0}, {1, -1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0}};
    a.triangles = a.points; SceneView::prepare(a);
    SceneData b = a; b.source = "cache-b";
    for (auto& v : b.triangles) { v.r = 0; v.b = 1; }
    auto present = [&](const SceneData& data, const QString& key) {
      frames.clear(); view.setScene(data, key);
      return !frames.isEmpty() || frames.wait(5000);
    };
    QVERIFY(present(a, "a-v1"));
    QVERIFY2(view.graphicsError().isEmpty(), qPrintable(view.graphicsError()));
    QVERIFY(frames.last()[2].toLongLong() > 0);
    const auto original = view.grabFramebuffer();
    QVERIFY(present(b, "b-v1"));
    QVERIFY(original != view.grabFramebuffer());
    QVERIFY(present(a, "a-v1"));
    QCOMPARE(frames.last()[2].toLongLong(), qint64(0));
    QCOMPARE(view.grabFramebuffer(), original);
    b.source = a.source;
    QVERIFY(present(b, "a-v2"));
    QVERIFY(frames.last()[2].toLongLong() > 0);
    QVERIFY(original != view.grabFramebuffer());
    // Reparenting destroys the widget's non-shared GL context. Cached GL
    // objects must be released there and rebuilt in the replacement context.
    firstHost.hide();
    secondLayout->addWidget(&view);
    secondHost.resize(600, 420); secondHost.show();
    QVERIFY(present(a, "a-v1"));
    QVERIFY2(view.graphicsError().isEmpty(), qPrintable(view.graphicsError()));
    QVERIFY(frames.last()[2].toLongLong() > 0);
  }
  void recoverySelectsDistinctRecordsWithIdenticalLabels() {
    QTemporaryDir dir;
    const auto root = dir.filePath("recovery");
    for (int n = 1; n <= 2; ++n) {
      Project project;
      project.name = "Same name";
      project.imageRoot = dir.filePath(QString("images%1").arg(n));
      QVERIFY(QDir().mkpath(project.imageRoot));
      for (int i = 0; i < n; ++i) {
        QImage image(20, 20, QImage::Format_RGB32); image.fill(Qt::green);
        QVERIFY(image.save(QDir(project.imageRoot).filePath(QString("%1.png").arg(i))));
      }
      RecoveryStore store(root);
      store.save(project);
    }
    const auto time = QDateTime::fromMSecsSinceEpoch(1700000000000LL);
    for (const auto& entry : QDir(root).entryInfoList({"*.colite.json"}, QDir::Files)) {
      QFile file(entry.absoluteFilePath()); QVERIFY(file.open(QIODevice::ReadWrite));
      QVERIFY(file.setFileTime(time, QFileDevice::FileModificationTime));
    }
    MainWindow window(root);
    window.enableRecovery();
    QTimer::singleShot(0, &window, [&] {
      auto* dialog = qobject_cast<QDialog*>(QApplication::activeModalWidget());
      if (!dialog) { QFAIL("Recovery dialog did not open"); return; }
      auto* choices = dialog->findChild<QComboBox*>("recoveryRecords");
      if (!choices || choices->count() != 2) { dialog->reject(); QFAIL("Missing recovery records"); return; }
      QCOMPARE(choices->itemText(0), choices->itemText(1));
      // Deliberately select the second entry, which text-based lookup confused
      // with the first. The identity must follow the stored path.
      choices->setCurrentIndex(1);
      const auto selected = Project::load(choices->currentData().toString());
      window.setProperty("expectedPhotos", Workspace::read(selected).images.size());
      dialog->accept();
    });
    window.offerRecovery();
    QTRY_COMPARE(window.findChild<QListWidget*>("photos")->count(), window.property("expectedPhotos").toInt());
  }
  void compactWindowKeepsPrimaryControlsReachable() {
    MainWindow window;
    window.show();
    QTRY_VERIFY(window.findChild<QTreeWidget*>("projectTree")->topLevelItemCount() > 0);
    for (const auto size : {QSize(1280, 660), QSize(1080, 600)}) {
      window.resize(size);
      QTRY_COMPARE_WITH_TIMEOUT(window.size(), size, 5000);
      for (const auto name : {"primary", "workflowParameters", "workflowPause", "addPhotos", "addFolder"}) {
        auto* button = window.findChild<QPushButton*>(name);
        QVERIFY(button && button->isVisible());
        QVERIFY(window.rect().contains(QRect(button->mapTo(&window, QPoint()), button->size())));
      }
      QVERIFY(window.findChild<QTreeWidget*>("projectTree")->height() >= 60);
      QVERIFY(window.findChild<QWidget*>("inspectorPanel")->height() >= 80);
      const auto capture = qEnvironmentVariable("COLITE_GUI_LAYOUT_CAPTURE");
      if (!capture.isEmpty()) {
        QDir().mkpath(capture);
        QVERIFY(window.grab().save(QDir(capture).filePath(QString("layout-%1x%2.png").arg(size.width()).arg(size.height()))));
        qInfo() << "Layout" << size << "dpr" << window.devicePixelRatioF()
                << "available" << window.screen()->availableGeometry().size();
      }
    }
  }
  void inspectionDiscardsFilesChangedBeforeDelivery() {
    QTemporaryDir dir;
    const auto path = dir.filePath("changing.png");
    QImage image(120, 80, QImage::Format_RGB32); image.fill(Qt::green);
    QVERIFY(image.save(path));
    Workspace workspace;
    workspace.images = {{1, 0, "changing.png", path}};
    FeatureView view;
    view.setWorkspace(workspace);
    view.setViewMode(FeatureView::Mode::Photos);
    // Let the worker run without delivering its queued completion to the GUI.
    QTest::qSleep(150);
    image = QImage(200, 100, QImage::Format_RGB32); image.fill(Qt::red);
    QVERIFY(image.save(path));
    auto* canvas = static_cast<InspectionCanvas*>(view.findChild<QWidget*>("inspectionCanvas"));
    QTRY_COMPARE(canvas->data.sourceA, QSize(200, 100));
    QCOMPARE(canvas->data.a.pixelColor(0, 0), QColor(Qt::red));
  }
  void inspectionPreviewPreservesSourceCoordinatesAndInvalidatesCache() {
    QTemporaryDir dir;
    QImage image(5000, 1000, QImage::Format_RGB32); image.fill(Qt::green);
    const auto first = dir.filePath("a.jpg"), second = dir.filePath("b.jpg");
    QVERIFY(image.save(first)); QVERIFY(image.save(second));
    Workspace workspace;
    workspace.images = {{1, 0, "a.jpg", first}, {2, 0, "b.jpg", second}};
    FeatureView view;
    view.setWorkspace(workspace); view.setViewMode(FeatureView::Mode::Photos);
    auto* canvas = static_cast<InspectionCanvas*>(view.findChild<QWidget*>("inspectionCanvas"));
    QTRY_VERIFY(!canvas->data.a.isNull());
    QCOMPARE(canvas->data.sourceA, QSize(5000, 1000));
    QVERIFY(canvas->data.a.width() <= 4096);
    view.selectImage(1);
    QTRY_COMPARE(canvas->data.nameA, QString("b.jpg"));
    view.selectImage(0);
    QCOMPARE(canvas->data.nameA, QString("a.jpg")); // Synchronous cache reuse.
    image = QImage(200, 100, QImage::Format_RGB32); image.fill(Qt::red);
    QVERIFY(image.save(first));
    view.selectImage(0);
    QTRY_COMPARE(canvas->data.sourceA, QSize(200, 100));
    view.selectImage(1); view.selectImage(0);
    QTRY_COMPARE(canvas->data.nameA, QString("a.jpg"));
    QCOMPARE(canvas->data.sourceA, QSize(200, 100));
  }
  void sceneCacheInvalidationAndLatestSelection() {
    QTemporaryDir dir;
    Project project;
    project.name = "Scene cache";
    project.imageRoot = dir.path();
    project.workspace = dir.path();
    project.dense = dir.path();
    project.mesh = dir.filePath("mesh.ply");
    project.sparse = dir.filePath("sparse");
    QImage texture(8, 8, QImage::Format_RGB32);
    texture.fill(Qt::green);
    QVERIFY(texture.save(dir.filePath("texture.png")));
    auto writePly = [&](const QString& path, int count) {
      QFile file(path); QVERIFY(file.open(QIODevice::WriteOnly));
      file.write(QString("ply\nformat ascii 1.0\n%1element vertex %2\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
                     .arg(path == project.mesh ? "comment TextureFile texture.png\n" : "").arg(count).toUtf8());
      for (int i = 0; i < count; ++i) file.write(QString("%1 0 0\n").arg(i).toUtf8());
    };
    writePly(dir.filePath("fused.ply"), 2);
    writePly(project.mesh, 3);
    project.parameters["gui_tab"] = 1;
    project.save(dir.filePath("cache.colite.json"));
    MainWindow window;
    window.openProject(project.file);
    QTRY_VERIFY(window.findChild<QTreeWidget*>("projectTree")->topLevelItemCount() > 0);
    QSignalSpy loaded(&window, &MainWindow::sceneLoaded);
    window.navigate(4);
    QTRY_COMPARE(loaded.size(), 1);
    QVERIFY(!loaded.last()[1].toBool());
    window.navigate(5);
    QTRY_COMPARE(loaded.size(), 2);
    window.navigate(4);
    QCOMPARE(loaded.size(), 3);
    QVERIFY(loaded.last()[1].toBool());
    writePly(dir.filePath("fused.ply"), 4);
    window.navigate(4);
    QTRY_COMPARE(loaded.size(), 4);
    QVERIFY(!loaded.last()[1].toBool());
    // A slower pending replacement must never overwrite the latest cached mesh.
    writePly(dir.filePath("fused.ply"), 20000);
    window.navigate(4);
    QVERIFY(window.sceneLoading());
    QVERIFY(!window.findChild<QProgressBar*>("sceneLoadProgress")->isHidden());
    window.navigate(5);
    QCOMPARE(loaded.size(), 5);
    QCOMPARE(loaded.last()[0].toString(), project.mesh);
    QVERIFY(loaded.last()[1].toBool());
    QTest::qWait(200);
    QCOMPARE(loaded.size(), 5);
    QVERIFY(!window.sceneLoading());
    QVERIFY(window.findChild<QProgressBar*>("sceneLoadProgress")->isHidden());
    texture = QImage(19, 13, QImage::Format_RGB32);
    texture.fill(Qt::red);
    QVERIFY(texture.save(dir.filePath("texture.png")));
    window.navigate(5);
    QTRY_COMPARE(loaded.size(), 6);
    QVERIFY(!loaded.last()[1].toBool());
  }
  void houseSceneSwitchBenchmark() {
    const auto path = qEnvironmentVariable("COLITE_GUI_SCENE_BENCHMARK");
    if (path.isEmpty()) QSKIP("Set COLITE_GUI_SCENE_BENCHMARK for real model loading timings");
    MainWindow window;
    window.openProject(path);
    window.show();
    QTRY_VERIFY_WITH_TIMEOUT(window.findChild<QListWidget*>("photos")->count() > 0, 30000);
    QSignalSpy loaded(&window, &MainWindow::sceneLoaded);
    auto* scene = window.findChild<SceneView*>();
    QSignalSpy presented(scene, &SceneView::firstFramePresented);
    const bool nativeFrames = qEnvironmentVariableIsSet("COLITE_GUI_NATIVE_FRAMES");
    QElapsedTimer firstFrameClock;
    qint64 firstFrameMs = 0;
    connect(scene, &SceneView::firstFramePresented, this, [&] { firstFrameMs = firstFrameClock.elapsed(); });
    connect(&window, &MainWindow::sceneLoaded, this, [&] { presented.clear(); firstFrameMs = 0; });
    QTRY_VERIFY_WITH_TIMEOUT(!window.sceneLoading(), 30000);
    QTest::qWait(100);
    for (int pass = 0; pass < 2; ++pass) {
      for (int stage : {4, 5, 6, 3}) {
        loaded.clear();
        presented.clear(); firstFrameClock.start();
        window.navigate(stage);
        QTRY_COMPARE_WITH_TIMEOUT(loaded.size(), 1, 30000);
        QVERIFY(!window.findChild<QLabel*>("sceneLoadStatus")->text().contains(QString::fromUtf8("失败")));
        qInfo() << "Scene" << stage << "pass" << pass << "cache" << loaded[0][1].toBool()
                << "data_ready_ms" << loaded[0][2].toLongLong();
        if (pass == 1) QVERIFY(loaded[0][1].toBool());
        if (nativeFrames) {
          QTRY_VERIFY_WITH_TIMEOUT(!presented.isEmpty(), 30000);
          QVERIFY2(window.graphicsError().isEmpty(), qPrintable(window.graphicsError()));
          qInfo() << "Frame" << stage << "pass" << pass << "selection_to_frame_ms" << firstFrameMs
                  << "apply_to_frame_ms" << presented.last()[0].toLongLong()
                  << "upload_cpu_ms" << presented.last()[1].toLongLong()
                  << "upload_bytes" << presented.last()[2].toLongLong();
          if (pass == 1) QCOMPARE(presented.last()[2].toLongLong(), qint64(0));
          if (pass == 1 && stage == 6) {
            QVector<qint64> samples;
            QElapsedTimer interaction; interaction.start();
            const auto connection = connect(scene, &QOpenGLWidget::frameSwapped, this, [&] { samples << interaction.elapsed(); });
            const auto origin = scene->rect().center();
            QTest::mousePress(scene, Qt::LeftButton, Qt::NoModifier, origin);
            for (int i = 1; i <= 60; ++i) {
              QMouseEvent move(QEvent::MouseMove, origin + QPoint(i * 2, i / 2), Qt::NoButton, Qt::LeftButton, Qt::NoModifier);
              QApplication::sendEvent(scene, &move);
              QTest::qWait(16);
            }
            QTest::mouseRelease(scene, Qt::LeftButton, Qt::NoModifier, origin + QPoint(120, 30));
            const auto duration = interaction.elapsed();
            disconnect(connection);
            QVERIFY(samples.size() >= 5);
            QVector<qint64> intervals;
            for (int i = 1; i < samples.size(); ++i) intervals << samples[i] - samples[i - 1];
            std::sort(intervals.begin(), intervals.end());
            qInfo() << "Interaction" << "frames" << samples.size() << "duration_ms" << duration
                    << "observed_fps" << (samples.size() * 1000. / duration)
                    << "median_ms" << intervals[intervals.size() / 2]
                    << "p95_ms" << intervals[qMin(intervals.size()-1, int(intervals.size() * .95))];
            QTest::qWait(200); // Allow the interaction timer to settle before switching.
          }
        }
      }
    }
  }
  void appendPhotoPreservesOriginalsAndSwitchesInputs() {
    QTemporaryDir dir;
    const auto originals = dir.filePath("originals");
    const auto work = dir.filePath("work");
    QVERIFY(QDir().mkpath(originals));
    QImage photo(16, 16, QImage::Format_RGB32);
    photo.fill(Qt::white);
    QVERIFY(photo.save(originals + "/first.png"));
    const auto added = dir.filePath("second.png");
    QVERIFY(photo.save(added));
    Project p;
    p.name = "Import";
    p.imageRoot = originals;
    p.workspace = work + "/workspace";
    p.sparse = p.workspace + "/sparse";
    p.dense = work + "/mvs";
    p.mesh = work + "/surface/mesh.ply";
    p.backendMode = "wsl";
    p.backend = "/old/worker";
    p.save(dir.filePath("import.colite.json"));
    MainWindow window;
    window.openProject(p.file);
    auto* add = window.findChild<QPushButton*>("addPhotos");
    QVERIFY(add);
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    QTimer::singleShot(0, &window, [&] {
      auto* picker = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
      QVERIFY(picker);
      picker->selectFile(added);
      QMetaObject::invokeMethod(picker, "accept", Qt::DirectConnection);
    });
    add->click();
    QVERIFY(window.windowTitle().contains('*'));
    QTimer::singleShot(0, &window, [] {
      auto* prompt = QApplication::activeModalWidget()->findChild<QDialogButtonBox*>();
      QVERIFY(prompt);
      prompt->button(QDialogButtonBox::Cancel)->click();
    });
    for (auto* action : window.findChildren<QAction*>())
      if (action->shortcut() == QKeySequence::New) { action->trigger(); break; }
    QVERIFY(window.windowTitle().contains('*'));
    for (auto* action : window.findChildren<QAction*>())
      if (action->shortcut() == QKeySequence::Save) { action->trigger(); break; }
    const auto imported = Project::load(p.file);
    QCOMPARE(imported.backendMode, QString("native"));
    QVERIFY(imported.backend.isEmpty());
    QCOMPARE(imported.photoFiles.size(), 2);
    QVERIFY(imported.imageRoot.isEmpty());
    QVERIFY(imported.workspace != p.workspace);
    QVERIFY(!QFileInfo::exists(originals + "/second.png"));
    QCOMPARE(Workspace::read(imported).images.size(), 2);
    auto running = imported;
    running.assignOutputs(work);
    running.prepareInput();
    QVERIFY(QFileInfo::exists(QDir(running.imageRoot).filePath("first.png")));
    QVERIFY(QFileInfo::exists(QDir(running.imageRoot).filePath("second.png")));
    QCOMPARE(running.id, p.id);
  }
  void untitledHouseLifecycle() {
    const auto folder = qEnvironmentVariable("COLITE_GUI_UNTITLED_IMAGES");
    if (folder.isEmpty()) QSKIP("Set COLITE_GUI_UNTITLED_IMAGES to run the installed desktop lifecycle");
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QApplication::setAttribute(Qt::AA_DontUseNativeDialogs);
    MainWindow window;
    window.show();
    QVERIFY(window.windowTitle().contains(QString::fromUtf8("未命名")));
    auto choose = [&](QString path) {
      QTimer::singleShot(0, &window, [path] {
        auto* picker = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
        QVERIFY(picker);
        picker->selectFile(path);
        QMetaObject::invokeMethod(picker, "accept", Qt::DirectConnection);
      });
    };
    QPushButton* folderButton = nullptr;
    for (auto* button : window.findChildren<QPushButton*>())
      if (button->objectName() == "addFolder") { folderButton = button; break; }
    QVERIFY(folderButton);
    choose(folder); folderButton->click();
    auto* photos = window.findChild<QListWidget*>("photos");
    QTRY_VERIFY_WITH_TIMEOUT(photos->count() > 0, 30000);
    const int count = photos->count();
    QVERIFY(window.windowTitle().contains('*'));
    auto* start = window.findChild<QPushButton*>("primary");
    QVERIFY(start->isEnabled());
    QTimer::singleShot(0, &window, [&] {
      auto* dialog = dynamic_cast<ParameterDialog*>(QApplication::activeModalWidget());
      QVERIFY(dialog);
      dialog->findChild<QListWidget*>("parameterStages")->setCurrentRow(1);
      dialog->findChild<QLineEdit*>("parameterWorkDirectory")->setText(dir.path());
      auto* device = dialog->findChild<QComboBox*>("device");
      device->setCurrentIndex(device->findData("cpu"));
      dialog->findChild<QLineEdit*>("FeatureExtraction.max_image_size")->setText("800");
      dialog->findChild<QLineEdit*>("FeatureExtraction.num_threads")->setText("4");
      dialog->findChild<QPushButton*>("runStage")->click();
    });
    window.findChild<QPushButton*>("workflowParameters")->click();
    auto* state = window.findChild<QLabel*>("taskState");
    QTRY_VERIFY_WITH_TIMEOUT(state->text().contains(QString::fromUtf8("产物检查通过")),180000);
    const auto path = dir.filePath("House.colite.json");
    QVERIFY(!QFileInfo::exists(path));
    choose(path);
    for (auto* action : window.findChildren<QAction*>())
      if (action->shortcut() == QKeySequence::Save) { action->trigger(); break; }
    const auto saved = Project::load(path);
    QCOMPARE(saved.name, QString("House"));
    QCOMPARE(saved.workDirectory, dir.path());
    QVERIFY(QFileInfo::exists(QDir(saved.workspace).filePath("images.tsv")));
    window.openProject(path);
    QTRY_COMPARE(photos->count(), count);
    QVERIFY(state->text().contains(QString::fromUtf8("产物检查通过")));
    QCOMPARE(window.windowTitle(), QString::fromUtf8("House — Colite"));
    qInfo() << "Saved and reopened" << count << "photos with verified feature extraction history";
    QThreadPool::globalInstance()->waitForDone();
  }
  void objectSelectionKeepsInspectorContext() {
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QDir().mkpath(dir.filePath("images"));
    QDir().mkpath(dir.filePath("sparse"));
    for (int id = 1; id <= 2; ++id) {
      QImage image(640, 480, QImage::Format_RGB32);
      image.fill(Qt::white);
      QVERIFY(image.save(dir.filePath(QString("images/photo%1.png").arg(id))));
    }
    QFile points(dir.filePath("sparse/points3D.txt"));
    QVERIFY(points.open(QIODevice::WriteOnly));
    points.write("1 0 0 0 255 255 255 0\n");
    points.close();
    QFile poses(dir.filePath("sparse/images.txt"));
    QVERIFY(poses.open(QIODevice::WriteOnly));
    poses.close();
    Project project;
    project.name = "Inspector regression";
    project.imageRoot = dir.filePath("images");
    project.workspace = dir.path();
    project.sparse = dir.filePath("sparse");
    project.save(dir.filePath("test.colite.json"));
    MainWindow window;
    auto* start = window.findChild<QPushButton*>("primary");
    QVERIFY(start);
    QVERIFY(!start->isEnabled());
    window.resize(1280, 800);
    window.show();
    window.openProject(project.file);
    QTRY_VERIFY(start->isEnabled());
    auto* tree = window.findChild<QTreeWidget*>("projectTree");
    auto* views = window.findChild<QTabWidget*>("centralViews");
    QVERIFY(tree && views);
    QCOMPARE(views->count(), 2);
    QCOMPARE(views->tabText(1), QString::fromUtf8("2D视图"));
    QTRY_VERIFY(tree->topLevelItemCount() > 0);
    auto* menu = window.menuBar();
    QVERIFY(menu->isVisible());
    for (auto* action : menu->actions()) {
      QVERIFY(action->isVisible());
      QVERIFY(menu->rect().contains(menu->actionGeometry(action)));
    }
    auto find = [&](int kind, int index = -1) -> QTreeWidgetItem* {
      QTreeWidgetItemIterator it(tree);
      while (*it) {
        auto* item = *it;
        if (item->data(0, Qt::UserRole).toInt() == kind &&
            (index < 0 || item->data(0, Qt::UserRole + 1).toInt() == index))
          return item;
        ++it;
      }
      return nullptr;
    };
    auto title = [&] {
      const auto titles = window.findChildren<QLabel*>("objectTitle");
      return titles.isEmpty() ? QString() : titles.first()->text();
    };
    QVERIFY(find(12, 1) && find(11) && find(2));
    QVERIFY(find(1) && find(13));
    QCOMPARE(find(1)->text(0), QString::fromUtf8("特征提取"));
    QCOMPARE(find(13)->text(0), QString::fromUtf8("特征匹配"));
    // An image choice must leave the matching view and select the same photo.
    window.navigate(2);
    QCOMPARE(tree->currentItem(), find(13));
    QCOMPARE(window.features()->viewMode(), FeatureView::Mode::Matches);
    tree->setCurrentItem(find(12, 1));
    QCOMPARE(views->currentIndex(), 1);
    QCOMPARE(window.features()->imageIndex(), 1);
    QCOMPARE(window.features()->viewMode(), FeatureView::Mode::Photos);
    QTRY_COMPARE(title(), QString("photo2.png"));
    // Pending feature/model reads must not replace a newly selected camera.
    window.navigate(3);
    tree->setCurrentItem(find(11));
    QTRY_COMPARE(title(), QString::fromUtf8("待分组图像"));
    QTest::qWait(150);
    QCOMPARE(title(), QString::fromUtf8("待分组图像"));
    QCOMPARE(window.findChildren<QLabel*>("objectTitle").size(), 1);
    tree->setCurrentItem(find(12, 0));
    QTRY_COMPARE(title(), QString("photo1.png"));
    views->setCurrentIndex(0);
    QCOMPARE(tree->currentItem(), find(2));
    QCOMPARE(title(), QString::fromUtf8("稀疏主模型"));
    for (auto* button : window.findChildren<QPushButton*>())
      if (button->text() == QString::fromUtf8("稀疏重建")) QVERIFY(button->isChecked());
    // The eye control changes visibility without changing the selected model.
    QVERIFY(QMetaObject::invokeMethod(tree, "itemClicked", Qt::DirectConnection,
                                      Q_ARG(QTreeWidgetItem*, find(2)),
                                      Q_ARG(int, 1)));
    bool hidden = false;
    for (auto* label : window.findChildren<QLabel*>("propertyValue"))
      hidden |= label->text() == QString::fromUtf8("隐藏");
    QVERIFY(hidden);
    QCOMPARE(tree->currentItem(), find(2));
    tree->setCurrentItem(find(1));
    QCOMPARE(window.features()->viewMode(), FeatureView::Mode::Features);
    QCOMPARE(views->currentIndex(), 1);
    tree->setCurrentItem(find(13));
    QCOMPARE(window.features()->viewMode(), FeatureView::Mode::Matches);
    auto* photoMode = window.findChild<QPushButton*>("photoMode");
    photoMode->click();
    QCOMPARE(tree->currentItem(), find(12, window.features()->imageIndex()));
    // Pairing from a photo preserves that photo as A and assigns the target to B.
    window.features()->selectImage(1);
    window.features()->selectPairImage(0);
    QCOMPARE(window.features()->pairA(), 1);
    QCOMPARE(window.features()->pairB(), 0);
    QCOMPARE(tree->currentItem(), find(13));
    auto* photos = window.findChild<QListWidget*>("photos");
    QCOMPARE(photos->item(0)->sizeHint(), photos->gridSize());
    QCOMPARE(photos->item(1)->data(Qt::UserRole + 2).toString(), QString("A"));
    QCOMPARE(photos->item(0)->data(Qt::UserRole + 2).toString(), QString("B"));
    // Rapid changes leave the final photo mode and source image authoritative.
    window.navigate(1); window.navigate(2); window.navigate(0);
    QTRY_VERIFY(!static_cast<InspectionCanvas*>(window.features()->findChild<QWidget*>("inspectionCanvas"))->data.a.isNull());
    QCOMPARE(window.features()->viewMode(), FeatureView::Mode::Photos);
    QVERIFY(photos->item(0)->data(Qt::UserRole + 2).toString().isEmpty());
  }
  void preserveImageAndPoint() {
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QDir().mkpath(dir.filePath("features"));
    QDir().mkpath(dir.filePath("matches"));
    Workspace workspace;
    workspace.root = dir.path();
    for (int id = 1; id <= 2; ++id) {
      const auto image = dir.filePath(QString("photo %1.png").arg(id));
      QImage pixels(640, 480, QImage::Format_RGB32);
      pixels.fill(Qt::white);
      QVERIFY(pixels.save(image));
      workspace.images.push_back({quint32(id), 1, QString::number(id), image});
      QFile file(dir.filePath(QString("features/%1.bin").arg(id)));
      QVERIFY(file.open(QIODevice::WriteOnly));
      QDataStream out(&file);
      out.setByteOrder(QDataStream::LittleEndian);
      out << quint32(0x434c4654) << quint32(1) << quint64(2) << quint64(0)
          << quint64(0);
      out.setFloatingPointPrecision(QDataStream::SinglePrecision);
      for (int n = 0; n < 2; ++n)
        out << float(200 + n * 20) << 100.f << 3.f << 45.f << 1.f;
    }
    QFile raw(dir.filePath("matches/raw_matches.tsv"));
    QVERIFY(raw.open(QIODevice::WriteOnly));
    raw.write("1\t2\t0\t1\n");
    raw.close();
    FeatureView view;
    view.resize(850, 640);
    view.show();
    view.setWorkspace(workspace);
    view.setMode(true);
    auto* canvas = static_cast<InspectionCanvas*>(
        view.findChild<QWidget*>("inspectionCanvas"));
    QVERIFY(canvas);
    QTRY_COMPARE(canvas->data.pair.matches.size(), 1);
    canvas->picked(0, 1);
    QCOMPARE(view.imageIndex(), 1);
    auto* single = view.findChild<QPushButton*>("singleMode");
    auto* pair = view.findChild<QPushButton*>("pairMode");
    auto* detail = view.findChild<QLabel*>("inspectionDetail");
    QVERIFY(single && pair && detail);
    QTest::mouseClick(single, Qt::LeftButton);
    QTRY_VERIFY(detail->text().contains("#1"));
    QVERIFY(!view.matching());
    QCOMPARE(view.imageIndex(), 1);
    QTest::mouseClick(pair, Qt::LeftButton);
    QTRY_VERIFY(detail->text().contains("A #0 / B #1"));
    QVERIFY(view.matching());
    QCOMPARE(view.imageIndex(), 1);
    // Replacing the input with one image must not show a self-pair or stale matches.
    workspace.images.resize(1);
    view.setWorkspace(workspace);
    QVERIFY(canvas->data.a.isNull());
    QVERIFY(canvas->data.error.contains(QString::fromUtf8("至少两张")));
    view.setViewMode(FeatureView::Mode::Photos);
    QTRY_VERIFY(!canvas->data.a.isNull());
    QVERIFY(canvas->data.first.isEmpty());
    QVERIFY(!canvas->show);
  }
};
int main(int argc, char** argv) {
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
  QApplication app(argc, argv);
  QTemporaryDir settings;
  if (!settings.isValid()) return 1;
  app.setOrganizationName("ColiteTests");
  QSettings::setDefaultFormat(QSettings::IniFormat);
  QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings.path());
  configureApplicationUi(app);
  InspectionTests tests;
  return QTest::qExec(&tests, argc, argv);
}
#include "ui_tests.moc"
