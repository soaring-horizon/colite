#include <QtWidgets>

#include "main_window.h"
#include "widgets/ui_style.h"

namespace {
void capture(colite::gui::MainWindow* window, const QString& folder, int step);
void saveCapture(colite::gui::MainWindow* window, const QString& folder,
                 int step, const QString& filename, int retry = 0) {
  if (window->sceneLoading()) {
    if (retry >= 60) {
      qWarning("Scene did not finish loading before capture timeout");
      QCoreApplication::exit(2);
      return;
    }
    QTimer::singleShot(500, window, [window, folder, step, filename, retry] {
      saveCapture(window, folder, step, filename, retry + 1);
    });
    return;
  }
  if (window->graphicsInfo().isEmpty() || !window->graphicsError().isEmpty() ||
      !window->grab().save(QDir(folder).filePath(filename))) {
    qCritical() << "Capture failed:" << window->graphicsError();
    QCoreApplication::exit(2);
    return;
  }
  QSaveFile report(QDir(folder).filePath("render-info.json"));
  const QJsonObject metadata{{"qt_version", qVersion()},
      {"font", QFontInfo(window->font()).family()},
      {"platform", QGuiApplication::platformName()},
      {"kernel", QSysInfo::kernelVersion()},
      {"architecture", QSysInfo::currentCpuArchitecture()},
      {"os", QSysInfo::prettyProductName()}, {"graphics", window->graphicsInfo()},
      {"error", window->graphicsError()}};
  const auto bytes = QJsonDocument(metadata).toJson();
  if (!report.open(QIODevice::WriteOnly) || report.write(bytes) != bytes.size() || !report.commit()) {
    QCoreApplication::exit(2);
    return;
  }
  capture(window, folder, step + 1);
}
void capture(colite::gui::MainWindow* window, const QString& folder, int step) {
  static const QVector<QPair<int, QString>> shots = {
      {3, "native-3d.png"},
      {1, "native-features.png"},
      {2, "native-matches.png"},
      {-1, "native-compact.png"},
      {4, "native-dense.png"},
      {5, "native-surface.png"},
      {6, "native-texture.png"},
      {-2, "native-match-statistics.png"},
      {-3, "native-image-properties.png"}};
  if (step >= shots.size()) {
    QCoreApplication::quit();
    return;
  }
  const auto shot = shots[step];
  if (shot.first == -2) {
    window->resize(1536, 1024);
    window->navigate(2);
    if (auto* tabs = window->findChild<QTabBar*>("inspectorTabs"))
      tabs->setCurrentIndex(1);
  } else if (shot.first == -3) {
    if (auto* tabs = window->findChild<QTabBar*>("inspectorTabs"))
      tabs->setCurrentIndex(0);
    if (auto* tree = window->findChild<QTreeWidget*>("projectTree")) {
      QTreeWidgetItemIterator it(tree);
      while (*it) {
        if ((*it)->data(0, Qt::UserRole).toInt() == 12) {
          tree->setCurrentItem(*it);
          break;
        }
        ++it;
      }
    }
  } else if (shot.first < 0)
    window->resize(1280, 800);
  else {
    window->resize(1536, 1024);
    window->navigate(shot.first);
  }
  QTimer::singleShot(1500, window, [window, folder, step, shot] {
    saveCapture(window, folder, step, shot.second);
  });
}
}  // namespace

#include "colite_release.h"

int main(int argc, char** argv) {
  if (argc == 2 && QByteArray(argv[1]) == "--build-info") {
    QFile output; output.open(stdout, QIODevice::WriteOnly);
    output.write(COLITE_BUILD_JSON "\n"); output.flush();
    return 0;
  }
  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
  QCoreApplication::setAttribute(Qt::AA_DisableWindowContextHelpButton);
  QApplication app(argc, argv);
  app.setApplicationName("Colite");
  app.setApplicationDisplayName("Colite");
  app.setApplicationVersion(COLITE_RELEASE_VERSION);
  app.setWindowIcon(QIcon(":/ui/colite.png"));
  app.setOrganizationName("Colite");
  colite::gui::configureApplicationUi(app);
  QCommandLineParser parser;
  parser.addHelpOption();
  parser.addVersionOption();
  parser.addPositionalArgument("project", "Optional project document to open");
  parser.addOption({"project", "Open a native project document", "file"});
  parser.addOption({"capture",
                    "Capture native workbench views for verification",
                    "directory"});
  parser.process(app);
  colite::gui::MainWindow window;
  if (!parser.isSet("capture")) window.enableRecovery();
  if (parser.isSet("project")) window.openProject(parser.value("project"));
  else if (!parser.positionalArguments().isEmpty())
    window.openProject(parser.positionalArguments().first());
  window.show();
  if (!parser.isSet("capture") && !parser.isSet("project") && parser.positionalArguments().isEmpty())
    QTimer::singleShot(0, &window, &colite::gui::MainWindow::offerRecovery);
  if (parser.isSet("capture")) {
    const auto folder = parser.value("capture");
    QDir().mkpath(folder);
    QTimer::singleShot(3500, &window,
                       [&window, folder] { capture(&window, folder, 0); });
  }
  return app.exec();
}
