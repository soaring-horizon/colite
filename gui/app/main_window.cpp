#include "main_window.h"

#include <QImageReader>
#include <QJsonDocument>
#include <QSaveFile>
#include <QtConcurrent>
#include <cmath>

#include "widgets/ui_style.h"
#include "workflow/task_plan.h"
#include "workflow/workflow_state.h"
#include "project/scene_export.h"
#include "widgets/parameter_dialog.h"
#include "workflow/parameter_schema.h"
#include "colite_release.h"

namespace colite::gui {
namespace {
const QStringList stageNames = workflowNames();
constexpr auto kRecentProjectsKey = "projects/recent";
constexpr int kRecentProjectCount = 5;
QString recentProjectName(const QString& path) {
  auto name = QFileInfo(path).fileName();
  if (name.endsWith(".colite.json", Qt::CaseInsensitive)) name.chop(12);
  return name.isEmpty() ? QFileInfo(path).fileName() : name;
}
QPushButton* action(const QString& text, std::function<void()> fn) {
  auto* b = new QPushButton(text);
  QObject::connect(b, &QPushButton::clicked, b, std::move(fn));
  return b;
}
QWidget* panel() {
  auto* p = new QWidget;
  p->setObjectName("panel");
  return p;
}
QString sceneStamp(const QString& path, const QString& texture = {}) {
  QStringList files;
  if (QFileInfo(path).isDir()) {
    for (const auto& name : {"cameras", "images", "points3D"})
      for (const auto& extension : {".bin", ".txt"})
        files << QDir(path).filePath(QString(name) + extension);
  } else files << path;
  if (!texture.isEmpty()) files << texture;
  QString stamp;
  for (const auto& file : files) {
    const QFileInfo info(file);
    stamp += file + '\n' + QString::number(info.size()) + ':' +
             QString::number(info.lastModified().toMSecsSinceEpoch()) + ':' +
             QString::number(info.exists()) + '\n';
  }
  return stamp;
}
}  // namespace
MainWindow::MainWindow(QString recoveryRoot) : recovery_(std::move(recoveryRoot)) {
  scenePool_.setMaxThreadCount(1);
  thumbnailPool_.setMaxThreadCount(1);
  thumbnailTimer_.setSingleShot(true); thumbnailTimer_.setInterval(30);
  connect(&thumbnailTimer_, &QTimer::timeout, this, &MainWindow::loadVisibleThumbnails);
  setWindowTitle("Colite");
  setMinimumSize(1080, 600);
  const auto available = QGuiApplication::primaryScreen()->availableGeometry().size();
  resize(QSize(1536, 1024).boundedTo(available - QSize(40, 60)));
  auto* menus = new QMenuBar;
  menus->setNativeMenuBar(false); // Keep File/Export available without a desktop global-menu service.
  setMenuBar(menus);
  auto* file = menus->addMenu(tr("文件"));
  file->addAction(tr("新建项目…"), this, &MainWindow::createProject,
                  QKeySequence::New);
  file->addAction(
      tr("打开项目…"), this,
      [this] {
        const auto path = QFileDialog::getOpenFileName(
            this, tr("打开项目"), {}, "Colite Project (*.colite.json)");
        if (!path.isEmpty()) openProject(path);
      },
      QKeySequence::Open);
  recentProjects_ = file->addMenu(tr("最近工程"));
  recentProjects_->setObjectName("recentProjects");
  recentProjects_->setToolTipsVisible(true);
  connect(recentProjects_, &QMenu::aboutToShow, this, &MainWindow::refreshRecentProjects);
  file->addAction(tr("保存项目"), this, [this] { saveProject(); },
                  QKeySequence::Save);
  file->addAction(tr("工程另存为…"), this, [this] { saveProject(true); }, QKeySequence::SaveAs);
  file->addAction(tr("导出当前模型…"), this, &MainWindow::exportSelectedScene);
  file->addAction(tr("打开结果目录"), this, [this] {
    const auto path = project_.workDirectory.isEmpty() ? project_.workspace : project_.workDirectory;
    if (path.isEmpty() || !QFileInfo::exists(path)) { statusBar()->showMessage(tr("尚未生成计算结果"), 4000); return; }
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
  });
  file->addAction(tr("项目数据…"), this, &MainWindow::editProject);
  file->addAction(tr("恢复未保存工程…"), this, &MainWindow::offerRecovery);
  auto* edit = menus->addMenu(tr("编辑"));
  edit->addAction(tr("项目数据…"), this, &MainWindow::editProject);
  auto* view = menus->addMenu(tr("视图"));
  view->addAction(
      tr("适配三维视图"), this, [this] { scene_->fit(); }, QKeySequence("F"));
  view->addAction(tr("顶视图"), this, [this] { scene_->top(); });
  view->addAction(tr("重新读取产物"), this, &MainWindow::reload);
  auto* reconstruct = menus->addMenu(tr("重建"));
  for (int i = 1; i < stageNames.size(); ++i)
    reconstruct->addAction(stageNames[i], this, [this, i] {
      navigate(i);
      configure(true);
    });
  auto* utilities = menus->addMenu(tr("工具"));
  utilities->addAction(tr("重新读取工作区"), this, &MainWindow::reload);
  utilities->addAction(tr("流程参数…"), this, [this] { configure(); });
  auto* help = menus->addMenu(tr("帮助"));
  help->addAction(tr("使用说明"), this, [this] {
    QDialog dialog(this); dialog.setWindowTitle(tr("Colite 使用说明"));
    dialog.resize(QSize(800, 640).boundedTo(size()));
    auto* layout = new QVBoxLayout(&dialog);
    auto* browser = new QTextBrowser;
    QFile guide(":/ui/user-guide.md");
    if (guide.open(QIODevice::ReadOnly)) browser->setMarkdown(QString::fromUtf8(guide.readAll()));
    layout->addWidget(browser);
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons); dialog.exec();
  });
  help->addAction(tr("关于 Colite"), this, [this] {
    QMessageBox about(this);
    about.setWindowTitle(tr("关于 Colite"));
    about.setIconPixmap(windowIcon().pixmap(64, 64));
    about.setText("Colite " + QCoreApplication::applicationVersion());
    about.setInformativeText(
        tr("摄影测量与三维重建工作台\n\n从图像、特征匹配到稀疏重建、"
           "稠密点云、网格及纹理模型。\n\n版本：%1\n源码：%2\nQt：%3\n\n图标素材：Phosphor Icons（MIT）。")
            .arg(COLITE_RELEASE_VERSION, COLITE_SOURCE_REVISION, qVersion()));
    QFile identity(QDir(QCoreApplication::applicationDirPath()).filePath("../package-identity.json"));
    if (identity.open(QIODevice::ReadOnly) && identity.size() < 8 * 1024 * 1024) {
      const auto package = QJsonDocument::fromJson(identity.readAll()).object();
      about.setDetailedText(tr("发行包：%1\n计算能力：%2\nCUDA 架构：%3\n包源码：%4")
          .arg(package["version"].toString(), package["cuda"].toBool() ? "CPU + CUDA" : "CPU",
               package["cuda_architectures"].toString(), package["source"].toObject()["commit"].toString()));
    }
    about.exec();
  });
  auto* tools = addToolBar(tr("工作台"));
  tools->setMovable(false);
  tools->setFixedHeight(50);
  menus->setFixedHeight(30);
  tools->setIconSize({20, 20});
  auto* newAction = tools->addAction(fluentIcon(0xe8f4), tr("新建"), this,
                   &MainWindow::createProject);
  auto* openAction = tools->addAction(fluentIcon(0xe8e5), tr("打开"), this, [this] {
    const auto path = QFileDialog::getOpenFileName(
        this, tr("打开项目"), {}, "Colite Project (*.colite.json)");
    if (!path.isEmpty()) openProject(path);
  });
  auto* saveAction = tools->addAction(fluentIcon(0xe74e), tr("保存"), this,
                   [this] { saveProject(); });
  for (auto* entry : {newAction, openAction, saveAction})
    if (auto* button = qobject_cast<QToolButton*>(tools->widgetForAction(entry)))
      button->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  tools->addSeparator();
  utilities->addAction(fluentIcon(0xe8b9), tr("加载模型…"), this, [this] {
    const auto path =
        QFileDialog::getExistingDirectory(this, tr("选择模型目录"));
    if (!path.isEmpty()) loadScene(path);
  });
  utilities->addAction(fluentIcon(0xe7c1), tr("加载 PLY…"), this, [this] {
    const auto path = QFileDialog::getOpenFileName(this, tr("加载点云或网格"),
                                                   {}, "PLY (*.ply)");
    if (!path.isEmpty()) loadScene(path);
  });
  tools->addAction(fluentIcon(0xe7f4), tr("3D视图"), this,
                   [this] { views_->setCurrentIndex(0); });
  tools->addAction(fluentIcon(0xe91b), tr("2D视图"), this,
                   [this] { views_->setCurrentIndex(1); });
  tools->addAction(fluentIcon(0xe9a6), tr("适配视图"), this,
                   [this] {
                     if (views_->currentIndex() == 0) scene_->fit();
                     else if (auto* canvas = static_cast<InspectionCanvas*>(features_->findChild<QWidget*>("inspectionCanvas"))) {
                       canvas->fit();
                       if (canvas->zoomed) canvas->zoomed(1);
                     }
                   });
  for (auto* entry : tools->actions())
    if (auto* button = qobject_cast<QToolButton*>(tools->widgetForAction(entry)))
      button->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
  auto* body = new QWidget;
  auto* outer = new QVBoxLayout(body);
  outer->setContentsMargins(0, 0, 0, 0);
  outer->setSpacing(0);
  auto* flowBar = new QWidget;
  flowBar->setObjectName("workflowBar");
  flowBar->setFixedHeight(82);
  auto* flow = new QHBoxLayout(flowBar);
  flow->setContentsMargins(14, 7, 14, 7);
  flow->setSpacing(5);
  auto* stepScroll = new QScrollArea;
  stepScroll->setObjectName("workflowStepsScroll");
  stepScroll->setFrameShape(QFrame::NoFrame);
  stepScroll->setWidgetResizable(true);
  stepScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  auto* stepBody = new QWidget;
  auto* stepLayout = new QHBoxLayout(stepBody);
  stepLayout->setContentsMargins(0, 0, 0, 0); stepLayout->setSpacing(2);
  stepScroll->setWidget(stepBody);
  flow->addWidget(stepScroll, 1);
  for (int i = 0; i < stageNames.size(); ++i) {
    auto* b = new WorkflowStep(stageNames[i]);
    connect(b, &QPushButton::clicked, this, [this, i] { navigate(i); });
    b->setObjectName("stage" + QString::number(i));
    stages_ << b;
    b->setContextMenuPolicy(Qt::ActionsContextMenu);
    auto* configureStage = new QAction(tr("配置%1参数…").arg(stageNames[i]), b);
    configureStage->setObjectName("configureStage" + QString::number(i));
    b->addAction(configureStage);
    connect(configureStage, &QAction::triggered, this, [this, i] {
      navigate(i); configure(false, true);
    });
    connect(b, &QPushButton::toggled, stepScroll, [stepScroll, b](bool checked) {
      if (checked) QTimer::singleShot(0, stepScroll, [stepScroll, b] { stepScroll->ensureWidgetVisible(b, 4, 0); });
    });
    stepLayout->addWidget(b, 1);
    if (i < 6) {
      auto* arrow = new QLabel(">");
      arrow->setObjectName("workflowSeparator");
      stepLayout->addWidget(arrow);
    }
  }
  auto* parameters = action(tr("参数"), [this] { configure(); });
  parameters->setObjectName("workflowParameters");
  flow->addWidget(parameters);
  pause_ = action(tr("暂停"), [this] { runner_.pauseOrResume(); });
  pause_->setObjectName("workflowPause");
  pause_->setToolTip(
      tr("在当前 CLI 计算步骤完成后暂停后续队列；当前步骤不会被冻结。"));
  flow->addWidget(pause_);
  start_ = action(tr("启动"), [this] {
    if (runner_.busy()) {
      if (QMessageBox::question(
              this, tr("中止当前任务"),
              tr("结束本次计算并保留已有结果？未完成产物不会被标记为成功。")) ==
          QMessageBox::Yes)
        runner_.abort();
    } else
      startTask(true);
  });
  start_->setObjectName("primary");
  flow->addWidget(start_);
  outer->addWidget(flowBar);
  auto* split = new QSplitter;
  split->setObjectName("workspaceSplitter");
  split->setChildrenCollapsible(false);
  outer->addWidget(split, 1);
  split->setContentsMargins(0, 0, 0, 0);
  auto* left = new QSplitter(Qt::Vertical);
  left->setObjectName("leftSplitter");
  left->setChildrenCollapsible(false);
  split->addWidget(left);
  auto* projectPanel = panel();
  auto* projectLayout = new QVBoxLayout(projectPanel);
  projectLayout->setContentsMargins(0, 0, 0, 0);
  projectLayout->setSpacing(0);
  auto* projectHeading = new QLabel(tr("工作区"));
  projectHeading->setObjectName("panelHeading");
  projectLayout->addWidget(projectHeading);
  tree_ = new QTreeWidget;
  tree_->setItemDelegate(new UntintedIconDelegate(tree_));
  tree_->setHeaderHidden(true);
  tree_->setObjectName("projectTree");
  tree_->setColumnCount(2);
  tree_->header()->setStretchLastSection(false);
  tree_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
  tree_->header()->setSectionResizeMode(1, QHeaderView::Fixed);
  tree_->setColumnWidth(1, 28);
  tree_->setSelectionBehavior(QAbstractItemView::SelectRows);
  tree_->setUniformRowHeights(true);
  tree_->setIndentation(18);
  tree_->setIconSize({17, 17});
  projectLayout->addWidget(tree_, 1);
  auto* inputs = new QHBoxLayout;
  inputs->setContentsMargins(10, 8, 10, 8);
  auto* add = action(tr("添加照片"), [this] { addPhotos(false); });
  add->setObjectName("addPhotos");
  add->setIcon(fluentIcon(0xe91b));
  inputs->addWidget(add);
  auto* folder = action(tr("添加文件夹"), [this] { addPhotos(true); });
  folder->setObjectName("addFolder");
  folder->setIcon(fluentIcon(0xe8e5));
  inputs->addWidget(folder);
  projectLayout->addLayout(inputs);
  auto* relocate = action(tr("原图目录丢失 · 重新定位…"), [this] { editProject(); });
  relocate->setObjectName("managePhotos");
  relocate->hide();
  projectLayout->addWidget(relocate);
  left->addWidget(projectPanel);
  auto* propertyPanel = panel();
  auto* propertyLayout = new QVBoxLayout(propertyPanel);
  propertyLayout->setContentsMargins(0, 0, 0, 0);
  propertyLayout->setSpacing(0);
  inspector_ = new InspectorPanel;
  propertyLayout->addWidget(inspector_, 1);
  left->addWidget(propertyPanel);
  left->setSizes({480, 310});
  auto* center = panel();
  auto* centerLayout = new QVBoxLayout(center);
  centerLayout->setContentsMargins(0, 0, 0, 0);
  centerLayout->setSpacing(0);
  split->addWidget(center);
  views_ = new QTabWidget;
  views_->setObjectName("centralViews");
  scene_ = new SceneView;
  auto* scenePage = new QWidget;
  auto* sceneLayout = new QVBoxLayout(scenePage);
  sceneLayout->setContentsMargins(0, 0, 0, 0);
  sceneLayout->setSpacing(0);
  sceneLayout->addWidget(scene_, 1);
  views_->addTab(scenePage, tr("3D视图"));
  features_ = new FeatureView;
  views_->addTab(features_, tr("2D视图"));
  centerLayout->addWidget(views_, 1);
  auto* sceneTools = new QWidget;
  auto* checks = new QHBoxLayout(sceneTools);
  checks->setContentsMargins(9, 5, 9, 5);
  for (int n = 0; n < 3; ++n) {
    auto* check =
        new QCheckBox(QStringList{tr("点云"), tr("相机"), tr("地面网格")}[n]);
    check->setChecked(true);
    checks->addWidget(check);
    connect(check, &QCheckBox::toggled, this, [this, n](bool on) {
      if (n == 0)
        scene_->setCloud(on);
      else if (n == 1)
        scene_->setCameras(on);
      else
        scene_->setGrid(on);
    });
  }
  checks->addStretch();
  checks->addWidget(action(tr("适配"), [this] { scene_->fit(); }));
  checks->addWidget(action(tr("顶视"), [this] { scene_->top(); }));
  sceneLayout->insertWidget(0, sceneTools);
  connect(views_, &QTabWidget::currentChanged, this,
          [this, sceneTools](int i) { sceneTools->setVisible(i == 0); });
  auto* taskStrip = new QWidget;
  taskStrip->setObjectName("taskStrip");
  taskStrip->setFixedHeight(52);
  auto* taskRow = new QHBoxLayout(taskStrip);
  taskRow->setContentsMargins(12, 7, 12, 7);
  auto* taskIcon = new QLabel;
  taskIcon->setPixmap(fluentIcon(0xe895, QColor("#76b69c"), 23).pixmap(23));
  taskRow->addWidget(taskIcon);
  taskTitle_ = new QLabel(tr("尚无任务"));
  taskState_ = new QLabel(tr("就绪"));
  taskTitle_->setObjectName("taskTitle");
  taskState_->setObjectName("taskState");
  auto* taskLabels = new QVBoxLayout;
  taskLabels->addWidget(taskTitle_);
  taskLabels->addWidget(taskState_);
  taskRow->addLayout(taskLabels, 1);
  progress_ = new QProgressBar;
  progress_->setMaximumWidth(260);
  progress_->setRange(0, 1);
  progress_->setValue(0);
  taskRow->addWidget(progress_);
  auto* openLog = action(tr("打开日志"), [this] {
    if (!runner_.logPath().isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(runner_.logPath()));
  });
  openLog->setObjectName("openTaskLog");
  taskRow->addWidget(openLog);
  auto* errorDetails = action(tr("错误详情"), [this] {
    QMessageBox message(QMessageBox::Warning, tr("任务未完成"),
                        runner_.lastError().section('\n', 0, 0), QMessageBox::Ok, this);
    message.setDetailedText(runner_.lastError());
    message.exec();
  });
  errorDetails->setObjectName("taskErrorDetails");
  errorDetails->hide();
  taskRow->addWidget(errorDetails);
  centerLayout->addWidget(taskStrip);
  taskStrip->hide();
  logs_ = new QPlainTextEdit;
  logs_->setObjectName("taskLog");
  logs_->setReadOnly(true);
  logs_->setMaximumBlockCount(2000);
  logs_->setLineWrapMode(QPlainTextEdit::WidgetWidth);
  logs_->setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);
  logs_->setPlaceholderText(tr("运行流程后，这里显示计算日志。"));
  rightTabs_ = new QTabWidget;
  rightTabs_->setObjectName("rightTabs");
  auto* right = panel();
  auto* photoLayout = new QVBoxLayout(right);
  photoLayout->setContentsMargins(0, 0, 0, 0);
  photoLayout->setSpacing(6);
  auto* photoHeader = new QWidget;
  photoHeader->setObjectName("photoHeader");
  auto* photoHeaderRow = new QHBoxLayout(photoHeader);
  photoHeaderRow->setContentsMargins(14, 9, 14, 9);
  auto* photoSearch = new QLineEdit;
  photoSearch->setPlaceholderText(tr("搜索照片…"));
  photoHeaderRow->addWidget(photoSearch, 1);
  auto* photoCount = new QLabel(tr("0 张"));
  photoCount->setObjectName("photoCount");
  photoHeaderRow->addWidget(photoCount);
  photoLayout->addWidget(photoHeader);
  auto* pairHint = new QLabel(tr("选择照片 · Shift 选择配对"));
  pairHint->setObjectName("pairHint");
  photoLayout->addWidget(pairHint);
  connect(features_, &FeatureView::viewModeChanged, pairHint, [this, pairHint] {
    pairHint->setVisible(features_->matching());
  });
  pairHint->hide();
  photos_ = new PhotoList;
  static_cast<PhotoList*>(photos_)->visibleChanged = [this] { thumbnailTimer_.start(); };
  connect(photos_->verticalScrollBar(), &QScrollBar::valueChanged, this, [this] { thumbnailTimer_.start(); });
  photos_->setItemDelegate(new UntintedIconDelegate(photos_));
  photos_->setObjectName("photos");
  photos_->setViewMode(QListView::IconMode);
  photos_->setResizeMode(QListView::Adjust);
  photos_->setMovement(QListView::Static);
  photos_->setFlow(QListView::LeftToRight);
  photos_->setWrapping(true);
  photos_->setUniformItemSizes(true);
  photos_->setIconSize({146, 98});
  photos_->setGridSize({152, 138});
  photos_->setWordWrap(false);
  photoLayout->addWidget(photos_);
  rightTabs_->addTab(right, tr("照片"));
  auto* logPage = new QWidget;
  auto* logLayout = new QVBoxLayout(logPage);
  logLayout->setContentsMargins(10, 8, 10, 8);
  auto* logActions = new QHBoxLayout;
  auto* followLog = new QCheckBox(tr("跟随最新"));
  followLog->setObjectName("followLog"); followLog->setChecked(true);
  logActions->addWidget(followLog); logActions->addStretch();
  auto* logFile = action(tr("打开文件"), [this] {
    if (!runner_.logPath().isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(runner_.logPath()));
  });
  logFile->setObjectName("openLogFile"); logActions->addWidget(logFile);
  logLayout->addLayout(logActions); logLayout->addWidget(logs_, 1);
  rightTabs_->addTab(logPage, tr("日志"));
  auto* photoToggle = view->addAction(tr("显示照片 / 日志栏"));
  photoToggle->setObjectName("showRightPanel");
  photoToggle->setCheckable(true);
  photoToggle->setChecked(true);
  connect(photoToggle, &QAction::toggled, rightTabs_, &QWidget::setVisible);
  split->addWidget(rightTabs_);
  left->setMinimumWidth(284);
  rightTabs_->setMinimumWidth(240);
  split->setSizes({304, 866, 362});
  split->setStretchFactor(1, 1);
  connect(
      photoSearch, &QLineEdit::textChanged, this, [this](const QString& query) {
        for (int i = 0; i < photos_->count(); ++i)
          photos_->item(i)->setHidden(
              !photos_->item(i)->text().contains(query, Qt::CaseInsensitive));
        thumbnailTimer_.start();
      });
  connect(photos_, &QListWidget::currentRowChanged, this, [this](int i) {
    if (!updating_ && !(QApplication::keyboardModifiers() & Qt::ShiftModifier)) selectImage(i);
  });
  connect(photos_, &QListWidget::itemClicked, this, [this](QListWidgetItem* item) {
    if (!(QApplication::keyboardModifiers() & Qt::ShiftModifier)) return;
    features_->selectPairImage(photos_->row(item));
    views_->setCurrentIndex(1);
    sync2DSelection();
  });
  connect(photos_, &QListWidget::itemDoubleClicked, this,
          [this] { features_->setViewMode(FeatureView::Mode::Photos); views_->setCurrentIndex(1); sync2DSelection(); });
  connect(features_, &FeatureView::imageChanged, this, [this](int index) {
    updating_ = true;
    photos_->setCurrentRow(index);
    updating_ = false;
    displayImage(index);
  });
  connect(features_, &FeatureView::selectionChanged, this,
          [this](const QString& text) {
            syncPhotoBadges();
            if (views_->currentIndex() == 1 &&
                (selectedKind_ == 1 || selectedKind_ == 13)) {
              selectionDetail_ = text;
              refreshInspector();
            }
          });
  connect(features_, &FeatureView::viewModeChanged, this, [this] {
    if (views_->currentIndex() == 1) sync2DSelection();
  });
  connect(scene_, &SceneView::selected, this, [this](const QString& text) {
    selectionDetail_ = text;
    refreshInspector();
  });
  connect(tree_, &QTreeWidget::currentItemChanged, this,
          [this](QTreeWidgetItem* item, QTreeWidgetItem*) {
            if (!item || updating_) return;
            const int kind = item->data(0, Qt::UserRole).toInt();
            selectedKind_ = kind;
            selectionDetail_.clear();
            if (kind == 12) {
              features_->setViewMode(FeatureView::Mode::Photos);
              views_->setCurrentIndex(1);
              selectImage(item->data(0, Qt::UserRole + 1).toInt());
              return;
            }
            if (kind == 11) {
              selectedCamera_ = item->data(0, Qt::UserRole + 1).toUInt();
              refreshInspector();
              return;
            }
            if (kind == 1 || kind == 13) {
              navigate(kind == 13 ? 2 : 1);
              return;
            }
            if (kind == 10) {
              navigate(0);
              return;
            }
            if (kind >= 2 && kind <= 5) {
              if (kind == 2) {
                const int model = item->data(0, Qt::UserRole + 3).toInt();
                if (model != project_.parameters.value("sparse_model").toInt(-1)) {
                  if (runner_.busy()) { syncTreeSelection(selectedKind_); return; }
                  project_.parameters["sparse_model"] = model;
                  markDirty();
                }
              }
              views_->setCurrentIndex(0);
              if (project_.workspace.isEmpty()) return;
              loadScene(kind == 2   ? project_.activeSparse()
                        : kind == 3 ? QDir(project_.dense).filePath("fused.ply")
                        : kind == 4 ? project_.mesh
                                    : QFileInfo(project_.mesh)
                                          .dir()
                                          .filePath("textured/mesh.ply"));
              return;
            }
            refreshInspector();
          });
  connect(views_, &QTabWidget::currentChanged, this, [this](int index) {
    if (index == 0 && (selectedKind_ < 2 || selectedKind_ > 5)) {
      selectedKind_ =
          inspectedScene_ == project_.mesh ? 4
          : inspectedScene_ ==
                  QFileInfo(project_.mesh).dir().filePath("textured/mesh.ply")
              ? 5
          : inspectedScene_ == QDir(project_.dense).filePath("fused.ply") ? 3
                                                                          : 2;
      syncTreeSelection(selectedKind_);
      draft_ = selectedKind_ + 1;
      for (int i = 0; i < stages_.size(); ++i) stages_[i]->setChecked(i == draft_);
      updateStagePresentation();
    } else if (index == 1) {
      sync2DSelection();
    }
    refreshInspector();
    syncPhotoBadges();
  });
  connect(views_, &QTabWidget::currentChanged, pairHint, [this, pairHint](int index) {
    pairHint->setVisible(index == 1 && features_->matching());
  });
  connect(tree_, &QTreeWidget::itemClicked, this,
          [this](QTreeWidgetItem* item, int column) {
            const int kind = item->data(0, Qt::UserRole).toInt();
            if (column != 1 || kind < 2 || kind > 5) return;
            const bool visible = !layerVisibility_.value(kind, true);
            layerVisibility_[kind] = visible;
            item->setIcon(
                1, QIcon(visible ? ":/ui/Eye.png" : ":/ui/EyeSlash.png"));
            item->setToolTip(1, visible ? tr("隐藏模型") : tr("显示模型"));
            scene_->setLayerVisible(visible);
            refreshInspector();
          });
  connect(&runner_, &TaskRunner::output, this, [this](const QString& text) {
    const int scroll = logs_->verticalScrollBar()->value();
    QTextCursor cursor(logs_->document()); cursor.movePosition(QTextCursor::End);
    cursor.insertText(text);
    if (findChild<QCheckBox*>("followLog")->isChecked())
      logs_->verticalScrollBar()->setValue(logs_->verticalScrollBar()->maximum());
    else logs_->verticalScrollBar()->setValue(scroll);
  });
  connect(&runner_, &TaskRunner::changed, this, &MainWindow::updateTask);
  connect(&runner_, &TaskRunner::progressChanged, this, &MainWindow::updateTask);
  connect(&runner_, &TaskRunner::finished, this, [this](bool ok) {
    if (!ok) { findChild<QAction*>("showRightPanel")->setChecked(true); rightTabs_->setCurrentIndex(1); }
    if (ok) reload();
    else refreshStageStates();
  });
  setCentralWidget(body);
  setWindowIcon(QIcon(":/ui/colite.png"));
  auto* statusState = new QLabel(tr("就绪"));
  statusState->setObjectName("statusState");
  statusBar()->addWidget(statusState, 1);
  sceneStatus_ = new QLabel;
  sceneStatus_->setObjectName("sceneLoadStatus");
  sceneStatus_->setMinimumWidth(200);
  statusBar()->addPermanentWidget(sceneStatus_);
  sceneProgress_ = new QProgressBar;
  sceneProgress_->setObjectName("sceneLoadProgress");
  sceneProgress_->setFixedWidth(110);
  sceneProgress_->setTextVisible(false);
  sceneProgress_->hide();
  statusBar()->addPermanentWidget(sceneProgress_);
  auto* projectStatus = new QLabel(tr("未打开项目"));
  projectStatus->setObjectName("projectStatus");
  statusBar()->addPermanentWidget(projectStatus);
  auto* taskToggle = action(tr("任务"), [taskStrip] { taskStrip->setVisible(!taskStrip->isVisible()); });
  taskToggle->setObjectName("taskToggle");
  statusBar()->addPermanentWidget(taskToggle);
  QFile theme(":/ui/theme.qss");
  if (theme.open(QIODevice::ReadOnly))
    setStyleSheet(QString::fromUtf8(theme.readAll()));
  createProject();
}
MainWindow::~MainWindow() {
  if (stageStateCancel_) stageStateCancel_->store(true);
  if (thumbnailCancel_) thumbnailCancel_->store(true);
  thumbnailPool_.waitForDone();
  if (sceneCancel_) sceneCancel_->store(true);
  scenePool_.waitForDone();
}
void MainWindow::openProject(const QString& path) {
  if (runner_.busy()) {
    QMessageBox::information(this, tr("任务正在执行"),
                             tr("请先结束当前任务，再切换项目。"));
    return;
  }
  try {
    auto candidate = Project::load(path);
    candidate.backend.clear();
    candidate.backendMode = "native";
    if (!allowReplacement()) return;
    clearViews();
    logs_->clear();
    if (!runner_.restoreHistory(candidate))
      throw std::runtime_error("The previous backend is still stopping");
    project_ = candidate;
    dirty_ = false;
    updateTask();
    if (auto* split = findChild<QSplitter*>("workspaceSplitter"))
      split->restoreState(QByteArray::fromBase64(
          project_.parameters["gui_layout"].toString().toLatin1()));
    if (auto* split = findChild<QSplitter*>("leftSplitter"))
      split->restoreState(QByteArray::fromBase64(
          project_.parameters["gui_left"].toString().toLatin1()));
    const int oldTab = project_.parameters["gui_tab"].toInt();
    const int mode = project_.parameters.value("gui_2d_mode").toInt(oldTab == 2 ? 1 : 0);
    features_->setViewMode(static_cast<FeatureView::Mode>(qBound(0, mode, 2)));
    views_->setCurrentIndex(qBound(0, oldTab, 1));
    setWindowTitle(project_.name + tr(" — Colite"));
    reload();
    rememberProject(project_.file);
  } catch (const std::exception& e) {
    QMessageBox::critical(this, tr("打开项目失败"),
                          QString::fromUtf8(e.what()));
  }
}
void MainWindow::markDirty() {
  dirty_ = true;
  setWindowTitle(project_.name + tr(" * — Colite"));
  snapshotProject();
  refreshStageStates();
}
void MainWindow::refreshStageStates() {
  if (stageStateCancel_) stageStateCancel_->store(true);
  stageStateCancel_ = std::make_shared<std::atomic_bool>(false);
  const auto cancelled = stageStateCancel_;
  const auto project = project_;
  const bool hasImages = !workspace_.images.isEmpty();
  using States = QVector<QPair<OutputState, QString>>;
  auto* watcher = new QFutureWatcher<States>(this);
  connect(watcher, &QFutureWatcher<States>::finished, this, [this, watcher, cancelled] {
    const auto states = watcher->result(); watcher->deleteLater();
    if (cancelled->load()) return;
    for (int i = 0; i < states.size(); ++i) {
      const auto state = states[i].first;
      stages_[i]->setProperty("hasOutput", state != OutputState::Missing);
      stages_[i]->setProperty("outputStale", state == OutputState::Stale);
      stages_[i]->setProperty("outputUnverified", state == OutputState::Unverified);
      stages_[i]->setToolTip(stageNames[i] + " · " + states[i].second);
      stages_[i]->update();
    }
    updateStagePresentation();
  });
  watcher->setFuture(QtConcurrent::run([project, hasImages, cancelled] {
    States states;
    for (int i = 0; i <= 6; ++i) {
      if (cancelled->load()) return States{};
      try {
        const auto state = i == 0 ? (hasImages ? OutputState::Current : OutputState::Missing) : outputState(project, i);
        states.append({state, outputStateText(state)});
      } catch (const std::exception& e) { states.append({OutputState::Missing, QString::fromUtf8(e.what())}); }
    }
    return states;
  }));
}
void MainWindow::updateStagePresentation() {
  const auto taskState = runner_.state();
  for (int i = 0; i < stages_.size(); ++i) {
    auto* step = static_cast<WorkflowStep*>(stages_[i]);
    QString state = "pending", label = i == 0 ? tr("待添加") : tr("待运行");
    if (step->property("hasOutput").toBool()) {
      state = "complete"; label = i == 0 ? tr("已添加") : tr("已完成");
      if (step->property("outputStale").toBool()) { state = "stale"; label = tr("待更新"); }
      else if (step->property("outputUnverified").toBool()) { state = "unverified"; label = tr("待核验"); }
    }
    if (i > 0 && runner_.busy() && runner_.containsStage(i)) {
      if (runner_.completedStage(i)) { state = "complete"; label = tr("已完成"); }
      else { state = "queued"; label = tr("等待中"); }
    }
    if (i > 0 && runner_.currentStage() == i) {
      if (taskState == TaskRunner::State::Running) {
        state = "running"; label = runner_.validating() ? tr("校验中") : tr("运行中");
      } else if (taskState == TaskRunner::State::PauseRequested) { state = "running"; label = tr("等待暂停"); }
      else if (taskState == TaskRunner::State::Paused) { state = "paused"; label = tr("已暂停"); }
      else if (taskState == TaskRunner::State::Aborting) { state = "running"; label = tr("中止中"); }
      else if (taskState == TaskRunner::State::Failed) { state = "failed"; label = tr("失败"); }
      else if (taskState == TaskRunner::State::Aborted) { state = "aborted"; label = tr("已中止"); }
    }
    step->setStatus(state, label);
    step->setToolTip(stageNames[i] + " · " + label + tr("\n单击查看；顶部“参数”或右键可配置当前节点。"));
  }
  if (auto* button = findChild<QPushButton*>("workflowParameters"))
    button->setToolTip(tr("配置当前节点：%1；可切换其他流程参数页。").arg(stageNames[qBound(0, draft_, 6)]));
  if (start_ && !runner_.busy())
    start_->setToolTip(tr("直接从%1运行到纹理映射，使用已保存的参数。").arg(stageNames[qBound(1, draft_, 6)]));
}
void MainWindow::snapshotProject() {
  if (!recoveryEnabled_ || !dirty_) return;
  try { recovery_.save(project_); }
  catch (const std::exception& e) {
    statusBar()->showMessage(tr("恢复草稿保存失败，请手动保存工程：%1").arg(QString::fromUtf8(e.what())), 15000);
  }
}
void MainWindow::clearRecovery() {
  if (!recoveryEnabled_) return;
  try { recovery_.clear(); }
  catch (const std::exception& e) { statusBar()->showMessage(QString::fromUtf8(e.what()), 15000); }
}
void MainWindow::offerRecovery() {
  if (!recoveryEnabled_ || runner_.busy()) return;
  const auto paths = recovery_.available();
  if (paths.isEmpty()) { statusBar()->showMessage(tr("没有可恢复的未保存工程"), 4000); return; }
  QStringList labels;
  for (const auto& path : paths) {
    try {
      const auto project = Project::load(path);
      labels << tr("%1 · %2").arg(project.name, QFileInfo(path).lastModified().toString("yyyy-MM-dd HH:mm:ss"));
    } catch (...) { labels << tr("无法读取的恢复记录 · %1").arg(QFileInfo(path).fileName()); }
  }
  QDialog dialog(this);
  dialog.setWindowTitle(tr("恢复未保存工程"));
  auto* layout = new QVBoxLayout(&dialog);
  auto* explanation = new QLabel(tr("发现上次未正常关闭的工程。恢复后可检查并保存；取消会保留草稿。"));
  explanation->setWordWrap(true);
  layout->addWidget(explanation);
  auto* choices = new QComboBox;
  choices->setObjectName("recoveryRecords");
  for (int i = 0; i < paths.size(); ++i) {
    choices->addItem(labels[i], paths[i]);
    choices->setItemData(i, paths[i], Qt::ToolTipRole);
  }
  layout->addWidget(choices);
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Ok)->setText(tr("恢复"));
  buttons->button(QDialogButtonBox::Cancel)->setText(tr("取消"));
  connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  layout->addWidget(buttons);
  if (dialog.exec() != QDialog::Accepted || !allowReplacement()) return;
  try {
    auto project = recovery_.restore(choices->currentData().toString());
    clearViews(); logs_->clear();
    project_ = project;
    project_.backend.clear(); project_.backendMode = "native";
    runner_.restoreHistory(project_);
    markDirty(); reload(); updateTask();
  } catch (const std::exception& e) {
    QMessageBox::warning(this, tr("恢复失败"), QString::fromUtf8(e.what()));
  }
}
bool MainWindow::allowReplacement() {
  if (!dirty_) return true;
  auto message = tr("是否保存「%1」的更改？").arg(project_.name);
  if (!project_.workspace.isEmpty())
    message += tr("\n放弃工程更改会保留磁盘上的计算结果：%1").arg(project_.workspace);
  QDialog dialog(this);
  dialog.setWindowTitle(tr("保存工程"));
  auto* layout = new QVBoxLayout(&dialog);
  auto* label = new QLabel(message);
  label->setTextFormat(Qt::PlainText);
  label->setWordWrap(true);
  layout->addWidget(label);
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Discard |
                                       QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Save)->setText(tr("保存"));
  buttons->button(QDialogButtonBox::Save)->setDefault(true);
  buttons->button(QDialogButtonBox::Discard)->setText(tr("放弃更改"));
  buttons->button(QDialogButtonBox::Cancel)->setText(tr("取消"));
  layout->addWidget(buttons);
  connect(buttons, &QDialogButtonBox::clicked, &dialog, [&](QAbstractButton* button) {
    dialog.done(buttons->standardButton(button));
  });
  const int answer = dialog.exec();
  if (answer == QDialogButtonBox::Discard) { clearRecovery(); return true; }
  return answer == QDialogButtonBox::Save && saveProject();
}
void MainWindow::clearViews() {
  if (stageStateCancel_) stageStateCancel_->store(true);
  if (thumbnailCancel_) thumbnailCancel_->store(true);
  thumbnailCache_.clear();
  if (sceneCancel_) sceneCancel_->store(true);
  sceneCache_.clear();
  displayedScene_.clear(); displayedStamp_.clear();
  sceneStatus_->clear(); sceneProgress_->hide();
  ++loadGeneration_; ++sceneGeneration_;
  workspace_ = {};
  updating_ = true;
  photos_->clear(); tree_->clear();
  updating_ = false;
  selectedImage_ = -1;
  inspectedScene_.clear(); sceneError_.clear(); selectionDetail_.clear();
  scenePoints_ = sceneTotalTriangles_ = sceneDisplayed_ = sceneTriangles_ = sceneCameras_ = 0;
  sceneLoading_ = false;
  scene_->setScene({});
  features_->setWorkspace(workspace_);
}
void MainWindow::createProject() {
  if (runner_.busy() || !allowReplacement()) return;
  project_ = Project{};
  project_.name = tr("未命名工程");
  dirty_ = false;
  clearViews();
  logs_->clear();
  runner_.restoreHistory(project_);
  setWindowTitle(project_.name + tr(" — Colite"));
  navigate(0);
  reload();
  updateTask();
}
bool MainWindow::saveProject(bool saveAs) {
  QString path = saveAs ? QString() : project_.file;
  auto saved = project_;
  if (path.isEmpty()) {
    path = QFileDialog::getSaveFileName(this, tr("保存工程"),
        QDir(project_.workDirectory).filePath(project_.name + ".colite.json"),
        tr("Colite 工程 (*.colite.json)"));
    if (path.isEmpty()) return false;
    if (!path.endsWith(".colite.json", Qt::CaseInsensitive)) {
      path += ".colite.json";
      if (QFileInfo::exists(path) && QMessageBox::question(this, tr("覆盖工程"),
          tr("文件已存在，是否替换 %1？").arg(path)) != QMessageBox::Yes) return false;
    }
    saved.name = QFileInfo(path).fileName().chopped(QString(".colite.json").size());
  }
  try {
    if (auto* split = findChild<QSplitter*>("workspaceSplitter"))
      saved.parameters["gui_layout"] =
          QString::fromLatin1(split->saveState().toBase64());
    if (auto* split = findChild<QSplitter*>("leftSplitter"))
      saved.parameters["gui_left"] =
          QString::fromLatin1(split->saveState().toBase64());
    saved.parameters["gui_tab"] = views_->currentIndex();
    saved.parameters["gui_2d_mode"] = int(features_->viewMode());
    saved.save(path);
    project_ = saved;
    dirty_ = false;
    clearRecovery();
    if (tree_->topLevelItemCount()) tree_->topLevelItem(0)->setText(0, project_.name);
    setWindowTitle(project_.name + tr(" — Colite"));
    rememberProject(project_.file);
    statusBar()->showMessage(tr("项目已保存"), 3000);
    return true;
  } catch (const std::exception& e) {
    QMessageBox::critical(this, tr("保存失败"), QString::fromUtf8(e.what()));
    return false;
  }
}
void MainWindow::rememberProject(const QString& path) {
  if (path.isEmpty()) return;
  const auto absolute = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
  QSettings settings;
  auto paths = settings.value(kRecentProjectsKey).toStringList();
#ifdef Q_OS_WIN
  constexpr auto caseSensitivity = Qt::CaseInsensitive;
#else
  constexpr auto caseSensitivity = Qt::CaseSensitive;
#endif
  for (int i = paths.size() - 1; i >= 0; --i)
    if (paths[i].compare(absolute, caseSensitivity) == 0) paths.removeAt(i);
  paths.prepend(absolute);
  while (paths.size() > kRecentProjectCount) paths.removeLast();
  settings.setValue(kRecentProjectsKey, paths);
}
void MainWindow::refreshRecentProjects() {
  recentProjects_->clear();
  QSettings settings;
  const auto paths = settings.value(kRecentProjectsKey).toStringList().mid(0, kRecentProjectCount);
  if (paths.isEmpty()) {
    recentProjects_->addAction(tr("暂无最近工程"))->setEnabled(false);
    return;
  }
  QHash<QString, int> names;
  for (const auto& path : paths) ++names[recentProjectName(path).toCaseFolded()];
  QStringList available;
  for (const auto& path : paths) {
    const auto name = recentProjectName(path);
    const auto nativePath = QDir::toNativeSeparators(path);
    const bool exists = QFileInfo(path).isFile();
    QString label = names.value(name.toCaseFolded()) > 1 ? tr("%1  ·  %2").arg(name, nativePath) : name;
    if (!exists) label = tr("%1（文件已移动或删除）").arg(label);
    label.replace('&', "&&");
    auto* item = recentProjects_->addAction(label, this, [this, path] { openProject(path); });
    item->setData(path);
    item->setToolTip(nativePath);
    item->setEnabled(exists);
    if (exists) available << path;
  }
  if (available.size() != paths.size()) {
    recentProjects_->addSeparator();
    recentProjects_->addAction(tr("移除失效记录"), this, [available] {
      QSettings settings;
      settings.setValue(kRecentProjectsKey, available);
    });
  }
}
void MainWindow::exportSelectedScene() {
  if (runner_.busy() || sceneLoading_) { statusBar()->showMessage(tr("请等待计算或模型加载完成后导出"), 5000); return; }
  if (views_->currentIndex() != 0 || inspectedScene_.isEmpty() || !QFileInfo::exists(inspectedScene_)) {
    QMessageBox::information(this, tr("导出模型"), tr("请先选择一个稀疏模型、点云、网格或纹理模型。")); return;
  }
  const auto destination = QFileDialog::getSaveFileName(this, tr("导出到新文件夹"),
      QDir(project_.workDirectory).filePath("colite-export"), tr("导出文件夹 (*)"), nullptr, QFileDialog::DontConfirmOverwrite);
  if (destination.isEmpty()) return;
  QProgressDialog progress(tr("正在复制并检查导出模型…"), tr("取消"), 0, 0, this);
  progress.setWindowModality(Qt::WindowModal);
  auto cancelled = std::make_shared<std::atomic_bool>(false);
  connect(&progress, &QProgressDialog::canceled, &progress, [cancelled] { cancelled->store(true); });
  QFutureWatcher<QString> watcher;
  connect(&watcher, &QFutureWatcher<QString>::finished, &progress, &QDialog::accept);
  const auto source = inspectedScene_;
  watcher.setFuture(QtConcurrent::run([source, destination, cancelled] {
    try { exportScene(source, destination, [cancelled] { return cancelled->load(); }); return QString(); }
    catch (const std::exception& e) { return QString::fromUtf8(e.what()); }
  }));
  progress.exec(); watcher.waitForFinished();
  if (watcher.result().isEmpty()) statusBar()->showMessage(tr("模型已导出：%1").arg(destination), 15000);
  else if (cancelled->load()) statusBar()->showMessage(tr("已取消导出"), 5000);
  else QMessageBox::warning(this, tr("导出失败"), watcher.result());
}
void MainWindow::addPhotos(bool folder) {
  if (runner_.busy()) return;
  QString directory;
  QStringList added;
  if (folder) {
    directory = QFileDialog::getExistingDirectory(this, tr("添加照片文件夹"));
    if (directory.isEmpty()) return;
    QDirIterator files(directory, QDir::Files, QDirIterator::Subdirectories);
    while (files.hasNext()) {
      const auto file = files.next();
      if (QStringList{"jpg","jpeg","png"}.contains(QFileInfo(file).suffix().toLower()))
        added << file;
    }
  } else {
    added = QFileDialog::getOpenFileNames(this, tr("添加照片"), {},
        tr("照片 (*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG)"));
  }
  if (added.isEmpty()) {
    if (folder) statusBar()->showMessage(tr("所选目录没有 JPEG 或 PNG 照片。"), 5000);
    return;
  }
  auto files = project_.photoFiles;
  if (files.isEmpty() && !project_.imageRoot.isEmpty()) {
    try {
      for (const auto& image : Workspace::read(project_).images) files.insert(image.name, image.path);
    } catch (const std::exception& e) {
      QMessageBox::warning(this, tr("读取照片失败"), QString::fromUtf8(e.what())); return;
    }
  }
  QSet<QString> existing;
  for (const auto& file : files) existing.insert(QFileInfo(file).canonicalFilePath());
  const int before = files.size();
  for (const auto& file : added) {
    const auto canonical = QFileInfo(file).canonicalFilePath();
    if (existing.contains(canonical)) continue;
    const auto name = folder ? QFileInfo(directory).fileName() + "/" + QDir(directory).relativeFilePath(file)
                             : QFileInfo(file).fileName();
    if (files.contains(name)) {
      QMessageBox::warning(this, tr("同名照片"), tr("已存在另一张名为 %1 的照片。可按文件夹添加以保留分组。").arg(name));
      return;
    }
    files.insert(name, file);
    existing.insert(canonical);
  }
  if (before == files.size()) return;
  if (folder && before == 0) {
    project_.photoFiles.clear();
    project_.imageRoot = directory;
  } else {
    project_.photoFiles = files;
    project_.imageRoot.clear();
  }
  if (!project_.workDirectory.isEmpty()) project_.assignOutputs(project_.workDirectory);
  else { project_.workspace.clear(); project_.sparse.clear(); project_.dense.clear(); project_.mesh.clear(); }
  clearViews();
  markDirty();
  navigate(0); reload(); updateTask();
}
bool MainWindow::editProject() {
  if (runner_.busy()) return false;
  QDialog dialog(this);
  dialog.setWindowTitle(tr("项目数据"));
  auto* form = new QFormLayout(&dialog);
  auto row = [&](const QString& label, const QString& value) {
    auto* layout = new QHBoxLayout;
    auto* edit = new QLineEdit(value);
    layout->addWidget(edit);
    layout->addWidget(action(tr("浏览…"), [&, edit, label] {
      const auto p = QFileDialog::getExistingDirectory(&dialog, label, edit->text());
      if (!p.isEmpty()) edit->setText(p);
    }));
    form->addRow(label, layout); return edit;
  };
  auto* source = row(tr("数据源目录"), project_.photoFiles.isEmpty() ? project_.imageRoot : QString());
  if (!project_.photoFiles.isEmpty()) source->setPlaceholderText(tr("已引用 %1 张照片，选择目录可替换").arg(project_.photoFiles.size()));
  auto* work = row(tr("工作目录"), project_.workDirectory.isEmpty()
      ? Project::defaultWorkDirectory() : project_.workDirectory);
  work->setObjectName("projectWorkDirectory");
  work->setPlaceholderText(Project::defaultWorkDirectory());
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  form->addRow(buttons);
  connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
  if (dialog.exec() != QDialog::Accepted) return false;
  const auto images = source->text().trimmed();
  const auto root = work->text().trimmed();
  if ((!images.isEmpty() && (!QFileInfo(images).isDir() || !QDir::isAbsolutePath(images))) ||
      (!root.isEmpty() && !QDir::isAbsolutePath(root))) {
    QMessageBox::warning(this, tr("路径无效"), tr("请选择有效的数据源目录和绝对工作路径。")); return false;
  }
  const bool replace = !images.isEmpty() && (images != project_.imageRoot || !project_.photoFiles.isEmpty());
  const bool relocate = replace && project_.photoFiles.isEmpty() && !QFileInfo(project_.imageRoot).isDir()
      && !project_.imageRoot.isEmpty();
  if (!replace && root == project_.workDirectory) return true;
  if (replace) { project_.imageRoot = images; project_.photoFiles.clear(); }
  if (root != project_.workDirectory || (replace && !relocate)) {
    if (!root.isEmpty()) project_.assignOutputs(root);
    else { project_.workDirectory.clear(); project_.workspace.clear(); project_.sparse.clear(); project_.dense.clear(); project_.mesh.clear(); }
  }
  clearViews(); markDirty(); reload(); updateTask(); return true;
}
void MainWindow::reload() {
  if (thumbnailCancel_) thumbnailCancel_->store(true);
  if (sceneCancel_) sceneCancel_->store(true);
  ++sceneGeneration_;
  sceneLoading_ = false;
  sceneCache_.clear(); displayedStamp_.clear();
  scene_->setLoading({});
  sceneProgress_->hide(); sceneStatus_->clear();
  const auto project = project_;
  const bool missing = !project.imageRoot.isEmpty() && !QFileInfo(project.imageRoot).isDir();
  if (auto* relocate = findChild<QPushButton*>("managePhotos"))
    relocate->setVisible(missing);
  const int token = ++loadGeneration_;
  auto* watcher = new QFutureWatcher<Workspace>(this);
  statusBar()->showMessage(tr("正在读取工作区…"));
  connect(watcher, &QFutureWatcher<Workspace>::finished, this,
          [this, watcher, token] {
            try {
              auto result = watcher->result();
              watcher->deleteLater();
              if (token != loadGeneration_) return;
              if (!result.error.isEmpty())
                throw std::runtime_error(result.error.toStdString());
              workspace_ = std::move(result);
              updateTask();
              refreshStageStates();
              if (auto* status = findChild<QLabel*>("projectStatus"))
                status->setText(tr("%1 张图像").arg(workspace_.images.size()));
              if (auto* photoCount = findChild<QLabel*>("photoCount"))
                photoCount->setText(tr("%1 张").arg(workspace_.images.size()));
              updating_ = true;
              photos_->clear();
              for (const auto& image : workspace_.images) {
                auto* item = new QListWidgetItem(
                    QIcon(":/ui/Image.png"),
                    image.name);
                item->setToolTip(image.path);
                item->setSizeHint(photos_->gridSize());
                photos_->addItem(item);
              }
              updating_ = false;
              features_->setWorkspace(workspace_);
              if (!workspace_.images.isEmpty()) displayImage(0);
              updating_ = true;
              tree_->clear();
              auto* root = new QTreeWidgetItem(tree_, {project_.name});
              root->setData(0, Qt::UserRole, 0);
              root->setIcon(0, fluentIcon(0xe8e5));
              auto* images = new QTreeWidgetItem(
                  root, {tr("图像（%1）").arg(workspace_.images.size())});
              images->setData(0, Qt::UserRole, 10);
              images->setIcon(0, fluentIcon(0xe8f4));
              QMap<quint32, QTreeWidgetItem*> cameras;
              for (int i = 0; i < workspace_.images.size(); ++i) {
                const auto& record = workspace_.images[i];
                if (!cameras.contains(record.camera)) {
                  auto* camera = new QTreeWidgetItem(
                      images,
                      {record.camera ? QString("Camera %1").arg(record.camera)
                                     : tr("待分组图像")});
                  camera->setIcon(0, QIcon(":/ui/Camera.png"));
                  camera->setData(0, Qt::UserRole, 11);
                  camera->setData(0, Qt::UserRole + 1, record.camera);
                  cameras[record.camera] = camera;
                }
                auto* image =
                    new QTreeWidgetItem(cameras[record.camera], {record.name});
                image->setIcon(0, QIcon(":/ui/Image.png"));
                image->setData(0, Qt::UserRole, 12);
                image->setData(0, Qt::UserRole + 1, i);
                image->setToolTip(0, record.path);
              }
              for (const auto& entry : QList<QPair<int, QString>>{
                       {1, tr("特征提取")}, {13, tr("特征匹配")}}) {
                auto* inspection = new QTreeWidgetItem(images, {entry.second});
                inspection->setIcon(0, fluentIcon(0xe80a));
                inspection->setData(0, Qt::UserRole, entry.first);
              }
              auto* results = new QTreeWidgetItem(root, {tr("重建结果")});
              results->setIcon(0, fluentIcon(0xe8e5));
              results->setData(0, Qt::UserRole, 20);
              const QStringList names = {tr("稀疏主模型"), tr("稠密点云"),
                                         tr("网格"), tr("纹理模型")};
              for (int i = 0; i < names.size(); ++i) {
                auto* item = new QTreeWidgetItem(results, {names[i]});
                item->setData(0, Qt::UserRole, i + 2);
                item->setIcon(0, fluentIcon(0xe7f4));
                item->setIcon(1, QIcon(layerVisibility_.value(i + 2, true)
                                           ? ":/ui/Eye.png"
                                           : ":/ui/EyeSlash.png"));
                item->setToolTip(1, tr("显示 / 隐藏模型"));
                item->setData(0, Qt::UserRole + 3, -1);
                if (i == 0) {
                  QFile manifest(QDir(project_.sparse).filePath("models.txt"));
                  if (manifest.open(QIODevice::ReadOnly) && manifest.size() < 1024 * 1024) {
                    QTextStream text(&manifest);
                    while (!text.atEnd()) {
                      const auto line = text.readLine();
                      if (line.startsWith('#')) continue;
                      const auto fields = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
                      bool valid = false;
                      const int model = fields.value(0).toInt(&valid);
                      if (!valid || model < 0 || fields.size() < 4) continue;
                      auto* child = new QTreeWidgetItem(item, {tr("子模型 %1 · %2 张照片").arg(model).arg(fields[1])});
                      child->setData(0, Qt::UserRole, 2);
                      child->setData(0, Qt::UserRole + 3, model);
                      child->setIcon(0, fluentIcon(0xe7f4));
                      child->setToolTip(0, tr("选择后作为稠密重建的输入；已有下游结果会标记为过期。"));
                    }
                  }
                  item->setExpanded(true);
                }
              }
              root->setExpanded(true);
              images->setExpanded(true);
              results->setExpanded(true);
              updating_ = false;
              if (views_->currentIndex() == 1) sync2DSelection();
              else { selectedKind_ = 2; syncTreeSelection(selectedKind_); }
              syncPhotoBadges();
              refreshInspector();
              statusBar()->showMessage(tr("工作区已载入"), 3000);
              thumbnailTimer_.start();
              if (views_->currentIndex() == 0 && QFileInfo::exists(project_.sparse))
                loadScene(project_.activeSparse());
            } catch (const std::exception& e) {
              statusBar()->showMessage(QString::fromUtf8(e.what()));
            }
          });
  watcher->setFuture(QtConcurrent::run([project] {
    try {
      return Workspace::read(project);
    } catch (const std::exception& e) {
      Workspace w;
      w.error = QString::fromUtf8(e.what());
      return w;
    }
  }));
}
void MainWindow::loadVisibleThumbnails() {
  if (thumbnailsLoading_ || !photos_->isVisible()) return;
  QMap<int, QString> requested, keys;
  const auto visible = photos_->viewport()->rect().adjusted(0, -150, 0, 150);
  for (int i = 0; i < photos_->count() && i < workspace_.images.size(); ++i) {
    auto* item = photos_->item(i);
    if (item->isHidden() || !photos_->visualItemRect(item).intersects(visible)) {
      if (!item->data(Qt::UserRole + 4).toString().isEmpty()) {
        item->setIcon(QIcon(":/ui/Image.png")); item->setData(Qt::UserRole + 4, QString());
      }
      continue;
    }
    const auto& path = workspace_.images[i].path;
    const auto key = sceneStamp(path);
    if (item->data(Qt::UserRole + 4).toString() == key) continue;
    if (auto* cached = thumbnailCache_.object(key)) {
      item->setIcon(cached->isNull() ? QIcon(":/ui/Image.png") : QIcon(QPixmap::fromImage(*cached)));
      item->setData(Qt::UserRole + 4, key);
    } else if (requested.size() < 16) { requested[i] = path; keys[i] = key; }
  }
  if (requested.isEmpty()) return;
  thumbnailsLoading_ = true;
  thumbnailCancel_ = std::make_shared<std::atomic_bool>(false);
  const auto cancelled = thumbnailCancel_;
  const int token = loadGeneration_;
  auto* watcher = new QFutureWatcher<QMap<int, QImage>>(this);
  connect(watcher, &QFutureWatcher<QMap<int, QImage>>::finished, this, [this, watcher, token, keys] {
    const auto result = watcher->result(); watcher->deleteLater();
    thumbnailsLoading_ = false;
    if (token == loadGeneration_) {
      for (auto it = result.begin(); it != result.end(); ++it) {
        const int cost = int(qMax<qint64>(1, (it.value().sizeInBytes() + 1023) / 1024));
        thumbnailCache_.insert(keys.value(it.key()), new QImage(it.value()), cost);
      }
    }
    thumbnailTimer_.start();
  });
  watcher->setFuture(QtConcurrent::run(&thumbnailPool_, [requested, cancelled] {
    QMap<int, QImage> result;
    for (auto it = requested.begin(); it != requested.end(); ++it) {
      if (cancelled->load()) break;
      QImageReader reader(it.value());
      const auto size = reader.size();
      if (size.isValid()) reader.setScaledSize(size.scaled(300, 200, Qt::KeepAspectRatio));
      result[it.key()] = reader.read().scaled(300, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    return result;
  }));
}
void MainWindow::selectImage(int index) {
  if (index < 0 || index >= workspace_.images.size()) return;
  features_->selectImage(index);
}
void MainWindow::displayImage(int index) {
  if (index < 0 || index >= workspace_.images.size()) return;
  const QSignalBlocker blockPhotoSelection(photos_);
  photos_->setCurrentRow(index);
  selectedImage_ = index;
  if (views_->currentIndex() == 1) sync2DSelection();
  else {
    selectedKind_ = 12;
    syncTreeSelection(12, index);
  }
  syncPhotoBadges();
  refreshInspector();
}
void MainWindow::syncPhotoBadges() {
  const bool paired = features_->matching() && views_->currentIndex() == 1;
  for (int i = 0; i < photos_->count(); ++i)
    photos_->item(i)->setData(Qt::UserRole + 2, paired && i == features_->pairA()
        ? QString("A") : paired && i == features_->pairB() ? QString("B") : QString());
}
void MainWindow::sync2DSelection() {
  const auto mode = features_->viewMode();
  draft_ = int(mode);
  selectedKind_ = mode == FeatureView::Mode::Photos ? (selectedImage_ >= 0 ? 12 : 10)
                  : mode == FeatureView::Mode::Matches ? 13 : 1;
  syncTreeSelection(selectedKind_, selectedImage_);
  for (int i = 0; i < stages_.size(); ++i) stages_[i]->setChecked(i == draft_);
  updateStagePresentation();
  syncPhotoBadges();
  refreshInspector();
}
void MainWindow::loadScene(const QString& path) {
  if (path.isEmpty()) return;
  if (views_->currentIndex() == 0) {
    selectedKind_ =
        path == project_.mesh ? 4
        : path == QFileInfo(project_.mesh).dir().filePath("textured/mesh.ply")
            ? 5
        : path == QDir(project_.dense).filePath("fused.ply") ? 3
                                                             : 2;
    syncTreeSelection(selectedKind_);
    draft_ = selectedKind_ + 1;
    for (int i = 0; i < stages_.size(); ++i) stages_[i]->setChecked(i == draft_);
    updateStagePresentation();
  }
  if (sceneLoading_ && inspectedScene_ == path) return;
  if (sceneCancel_) sceneCancel_->store(true);
  sceneCancel_ = std::make_shared<std::atomic_bool>(false);
  const auto cancelled = sceneCancel_;
  inspectedScene_ = path;
  sceneClock_.start();
  sceneLoading_ = true;
  sceneError_.clear();
  selectionDetail_.clear();
  refreshInspector();
  const int token = ++sceneGeneration_;
  const int layerKind =
      path == project_.mesh                                                  ? 4
      : path == QFileInfo(project_.mesh).dir().filePath("textured/mesh.ply") ? 5
      : path == QDir(project_.dense).filePath("fused.ply") ? 3
                                                           : 2;
  const auto cached = sceneCache_.object(path);
  if (cached && cached->stamp == sceneStamp(path, cached->data.texture)) {
    applyScene(cached->data, layerKind, true, cached->stamp);
    return;
  }
  sceneCache_.remove(path);
  const QString initialStamp = sceneStamp(path);
  sceneStatus_->setText(tr("%1 · 等待加载").arg(stageNames[layerKind + 1]));
  sceneProgress_->setRange(0, 0);
  sceneProgress_->show();
  scene_->setLoading(tr("正在加载%1…").arg(stageNames[layerKind + 1]));
  auto* watcher = new QFutureWatcher<SceneData>(this);
  connect(watcher, &QFutureWatcher<SceneData>::finished, this,
          [this, watcher, token, layerKind, path, initialStamp] {
            const auto data = watcher->result();
            watcher->deleteLater();
            if (token != sceneGeneration_) return;
            if (initialStamp != sceneStamp(path) ||
                (!data.textureRevision.isEmpty() && data.textureRevision != fileRevision(data.texture))) {
              sceneLoading_ = false;
              loadScene(path);
              return;
            }
            const auto stamp = sceneStamp(path, data.texture);
            if (data.error.isEmpty() && initialStamp == sceneStamp(path)) {
              const qint64 bytes = (qint64(data.points.capacity()) + data.triangles.capacity() +
                  data.cameras.capacity() + data.navigationPoints.capacity()) * sizeof(SceneVertex) +
                  data.texturePreview.sizeInBytes();
              const int cost = int(qMax<qint64>(1, (bytes + 1048575) / 1048576));
              // Oversized results are still displayed, but do not evict the whole cache.
              if (cost <= sceneCache_.maxCost())
                sceneCache_.insert(path, new CachedScene{data, stamp}, cost);
            }
            applyScene(data, layerKind, false, stamp);
          });
  auto progress = [this, token, cancelled, last = QString(), clock = QElapsedTimer()]
      (const QString& stage, int percent) mutable {
    if (cancelled->load()) return;
    if (last == stage && percent != 100 && clock.isValid() && clock.elapsed() < 50) return;
    last = stage; clock.start();
    QMetaObject::invokeMethod(this, [this, token, stage, percent] {
      if (token != sceneGeneration_ || !sceneLoading_) return;
      sceneStatus_->setText(percent < 0 ? stage + tr("…") : stage + QString(" · %1%").arg(percent));
      sceneProgress_->setRange(0, percent < 0 ? 0 : 100);
      if (percent >= 0) sceneProgress_->setValue(percent);
    }, Qt::QueuedConnection);
  };
  watcher->setFuture(QtConcurrent::run(&scenePool_, [path, cancelled, progress]() mutable {
    try {
      auto scene = readScene(path, progress, [cancelled] { return cancelled->load(); });
      if (cancelled->load()) return SceneData{};
      progress(QString::fromUtf8("准备显示"), -1);
      SceneView::prepare(scene);
      return scene;
    } catch (const std::exception& e) {
      SceneData d;
      d.source = path;
      d.error = QString::fromUtf8(e.what());
      return d;
    }
  }));
}
void MainWindow::applyScene(const SceneData& model, int layerKind, bool cached, const QString& stamp) {
  if (!cached || displayedScene_ != model.source || displayedStamp_ != stamp)
    scene_->setScene(model, model.error.isEmpty() ? model.source + '\n' + stamp : QString());
  scene_->setLoading({});
  displayedScene_ = model.source;
  displayedStamp_ = stamp;
  scene_->setLayerVisible(layerVisibility_.value(layerKind, true));
  sceneLoading_ = false;
  sceneError_ = model.error;
  scenePoints_ = model.totalPoints;
  sceneTotalTriangles_ = model.totalTriangles;
  sceneDisplayed_ = model.points.size();
  sceneTriangles_ = model.triangles.size() / 3;
  sceneCameras_ = model.cameras.size() / 16;
  sceneProgress_->hide();
  const auto elapsed = sceneClock_.elapsed();
  const bool missingPreviewTexture = !model.texture.isEmpty() && model.texturePreview.isNull();
  const auto viewStatus = model.sampledMesh ? tr("网格采样预览") : tr("已加载");
  sceneStatus_->setText(model.error.isEmpty()
      ? tr("%1 · %2%3 · %4 s").arg(stageNames[layerKind + 1], viewStatus,
                                  cached ? tr("（缓存）") : QString(), QString::number(elapsed / 1000., 'f', 2))
      : tr("模型加载失败"));
  auto detail = model.error;
  if (detail.isEmpty()) {
    detail = model.sampledMesh ? tr("显示 %1 / %2 个三角面。导出使用原始模型。").arg(sceneTriangles_).arg(model.totalTriangles)
                               : tr("显示数据已就绪。耗时不含显卡绘制。");
    if (missingPreviewTexture) {
      sceneStatus_->setText(sceneStatus_->text() + tr(" · 无纹理预览"));
      detail += tr(" 当前纹理格式无法按预览尺寸解码。导出保留原纹理。");
    }
  }
  sceneStatus_->setToolTip(detail);
  refreshInspector();
  emit sceneLoaded(model.source, cached, elapsed);
}
void MainWindow::navigate(int stage) {
  draft_ = stage;
  for (int i = 0; i < stages_.size(); ++i) stages_[i]->setChecked(i == stage);
  updateStagePresentation();
  selectedKind_ = stage == 0 ? 10 : stage == 1 ? 1 : stage == 2 ? 13 : stage - 1;
  selectionDetail_.clear();
  syncTreeSelection(selectedKind_);
  refreshInspector();
  if (stage == 0) {
    features_->setViewMode(FeatureView::Mode::Photos);
    views_->setCurrentIndex(1);
    sync2DSelection();
  } else if (stage < 3) {
    features_->setMode(stage == 2);
    views_->setCurrentIndex(1);
    sync2DSelection();
  } else {
    views_->setCurrentIndex(0);
    if (project_.workspace.isEmpty()) return;
    if (stage == 3)
      loadScene(project_.activeSparse());
    else if (stage == 4)
      loadScene(QDir(project_.dense).filePath("fused.ply"));
    else if (stage == 5)
      loadScene(project_.mesh);
    else
      loadScene(QFileInfo(project_.mesh).dir().filePath("textured/mesh.ply"));
  }
}
void MainWindow::configure(bool confirmRun, bool singleStage) {
  if (runner_.busy() || (confirmRun && workspace_.images.isEmpty())) return;
  ParameterDialog dialog(project_, confirmRun ? qMax(1, draft_) : draft_,
                         singleStage, !workspace_.images.isEmpty(), this);
  if (dialog.exec() != QDialog::Accepted) return;
  const auto before = project_.parameters;
  const bool directoryChanged = project_.workspace != dialog.project().workspace;
  project_ = dialog.project();
  if (directoryChanged) {
    clearViews(); logs_->clear(); runner_.restoreHistory(Project{});
    reload(); updateTask();
  }
  if (before != project_.parameters || directoryChanged) markDirty();
  if (dialog.runRequested()) {
    navigate(dialog.selectedStage());
    startTask(dialog.followStages());
  }
}
void MainWindow::startTask(bool all) {
  if (runner_.busy() || (project_.imageRoot.isEmpty() && project_.photoFiles.isEmpty())) return;
  bool cancelLaunch = false;
  auto candidate = project_;
  if (qMax(1, draft_) <= 3 && (all || draft_ == 3)) candidate.parameters.remove("sparse_model");
  try {
    candidate.ensureOutputs();
    if (!candidate.workDirectory.isEmpty() && !QDir().mkpath(candidate.workDirectory))
      throw std::runtime_error(tr("无法创建工作目录：%1。请在“文件 → 项目数据”中修改工作目录。")
          .arg(candidate.workDirectory).toStdString());
    if (candidate.imageRoot.isEmpty() && !candidate.photoFiles.isEmpty()) {
      QProgressDialog progress(tr("正在准备计算输入…"), tr("取消"), 0, 100, this);
      progress.setWindowModality(Qt::WindowModal);
      progress.setAutoClose(false); progress.setAutoReset(false);
      auto cancelled = std::make_shared<std::atomic_bool>(false);
      connect(&progress, &QProgressDialog::canceled, &progress, [cancelled] { cancelled->store(true); });
      QFutureWatcher<QString> watcher;
      connect(&watcher, &QFutureWatcher<QString>::finished, &progress, &QDialog::accept);
      watcher.setFuture(QtConcurrent::run([&candidate, &progress, cancelled] {
        try {
          QElapsedTimer clock; clock.start();
          candidate.prepareInput([&](qint64 done, qint64 total, const QString& name) {
            if (clock.elapsed() < 50 && done != total) return;
            clock.restart();
            QMetaObject::invokeMethod(&progress, [&progress, done, total, name] {
              progress.setLabelText(tr("正在准备 %1\n%2 / %3 MiB").arg(name)
                  .arg(done / 1048576.).arg(total / 1048576.));
              progress.setValue(total ? int(100. * done / total) : 100);
            }, Qt::QueuedConnection);
          }, [cancelled] { return cancelled->load(); });
          return QString();
        }
        catch (const std::exception& e) { return QString::fromUtf8(e.what()); }
      }));
      progress.exec();
      watcher.waitForFinished();
      if (cancelled->load() && !watcher.result().isEmpty()) {
        statusBar()->showMessage(tr("已取消照片准备，工程保持不变"), 5000); return;
      }
      if (!watcher.result().isEmpty()) throw std::runtime_error(watcher.result().toStdString());
      cancelLaunch = cancelled->load();
    }
    const bool changed = candidate.workspace != project_.workspace || candidate.imageRoot != project_.imageRoot || candidate.parameters != project_.parameters;
    project_ = candidate;
    if (changed) markDirty();
    snapshotProject();
    if (cancelLaunch) { statusBar()->showMessage(tr("照片已准备完成，已取消启动计算"), 5000); return; }
    runner_.startWorkflow(project_, Workflow::from(qMax(1,draft_), all));
    findChild<QAction*>("showRightPanel")->setChecked(true);
    rightTabs_->setCurrentIndex(1);
  } catch (const std::exception& e) {
    QMessageBox::critical(this, tr("启动失败"), QString::fromUtf8(e.what()));
  }
}
void MainWindow::updateTask() {
  const auto state = runner_.state();
  const bool busy = runner_.busy();
  start_->setText(busy ? tr("中止") : tr("启动"));
  findChild<QPushButton*>("workflowParameters")->setEnabled(!busy);
  for (int i = 0; i < stages_.size(); ++i)
    findChild<QAction*>("configureStage" + QString::number(i))->setEnabled(!busy);
  updateStagePresentation();
  start_->setEnabled(busy || !workspace_.images.isEmpty());
  pause_->setEnabled(state == TaskRunner::State::Running ||
                     state == TaskRunner::State::PauseRequested ||
                     state == TaskRunner::State::Paused);
  pause_->setText(state == TaskRunner::State::Paused           ? tr("继续")
                  : state == TaskRunner::State::PauseRequested ? tr("取消等待")
                                                               : state == TaskRunner::State::Running ? tr("步后暂停") : tr("暂停"));
  taskTitle_->setText(runner_.title());
  QString text;
  switch (state) {
    case TaskRunner::State::Idle:
      text = tr("就绪");
      break;
    case TaskRunner::State::Running:
      if (runner_.progressTotal() > 0) {
        const QMap<QString, QString> phases{
          {"descriptor_matching", tr("描述子匹配")}, {"matching_pipeline", tr("匹配与几何验证")},
          {"geometry_verification", tr("几何验证")},
          {"patch_match_photometric", tr("光度深度图")}, {"patch_match_geometric", tr("几何深度图")}};
        text = tr("%1 %2 / %3").arg(phases.value(runner_.progressPhase(), tr("已处理")))
            .arg(runner_.progressDone()).arg(runner_.progressTotal());
      } else text = tr("运行中 · 等待当前步骤完成");
      break;
    case TaskRunner::State::PauseRequested:
      text = tr("等待当前步骤结束后暂停");
      break;
    case TaskRunner::State::Paused:
      text = tr("已暂停后续队列");
      break;
    case TaskRunner::State::Succeeded:
      text = tr("已完成 · 产物检查通过");
      break;
    case TaskRunner::State::Failed:
      text = tr("未完成 · 查看错误详情");
      break;
    case TaskRunner::State::Aborting:
      text = tr("正在中止本任务进程");
      break;
    case TaskRunner::State::Aborted:
      text = tr("已中止 · 未完成产物需重新检查");
      break;
  }
  if (runner_.validating()) text = state == TaskRunner::State::Aborting
      ? tr("正在结束产物检查") : tr("正在后台校验产物");
  if (state != TaskRunner::State::Idle) {
    const auto seconds = runner_.elapsedMilliseconds() / 1000;
    text += tr(" · 用时 %1:%2").arg(seconds / 60).arg(seconds % 60, 2, 10, QChar('0'));
  }
  taskState_->setToolTip(state == TaskRunner::State::Failed ? runner_.lastError() : runner_.progressDetail());
  for (const auto* name : {"openTaskLog", "openLogFile"})
    if (auto* openLog = findChild<QPushButton*>(name)) openLog->setEnabled(!runner_.logPath().isEmpty());
  if (auto* details = findChild<QPushButton*>("taskErrorDetails")) details->setVisible(state == TaskRunner::State::Failed);
  taskState_->setText(
      text +
      QString("  [%1 / %2]").arg(runner_.completed()).arg(runner_.total()));
  if (auto* statusState = findChild<QLabel*>("statusState")) statusState->setText(text);
  progress_->setVisible(state != TaskRunner::State::Idle);
  if (state == TaskRunner::State::Idle) taskState_->setText(tr("就绪"));
  if (auto* strip = findChild<QWidget*>("taskStrip")) {
    if (busy) strip->show();
    if (state == TaskRunner::State::Idle || state == TaskRunner::State::Succeeded) strip->hide();
  }
  if (auto* button = findChild<QPushButton*>("taskToggle")) {
    button->setText(busy ? tr("任务 · %1 / %2").arg(runner_.completed()).arg(runner_.total()) : tr("任务"));
    button->setToolTip(taskState_->text());
  }
  if (!runner_.validating() && runner_.progressTotal() > 0 &&
      (state == TaskRunner::State::Running || state == TaskRunner::State::PauseRequested)) {
    progress_->setRange(0, 1000);
    progress_->setValue(int(1000. * runner_.progressDone() / runner_.progressTotal()));
  } else if (state == TaskRunner::State::Running ||
      state == TaskRunner::State::PauseRequested)
    progress_->setRange(0, 0);
  else {
    progress_->setRange(0, qMax(1, runner_.total()));
    progress_->setValue(runner_.completed());
  }
}
void MainWindow::closeEvent(QCloseEvent* event) {
  if (runner_.busy()) {
    QMessageBox::information(
        this, tr("任务仍在运行"),
        tr("请等待任务结束，或先点击中止并等待退出，再关闭工作台。"));
    event->ignore();
    return;
  }
  if (allowReplacement()) event->accept(); else event->ignore();
}
}  // namespace colite::gui

void colite::gui::MainWindow::setProperties(const QString& text) {
  selectionDetail_ = text;
  refreshInspector();
}
void colite::gui::MainWindow::syncTreeSelection(int kind, int index) {
  updating_ = true;
  QTreeWidgetItemIterator it(tree_);
  while (*it) {
    auto* item = *it;
    if (item->data(0, Qt::UserRole).toInt() == kind &&
        (kind != 12 || item->data(0, Qt::UserRole + 1).toInt() == index) &&
        (kind != 2 || item->data(0, Qt::UserRole + 3).toInt() == project_.parameters.value("sparse_model").toInt(-1))) {
      tree_->setCurrentItem(item);
      if (item->parent()) item->parent()->setExpanded(true);
      tree_->scrollToItem(item);
      break;
    }
    ++it;
  }
  updating_ = false;
}
void colite::gui::MainWindow::refreshInspector() {
  if (!inspector_) return;
  InspectorContent c;
  auto row = [&](const QString& key, const QString& value) {
    c.properties.append({key, value});
  };
  auto stat = [&](const QString& key, const QString& value) {
    c.statistics.append({key, value});
  };
  auto count = [](quint64 n) { return QLocale().toString(n); };
  QSet<quint32> cameras;
  for (const auto& image : workspace_.images)
    if (image.camera) cameras.insert(image.camera);
  if (selectedKind_ == 1 || selectedKind_ == 13) {
    c.title = features_->matching() ? tr("特征匹配") : tr("特征提取");
    row(tr("对象类型"), c.title);
    auto* canvas = static_cast<InspectionCanvas*>(
        features_->findChild<QWidget*>("inspectionCanvas"));
    if (canvas && features_->matching()) {
      const auto& pair = canvas->data.pair;
      auto name = [&](quint32 id) {
        for (const auto& image : workspace_.images)
          if (image.id == id) return image.name;
        return QString::fromUtf8("—");
      };
      row(tr("图像 A"), name(pair.a));
      row(tr("图像 B"), name(pair.b));
      row(tr("候选匹配"), count(pair.matches.size()));
      row(tr("有效匹配"),
          pair.verified ? count(pair.inliers) : QString::fromUtf8("—"));
      row(tr("内点比例"), pair.verified && !pair.matches.isEmpty()
          ? QString::number(100. * pair.inliers / pair.matches.size(), 'f', 1) + "%"
          : QString::fromUtf8("—"));
      stat(tr("非内点"), pair.verified
                             ? count(pair.matches.size() - pair.inliers)
                             : QString::fromUtf8("—"));
      c.metricTitle = pair.verified ? tr("几何内点占比") : tr("原始匹配对");
      c.metricValue =
          pair.verified && pair.matches.size()
              ? QString::number(100. * pair.inliers / pair.matches.size(), 'f',
                                1) +
                    "%"
              : count(pair.matches.size());
      if (pair.verified && !pair.matches.isEmpty()) {
        c.ratio = double(pair.inliers) / pair.matches.size();
        c.ratioTitle = tr("内点 / 全部匹配");
      }
    } else if (canvas) {
      const int i = features_->imageIndex();
      if (i >= 0 && i < workspace_.images.size()) {
        c.title = workspace_.images[i].name;
        c.path = workspace_.images[i].path;
      }
      row(tr("检查内容"), tr("点位 / 尺度 / 方向"));
      row(tr("特征点数"), count(canvas->data.first.size()));
      stat(tr("显示上限"), count(canvas->limit));
      c.metricValue = count(canvas->data.first.size());
      c.metricTitle = tr("已载入特征点");
    }
    if (!selectionDetail_.isEmpty()) row(tr("选中项"), selectionDetail_);
  } else if (selectedKind_ == 12 && selectedImage_ >= 0 &&
             selectedImage_ < workspace_.images.size()) {
    const auto& image = workspace_.images[selectedImage_];
    c.title = image.name;
    c.subtitle = tr("图像");
    c.path = image.path;
    const auto size = QImageReader(image.path).size();
    row(tr("Image ID"), QString::number(image.id));
    row(tr("相机组"),
        image.camera ? QString("Camera %1").arg(image.camera) : tr("尚未分组"));
    row(tr("尺寸"),
        size.isValid()
            ? QString("%1 × %2 px").arg(size.width()).arg(size.height())
            : tr("无法读取"));
    row(tr("格式"), QFileInfo(image.path).suffix().toUpper());
    row(tr("文件大小"),
        QLocale().formattedDataSize(QFileInfo(image.path).size()));
    c.metricValue =
        size.isValid()
            ? QString::number(size.width() * double(size.height()) / 1000000,
                              'f', 1)
            : QString::fromUtf8("—");
    c.metricTitle = tr("百万像素");
  } else if (selectedKind_ >= 2 && selectedKind_ <= 5) {
    c.title = QStringList{project_.parameters.value("sparse_model").toInt(-1) < 0 ? tr("稀疏主模型")
                         : tr("稀疏子模型 %1").arg(project_.parameters.value("sparse_model").toInt()), tr("稠密点云"), tr("网格"),
                          tr("纹理模型")}[selectedKind_ - 2];
    c.subtitle = selectedKind_ == 2   ? tr("COLMAP TXT / BIN")
                 : selectedKind_ == 5 ? tr("PLY + 纹理")
                                      : tr("PLY");
    c.path = inspectedScene_;
    row(tr("可见性"),
        layerVisibility_.value(selectedKind_, true) ? tr("显示") : tr("隐藏"));
    row(tr("加载状态"), sceneLoading_           ? tr("读取中…")
                        : sceneError_.isEmpty() ? tr("已载入")
                                                : tr("读取失败"));
    const bool ready = !sceneLoading_ && sceneError_.isEmpty();
    auto n = [&](quint64 value) {
      return ready ? count(value) : QString::fromUtf8("—");
    };
    row(tr("点 / 顶点"), n(scenePoints_));
    if (selectedKind_ >= 4) {
      row(tr("三角形"), n(sceneTotalTriangles_));
      if (sceneTotalTriangles_ > quint64(sceneTriangles_))
        row(tr("预览三角形"), n(sceneTriangles_));
    } else
      row(tr("显示点数"), n(sceneDisplayed_));
    if (selectedKind_ == 2) {
      row(tr("已注册照片"), n(sceneCameras_));
      row(tr("未注册照片"), n(qMax(0, workspace_.images.size() - sceneCameras_)));
    }
    c.metricTitle = selectedKind_ >= 4 ? tr("三角形") : tr("源点数");
    c.metricValue = n(selectedKind_ >= 4 ? sceneTotalTriangles_ : scenePoints_);
    c.note = sceneError_.isEmpty() ? selectionDetail_ : sceneError_;
  } else if (selectedKind_ == 11) {
    c.title = selectedCamera_ ? QString("Camera %1").arg(selectedCamera_)
                              : tr("待分组图像");
    c.subtitle = tr("相机内参组");
    int images = 0;
    for (const auto& image : workspace_.images)
      if (image.camera == selectedCamera_) ++images;
    row(tr("Camera ID"), QString::number(selectedCamera_));
    row(tr("关联图像"), count(images));
    stat(tr("项目图像"), count(workspace_.images.size()));
    c.metricValue = count(images);
    c.metricTitle = tr("关联图像");
  } else {
    c.title = selectedKind_ == 10       ? tr("图像")
              : selectedKind_ == 20     ? tr("重建结果")
              : project_.name.isEmpty() ? tr("尚未打开项目")
                                        : project_.name;
    c.subtitle = tr("项目概览");
    c.path = project_.file;
    row(tr("图像数量"), count(workspace_.images.size()));
    row(tr("相机组数"), count(cameras.size()));
    c.metricValue = count(workspace_.images.size());
    c.metricTitle = tr("项目图像");
  }
  inspector_->setContent(c);
}
