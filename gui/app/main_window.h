#pragma once
#include <QtWidgets>
#include <atomic>
#include <memory>

#include "widgets/feature_view.h"
#include "widgets/inspector_panel.h"
#include "widgets/scene_view.h"
#include "workflow/task_runner.h"
#include "project/recovery_store.h"

namespace colite::gui {
class MainWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit MainWindow(QString recoveryRoot = {});
  ~MainWindow() override;
  void openProject(const QString& path);
  void navigate(int stage);
  void enableRecovery() { recoveryEnabled_ = true; }
  void offerRecovery();
  FeatureView* features() const { return features_; }
  bool sceneLoading() const { return sceneLoading_; }
  QString graphicsInfo() const { return scene_->graphicsInfo(); }
  QString graphicsError() const { return scene_->graphicsError(); }
 signals:
  // Time to read/prepare/apply display data; does not include GPU presentation.
  void sceneLoaded(const QString& path, bool cached, qint64 milliseconds);

 protected:
  void closeEvent(QCloseEvent*) override;

 private:
  Project project_;
  Workspace workspace_;
  TaskRunner runner_;
  SceneView* scene_ = nullptr;
  FeatureView* features_ = nullptr;
  QTabWidget* views_ = nullptr;
  QTabWidget* rightTabs_ = nullptr;
  QTreeWidget* tree_ = nullptr;
  QListWidget* photos_ = nullptr;
  QTimer thumbnailTimer_;
  QThreadPool thumbnailPool_;
  QCache<QString, QImage> thumbnailCache_{64 * 1024}; // KiB
  std::shared_ptr<std::atomic_bool> thumbnailCancel_;
  bool thumbnailsLoading_ = false;
  void loadVisibleThumbnails();
  void syncPhotoBadges();
  void sync2DSelection();
  QLabel *taskTitle_ = nullptr, *taskState_ = nullptr;
  InspectorPanel* inspector_ = nullptr;
  int selectedKind_ = 0, selectedImage_ = -1;
  quint32 selectedCamera_ = 0;
  QString inspectedScene_, sceneError_;
  quint64 scenePoints_ = 0, sceneTotalTriangles_ = 0;
  int sceneDisplayed_ = 0, sceneTriangles_ = 0, sceneCameras_ = 0;
  bool sceneLoading_ = false;
  struct CachedScene { SceneData data; QString stamp; };
  QCache<QString, CachedScene> sceneCache_{384}; // MiB, CPU display data only
  QThreadPool scenePool_;
  std::shared_ptr<std::atomic_bool> sceneCancel_;
  QElapsedTimer sceneClock_;
  QLabel* sceneStatus_ = nullptr;
  QProgressBar* sceneProgress_ = nullptr;
  QString displayedScene_, displayedStamp_;
  void applyScene(const SceneData& data, int layerKind, bool cached, const QString& stamp);
  QString selectionDetail_;
  QMap<int, bool> layerVisibility_;
  void refreshInspector();
  void syncTreeSelection(int kind, int index = -1);
  QProgressBar* progress_ = nullptr;
  QPlainTextEdit* logs_ = nullptr;
  QMenu* recentProjects_ = nullptr;
  QPushButton *pause_ = nullptr, *start_ = nullptr;
  QVector<QPushButton*> stages_;
  int draft_ = 1, loadGeneration_ = 0, sceneGeneration_ = 0;
  bool updating_ = false;
  bool dirty_ = false;
  bool recoveryEnabled_ = false;
  RecoveryStore recovery_;
  void snapshotProject();
  void clearRecovery();
  void refreshStageStates();
  void updateStagePresentation();
  std::shared_ptr<std::atomic_bool> stageStateCancel_;
  void markDirty();
  bool allowReplacement();
  void clearViews();
  void addPhotos(bool folder);
  void createProject();
  bool editProject();
  bool saveProject(bool saveAs = false);
  void rememberProject(const QString& path);
  void refreshRecentProjects();
  void exportSelectedScene();
  void reload();
  void selectImage(int index);
  void displayImage(int index);
  void loadScene(const QString& path);
  void configure(bool confirmRun = false, bool singleStage = false);
  void setProperties(const QString& text);
  void startTask(bool all = false);
  void updateTask();

};
}  // namespace colite::gui
