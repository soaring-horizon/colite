#pragma once
#include <QFutureWatcher>
#include <QtWidgets>
#include <atomic>
#include <memory>

#include "project/workspace.h"

namespace colite::gui {
struct InspectionData {
  QImage a, b;
  QSize sourceA, sourceB;
  QString nameA, nameB;
  QVector<Keypoint> first, second;
  PairData pair;
  QVector<int> displayOrder;
  QString error;
};
class InspectionCanvas : public QWidget {
 public:
  explicit InspectionCanvas(QWidget* parent = nullptr);
  InspectionData data;
  bool matching = false, show = true, orientation = false;
  int limit = 10000, filter = 0, focus = 0, point = -1, match = -1;
  qreal zoom = 1;
  std::function<void(int, int)> picked;
  std::function<void(qreal)> zoomed;
  void fit();

 protected:
  void paintEvent(QPaintEvent*) override;
  void mousePressEvent(QMouseEvent*) override;
  void mouseMoveEvent(QMouseEvent*) override;
  void mouseReleaseEvent(QMouseEvent*) override;
  void mouseDoubleClickEvent(QMouseEvent*) override;
  void wheelEvent(QWheelEvent*) override;

 private:
  struct Hit {
    QPointF a, b;
    int id = 0;
  };
  QVector<Hit> hits_;
  QPoint start_, last_;
  QPointF pan_;
};
class FeatureView : public QWidget {
  Q_OBJECT
 public:
  explicit FeatureView(QWidget* parent = nullptr);
  ~FeatureView() override;
  enum class Mode { Photos, Features, Matches };
  void setWorkspace(const Workspace& workspace);
  void setMode(bool matching);
  void setViewMode(Mode mode);
  Mode viewMode() const { return mode_; }
  void selectPairImage(int index);
  int pairA() const { return pairA_; }
  int pairB() const { return pairB_; }
  void selectImage(int index);
  int imageIndex() const { return image_; }
  bool matching() const { return matching_; }
 signals:
  void selectionChanged(const QString& text);
  void imageChanged(int index);
  void modeChanged(bool matching);
  void viewModeChanged();

 private:
  Workspace workspace_;
  InspectionCanvas* canvas_ = nullptr;
  QComboBox *a_ = nullptr, *b_ = nullptr, *filter_ = nullptr;
  QPushButton *photoMode_ = nullptr, *single_ = nullptr, *pair_ = nullptr, *focusA_ = nullptr,
              *focusB_ = nullptr, *swap_ = nullptr;
  QCheckBox *points_ = nullptr, *lines_ = nullptr, *orientation_ = nullptr;
  QSlider* limit_ = nullptr;
  QLabel *context_ = nullptr, *counts_ = nullptr, *detail_ = nullptr,
         *zoom_ = nullptr, *bLabel_ = nullptr;
  bool matching_ = false, changing_ = false;
  Mode mode_ = Mode::Photos;
  QWidget* options_ = nullptr;
  QLabel* imageLabel_ = nullptr;
  int image_ = 0, pairA_ = 0, pairB_ = 1, generation_ = 0;
  int singleLimit_ = 10000, pairLimit_ = 2000, selectedMatch_ = -1;
  qreal singleZoom_ = 1, pairZoom_ = 1;
  QMap<quint32, int> selectedPoints_;
  QCache<QString, InspectionData> cache_{192}; // MiB of decoded preview and inspection data
  QThreadPool pool_;
  std::shared_ptr<std::atomic_bool> cancel_;
  void load();
  void refresh();
  void choose(int match, int side);
};
}  // namespace colite::gui
