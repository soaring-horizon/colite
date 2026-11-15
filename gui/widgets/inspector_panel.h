#pragma once
#include <QtWidgets>

namespace colite::gui {
struct InspectorContent {
  QString title, subtitle, path, note;
  QList<QPair<QString, QString>> properties, statistics;
  QString metricTitle, metricValue, ratioTitle;
  double ratio = -1;
};
class InspectorPanel : public QWidget {
 public:
  explicit InspectorPanel(QWidget* parent = nullptr);
  void setContent(const InspectorContent& content);

 private:
  QStackedWidget* pages_;
  InspectorContent content_;
  QWidget* page();
};
}  // namespace colite::gui
