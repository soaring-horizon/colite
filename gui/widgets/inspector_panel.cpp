#include "inspector_panel.h"

namespace colite::gui {
InspectorPanel::InspectorPanel(QWidget* parent) : QWidget(parent) {
  setObjectName("inspectorPanel");
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);
  auto* heading = new QLabel(QString::fromUtf8("属性"));
  heading->setObjectName("panelHeading");
  layout->addWidget(heading);
  pages_ = new QStackedWidget;
  layout->addWidget(pages_, 1);
  InspectorContent empty;
  empty.title = QString::fromUtf8("尚未选择对象");
  empty.note =
      QString::fromUtf8("选择左侧图像、相机或模型，查看其属性与统计。");
  setContent(empty);
}
void InspectorPanel::setContent(const InspectorContent& content) {
  content_ = content;
  while (pages_->count()) {
    auto* old = pages_->widget(0);
    pages_->removeWidget(old);
    delete old;
  }
  pages_->addWidget(page());
}
QWidget* InspectorPanel::page() {
  auto* scroll = new QScrollArea;
  scroll->setWidgetResizable(true);
  scroll->setFrameShape(QFrame::NoFrame);
  auto* body = new QWidget;
  body->setObjectName("inspectorBody");
  auto* layout = new QVBoxLayout(body);
  layout->setContentsMargins(14, 8, 14, 8);
  layout->setSpacing(0);
  auto* title = new QLabel(content_.title);
  title->setObjectName("objectTitle");
  title->setWordWrap(true);
  title->setTextFormat(Qt::PlainText);
  layout->addWidget(title);
  if (!content_.subtitle.isEmpty()) {
    auto* subtitle = new QLabel(content_.subtitle);
    subtitle->setObjectName("objectSubtitle");
    subtitle->setWordWrap(true);
    layout->addWidget(subtitle);
  }
  layout->addSpacing(8);
  auto rows = content_.properties;
  QSet<QString> keys;
  for (const auto& row : rows) keys.insert(row.first);
  for (const auto& row : content_.statistics) if (!keys.contains(row.first)) {
    rows.append(row); keys.insert(row.first);
  }
  for (const auto& row : rows) {
    auto* frame = new QFrame;
    frame->setObjectName("inspectorRow");
    auto* line = new QHBoxLayout(frame);
    line->setContentsMargins(0, 6, 0, 6);
    line->setSpacing(12);
    auto* key = new QLabel(row.first);
    key->setObjectName("propertyKey");
    key->setMinimumWidth(68);
    auto* value = new QLabel(row.second);
    value->setObjectName("propertyValue");
    value->setTextFormat(Qt::PlainText);
    value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value->setWordWrap(true);
    value->setTextInteractionFlags(Qt::TextSelectableByMouse);
    value->setToolTip(row.second);
    line->addWidget(key);
    line->addWidget(value, 1);
    layout->addWidget(frame);
  }
  if (content_.ratio >= 0) {
    layout->addSpacing(12);
    auto* label = new QLabel(content_.ratioTitle);
    label->setObjectName("objectSubtitle");
    layout->addWidget(label);
    auto* bar = new QProgressBar;
    bar->setRange(0, 1000);
    bar->setValue(qBound(0, qRound(content_.ratio * 1000), 1000));
    bar->setTextVisible(false);
    bar->setFixedHeight(7);
    layout->addWidget(bar);
  }
  if (!content_.note.isEmpty()) {
    layout->addSpacing(12);
    auto* note = new QLabel(content_.note);
    note->setObjectName("inspectorNote");
    note->setWordWrap(true);
    layout->addWidget(note);
  }
  if (!content_.path.isEmpty()) {
    layout->addSpacing(10);
    auto* location = new QPushButton(QString::fromUtf8("文件位置"));
    location->setObjectName("locationButton");
    location->setCheckable(true);
    location->setToolTip(content_.path);
    auto* path = new QLabel(QDir::toNativeSeparators(content_.path));
    path->setObjectName("locationText");
    path->setTextFormat(Qt::PlainText);
    path->setWordWrap(true);
    path->setTextInteractionFlags(Qt::TextSelectableByMouse);
    path->hide();
    connect(location, &QPushButton::toggled, path, &QLabel::setVisible);
    layout->addWidget(location);
    layout->addWidget(path);
  }
  layout->addStretch();
  scroll->setWidget(body);
  return scroll;
}
}  // namespace colite::gui
