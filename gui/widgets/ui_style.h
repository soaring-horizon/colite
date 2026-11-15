#pragma once
#include <QtWidgets>
namespace colite::gui {
inline void configureApplicationUi(QApplication& app) {
  app.setStyle("Fusion");
  const auto families = QFontDatabase().families();
  QFont font = app.font();
  for (const auto& family : {QStringLiteral("Microsoft YaHei UI"), QStringLiteral("Noto Sans CJK SC"), QStringLiteral("WenQuanYi Micro Hei")}) {
    if (families.contains(family)) { font.setFamily(family); break; }
  }
  font.setPixelSize(16);
  font.setStyleHint(QFont::SansSerif);
  font.setStyleStrategy(QFont::PreferAntialias);
  app.setFont(font);
  QSurfaceFormat format;
  format.setDepthBufferSize(24);
  if (QOpenGLContext::openGLModuleType() == QOpenGLContext::LibGLES) {
    format.setRenderableType(QSurfaceFormat::OpenGLES); format.setVersion(2, 0);
  } else format.setVersion(2, 1);
  format.setProfile(QSurfaceFormat::NoProfile);
  QSurfaceFormat::setDefaultFormat(format);
}
class PhotoList : public QListWidget {
 public:
  using QListWidget::QListWidget;
  std::function<void()> visibleChanged;
 protected:
  void resizeEvent(QResizeEvent* event) override {
    QListWidget::resizeEvent(event);
    const int available = viewport()->width() - 16;
    const int columns = available >= 300 ? 2 : 1;
    const int cell = available / columns;
    const int photoWidth = qMax(100, cell - 12);
    setIconSize({photoWidth, photoWidth * 2 / 3});
    setGridSize({cell, photoWidth * 2 / 3 + 42});
    for (int i = 0; i < count(); ++i) item(i)->setSizeHint(gridSize());
    if (visibleChanged) visibleChanged();
  }
};
// Keep photograph colors intact when selected; selection is shown by the
// row/border.
class UntintedIconDelegate : public QStyledItemDelegate {
 public:
  using QStyledItemDelegate::QStyledItemDelegate;

 protected:
  void paint(QPainter* painter, const QStyleOptionViewItem& option,
             const QModelIndex& index) const override {
    QStyleOptionViewItem styled(option);
    const auto badge = index.data(Qt::UserRole + 2).toString();
    if (!badge.isEmpty()) styled.state |= QStyle::State_Selected;
    QStyledItemDelegate::paint(painter, styled, index);
    if (badge.isEmpty()) return;
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing);
    const QRect box(option.rect.topLeft() + QPoint(7, 7), QSize(23, 23));
    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor("#226f56"));
    painter->drawRoundedRect(box, 4, 4);
    painter->setPen(Qt::white);
    painter->drawText(box, Qt::AlignCenter, badge);
    painter->restore();
  }
  void initStyleOption(QStyleOptionViewItem* option,
                       const QModelIndex& index) const override {
    QStyledItemDelegate::initStyleOption(option, index);
    if (!option->icon.isNull()) {
      const auto normal = option->icon.pixmap(option->decorationSize,
                                              QIcon::Normal, QIcon::Off);
      QIcon icon(normal);
      icon.addPixmap(normal, QIcon::Selected, QIcon::Off);
      option->icon = icon;
    }
  }
};
// Bundled monochrome assets and painted primitives work on both platforms.
inline QIcon fluentIcon(ushort glyph, QColor color = QColor("#617c8c"),
                        int size = 20) {
  QString asset;
  switch (glyph) {
    case 0xe7f4:
      asset = "Cube";
      break;
    case 0xe8e5:
      asset = "FolderOpen";
      break;
    case 0xe8f4:
      asset = "Folder";
      break;
    case 0xe74e:
      asset = "FloppyDisk";
      break;
    case 0xe8b9:
      asset = "Cube";
      break;
    case 0xe7c1:
      asset = "Image";
      break;
    case 0xe9a6:
      asset = "ArrowsOut";
      break;
    case 0xe91b:
      asset = "Image";
      break;
    case 0xe80a:
      asset = "SquaresFour";
      break;
    case 0xe895:
      asset = "CircleNotch";
      break;
  }
  if (!asset.isEmpty()) return QIcon(":/ui/" + asset + ".png");
  QPixmap pix(size * 2, size * 2);
  pix.fill(Qt::transparent);
  QPainter p(&pix);
  p.setRenderHint(QPainter::Antialiasing);
  p.setPen(QPen(color, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  if (glyph == 0xe76c) {
    QPainterPath chevron;
    chevron.moveTo(size * .7, size * .5);
    chevron.lineTo(size * 1.2, size);
    chevron.lineTo(size * .7, size * 1.5);
    p.drawPath(chevron);
  }
  if (glyph == 0xe713) {
    p.translate(size, size);
    p.drawEllipse(QPointF(0, 0), size * .60, size * .60);
    p.drawEllipse(QPointF(0, 0), size * .24, size * .24);
    for (int i = 0; i < 8; ++i) {
      p.drawLine(QPointF(0, size * .60), QPointF(0, size * .85));
      p.rotate(45);
    }
  }
  pix.setDevicePixelRatio(2);
  return QIcon(pix);
}
class WorkflowStep : public QPushButton {
 public:
  explicit WorkflowStep(const QString& title) : title_(title) {
    setCheckable(true); setFixedHeight(60); setMinimumWidth(120);
    setText(title); setCursor(Qt::PointingHandCursor); setAccessibleName(title);
    animation_.setInterval(40);
    connect(&animation_, &QTimer::timeout, this, [this] {
      rotation_ = (rotation_ + 12) % 360;
      if (isVisible()) update();
    });
  }
  void setStatus(const QString& state, const QString& label) {
    setProperty("workflowState", state); setProperty("statusText", label);
    setAccessibleDescription(label);
    if (state == "running") { if (!animation_.isActive()) animation_.start(); }
    else animation_.stop();
    update();
  }
 protected:
  void paintEvent(QPaintEvent*) override {
    QPainter p(this); p.setRenderHint(QPainter::Antialiasing);
    if (isChecked()) {
      p.fillRect(rect(), QColor("#e1f1e8"));
      p.setPen(Qt::NoPen); p.setBrush(QColor("#379977"));
      p.drawRoundedRect(QRect(8, height() - 4, width() - 16, 3), 1, 1);
    } else if (underMouse() || hasFocus()) p.fillRect(rect(), QColor("#eef7f2"));
    auto font = this->font(); font.setPixelSize(18); font.setWeight(QFont::DemiBold);
    p.setFont(font);
    const int textWidth = QFontMetrics(font).horizontalAdvance(title_);
    const qreal left = (width() - textWidth - 30) / 2.;
    const qreal top = (height() - 22) / 2.;
    p.setPen(QColor(isChecked() ? "#226f56" : "#26352f"));
    p.drawText(QRectF(left + 30, 0, textWidth + 1, height() - 2), Qt::AlignVCenter, title_);
    const auto state = property("workflowState").toString();
    const bool complete = state == "complete", failed = state == "failed";
    const bool running = state == "running";
    const bool warning = state == "stale" || state == "paused" || state == "aborted";
    const QColor color(complete ? "#2e8868" : failed ? "#b44e43" : running ? "#3278ab" : warning ? "#a87225" : "#8c9992");
    p.translate(left, top);
    p.setPen(QPen(color, 1.8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);
    const QRectF ring(1, 1, 20, 20);
    if (running) {
      p.setPen(QPen(QColor("#d6e5e0"), 1.8)); p.drawEllipse(ring);
      p.setPen(QPen(color, 2, Qt::SolidLine, Qt::RoundCap));
      p.drawArc(ring, -rotation_ * 16, 260 * 16);
    } else {
      p.drawEllipse(ring);
      if (complete) {
        QPainterPath check; check.moveTo(6, 11); check.lineTo(9.5, 14.5); check.lineTo(16, 8); p.drawPath(check);
      } else if (failed) {
        p.drawLine(QPointF(7, 7), QPointF(15, 15)); p.drawLine(QPointF(15, 7), QPointF(7, 15));
      } else if (state == "paused") {
        p.drawLine(QPointF(8, 7), QPointF(8, 15)); p.drawLine(QPointF(14, 7), QPointF(14, 15));
      } else if (state == "aborted") {
        p.setPen(Qt::NoPen); p.setBrush(color); p.drawRoundedRect(QRectF(7, 7, 8, 8), 1, 1);
      } else if (state == "stale") {
        p.drawLine(QPointF(11, 6), QPointF(11, 12)); p.drawPoint(QPointF(11, 16));
      } else if (state == "queued") {
        p.drawLine(QPointF(11, 6), QPointF(11, 11)); p.drawLine(QPointF(11, 11), QPointF(15, 11));
      } else if (state == "unverified") {
        auto questionFont = font; questionFont.setPixelSize(15); p.setFont(questionFont);
        p.drawText(QRectF(1, 0, 20, 21), Qt::AlignCenter, "?");
      }
    }
  }
 private:
  QString title_;
  QTimer animation_;
  int rotation_ = 0;
};
}  // namespace colite::gui
