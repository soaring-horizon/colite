#include "feature_view.h"

#include <QtConcurrent>
#include <cmath>

namespace colite::gui {
namespace {
QString inspectionStamp(const QStringList& files) {
  QString key;
  for (const auto& path : files) {
    const QFileInfo info(path);
    key += path + '\n' + QString::number(info.size()) + ':' + QString::number(info.lastModified().toMSecsSinceEpoch()) + '\n';
  }
  return key;
}
}
InspectionCanvas::InspectionCanvas(QWidget* parent) : QWidget(parent) {
  setMinimumSize(260, 140);
  setCursor(Qt::CrossCursor);
  setFocusPolicy(Qt::StrongFocus);
  setAccessibleName(QString::fromUtf8("二维图像画布"));
}
void InspectionCanvas::fit() {
  pan_ = {};
  zoom = 1;
  update();
}
void InspectionCanvas::paintEvent(QPaintEvent*) {
  QPainter p(this);
  p.fillRect(rect(), QColor("#f7f9f8"));
  p.setRenderHint(QPainter::Antialiasing);
  hits_.clear();
  if (data.a.isNull()) {
    p.setPen(QColor("#617786"));
    p.drawText(
        rect().adjusted(25, 15, -25, -15), Qt::AlignCenter | Qt::TextWordWrap,
        data.error.isEmpty()
            ? QString::fromUtf8("选择项目中的照片，查看真实特征和匹配结果")
            : data.error);
    return;
  }
  const qreal gap = 18,
              slot = (width() - 40 - (matching ? gap : 0)) / (matching ? 2 : 1);
  auto photoRect = [&](const QImage& image, int side) {
    QSizeF size = image.size();
    size.scale(QSizeF(slot, qMax(1, height() - 56)), Qt::KeepAspectRatio);
    size *= zoom;
    const qreal center = matching ? (side == 0 ? (width() - gap) / 2 - slot / 2
                                               : (width() + gap) / 2 + slot / 2)
                                  : width() / 2.;
    return QRectF(QPointF(center - size.width() / 2 + pan_.x(),
                          (height() - size.height()) / 2 + pan_.y()),
                  size);
  };
  const auto first = photoRect(data.a, 0),
             second = photoRect(data.b.isNull() ? data.a : data.b, 1);
  auto drawPhoto = [&](const QImage& image, const QRectF& box, int side) {
    if (image.isNull()) return;
    p.drawImage(box, image);
    p.setPen(
        QPen(matching && focus == side ? QColor("#79b69f") : QColor("#d3dfe5"),
             matching && focus == side ? 2 : 1));
    p.drawRect(box.adjusted(-2, -2, 2, 2));
    if (matching) {
      p.setPen(QColor("#53645c"));
      const auto label = (side == 0 ? QString("A · ") + data.nameA : QString("B · ") + data.nameB);
      p.drawText(QRectF(box.left(), box.bottom() + 7, box.width(), 24), Qt::AlignHCenter | Qt::AlignTop,
                 p.fontMetrics().elidedText(label, Qt::ElideMiddle, int(box.width())));
    }
  };
  drawPhoto(data.a, first, 0);
  if (matching) drawPhoto(data.b, second, 1);
  auto xy = [](const Keypoint& k, const QRectF& box, const QSize& source) {
    return QPointF(box.left() + k.x / qMax(1, source.width()) * box.width(),
                   box.top() + k.y / qMax(1, source.height()) * box.height());
  };
  int displayed = 0;
  if (matching) {
    for (const int i : data.displayOrder) {
      const auto& m = data.pair.matches[i];
      const bool chosen = i == match;
      const bool visible =
          filter == 0 ? (m.inlier)
                      : (filter == 1 ? true : (m.verified && !m.inlier));
      if ((!visible || displayed >= limit) && !chosen) continue;
      if (m.a >= quint32(data.first.size()) ||
          m.b >= quint32(data.second.size()))
        continue;
      const auto x = xy(data.first[int(m.a)], first, data.sourceA),
                 y = xy(data.second[int(m.b)], second, data.sourceB);
      const QColor color = chosen     ? QColor("#c48c24")
                           : m.inlier ? QColor(35, 143, 112, 155)
                                      : QColor(190, 105, 78, 140);
      p.setPen(QPen(color, chosen ? 2.2 : .9));
      if (show || chosen) p.drawLine(x, y);
      p.setBrush(chosen ? QColor("#f3c664") : QColor("#a5e4ce"));
      p.drawEllipse(x, chosen ? 4 : 2.4, chosen ? 4 : 2.4);
      p.drawEllipse(y, chosen ? 4 : 2.4, chosen ? 4 : 2.4);
      hits_.push_back({x, y, i});
      ++displayed;
    }
  } else if (show) {
    for (int i = 0; i < data.first.size(); ++i) {
      if (i >= limit && i != point) continue;
      const auto& k = data.first[i];
      const auto x = xy(k, first, data.sourceA);
      const bool chosen = i == point;
      const qreal radius =
          chosen ? 7
          : orientation
              ? qBound(2., k.scale * first.width() / qMax(1, data.sourceA.width()), 12.)
              : 2.3;
      p.setPen(QPen(chosen ? QColor("#a77119") : QColor("#2c7a63"), 1));
      p.setBrush(orientation
                     ? Qt::NoBrush
                     : QBrush(chosen ? QColor("#e9b550") : QColor("#9ce5cb")));
      p.drawEllipse(x, radius, radius);
      if (orientation) {
        const auto angle = k.orientation * 3.141592653589793 / 180;
        p.drawLine(
            x, x + QPointF(std::cos(angle), std::sin(angle)) * radius * 1.8);
      }
      hits_.push_back({x, x, i});
    }
  }
}
void InspectionCanvas::mousePressEvent(QMouseEvent* e) {
  start_ = last_ = e->pos();
}
void InspectionCanvas::mouseMoveEvent(QMouseEvent* e) {
  if (e->buttons() & (Qt::LeftButton | Qt::MiddleButton)) {
    pan_ += e->pos() - last_;
    last_ = e->pos();
    update();
  }
}
void InspectionCanvas::mouseReleaseEvent(QMouseEvent* e) {
  if ((e->pos() - start_).manhattanLength() > 4) return;
  qreal best = 10;
  int id = -1, side = 0;
  for (const auto& h : hits_) {
    const QPointF delta = h.b - h.a, pos = e->pos() - h.a;
    const qreal length = QPointF::dotProduct(delta, delta);
    const qreal t =
        length > 0 ? qBound(0., QPointF::dotProduct(pos, delta) / length, 1.)
                   : 0.;
    const qreal distance = matching && show
                               ? QLineF(e->pos(), h.a + t * delta).length()
                               : qMin(QLineF(e->pos(), h.a).length(),
                                      QLineF(e->pos(), h.b).length());
    if (distance < best) {
      best = distance;
      id = h.id;
      side = QLineF(e->pos(), h.a).length() <= QLineF(e->pos(), h.b).length()
                 ? 0
                 : 1;
    }
  }
  if (id >= 0 && picked) picked(id, side);
}
void InspectionCanvas::wheelEvent(QWheelEvent* e) {
  if (zoomed)
    zoomed(qBound(.2, zoom * std::pow(1.001, e->angleDelta().y()), 8.));
  e->accept();
}
void InspectionCanvas::mouseDoubleClickEvent(QMouseEvent* e) {
  fit();
  if (zoomed) zoomed(1);
  e->accept();
}

FeatureView::FeatureView(QWidget* parent) : QWidget(parent) {
  pool_.setMaxThreadCount(1);
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(0);
  auto* mode = new QWidget;
  mode->setObjectName("inspectionModes");
  auto* modes = new QHBoxLayout(mode);
  mode->setFixedHeight(46);
  modes->setContentsMargins(12, 7, 12, 7);
  modes->setSpacing(0);
  photoMode_ = new QPushButton(QString::fromUtf8("照片"));
  photoMode_->setCheckable(true);
  photoMode_->setObjectName("photoMode");
  single_ = new QPushButton(QString::fromUtf8("特征点"));
  pair_ = new QPushButton(QString::fromUtf8("特征匹配"));
  single_->setCheckable(true);
  pair_->setCheckable(true);
  single_->setObjectName("singleMode");
  pair_->setObjectName("pairMode");
  modes->addWidget(photoMode_);
  modes->addWidget(single_);
  modes->addWidget(pair_);
  context_ = new QLabel;
  modes->addSpacing(12);
  modes->addWidget(context_);
  modes->addStretch();
  layout->addWidget(mode);
  connect(photoMode_, &QPushButton::clicked, this, [this] { setViewMode(Mode::Photos); });
  connect(single_, &QPushButton::clicked, this, [this] { setMode(false); });
  connect(pair_, &QPushButton::clicked, this, [this] { setMode(true); });
  auto* controls = new QWidget;
  controls->setObjectName("featureControls");
  controls->setFixedHeight(46);
  auto* row = new QHBoxLayout(controls);
  row->setContentsMargins(10, 7, 10, 7);
  row->setSpacing(7);
  imageLabel_ = new QLabel(QString::fromUtf8("图像"));
  row->addWidget(imageLabel_);
  a_ = new QComboBox;
  a_->setObjectName("featureImageA");
  a_->setMinimumWidth(95);
  a_->setMaximumWidth(165);
  a_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  row->addWidget(a_);
  swap_ = new QPushButton(QString::fromUtf8("交换"));
  row->addWidget(swap_);
  bLabel_ = new QLabel("B");
  row->addWidget(bLabel_);
  b_ = new QComboBox;
  b_->setObjectName("featureImageB");
  b_->setMinimumWidth(95);
  b_->setMaximumWidth(165);
  b_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  row->addWidget(b_);
  focusA_ = new QPushButton(QString::fromUtf8("检查 A"));
  focusB_ = new QPushButton(QString::fromUtf8("检查 B"));
  focusA_->setCheckable(true);
  focusB_->setCheckable(true);
  // The canvas endpoint and the photo sidebar select A/B without extra buttons.
  focusA_->setParent(this);
  focusB_->setParent(this);
  focusA_->hide();
  focusB_->hide();
  row->addStretch();
  filter_ = new QComboBox;
  filter_->addItems({QString::fromUtf8("几何验证"),
                     QString::fromUtf8("全部匹配"),
                     QString::fromUtf8("非内点")});
  row->addWidget(filter_);
  orientation_ = new QCheckBox(QString::fromUtf8("尺度与方向"));
  points_ = new QCheckBox(QString::fromUtf8("特征点"));
  lines_ = new QCheckBox(QString::fromUtf8("连线"));
  points_->setChecked(true);
  lines_->setChecked(true);
  row->addWidget(orientation_);
  row->addWidget(points_);
  row->addWidget(lines_);
  layout->addWidget(controls);
  auto* options = new QWidget;
  options_ = options;
  options->setObjectName("featureOptions");
  options->setFixedHeight(42);
  auto* tools = new QHBoxLayout(options);
  tools->setContentsMargins(10, 5, 10, 5);
  tools->addWidget(new QLabel(QString::fromUtf8("显示数量")));
  limit_ = new QSlider(Qt::Horizontal);
  limit_->setRange(1, 10000);
  limit_->setValue(singleLimit_);
  limit_->setFixedWidth(75);
  tools->addWidget(limit_);
  counts_ = new QLabel;
  tools->addWidget(counts_);
  tools->addStretch();
  zoom_ = new QLabel;
  tools->addWidget(zoom_);
  auto* minus = new QPushButton("−");
  auto* fit = new QPushButton(QString::fromUtf8("适配"));
  auto* plus = new QPushButton("+");
  tools->addWidget(minus);
  tools->addWidget(fit);
  tools->addWidget(plus);
  layout->addWidget(options);
  canvas_ = new InspectionCanvas;
  canvas_->setObjectName("inspectionCanvas");
  layout->addWidget(canvas_, 1);
  detail_ = new QLabel;
  detail_->setObjectName("inspectionDetail");
  detail_->setAlignment(Qt::AlignCenter);
  detail_->setWordWrap(true);
  detail_->setMinimumHeight(30);
  detail_->setMaximumHeight(52);
  layout->addWidget(detail_);
  auto setZoom = [this](qreal value) {
    if (matching_)
      pairZoom_ = value;
    else
      singleZoom_ = value;
    refresh();
  };
  canvas_->zoomed = setZoom;
  connect(minus, &QPushButton::clicked, this,
          [this, setZoom] { setZoom(qMax(.2, canvas_->zoom / 1.2)); });
  connect(plus, &QPushButton::clicked, this,
          [this, setZoom] { setZoom(qMin(8., canvas_->zoom * 1.2)); });
  connect(fit, &QPushButton::clicked, this, [this, setZoom] {
    canvas_->fit();
    setZoom(1);
  });
  connect(a_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int i) {
            if (changing_ || i < 0) return;
            if (matching_) {
              pairA_ = i;
              if (pairB_ == i) pairB_ = i == 0 ? 1 : 0;
              selectedMatch_ = -1;
            }
            selectImage(i);
          });
  connect(b_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this](int i) {
            if (changing_ || i < 0) return;
            const int index = b_->itemData(i).toInt();
            pairB_ = index;
            selectedMatch_ = -1;
            selectImage(index);
          });
  connect(swap_, &QPushButton::clicked, this, [this] {
    std::swap(pairA_, pairB_);
    selectedMatch_ = -1;
    load();
  });
  connect(focusA_, &QPushButton::clicked, this, [this] {
    image_ = pairA_;
    refresh();
    emit imageChanged(image_);
  });
  connect(focusB_, &QPushButton::clicked, this, [this] {
    image_ = pairB_;
    refresh();
    emit imageChanged(image_);
  });
  connect(filter_, qOverload<int>(&QComboBox::currentIndexChanged), this,
          [this] {
            selectedMatch_ = -1;
            refresh();
          });
  for (auto* box : {orientation_, points_, lines_})
    connect(box, &QCheckBox::toggled, this, [this] { refresh(); });
  connect(limit_, &QSlider::valueChanged, this, [this](int value) {
    if (changing_) return;
    if (matching_)
      pairLimit_ = value;
    else
      singleLimit_ = value;
    refresh();
  });
  canvas_->picked = [this](int id, int side) { choose(id, side); };
  refresh();
}
FeatureView::~FeatureView() {
  if (cancel_) cancel_->store(true);
  pool_.waitForDone();
}
void FeatureView::setWorkspace(const Workspace& w) {
  cache_.clear();
  workspace_ = w;
  image_ = pairA_ = 0;
  pairB_ = w.images.size() > 1 ? 1 : 0;
  selectedMatch_ = -1;
  selectedPoints_.clear();
  changing_ = true;
  a_->clear();
  for (const auto& image : w.images) a_->addItem(image.name);
  changing_ = false;
  load();
}
void FeatureView::selectImage(int index) {
  if (index < 0 || index >= workspace_.images.size()) return;
  image_ = index;
  if (matching_ && index != pairA_ && index != pairB_) {
    pairA_ = index;
    selectedMatch_ = -1;
  }
  canvas_->fit();
  emit imageChanged(index);
  load();
}
void FeatureView::setMode(bool matching) {
  setViewMode(matching ? Mode::Matches : Mode::Features);
}
void FeatureView::selectPairImage(int index) {
  if (mode_ != Mode::Matches) pairA_ = image_;
  if (index < 0 || index >= workspace_.images.size() || index == pairA_) return;
  pairB_ = index;
  selectedMatch_ = -1;
  if (mode_ != Mode::Matches) setViewMode(Mode::Matches);
  else load();
}
void FeatureView::setViewMode(Mode mode) {
  if (mode_ == mode) {
    refresh();
    return;
  }
  mode_ = mode;
  matching_ = mode == Mode::Matches;
  if (matching_ && image_ != pairA_ && image_ != pairB_) {
    pairA_ = image_;
    selectedMatch_ = -1;
  }
  canvas_->fit();
  load();
  emit modeChanged(matching_);
  emit viewModeChanged();
}
void FeatureView::load() {
  const int token = ++generation_;
  if (cancel_) cancel_->store(true);
  cancel_ = std::make_shared<std::atomic_bool>(false);
  const auto cancelled = cancel_;
  canvas_->data = {};
  canvas_->data.error = QString::fromUtf8("正在读取图像与工作区…");
  canvas_->update();
  refresh();
  if (workspace_.images.isEmpty()) {
    canvas_->data.error =
        QString::fromUtf8("未找到照片。请在工作区添加照片，或重新定位数据源目录。");
    refresh();
    return;
  }
  if (matching_ && workspace_.images.size() < 2) {
    canvas_->data.error = tr("特征匹配需要至少两张照片，请先添加照片或文件夹。");
    refresh();
    return;
  }
  const auto workspace = workspace_;
  const int first = matching_ ? pairA_ : image_, second = pairB_;
  const bool matching = matching_;
  const bool photo = mode_ == Mode::Photos;
  QStringList files{workspace.images[first].path};
  if (!photo) files << QDir(workspace.root).filePath(QString("features/%1.bin").arg(workspace.images[first].id));
  if (matching) files << workspace.images[second].path
      << QDir(workspace.root).filePath(QString("features/%1.bin").arg(workspace.images[second].id))
      << QDir(workspace.root).filePath("matches/raw_matches.tsv")
      << QDir(workspace.root).filePath("matches/verified_matches.tsv");
  const auto key = QString::number(int(mode_)) + ':' + inspectionStamp(files);
  if (const auto* cached = cache_.object(key)) { canvas_->data = *cached; refresh(); return; }
  auto* watcher = new QFutureWatcher<InspectionData>(this);
  connect(watcher, &QFutureWatcher<InspectionData>::finished, this,
          [this, watcher, token, key, files] {
            auto result = watcher->result();
            watcher->deleteLater();
            if (token != generation_) return;
            // A worker may finish before the GUI handles its completion. Never
            // display a result whose files changed during that interval.
            if (key != QString::number(int(mode_)) + ':' + inspectionStamp(files)) {
              load();
              return;
            }
            if (result.error.isEmpty()) {
              const qint64 bytes = result.a.sizeInBytes() + result.b.sizeInBytes() +
                  qint64(result.first.capacity() + result.second.capacity()) * sizeof(Keypoint) +
                  qint64(result.pair.matches.capacity()) * sizeof(Correspondence) +
                  qint64(result.displayOrder.capacity()) * sizeof(int);
              const int cost = int(qMax<qint64>(1, (bytes + 1048575) / 1048576));
              if (cost <= cache_.maxCost()) cache_.insert(key, new InspectionData(result), cost);
            }
            canvas_->data = std::move(result);
            refresh();
          });
  watcher->setFuture(QtConcurrent::run(&pool_, [workspace, first, second, matching, photo, cancelled] {
    InspectionData result;
    try {
      auto stopped = [cancelled] { return cancelled->load(); };
      if (stopped()) return result;
      const auto& a = workspace.images[first];
      result.nameA = a.name;
      QImageReader reader(a.path);
      reader.setAutoTransform(false);
      result.sourceA = reader.size();
      if (result.sourceA.width() > 4096 || result.sourceA.height() > 4096)
        reader.setScaledSize(result.sourceA.scaled(4096, 4096, Qt::KeepAspectRatio));
      result.a = reader.read();
      if (result.a.isNull())
        throw std::runtime_error(reader.errorString().toStdString());
      if (!result.sourceA.isValid()) result.sourceA = result.a.size();
      if (stopped()) return InspectionData{};
      if (!photo) result.first = workspace.features(a.id, stopped);
      if (matching) {
        const auto& b = workspace.images[second];
        result.nameB = b.name;
        QImageReader other(b.path);
        result.sourceB = other.size();
        if (result.sourceB.width() > 4096 || result.sourceB.height() > 4096)
          other.setScaledSize(result.sourceB.scaled(4096, 4096, Qt::KeepAspectRatio));
        result.b = other.read();
        if (result.b.isNull())
          throw std::runtime_error(other.errorString().toStdString());
        if (!result.sourceB.isValid()) result.sourceB = result.b.size();
        if (stopped()) return InspectionData{};
        result.second = workspace.features(b.id, stopped);
        result.pair = workspace.pair(a.id, b.id, result.first.size(),
                                     result.second.size(), stopped);
        QVector<QVector<int>> cells(32);
        for (int i = 0; i < result.pair.matches.size(); ++i) {
          const auto& point = result.first[int(result.pair.matches[i].a)];
          const int x = qBound(0, int(8 * point.x / qMax(1, result.sourceA.width())), 7);
          const int y = qBound(0, int(4 * point.y / qMax(1, result.sourceA.height())), 3);
          cells[y * 8 + x].append(i);
        }
        for (int offset = 0; result.displayOrder.size() < result.pair.matches.size(); ++offset)
          for (const auto& cell : cells)
            if (offset < cell.size()) result.displayOrder.append(cell[offset]);
      }
    } catch (const std::exception& e) {
      result.error = QString::fromUtf8(e.what());
      result.a = {};
    }
    return result;
  }));
}
void FeatureView::choose(int id, int side) {
  if (workspace_.images.isEmpty()) return;
  if (matching_) {
    if (id < 0 || id >= canvas_->data.pair.matches.size()) return;
    selectedMatch_ = id;
    const auto& m = canvas_->data.pair.matches[id];
    selectedPoints_[workspace_.images[pairA_].id] = int(m.a);
    selectedPoints_[workspace_.images[pairB_].id] = int(m.b);
    image_ = side == 0 ? pairA_ : pairB_;
    emit imageChanged(image_);
  } else
    selectedPoints_[workspace_.images[image_].id] = id;
  refresh();
}
void FeatureView::refresh() {
  changing_ = true;
  const bool photo = mode_ == Mode::Photos;
  photoMode_->setChecked(photo);
  single_->setChecked(mode_ == Mode::Features);
  pair_->setChecked(matching_);
  pair_->setEnabled(workspace_.images.size() > 1);
  for (QWidget* widget :
       QList<QWidget*>{b_, bLabel_, swap_, filter_, lines_})
    widget->setVisible(matching_);
  orientation_->setVisible(mode_ == Mode::Features);
  points_->setVisible(mode_ == Mode::Features);
  options_->setVisible(!photo);
  imageLabel_->setText(matching_ ? QString("A") : tr("图像"));
  a_->setCurrentIndex(matching_ ? pairA_ : image_);
  b_->clear();
  for (int i = 0; i < workspace_.images.size(); ++i)
    if (i != pairA_) b_->addItem(workspace_.images[i].name, i);
  b_->setCurrentIndex(b_->findData(pairB_));
  focusA_->setChecked(image_ == pairA_);
  focusB_->setChecked(image_ == pairB_);
  context_->setText(QString());
  limit_->setRange(1, matching_ ? 2000 : 10000);
  limit_->setValue(matching_ ? pairLimit_ : singleLimit_);
  changing_ = false;
  canvas_->matching = matching_;
  canvas_->show = !photo && (matching_ ? lines_->isChecked() : points_->isChecked());
  canvas_->orientation = orientation_->isChecked();
  canvas_->limit = limit_->value();
  canvas_->filter = filter_->currentIndex();
  canvas_->focus = image_ == pairB_ ? 1 : 0;
  canvas_->zoom = matching_ ? pairZoom_ : singleZoom_;
  canvas_->match = selectedMatch_;
  canvas_->point =
      workspace_.images.isEmpty()
          ? -1
          : selectedPoints_.value(workspace_.images[image_].id, -1);
  zoom_->setText(QString::number(qRound(canvas_->zoom * 100)) + "%");
  QString detail;
  if (photo) {
    counts_->clear();
  } else if (matching_) {
    const auto& pair = canvas_->data.pair;
    counts_->setText(
        QString::fromUtf8("上限 %1 · 匹配 %2 · 内点 %3%4")
            .arg(pairLimit_)
            .arg(pair.matches.size())
            .arg(pair.inliers)
            .arg(pair.verified ? QString() : QString::fromUtf8(" · 未验证")));
    if (selectedMatch_ >= 0 && selectedMatch_ < pair.matches.size()) {
      const auto& m = pair.matches[selectedMatch_];
      detail = QString::fromUtf8("点对 #%1 · A #%2 / B #%3 · %4")
                   .arg(selectedMatch_)
                   .arg(m.a)
                   .arg(m.b)
                   .arg(m.inlier     ? QString::fromUtf8("几何内点")
                        : m.verified ? QString::fromUtf8("非内点")
                                     : QString::fromUtf8("未验证"));
    }
  } else {
    counts_->setText(QString::fromUtf8("上限 %1 · 已载入 %2 个特征点")
                         .arg(singleLimit_)
                         .arg(canvas_->data.first.size()));
    const int id = canvas_->point;
    if (id >= 0 && id < canvas_->data.first.size()) {
      const auto& p = canvas_->data.first[id];
      detail =
          QString::fromUtf8("特征点 #%1 · (%2, %3) · 尺度 %4 px · 方向 %5°")
              .arg(id)
              .arg(p.x, 0, 'f', 1)
              .arg(p.y, 0, 'f', 1)
              .arg(p.scale, 0, 'f', 2)
              .arg(p.orientation, 0, 'f', 1);
    }
  }
  if (!photo && canvas_->data.error.isEmpty() && !canvas_->data.a.isNull() && detail.isEmpty()) {
    if (matching_ && canvas_->data.pair.matches.isEmpty())
      detail = tr("此图像对暂无匹配记录，可更换配对或执行特征匹配");
    else if (!matching_ && canvas_->data.first.isEmpty())
      detail = tr("此照片暂无特征点，请先执行特征提取");
  }
  detail_->setText(!canvas_->data.error.isEmpty() ? canvas_->data.error :
      photo ? tr("滚轮缩放 · 拖动平移 · 双击适应视图") : detail.isEmpty()
          ? (matching_
                 ? QString::fromUtf8(
                       "点击连线或端点检查匹配；拖动平移，滚轮缩放")
                 : QString::fromUtf8("点击特征点查看信息；拖动平移，滚轮缩放"))
          : detail);
  emit selectionChanged(detail);
  canvas_->update();
}
}  // namespace colite::gui
