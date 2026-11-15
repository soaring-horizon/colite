#include "scene_view.h"

#include <QtWidgets>
#include <cmath>
#include <limits>

namespace colite::gui {
SceneView::SceneView(QWidget* parent) : QOpenGLWidget(parent) {
  setMinimumSize(300, 180);
  connect(this, &QOpenGLWidget::frameSwapped, this, [this] {
    if (!awaitingFrame_) return;
    awaitingFrame_ = false;
    emit firstFramePresented(presentationClock_.elapsed(), uploadMilliseconds_, uploadBytes_);
  });
  settle_.setSingleShot(true);
  settle_.setInterval(150);
  connect(&settle_, &QTimer::timeout, this, [this] { update(); });
  setFocusPolicy(Qt::StrongFocus);
  setCursor(Qt::OpenHandCursor);
  for (int i = -6; i <= 6; ++i) {
    const float n = float(i);
    for (auto p : {QVector3D(n, -3, -6), QVector3D(n, -3, 6),
                   QVector3D(-6, -3, n), QVector3D(6, -3, n)})
      gridVertices_.push_back({p.x(), p.y(), p.z(), .82f, .86f, .88f});
  }
}
SceneView::~SceneView() {
  if (context()) disconnect(context(), nullptr, this, nullptr);
  cleanupGL();
}
void SceneView::cleanupGL() {
  if (context()) {
    makeCurrent();
    gpu_ = nullptr;
    gpuCache_.clear();
    uncachedGpu_.reset();
    shader_.reset();
    gridBuffer_.destroy();
    doneCurrent();
  }
}
void SceneView::prepare(SceneData& scene) {
  QVector3D low(1e30f, 1e30f, 1e30f), high(-1e30f, -1e30f, -1e30f);
  for (const auto& p : scene.points) {
    low.setX(qMin(low.x(), p.x));
    low.setY(qMin(low.y(), p.y));
    low.setZ(qMin(low.z(), p.z));
    high.setX(qMax(high.x(), p.x));
    high.setY(qMax(high.y(), p.y));
    high.setZ(qMax(high.z(), p.z));
  }
  const auto center = (low + high) / 2;
  const float scale =
      scene.points.isEmpty() ? 1 : 8 / qMax(.001f, (high - low).length());
  for (auto* vertices : {&scene.points, &scene.triangles, &scene.cameras})
    for (auto& p : *vertices) {
      const auto v = (QVector3D(p.x, p.y, p.z) - center) * scale;
      p.x = v.x();
      p.y = v.y();
      p.z = v.z();
    }
  // Point-cloud navigation may be sampled, but a surface must retain its
  // triangles and UVs while moving. Textured PLY vertices need not have RGB.
  if (scene.triangles.isEmpty() && scene.points.size() > 80000) {
    const int stride = qMax(1, (scene.points.size() + 79999) / 80000);
    for (int i = 0; i < scene.points.size(); i += stride)
      scene.navigationPoints.push_back(scene.points[i]);
  }
}
void SceneView::setScene(SceneData scene, QString cacheKey) {
  presentationClock_.start(); awaitingFrame_ = true;
  uploadMilliseconds_ = uploadBytes_ = 0;
  loading_.clear();
  scene_ = std::move(scene);
  cacheKey_ = std::move(cacheKey);
  dirty_ = true;
  surface_ = !scene_.triangles.isEmpty();
  fit();
}
void SceneView::fit() {
  yaw_ = -30;
  pitch_ = 25;
  // Fit the rotated geometry to the current aspect ratio, with an 18% margin.
  QMatrix4x4 rotation;
  rotation.rotate(pitch_, 1, 0, 0);
  rotation.rotate(yaw_, 0, 1, 0);
  const float tangent = std::tan(20.f * 3.14159265358979323846f / 180.f);
  const float aspect = float(qMax(1, width())) / qMax(1, height());
  distance_ = 1;
  for (const auto& point : qAsConst(scene_.points)) {
    const auto v = rotation * QVector3D(point.x, point.y, point.z);
    distance_ = qMax(distance_, v.z() + 1.18f * qMax(std::abs(v.y()) / tangent,
                                                    std::abs(v.x()) / (tangent * aspect)));
  }
  if (scene_.points.isEmpty()) distance_ = 16;
  pan_ = {};
  update();
}
void SceneView::top() {
  pitch_ = 89;
  yaw_ = 0;
  update();
}
QMatrix4x4 SceneView::matrix() const {
  QMatrix4x4 p, v;
  p.perspective(40, float(width()) / qMax(1, height()), .05f, 200.f);
  v.translate(float(pan_.x()), float(pan_.y()), -distance_);
  v.rotate(pitch_, 1, 0, 0);
  v.rotate(yaw_, 0, 1, 0);
  return p * v;
}
void SceneView::initializeGL() {
  initializeOpenGLFunctions();
  connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &SceneView::cleanupGL, Qt::DirectConnection);
  graphics_ = QString::fromLatin1(
      reinterpret_cast<const char*>(glGetString(GL_VERSION)));
  graphics_ += " | " + QString::fromLatin1(
      reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
  shader_ = std::make_unique<QOpenGLShaderProgram>();
  const QByteArray version = context()->isOpenGLES()
      ? "#version 100\nprecision mediump float;\n" : "#version 120\n";
  bool ok = shader_->addShaderFromSourceCode(
      QOpenGLShader::Vertex,
      version + "attribute vec3 pos;attribute vec3 color;attribute vec2 "
      "uv;uniform mat4 mvp;varying vec3 c;varying vec2 t;void "
      "main(){gl_Position=mvp*vec4(pos,1.);gl_PointSize=2.2;c=color;t=uv;}");
  ok &= shader_->addShaderFromSourceCode(
      QOpenGLShader::Fragment,
      version + "varying vec3 c;varying vec2 t;uniform bool "
      "textured;uniform sampler2D tex;void "
      "main(){gl_FragColor=textured?texture2D(tex,t):vec4(c,1.);}");
  ok &= shader_->link();
  if (!ok) error_ = shader_->log();
  gridBuffer_.create();
  gridBuffer_.bind();
  gridBuffer_.allocate(gridVertices_.constData(), gridVertices_.size() * int(sizeof(SceneVertex)));
  gridBuffer_.release();
  dirty_ = true;
}
void SceneView::draw(QOpenGLBuffer& buffer, int count, GLenum primitive,
                     bool textured) {
  if (!count) return;
  buffer.bind();
  const int size = sizeof(SceneVertex);
  shader_->enableAttributeArray("pos");
  shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3, size);
  shader_->enableAttributeArray("color");
  shader_->setAttributeBuffer("color", GL_FLOAT, 3 * sizeof(float), 3, size);
  shader_->enableAttributeArray("uv");
  shader_->setAttributeBuffer("uv", GL_FLOAT, 6 * sizeof(float), 2, size);
  shader_->setUniformValue("textured", textured);
  shader_->setUniformValue("tex", 0);
  if (textured) gpu_->texture->bind();
  glDrawArrays(primitive, 0, count);
  if (textured) gpu_->texture->release();
  shader_->disableAttributeArray("pos");
  shader_->disableAttributeArray("color");
  shader_->disableAttributeArray("uv");
  buffer.release();
}
void SceneView::paintGL() {
  glClearColor(247 / 255.f, 249 / 255.f, 248 / 255.f, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (error_.isEmpty() && shader_ && shader_->isLinked()) {
    if (dirty_) {
      QElapsedTimer uploadClock; uploadClock.start();
      gpu_ = cacheKey_.isEmpty() ? nullptr : gpuCache_.object(cacheKey_);
      uncachedGpu_.reset();
      if (!gpu_) {
        auto resource = std::make_unique<GpuScene>();
        uploadBytes_ = (qint64(scene_.points.size()) + scene_.navigationPoints.size() +
            scene_.triangles.size() + scene_.cameras.size()) * sizeof(SceneVertex) +
            scene_.texturePreview.sizeInBytes();
        auto upload = [](QOpenGLBuffer& buffer, const QVector<SceneVertex>& v) {
          buffer.create(); buffer.bind();
          buffer.allocate(v.constData(), v.size() * int(sizeof(SceneVertex)));
          buffer.release();
        };
        upload(resource->points, scene_.points);
        upload(resource->navigation, scene_.navigationPoints);
        upload(resource->triangles, scene_.triangles);
        upload(resource->cameras, scene_.cameras);
        if (!scene_.texturePreview.isNull()) {
          resource->texture = std::make_unique<QOpenGLTexture>(
                scene_.texturePreview, QOpenGLTexture::DontGenerateMipMaps);
          resource->texture->setMinificationFilter(QOpenGLTexture::Linear);
        }
        gpu_ = resource.get();
        const int cost = int(qMax<qint64>(1, (uploadBytes_ + 1048575) / 1048576));
        if (!cacheKey_.isEmpty() && cost <= gpuCache_.maxCost()) gpuCache_.insert(cacheKey_, resource.release(), cost);
        else uncachedGpu_ = std::move(resource);
      }
      dirty_ = false;
      uploadMilliseconds_ = uploadClock.elapsed();
    }
    glEnable(GL_DEPTH_TEST);
    shader_->bind();
    shader_->setUniformValue("mvp", matrix());
    if (grid_) draw(gridBuffer_, gridVertices_.size(), GL_LINES);
    const bool preview = !surface_ && settle_.isActive() && !scene_.navigationPoints.isEmpty();
    if (layerVisible_ && preview && cloud_)
      draw(gpu_->navigation, scene_.navigationPoints.size(), GL_POINTS);
    if (layerVisible_ && !preview && cloud_ && !surface_) {
      if (!context()->isOpenGLES()) glPointSize(2.2f);
      draw(gpu_->points, scene_.points.size(), GL_POINTS);
    }
    if (layerVisible_ && surface_)
      draw(gpu_->triangles, scene_.triangles.size(), GL_TRIANGLES, bool(gpu_->texture));
    if (cameras_) draw(gpu_->cameras, scene_.cameras.size(), GL_LINES);
    shader_->release();
    glDisable(GL_DEPTH_TEST);
  }
  QPainter painter(this);
  painter.setPen(QColor("#365463"));
  painter.drawText(16, 25,
                   scene_.source.isEmpty()
                       ? QString::fromUtf8("尚未加载模型")
                       : QFileInfo(scene_.source).fileName());
  painter.drawText(16, 45,
                   QString::fromUtf8("显示 %1 / %2 点 · %3 个三角形")
                       .arg(scene_.points.size())
                       .arg(scene_.totalPoints)
                       .arg(scene_.triangles.size() / 3));
  painter.drawText(
      16, height() - 15,
      QString::fromUtf8("左拖旋转 · 右拖平移 · 滚轮缩放 · 双击选点 · F 适配"));
  if (!error_.isEmpty() || !scene_.error.isEmpty()) {
    painter.setPen(QColor("#ae4f44"));
    painter.drawText(rect().adjusted(20, 70, -20, -55),
                     Qt::AlignCenter | Qt::TextWordWrap, error_ + scene_.error);
  }
  if (!loading_.isEmpty()) {
    const QRect box = rect().adjusted(30, height() / 2 - 30, -30, -height() / 2 + 30);
    painter.fillRect(box, QColor(247, 249, 248, 235));
    painter.setPen(QColor("#226f56"));
    painter.drawText(box, Qt::AlignCenter, loading_);
  }
}
void SceneView::mousePressEvent(QMouseEvent* e) {
  last_ = e->pos();
  setFocus();
}
void SceneView::mouseMoveEvent(QMouseEvent* e) {
  if (!e->buttons()) return;
  settle_.start();
  const auto d = e->pos() - last_;
  if (e->buttons() & Qt::LeftButton) {
    yaw_ += d.x() * .4f;
    pitch_ = qBound(-89.f, pitch_ + d.y() * .4f, 89.f);
  }
  if (e->buttons() & Qt::RightButton)
    pan_ += QPointF(d.x() * distance_ / 1000, -d.y() * distance_ / 1000);
  last_ = e->pos();
  update();
}
void SceneView::wheelEvent(QWheelEvent* e) {
  settle_.start();
  distance_ = qBound(
      2.f, distance_ * float(std::pow(.999, e->angleDelta().y())), 100.f);
  update();
}
void SceneView::keyPressEvent(QKeyEvent* e) {
  if (e->key() == Qt::Key_F)
    fit();
  else
    QOpenGLWidget::keyPressEvent(e);
}
void SceneView::mouseDoubleClickEvent(QMouseEvent* e) {
  double best = 100;
  int index = -1;
  const auto transform = matrix();
  for (int i = 0; i < scene_.points.size(); ++i) {
    const auto& p = scene_.points[i];
    const auto v = transform * QVector4D(p.x, p.y, p.z, 1);
    if (v.w() <= 0) continue;
    const QPointF screen((v.x() / v.w() + 1) * width() / 2,
                         (1 - v.y() / v.w()) * height() / 2);
    const auto delta = screen - e->pos();
    const auto d = QPointF::dotProduct(delta, delta);
    if (d < best) {
      best = d;
      index = i;
    }
  }
  emit selected(index >= 0
                    ? QString::fromUtf8("显示点 #%1 · 只读选择").arg(index)
                    : QString::fromUtf8("未命中显示点"));
}
}  // namespace colite::gui
