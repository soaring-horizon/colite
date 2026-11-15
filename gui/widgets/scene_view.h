#pragma once
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>
#include <memory>
#include <QTimer>
#include <QElapsedTimer>
#include <QCache>

#include "project/workspace.h"

namespace colite::gui {
class SceneView : public QOpenGLWidget, protected QOpenGLFunctions {
  Q_OBJECT
 public:
  explicit SceneView(QWidget* parent = nullptr);
  ~SceneView() override;
  static void prepare(SceneData& scene);
  void setScene(SceneData data, QString cacheKey = {});
  void setLoading(const QString& text) { loading_ = text; update(); }
  void fit();
  void top();
  void setGrid(bool value) {
    grid_ = value;
    update();
  }
  void setCameras(bool value) {
    cameras_ = value;
    update();
  }
  void setCloud(bool value) {
    cloud_ = value;
    update();
  }
  void setSurface(bool value) {
    surface_ = value;
    update();
  }
  void setLayerVisible(bool visible) {
    layerVisible_ = visible;
    update();
  }
  QString graphicsInfo() const { return graphics_; }
  QString graphicsError() const {
    if (!error_.isEmpty()) return error_;
    if (shader_ && !shader_->isLinked())
      return QStringLiteral("OpenGL shader program is not linked: ") + shader_->log();
    return {};
  }
 signals:
  void selected(const QString& text);
  // Qt's first swapped frame after setScene; upload time is CPU/driver time,
  // not a GPU timer query or a monitor scanout measurement.
  void firstFramePresented(qint64 milliseconds, qint64 uploadMilliseconds, qint64 uploadBytes);

 protected:
  void initializeGL() override;
  void paintGL() override;
  void mousePressEvent(QMouseEvent*) override;
  void mouseMoveEvent(QMouseEvent*) override;
  void mouseDoubleClickEvent(QMouseEvent*) override;
  void wheelEvent(QWheelEvent*) override;
  void keyPressEvent(QKeyEvent*) override;

 private:
  SceneData scene_;
  QTimer settle_;
  QElapsedTimer presentationClock_;
  bool awaitingFrame_ = false;
  qint64 uploadMilliseconds_ = 0, uploadBytes_ = 0;
  struct GpuScene {
    QOpenGLBuffer navigation, points, triangles, cameras;
    std::unique_ptr<QOpenGLTexture> texture;
  };
  QCache<QString, GpuScene> gpuCache_{192}; // MiB; destroy only with this GL context current.
  std::unique_ptr<GpuScene> uncachedGpu_;
  GpuScene* gpu_ = nullptr;
  QString cacheKey_;
  QOpenGLBuffer gridBuffer_;
  std::unique_ptr<QOpenGLShaderProgram> shader_;
  QString graphics_, error_, loading_;
  QVector<SceneVertex> gridVertices_;
  bool dirty_ = true, grid_ = true, cameras_ = true, cloud_ = true,
       surface_ = false, layerVisible_ = true;
  float yaw_ = -30, pitch_ = 25, distance_ = 16;
  QPoint last_;
  QPointF pan_;
  QMatrix4x4 matrix() const;
  void cleanupGL();
  void draw(QOpenGLBuffer& buffer, int count, GLenum primitive,
            bool textured = false);
};
}  // namespace colite::gui
