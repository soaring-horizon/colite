#pragma once
#include <QColor>
#include <QImage>
#include <QMap>
#include <QPointF>
#include <QVector3D>
#include <QVector>
#include <functional>

#include "project.h"

namespace colite::gui {
struct Keypoint {
  float x = 0, y = 0, scale = 0, orientation = 0, response = 0;
};
struct ImageRecord {
  quint32 id = 0, camera = 0;
  QString name, path;
};
struct Correspondence {
  quint32 a = 0, b = 0;
  bool inlier = false, verified = false;
};
struct PairData {
  quint32 a = 0, b = 0;
  QVector<Correspondence> matches;
  bool verified = false;
  int inliers = 0;
};
struct Workspace {
  QVector<ImageRecord> images;
  QString root;
  QString error;
  QMap<quint32, QImage> thumbnails;
  static Workspace read(const Project& project);
  QVector<Keypoint> features(quint32 image, const std::function<bool()>& cancelled = {}) const;
  PairData pair(quint32 a, quint32 b, int countA, int countB,
                const std::function<bool()>& cancelled = {}) const;
};
struct SceneVertex {
  float x = 0, y = 0, z = 0, r = .5f, g = .5f, b = .5f, u = 0, v = 0;
};
struct SceneData {
  QVector<SceneVertex> points, triangles, cameras, navigationPoints;
  QString source, texture, error, textureRevision;
  QImage texturePreview;
  quint64 totalPoints = 0;
  quint64 totalTriangles = 0;
  bool sampledMesh = false;
};
struct ArtifactSummary { quint64 points = 0, triangles = 0; QString texture; };
QString fileRevision(const QString& path);
// Streams all geometry and validates texture contents without retaining display data.
ArtifactSummary validateScene(const QString& path, const std::function<bool()>& cancelled = {});
using SceneProgress = std::function<void(const QString&, int)>;
SceneData readScene(const QString& path, const SceneProgress& progress = {},
                   const std::function<bool()>& cancelled = {});
}  // namespace colite::gui
