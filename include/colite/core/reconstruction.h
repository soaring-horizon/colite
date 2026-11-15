#pragma once

#include <set>
#include <unordered_map>

#include "colite/core/camera.h"
#include "colite/core/image.h"
#include "colite/core/point3d.h"

namespace colite {

// 一个稀疏重建模型。主体结构参考 COLMAP Reconstruction。
class Reconstruction {
 public:
  const std::unordered_map<CameraId, Camera>& Cameras() const;
  const std::unordered_map<ImageId, Image>& Images() const;
  const std::unordered_map<Point3DId, Point3D>& Points3D() const;
  const std::set<ImageId>& RegisteredImageIds() const;

  // Automatic point IDs are monotonic, including after imports and deletion.
  // A saved boundary can delimit points created by a mapping operation without
  // snapshotting every point in the reconstruction.
  Point3DId NextPoint3DId() const { return next_point3d_id_; }

  // Optional operation-scoped track/point change collection for incremental
  // optimization. It is transient state and is not part of model IO.
  void BeginPoint3DChangeTracking();
  std::set<Point3DId> EndPoint3DChangeTracking();
  void MarkPoint3DChanged(Point3DId point3d_id);
  // Transient mutation epoch for caches scoped to synchronous track updates.
  // Direct parameter writes by BA are outside those scopes.
  std::uint64_t Point3DChangeVersion() const { return point3d_change_version_; }

  bool ExistsCamera(CameraId camera_id) const;
  bool ExistsImage(ImageId image_id) const;
  bool ExistsPoint3D(Point3DId point3d_id) const;
  bool IsImageRegistered(ImageId image_id) const;

  const Camera& GetCamera(CameraId camera_id) const;
  Camera& GetCamera(CameraId camera_id);
  const Image& GetImage(ImageId image_id) const;
  Image& GetImage(ImageId image_id);
  const Point3D& GetPoint3D(Point3DId point3d_id) const;
  Point3D& GetPoint3D(Point3DId point3d_id);

  void AddCamera(Camera camera);
  void AddImage(Image image);

  Point3DId AddPoint3D(const Eigen::Vector3d& xyz, Track track);
  void AddPoint3D(Point3D point3d);
  bool DeleteObservation(Point3DId point3d_id,
                         ImageId image_id,
                         Point2DIdx point2d_idx);
  bool DeletePoint3D(Point3DId point3d_id);

  void RegisterImage(ImageId image_id);
  void DeregisterImage(ImageId image_id);

 private:
  std::unordered_map<CameraId, Camera> cameras_;
  std::unordered_map<ImageId, Image> images_;
  std::unordered_map<Point3DId, Point3D> points3d_;
  std::set<ImageId> registered_image_ids_;
  Point3DId next_point3d_id_ = 1;
  bool track_point3d_changes_ = false;
  std::uint64_t point3d_change_version_ = 0;
  std::set<Point3DId> changed_point3d_ids_;
};

}  // namespace colite
