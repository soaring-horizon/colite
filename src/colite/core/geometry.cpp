#include "colite/core/geometry.h"

#include <algorithm>
#include <cmath>

#include <Eigen/SVD>

#include "colite/core/reconstruction.h"

namespace colite {
namespace {

constexpr double kMinTriangulationSingularGap = 1e-12;
constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;

Eigen::Matrix<double, 3, 4> ProjectionMatrixFromPose(
    const Pose3& cam_from_world) {
  Eigen::Matrix<double, 3, 4> projection;
  projection.leftCols<3>() = cam_from_world.q_cw.toRotationMatrix();
  projection.rightCols<1>() = cam_from_world.t_cw;
  return projection;
}

}  // namespace

Eigen::Vector3d PixelToBearing(const Camera& camera,
                               const Eigen::Vector2d& pixel) {
  return UnprojectPixel(camera, pixel, 1.0).normalized();
}

std::optional<Eigen::Vector3d> TryPixelToBearing(
    const Camera& camera, const Eigen::Vector2d& pixel) {
  const auto ray = TryUnprojectPixel(camera, pixel, 1.0);
  if (!ray || !std::isfinite(ray->norm()) || ray->norm() == 0) return std::nullopt;
  return ray->normalized();
}

std::optional<Eigen::Vector2d> ProjectWorldPoint(
    const Camera& camera,
    const Pose3& cam_from_world,
    const Eigen::Vector3d& point_world) {
  const Eigen::Vector3d point_camera =
      cam_from_world.TransformPoint(point_world);
  Eigen::Vector2d projected;
  if (!camera.VerifyParams() ||
      !ProjectPointWithCameraParams(camera.model_id, camera.params.data(),
                                    point_camera, &projected)) {
    return std::nullopt;
  }
  return projected;
}

std::optional<double> ReprojectionError(
    const Camera& camera,
    const Pose3& cam_from_world,
    const Eigen::Vector3d& point_world,
    const Eigen::Vector2d& observed_pixel) {
  const std::optional<Eigen::Vector2d> projected =
      ProjectWorldPoint(camera, cam_from_world, point_world);
  if (!projected) {
    return std::nullopt;
  }
  return (*projected - observed_pixel).norm();
}

std::optional<double> Point3DTrackReprojectionError(
    const Reconstruction& reconstruction,
    const Point3D& point3d) {
  double error_sum = 0.0;
  std::size_t num_observations = 0;
  for (const TrackElement& element : point3d.track.elements) {
    if (!reconstruction.ExistsImage(element.image_id)) {
      continue;
    }
    const Image& image = reconstruction.GetImage(element.image_id);
    if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
        element.point2d_idx >= image.points2d.size()) {
      continue;
    }
    const std::optional<double> error = ReprojectionError(
        reconstruction.GetCamera(image.camera_id), *image.cam_from_world,
        point3d.xyz, image.points2d[element.point2d_idx].xy);
    if (!error || !std::isfinite(*error)) {
      continue;
    }
    error_sum += *error;
    ++num_observations;
  }
  if (num_observations == 0) {
    return std::nullopt;
  }
  return error_sum / static_cast<double>(num_observations);
}

void UpdatePoint3DReprojectionErrors(Reconstruction* reconstruction) {
  for (const auto& [id, point] : reconstruction->Points3D()) {
    reconstruction->GetPoint3D(id).reprojection_error =
        Point3DTrackReprojectionError(*reconstruction, point).value_or(-1.0);
  }
}

std::optional<Eigen::Vector3d> TriangulatePoint(
    const Camera& camera1,
    const Pose3& cam1_from_world,
    const Eigen::Vector2d& pixel1,
    const Camera& camera2,
    const Pose3& cam2_from_world,
    const Eigen::Vector2d& pixel2) {
  const auto ray1 = TryUnprojectPixel(camera1, pixel1, 1.0);
  const auto ray2 = TryUnprojectPixel(camera2, pixel2, 1.0);
  if (!ray1 || !ray2) return std::nullopt;
  const Eigen::Vector2d x1(ray1->x() / ray1->z(), ray1->y() / ray1->z());
  const Eigen::Vector2d x2(ray2->x() / ray2->z(), ray2->y() / ray2->z());
  const Eigen::Matrix<double, 3, 4> p1 =
      ProjectionMatrixFromPose(cam1_from_world);
  const Eigen::Matrix<double, 3, 4> p2 =
      ProjectionMatrixFromPose(cam2_from_world);

  return TriangulatePointFromNormalized(p1, x1, p2, x2);
}

std::optional<Eigen::Vector3d> TriangulatePointFromNormalized(
    const Eigen::Matrix<double, 3, 4>& p1,
    const Eigen::Vector2d& x1,
    const Eigen::Matrix<double, 3, 4>& p2,
    const Eigen::Vector2d& x2) {

  Eigen::Matrix4d a;
  a.row(0) = x1.x() * p1.row(2) - p1.row(0);
  a.row(1) = x1.y() * p1.row(2) - p1.row(1);
  a.row(2) = x2.x() * p2.row(2) - p2.row(0);
  a.row(3) = x2.y() * p2.row(2) - p2.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(a, Eigen::ComputeFullV);
  const auto singular_values = svd.singularValues();
  if (singular_values(2) - singular_values(3) < kMinTriangulationSingularGap) {
    return std::nullopt;
  }

  const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
  if (std::abs(homogeneous.w()) < 1e-12) {
    return std::nullopt;
  }
  return homogeneous.hnormalized();
}

double TriangulationAngleDegrees(const Pose3& cam1_from_world,
                                 const Eigen::Vector3d& bearing1,
                                 const Pose3& cam2_from_world,
                                 const Eigen::Vector3d& bearing2) {
  const Eigen::Vector3d ray1_world =
      (cam1_from_world.q_cw.conjugate() * bearing1).normalized();
  const Eigen::Vector3d ray2_world =
      (cam2_from_world.q_cw.conjugate() * bearing2).normalized();
  const double cos_angle =
      std::clamp(ray1_world.dot(ray2_world), -1.0, 1.0);
  return std::acos(cos_angle) * kRadiansToDegrees;
}

double TriangulationAngleDegrees(const Eigen::Vector3d& camera_center1,
                                 const Eigen::Vector3d& camera_center2,
                                 const Eigen::Vector3d& point_world) {
  const Eigen::Vector3d ray1 = point_world - camera_center1;
  const Eigen::Vector3d ray2 = point_world - camera_center2;
  const double length1 = ray1.norm();
  const double length2 = ray2.norm();
  if (!(length1 > 0) || !(length2 > 0) ||
      !std::isfinite(length1) || !std::isfinite(length2)) return 0;
  const double cosine = std::clamp(std::abs((ray1 / length1).dot(ray2 / length2)), 0.0, 1.0);
  return std::acos(cosine) * kRadiansToDegrees;
}

}  // namespace colite
