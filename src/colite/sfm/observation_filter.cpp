#include "colite/sfm/observation_filter.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <vector>

#include "colite/core/geometry.h"

namespace colite {
namespace {

struct ObservationRef {
  Point3DId point3d_id = kInvalidPoint3DId;
  ImageId image_id = kInvalidImageId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;
};

bool HasUsableObservation(const Reconstruction& reconstruction,
                          Point3DId point3d_id,
                          const TrackElement& element) {
  if (!reconstruction.ExistsImage(element.image_id)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(element.image_id);
  if (!image.registered || !image.HasPose() ||
      !reconstruction.ExistsCamera(image.camera_id) ||
      element.point2d_idx >= image.points2d.size()) {
    return false;
  }
  return image.points2d[element.point2d_idx].point3d_id == point3d_id;
}

bool HasLargeReprojectionError(const Reconstruction& reconstruction,
                               const Point3D& point3d,
                               const TrackElement& element,
                               double max_error_px) {
  const Image& image = reconstruction.GetImage(element.image_id);
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  const Point2D& point2d = image.points2d[element.point2d_idx];
  const std::optional<double> error =
      ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                        point2d.xy);
  return !error.has_value() || *error > max_error_px;
}

double MaxTriangulationAngleDegrees(const Reconstruction& reconstruction,
                                    const Point3D& point3d,
                                    double required_angle) {
  double max_angle = 0.0;
  for (std::size_t i = 0; i < point3d.track.elements.size(); ++i) {
    const TrackElement& element1 = point3d.track.elements[i];
    if (!HasUsableObservation(reconstruction, point3d.id, element1)) {
      continue;
    }
    const Image& image1 = reconstruction.GetImage(element1.image_id);
    const auto center1 = image1.cam_from_world->CameraCenter();

    for (std::size_t j = i + 1; j < point3d.track.elements.size(); ++j) {
      const TrackElement& element2 = point3d.track.elements[j];
      if (!HasUsableObservation(reconstruction, point3d.id, element2)) {
        continue;
      }
      const Image& image2 = reconstruction.GetImage(element2.image_id);
      max_angle = std::max(max_angle,
                           TriangulationAngleDegrees(center1,
                               image2.cam_from_world->CameraCenter(), point3d.xyz));
      // Only the threshold decision is consumed. A passing pair proves the
      // point is sufficiently constrained; longer tracks need no more pairs.
      if (max_angle >= required_angle) return max_angle;
    }
  }
  return max_angle;
}

std::vector<Point3DId> CollectPointIds(const Reconstruction& reconstruction) {
  std::vector<Point3DId> point_ids;
  point_ids.reserve(reconstruction.Points3D().size());
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d;
    point_ids.push_back(point3d_id);
  }
  return point_ids;
}

std::vector<Point3DId> CollectPointIdsInImages(
    const Reconstruction& reconstruction,
    const std::set<ImageId>& image_ids,
    const std::set<Point3DId>* additional_point_ids) {
  std::set<Point3DId> unique_point_ids;
  if (additional_point_ids != nullptr) unique_point_ids = *additional_point_ids;
  for (const ImageId image_id : image_ids) {
    if (!reconstruction.ExistsImage(image_id)) {
      continue;
    }
    const Image& image = reconstruction.GetImage(image_id);
    for (const Point2D& point2d : image.points2d) {
      if (point2d.HasPoint3D() &&
          reconstruction.ExistsPoint3D(point2d.point3d_id)) {
        unique_point_ids.insert(point2d.point3d_id);
      }
    }
  }
  return {unique_point_ids.begin(), unique_point_ids.end()};
}

ObservationFilterSummary FilterPointIds(
    const ObservationFilterOptions& options,
    const std::vector<Point3DId>& point_ids,
    Reconstruction* reconstruction) {
  ObservationFilterSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  std::vector<ObservationRef> observations_to_delete;
  for (const Point3DId point3d_id : point_ids) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) {
      continue;
    }
    const Point3D& point3d = reconstruction->GetPoint3D(point3d_id);
    for (const TrackElement& element : point3d.track.elements) {
      const bool invalid = !HasUsableObservation(*reconstruction, point3d_id, element);
      if (invalid || HasLargeReprojectionError(*reconstruction, point3d, element,
                                    options.max_reprojection_error_px)) {
        if (invalid) ++summary.num_invalid_observations;
        else ++summary.num_reprojection_observations;
        observations_to_delete.push_back(
            {point3d_id, element.image_id, element.point2d_idx});
      }
    }
  }

  for (const ObservationRef& observation : observations_to_delete) {
    if (reconstruction->DeleteObservation(observation.point3d_id,
                                          observation.image_id,
                                          observation.point2d_idx)) {
      ++summary.num_removed_observations;
    }
  }

  std::vector<Point3DId> points_to_delete;
  for (const Point3DId point3d_id : point_ids) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) {
      continue;
    }
    const Point3D& point3d = reconstruction->GetPoint3D(point3d_id);
    const bool short_track = point3d.track.Length() < options.min_track_length;
    if (short_track ||
        MaxTriangulationAngleDegrees(*reconstruction, point3d,
                                     options.min_triangulation_angle_deg) <
            options.min_triangulation_angle_deg) {
      points_to_delete.push_back(point3d_id);
      if (short_track) ++summary.num_short_track_points;
      else ++summary.num_low_angle_points;
    }
  }

  for (const Point3DId point3d_id : points_to_delete) {
    if (reconstruction->DeletePoint3D(point3d_id)) {
      ++summary.num_removed_points;
    }
  }

  summary.success = true;
  summary.message = "Observation filtering finished";
  return summary;
}

}  // namespace

NegativeDepthFilterSummary FilterObservationsWithNegativeDepth(
    Reconstruction* reconstruction) {
  NegativeDepthFilterSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    const Image& image = reconstruction->GetImage(image_id);
    const Pose3& pose = image.cam_from_world.value();
    Eigen::RowVector4d depth_row;
    depth_row.head<3>() = pose.q_cw.toRotationMatrix().row(2);
    depth_row(3) = pose.t_cw.z();
    for (std::size_t idx = 0; idx < image.points2d.size(); ++idx) {
      const Point2D& point2d = image.points2d[idx];
      if (!point2d.HasPoint3D()) continue;
      const Point3DId point_id = point2d.point3d_id;
      const Point3D& point = reconstruction->GetPoint3D(point_id);
      if (depth_row.dot(point.xyz.homogeneous()) >=
          std::numeric_limits<double>::epsilon()) continue;
      if (point.track.Length() <= 2) {
        reconstruction->DeletePoint3D(point_id);
        ++summary.num_removed_points;
      } else {
        reconstruction->DeleteObservation(
            point_id, image_id, static_cast<Point2DIdx>(idx));
      }
      ++summary.num_filtered_observations;
    }
  }
  // A later point deletion can also unlink an earlier positive observation.
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    for (const auto& point2d : reconstruction->GetImage(image_id).points2d) {
      if (point2d.HasPoint3D()) ++summary.num_remaining_observations;
    }
  }
  summary.success = true;
  summary.message = "Negative-depth observation filtering finished";
  return summary;
}

ObservationFilterSummary FilterObservationsAndPoints(
    const ObservationFilterOptions& options,
    Reconstruction* reconstruction) {
  if (reconstruction == nullptr) {
    ObservationFilterSummary summary;
    summary.message = "Reconstruction is null";
    return summary;
  }
  return FilterPointIds(options, CollectPointIds(*reconstruction),
                        reconstruction);
}

ObservationFilterSummary FilterObservationsAndPointsInImages(
    const ObservationFilterOptions& options,
    const std::set<ImageId>& image_ids,
    Reconstruction* reconstruction,
    const std::set<Point3DId>* additional_point_ids) {
  if (reconstruction == nullptr) {
    ObservationFilterSummary summary;
    summary.message = "Reconstruction is null";
    return summary;
  }
  return FilterPointIds(options,
                        CollectPointIdsInImages(*reconstruction, image_ids,
                                                additional_point_ids),
                        reconstruction);
}

}  // namespace colite
