#include "colite/sfm/incremental_triangulator.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/SVD>
#include <glog/logging.h>

#include "colite/core/geometry.h"
#include "colite/core/parallel.h"
#include "colite/sfm/observation_manager.h"

namespace colite {
namespace {

using ObservationKey = IncrementalTriangulatorCache::ObservationKey;
using CorrespondenceAdjacency =
    IncrementalTriangulatorCache::CorrespondenceAdjacency;

struct PointPairHash {
  std::size_t operator()(const std::pair<Point3DId, Point3DId>& pair) const {
    const auto first = std::hash<Point3DId>{}(pair.first);
    const auto second = std::hash<Point3DId>{}(pair.second);
    return first ^ (second + 0x9e3779b9U + (first << 6) + (first >> 2));
  }
};

// Scoped to synchronous completion/merge operations. A failed trial may be
// reused only until a point changes, including a partially edited failure.
class MergeTrialCache {
 public:
  bool Contains(Point3DId first, Point3DId second,
                const Reconstruction& reconstruction) {
    Refresh(reconstruction);
    return failures_.count(Key(first, second)) != 0;
  }
  void Remember(Point3DId first, Point3DId second,
                std::uint64_t version_before,
                const Reconstruction& reconstruction) {
    Refresh(reconstruction);
    if (version_before == version_ && failures_.size() < 65536) {
      failures_.insert(Key(first, second));
    }
  }
 private:
  static std::pair<Point3DId, Point3DId> Key(Point3DId a, Point3DId b) {
    return {std::min(a, b), std::max(a, b)};
  }
  void Refresh(const Reconstruction& reconstruction) {
    if (version_ != reconstruction.Point3DChangeVersion()) {
      failures_.clear();
      version_ = reconstruction.Point3DChangeVersion();
    }
  }
  std::uint64_t version_ = 0;
  std::unordered_set<std::pair<Point3DId, Point3DId>, PointPairHash> failures_;
};

std::optional<Eigen::Vector3d> TriangulateObservationsLinear(
    const std::vector<ObservationKey>& observations,
    const Reconstruction& reconstruction);

bool PairContainsImage(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id || pair.image_id2 == image_id;
}

ImageId OtherImageId(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id ? pair.image_id2 : pair.image_id1;
}

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.HasUsableCorrespondences();
}

bool PassesPairAngleLayer(const TwoViewGeometry& geometry,
                          const IncrementalTriangulatorOptions& options) {
  if (options.min_pair_median_triangulation_angle_deg <= 0.0 ||
      geometry.median_triangulation_angle < 0.0) {
    return true;
  }
  return geometry.median_triangulation_angle >=
         options.min_pair_median_triangulation_angle_deg;
}

bool HasPoseAndCamera(const Reconstruction& reconstruction, ImageId image_id) {
  if (!reconstruction.ExistsImage(image_id)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(image_id);
  return image.HasPose() && reconstruction.ExistsCamera(image.camera_id);
}

bool HasValidMatchIndices(const FeatureMatch& match,
                          const Image& image1,
                          const Image& image2) {
  return match.idx1 != kInvalidPoint2DIdx &&
         match.idx2 != kInvalidPoint2DIdx &&
         match.idx1 < image1.points2d.size() &&
         match.idx2 < image2.points2d.size();
}

bool IsFinitePoint(const Eigen::Vector3d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y()) &&
         std::isfinite(point.z());
}

struct MergePositionScore {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  double max_error_px = std::numeric_limits<double>::infinity();
  double squared_error_sum = std::numeric_limits<double>::infinity();
  bool valid = false;
};

MergePositionScore ScoreMergedPointPosition(
    const Reconstruction& reconstruction,
    const Track& first_track,
    const Track& second_track,
    const Eigen::Vector3d& xyz) {
  MergePositionScore score;
  score.xyz = xyz;
  score.max_error_px = 0.0;
  score.squared_error_sum = 0.0;
  score.valid = IsFinitePoint(xyz);
  if (!score.valid) {
    return score;
  }
  for (const Track* track : {&first_track, &second_track}) {
    for (const TrackElement& element : track->elements) {
      const Image& image = reconstruction.GetImage(element.image_id);
      const Camera& camera = reconstruction.GetCamera(image.camera_id);
      const std::optional<double> error =
          ReprojectionError(camera, *image.cam_from_world, xyz,
                            image.points2d[element.point2d_idx].xy);
      if (!error.has_value()) {
        score.valid = false;
        return score;
      }
      score.max_error_px = std::max(score.max_error_px, *error);
      score.squared_error_sum += *error * *error;
    }
  }
  return score;
}

bool IsBetterMergePosition(const MergePositionScore& candidate,
                           const MergePositionScore& best) {
  return candidate.valid &&
         (!best.valid ||
          candidate.max_error_px < best.max_error_px ||
          (candidate.max_error_px == best.max_error_px &&
           candidate.squared_error_sum < best.squared_error_sum));
}

Eigen::Vector3d WeightedMergedPointPosition(const Point3D& point1,
                                            const Point3D& point2) {
  const double weight1 =
      static_cast<double>(std::max<std::size_t>(1, point1.track.Length()));
  const double weight2 =
      static_cast<double>(std::max<std::size_t>(1, point2.track.Length()));
  return (weight1 * point1.xyz + weight2 * point2.xyz) / (weight1 + weight2);
}

bool HasTwoIncompatibleImageObservations(
    const Track& first, const Track& second,
    double max_error_px, const Reconstruction& reconstruction) {
  // Every accepted position fits all observations except, at most, one
  // explicitly pruned outlier. Two conflicting image groups require at least
  // two outliers: the same projected coordinate cannot lie within r of both
  // observations when one coordinate differs by more than 2*r.
  if (first.elements.empty() || second.elements.empty() ||
      !std::isfinite(max_error_px) || max_error_px < 0.0 ||
      first.Length() > 1024 / second.Length()) {
    return false;
  }
  ImageId conflicting_image = kInvalidImageId;
  for (const auto& a : first.elements) {
    for (const auto& b : second.elements) {
      if (a.image_id != b.image_id || a.point2d_idx == b.point2d_idx) continue;
      if (!reconstruction.ExistsImage(a.image_id)) return false;
      const auto& image = reconstruction.GetImage(a.image_id);
      if (a.point2d_idx >= image.points2d.size() ||
          b.point2d_idx >= image.points2d.size()) return false;
      const auto& xy1 = image.points2d[a.point2d_idx].xy;
      const auto& xy2 = image.points2d[b.point2d_idx].xy;
      if (!xy1.allFinite() || !xy2.allFinite()) continue;
      const double margin = 64 * std::numeric_limits<double>::epsilon() *
          (1.0 + max_error_px + xy1.cwiseAbs().sum() + xy2.cwiseAbs().sum());
      if ((xy1 - xy2).cwiseAbs().maxCoeff() > 2 * max_error_px + margin) {
        if (conflicting_image != kInvalidImageId &&
            conflicting_image != a.image_id) return true;
        conflicting_image = a.image_id;
      }
    }
  }
  return false;
}

bool TryMergePoint3Ds(Point3DId target_point3d_id,
                      Point3DId source_point3d_id,
                      const IncrementalTriangulatorOptions& options,
                      Reconstruction* reconstruction) {
  if (target_point3d_id == source_point3d_id ||
      !reconstruction->ExistsPoint3D(target_point3d_id) ||
      !reconstruction->ExistsPoint3D(source_point3d_id)) {
    return false;
  }

  const Point3D& target_point3d =
      reconstruction->GetPoint3D(target_point3d_id);
  const Point3D& source_point3d =
      reconstruction->GetPoint3D(source_point3d_id);
  if (HasTwoIncompatibleImageObservations(
          target_point3d.track, source_point3d.track,
          options.merge_max_reprojection_error_px, *reconstruction)) {
    return false;
  }
  MergePositionScore best = ScoreMergedPointPosition(
      *reconstruction, target_point3d.track, source_point3d.track,
      WeightedMergedPointPosition(target_point3d, source_point3d));
  std::vector<ObservationKey> observations;
  observations.reserve(target_point3d.track.Length() +
                       source_point3d.track.Length());
  for (const Track* track :
       {&target_point3d.track, &source_point3d.track}) {
    for (const TrackElement& element : track->elements) {
      observations.emplace_back(element.image_id, element.point2d_idx);
    }
  }
  const std::optional<Eigen::Vector3d> triangulated_xyz =
      TriangulateObservationsLinear(observations, *reconstruction);
  if (triangulated_xyz.has_value()) {
    const MergePositionScore triangulated = ScoreMergedPointPosition(
        *reconstruction, target_point3d.track, source_point3d.track,
        *triangulated_xyz);
    if (IsBetterMergePosition(triangulated, best)) {
      best = triangulated;
    }
  }
  std::optional<TrackElement> outlier_to_remove;
  if (best.valid &&
      best.max_error_px > options.merge_max_reprojection_error_px &&
      observations.size() >= 4) {
    const std::vector<Eigen::Vector3d> robust_seeds = {
        WeightedMergedPointPosition(target_point3d, source_point3d),
        target_point3d.xyz, source_point3d.xyz, best.xyz};
    for (const Eigen::Vector3d& seed_xyz : robust_seeds) {
      std::vector<ObservationKey> inlier_observations;
      inlier_observations.reserve(observations.size() - 1);
      std::size_t num_target_inliers = 0;
      std::size_t num_source_inliers = 0;
      std::size_t num_outliers = 0;
      std::optional<TrackElement> seed_outlier;
      for (std::size_t index = 0; index < observations.size(); ++index) {
        const ObservationKey& observation = observations[index];
        const Image& image = reconstruction->GetImage(observation.first);
        const Camera& camera = reconstruction->GetCamera(image.camera_id);
        const std::optional<double> error =
            ReprojectionError(camera, *image.cam_from_world, seed_xyz,
                              image.points2d[observation.second].xy);
        if (!error.has_value() ||
            *error > options.merge_max_reprojection_error_px) {
          ++num_outliers;
          seed_outlier =
              TrackElement{observation.first, observation.second};
        } else {
          inlier_observations.push_back(observation);
          if (index < target_point3d.track.Length()) {
            ++num_target_inliers;
          } else {
            ++num_source_inliers;
          }
        }
      }

      if (num_outliers != 1 || num_target_inliers == 0 ||
          num_source_inliers == 0 || inlier_observations.size() < 3) {
        continue;
      }
      const std::optional<Eigen::Vector3d> robust_xyz =
          TriangulateObservationsLinear(inlier_observations,
                                        *reconstruction);
      if (robust_xyz.has_value()) {
        Track inlier_track;
        for (const ObservationKey& observation : inlier_observations) {
          inlier_track.AddElement(observation.first, observation.second);
        }
        const MergePositionScore robust_score = ScoreMergedPointPosition(
            *reconstruction, inlier_track, Track{}, *robust_xyz);
        if (robust_score.valid &&
            robust_score.max_error_px <=
                options.merge_max_reprojection_error_px) {
          best = robust_score;
          outlier_to_remove = seed_outlier;
          break;
        }
      }
    }
  }
  if (!best.valid ||
      best.max_error_px > options.merge_max_reprojection_error_px) {
    return false;
  }

  if (outlier_to_remove.has_value()) {
    const Image& image =
        reconstruction->GetImage(outlier_to_remove->image_id);
    const Point3DId owner_id =
        image.points2d[outlier_to_remove->point2d_idx].point3d_id;
    if ((owner_id != target_point3d_id && owner_id != source_point3d_id) ||
        !reconstruction->DeleteObservation(
            owner_id, outlier_to_remove->image_id,
            outlier_to_remove->point2d_idx)) {
      return false;
    }
  }

  ObservationManager observation_manager(reconstruction);
  return observation_manager.MergePoints(target_point3d_id, source_point3d_id,
                                         best.xyz);
}

bool TryCompleteObservation(const ImagePair& pair,
                            ImageId requested_image_id,
                            const FeatureMatch& match,
                            Image& image1,
                            Image& image2,
                            const IncrementalTriangulatorOptions& options,
                            Reconstruction* reconstruction) {
  const bool first_has_point = image1.points2d[match.idx1].HasPoint3D();
  const bool requested_is_first = pair.image_id1 == requested_image_id;
  const bool target_is_first = options.complete_neighbor_observations
                                   ? !first_has_point
                                   : requested_is_first;
  Image& target_image = target_is_first ? image1 : image2;
  const Image& source_image = target_is_first ? image2 : image1;
  const ImageId target_image_id = target_is_first ? pair.image_id1
                                                   : pair.image_id2;
  const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
  const Point2DIdx source_idx = target_is_first ? match.idx2 : match.idx1;

  Point2D& target_point2d = target_image.points2d[target_idx];
  const Point2D& source_point2d = source_image.points2d[source_idx];
  if (target_point2d.HasPoint3D() || !source_point2d.HasPoint3D() ||
      !reconstruction->ExistsPoint3D(source_point2d.point3d_id)) {
    return false;
  }

  const Camera& target_camera =
      reconstruction->GetCamera(target_image.camera_id);
  const Point3D& source_point3d =
      reconstruction->GetPoint3D(source_point2d.point3d_id);
  const auto error = ReprojectionError(target_camera,
                                       *target_image.cam_from_world,
                                       source_point3d.xyz,
                                       target_point2d.xy);
  if (!error.has_value() ||
      *error > options.complete_max_reprojection_error_px) {
    return false;
  }

  ObservationManager observation_manager(reconstruction);
  return observation_manager.AddObservation(source_point2d.point3d_id,
                                            target_image_id, target_idx);
}

bool TryMergeMatchedExistingPoints(Point3DId first_point_id,
                                   Point3DId second_point_id,
                                   const IncrementalTriangulatorOptions& options,
                                   Reconstruction* reconstruction,
                                   std::size_t* merged_observations) {
  const Point3DId kept_point3d_id =
      std::min(first_point_id, second_point_id);
  const Point3DId merged_point3d_id =
      std::max(first_point_id, second_point_id);
  if (kept_point3d_id == merged_point3d_id) return false;
  const std::size_t count =
      reconstruction->GetPoint3D(kept_point3d_id).track.Length() +
      reconstruction->GetPoint3D(merged_point3d_id).track.Length();
  if (!TryMergePoint3Ds(kept_point3d_id, merged_point3d_id, options,
                        reconstruction)) return false;
  *merged_observations = count;
  return true;
}

bool HasTrackUpdates(const IncrementalTriangulatorSummary& summary) {
  return summary.num_completed_observations > 0 ||
         summary.num_merged_points > 0 || summary.num_created_points > 0;
}

void AccumulateTrackUpdates(const IncrementalTriangulatorSummary& update,
                            IncrementalTriangulatorSummary* summary) {
  summary->num_created_points += update.num_created_points;
  summary->num_created_observations += update.num_created_observations;
  summary->num_completed_observations += update.num_completed_observations;
  summary->num_merged_points += update.num_merged_points;
  summary->num_merged_observations += update.num_merged_observations;
}

CorrespondenceAdjacency BuildCorrespondenceAdjacency(
    const MatchGraph& graph) {
  CorrespondenceAdjacency adjacency;
  for (const auto& [image_id, features] : graph.features) {
    adjacency.by_image[image_id].second = features.keypoints.size();
  }
  const auto for_each_edge = [&](auto action) {
    for (const auto& [pair, geometry] : graph.verified_geometries) {
      if (!IsUsableGeometry(geometry)) continue;
      for (const FeatureMatch& match : geometry.inlier_matches) {
        if (match.idx1 == kInvalidPoint2DIdx || match.idx2 == kInvalidPoint2DIdx) continue;
        action(ObservationKey{pair.image_id1, match.idx1},
               ObservationKey{pair.image_id2, match.idx2});
      }
    }
  };
  // Preserve the old graph's handling of matches without loaded features and
  // indices beyond their feature rows: retain nodes, validate geometry later.
  for_each_edge([&](const auto& first, const auto& second) {
    for (const auto& observation : {first, second}) {
      auto& count = adjacency.by_image[observation.first].second;
      count = std::max(count, static_cast<std::size_t>(observation.second) + 1);
    }
  });
  std::size_t count = 0;
  for (auto& [image_id, range] : adjacency.by_image) {
    range.first = count;
    count += range.second;
  }
  adjacency.observations.reserve(count);
  for (const auto& [image_id, range] : adjacency.by_image) {
    for (std::size_t row = 0; row < range.second; ++row) {
      adjacency.observations.emplace_back(image_id, static_cast<Point2DIdx>(row));
    }
  }
  adjacency.edge_offsets.resize(count + 1, 0);
  for_each_edge([&](const auto& first, const auto& second) {
    ++adjacency.edge_offsets[adjacency.Index(first) + 1];
    ++adjacency.edge_offsets[adjacency.Index(second) + 1];
  });
  std::partial_sum(adjacency.edge_offsets.begin(), adjacency.edge_offsets.end(),
                   adjacency.edge_offsets.begin());
  adjacency.edges.resize(adjacency.edge_offsets.back());
  auto next = adjacency.edge_offsets;
  for_each_edge([&](const auto& first, const auto& second) {
    const auto a = adjacency.Index(first), b = adjacency.Index(second);
    adjacency.edges[next[a]++] = b;
    adjacency.edges[next[b]++] = a;
  });
  return adjacency;
}

bool IsUsableObservation(const ObservationKey& observation,
                         const Reconstruction& reconstruction) {
  if (!reconstruction.IsImageRegistered(observation.first) ||
      !HasPoseAndCamera(reconstruction, observation.first)) {
    return false;
  }
  const Image& image = reconstruction.GetImage(observation.first);
  return observation.second < image.points2d.size();
}

std::vector<ObservationKey> CollectTransitiveCorrespondences(
    const ObservationKey& reference,
    const CorrespondenceAdjacency& adjacency,
    std::size_t max_transitivity,
    const Reconstruction& reconstruction,
    const std::unordered_map<ImageId, const Image*>& usable_images,
    bool reference_has_point3d) {
  std::vector<ObservationKey> correspondences;
  if (max_transitivity == 1) {
    // The default creation policy needs only direct neighbors. Preserve their
    // order and uniqueness without allocating a tree and two BFS frontiers
    // for every feature on every local retriangulation pass.
    const auto neighbors = adjacency.Find(reference);
    if (!neighbors) return correspondences;
    correspondences.reserve(neighbors->size());
    for (const auto& target : *neighbors) {
      const auto image_it = usable_images.find(target.first);
      if (image_it == usable_images.end() ||
          target.second >= image_it->second->points2d.size()) continue;
      // An assigned reference cannot be continued. Creation only consumes
      // unassigned neighbors, so omit assigned neighbors in this case.
      if (reference_has_point3d &&
          image_it->second->points2d[target.second].HasPoint3D()) continue;
      if (target != reference &&
          std::find(correspondences.begin(), correspondences.end(), target) ==
              correspondences.end()) {
        correspondences.push_back(target);
      }
    }
    if (reference_has_point3d && correspondences.size() < 2) {
      correspondences.clear();
    }
    return correspondences;
  }
  std::vector<ObservationKey> frontier{reference};
  std::set<ObservationKey> visited{reference};
  for (std::size_t depth = 0;
       depth < max_transitivity && !frontier.empty(); ++depth) {
    std::vector<ObservationKey> next_frontier;
    for (const ObservationKey& source : frontier) {
      const auto neighbors = adjacency.Find(source);
      if (!neighbors) {
        continue;
      }
      for (const ObservationKey& target : *neighbors) {
        if (!visited.insert(target).second) {
          continue;
        }
        next_frontier.push_back(target);
        if (IsUsableObservation(target, reconstruction)) {
          correspondences.push_back(target);
        }
      }
    }
    frontier = std::move(next_frontier);
  }
  return correspondences;
}

double AngularErrorDegrees(const ObservationKey& observation,
                           const Eigen::Vector3d& xyz,
                           const Reconstruction& reconstruction) {
  if (!IsUsableObservation(observation, reconstruction)) {
    return std::numeric_limits<double>::infinity();
  }
  const Image& image = reconstruction.GetImage(observation.first);
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  const Eigen::Vector3d point_camera =
      image.cam_from_world->TransformPoint(xyz);
  if (point_camera.z() <= 0.0 || !IsFinitePoint(point_camera)) {
    return std::numeric_limits<double>::infinity();
  }
  const auto bearing = TryPixelToBearing(camera, image.points2d[observation.second].xy);
  if (!bearing) return std::numeric_limits<double>::infinity();
  const double cosine = std::clamp(
      bearing->dot(point_camera.normalized()), -1.0, 1.0);
  constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
  return std::acos(cosine) * kRadiansToDegrees;
}

std::optional<Eigen::Vector3d> TriangulateObservationsLinear(
    const std::vector<ObservationKey>& observations,
    const Reconstruction& reconstruction) {
  if (observations.size() < 2) {
    return std::nullopt;
  }
  Eigen::MatrixXd equations(2 * observations.size(), 4);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const ObservationKey& observation = observations[index];
    const Image& image = reconstruction.GetImage(observation.first);
    const Camera& camera = reconstruction.GetCamera(image.camera_id);
    const auto bearing = TryPixelToBearing(camera, image.points2d[observation.second].xy);
    if (!bearing || std::abs(bearing->z()) < 1e-12) {
      return std::nullopt;
    }
    Eigen::Matrix<double, 3, 4> projection;
    projection.leftCols<3>() =
        image.cam_from_world->q_cw.toRotationMatrix();
    projection.rightCols<1>() = image.cam_from_world->t_cw;
    const double x = bearing->x() / bearing->z();
    const double y = bearing->y() / bearing->z();
    equations.row(2 * index) = x * projection.row(2) - projection.row(0);
    equations.row(2 * index + 1) =
        y * projection.row(2) - projection.row(1);
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(equations,
                                               Eigen::ComputeFullV);
  const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
  if (std::abs(homogeneous.w()) < 1e-12) {
    return std::nullopt;
  }
  const Eigen::Vector3d xyz = homogeneous.hnormalized();
  return IsFinitePoint(xyz) ? std::optional<Eigen::Vector3d>(xyz)
                            : std::nullopt;
}

struct RobustTrackEstimate {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  std::vector<bool> inlier_mask;
  std::size_t num_inliers = 0;
  double inlier_error_sum = std::numeric_limits<double>::infinity();
};

struct PreparedTriangulationObservation {
  bool valid = false;
  const Pose3* pose = nullptr;
  std::size_t image_slot = 0;
  Eigen::Vector3d bearing;
  Eigen::Vector3d world_ray;
  Eigen::Vector2d normalized;
  Eigen::Matrix<double, 3, 4> projection;
};

// Like COLMAP's EstimateTriangulation point/pose data, this is scoped to one
// estimation call. Camera calibration and poses cannot become stale after BA.
std::vector<PreparedTriangulationObservation> PrepareTriangulationObservations(
    const std::vector<ObservationKey>& observations,
    const Reconstruction& reconstruction) {
  std::vector<ImageId> image_ids;
  image_ids.reserve(observations.size());
  for (const auto& observation : observations) {
    image_ids.push_back(observation.first);
  }
  std::sort(image_ids.begin(), image_ids.end());
  image_ids.erase(std::unique(image_ids.begin(), image_ids.end()), image_ids.end());
  std::vector<PreparedTriangulationObservation> prepared;
  prepared.reserve(observations.size());
  for (const auto& observation : observations) {
    const Image& image = reconstruction.GetImage(observation.first);
    const Camera& camera = reconstruction.GetCamera(image.camera_id);
    PreparedTriangulationObservation data;
    data.pose = &*image.cam_from_world;
    data.image_slot = std::lower_bound(image_ids.begin(), image_ids.end(),
                                       observation.first) - image_ids.begin();
    const auto ray = TryUnprojectPixel(camera, image.points2d[observation.second].xy, 1.0);
    if (!ray) {
      prepared.push_back(data);
      continue;
    }
    data.valid = true;
    data.bearing = ray->normalized();
    data.world_ray = (data.pose->q_cw.conjugate() * data.bearing).normalized();
    data.normalized = {ray->x() / ray->z(), ray->y() / ray->z()};
    data.projection.leftCols<3>() = data.pose->q_cw.toRotationMatrix();
    data.projection.rightCols<1>() = data.pose->t_cw;
    prepared.push_back(data);
  }
  return prepared;
}

RobustTrackEstimate ScoreTrackHypothesis(
    const Eigen::Vector3d& xyz,
    const std::vector<ObservationKey>& observations,
    const std::vector<PreparedTriangulationObservation>& prepared,
    double max_angle_error_deg) {
  RobustTrackEstimate estimate;
  estimate.xyz = xyz;
  estimate.inlier_mask.assign(observations.size(), false);
  estimate.inlier_error_sum = 0.0;
  struct ImageInlier {
    std::size_t observation_index = std::numeric_limits<std::size_t>::max();
    double error = std::numeric_limits<double>::infinity();
  };
  // Reuse the sorted image slots across hypotheses. Duplicate-image matches
  // still compete geometrically, and summation retains ascending image order.
  std::vector<ImageInlier> best_by_image(observations.size());
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const auto& data = prepared[index];
    if (!data.valid) continue;
    const Eigen::Vector3d point_camera = data.pose->TransformPoint(xyz);
    double error = std::numeric_limits<double>::infinity();
    if (point_camera.z() > 0.0 && IsFinitePoint(point_camera)) {
      constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
      error = std::acos(std::clamp(
          data.bearing.dot(point_camera.normalized()), -1.0, 1.0)) *
          kRadiansToDegrees;
    }
    if (error > max_angle_error_deg) {
      continue;
    }
    auto& existing = best_by_image[data.image_slot];
    if (existing.observation_index == std::numeric_limits<std::size_t>::max() ||
        error < existing.error ||
        (error == existing.error &&
         observations[index] <
             observations[existing.observation_index])) {
      existing = {index, error};
    }
  }
  for (const auto& inlier : best_by_image) {
    if (inlier.observation_index == std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    estimate.inlier_mask[inlier.observation_index] = true;
    ++estimate.num_inliers;
    estimate.inlier_error_sum += inlier.error;
  }
  return estimate;
}

bool IsBetterTrackEstimate(const RobustTrackEstimate& candidate,
                           const RobustTrackEstimate& best) {
  return candidate.num_inliers > best.num_inliers ||
         (candidate.num_inliers == best.num_inliers &&
          candidate.inlier_error_sum < best.inlier_error_sum);
}

std::optional<RobustTrackEstimate> EstimateRobustTrack(
    const std::vector<ObservationKey>& observations,
    const IncrementalTriangulatorOptions& options,
    const Reconstruction& reconstruction) {
  if (observations.size() < 2) {
    return std::nullopt;
  }

  const auto prepared = PrepareTriangulationObservations(observations,
                                                       reconstruction);
  RobustTrackEstimate best;
  best.inlier_mask.assign(observations.size(), false);
  constexpr std::size_t kMaxPairHypotheses = 10000;
  std::size_t num_hypotheses = 0;
  for (std::size_t first = 0; first + 1 < observations.size(); ++first) {
    for (std::size_t second = first + 1; second < observations.size();
         ++second) {
      if (!prepared[first].valid || !prepared[second].valid) continue;
      if (++num_hypotheses > kMaxPairHypotheses) {
        break;
      }
      constexpr double kRadiansToDegrees = 180.0 / 3.14159265358979323846;
      const double angle = std::acos(std::clamp(
          prepared[first].world_ray.dot(prepared[second].world_ray),
          -1.0, 1.0)) * kRadiansToDegrees;
      if (angle <
          options.min_triangulation_angle_deg) {
        continue;
      }
      const std::optional<Eigen::Vector3d> xyz = TriangulatePointFromNormalized(
          prepared[first].projection, prepared[first].normalized,
          prepared[second].projection, prepared[second].normalized);
      if (!xyz || !IsFinitePoint(*xyz)) {
        continue;
      }
      const RobustTrackEstimate candidate = ScoreTrackHypothesis(
          *xyz, observations, prepared, options.create_max_angle_error_deg);
      if (candidate.num_inliers >= 2 &&
          IsBetterTrackEstimate(candidate, best)) {
        best = candidate;
      }
    }
    if (num_hypotheses > kMaxPairHypotheses) {
      break;
    }
  }
  if (best.num_inliers < 2) {
    return std::nullopt;
  }

  // COLMAP's LORANSAC locally re-estimates the model from all current
  // inliers. Two deterministic refinement rounds provide the same important
  // behavior without introducing another RANSAC framework dependency.
  for (int refinement = 0; refinement < 2; ++refinement) {
    std::vector<ObservationKey> inliers;
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (best.inlier_mask[index]) {
        inliers.push_back(observations[index]);
      }
    }
    const std::optional<Eigen::Vector3d> refined_xyz =
        TriangulateObservationsLinear(inliers, reconstruction);
    if (!refined_xyz) {
      break;
    }
    const RobustTrackEstimate refined = ScoreTrackHypothesis(
        *refined_xyz, observations, prepared, options.create_max_angle_error_deg);
    if (refined.num_inliers < 2 || !IsBetterTrackEstimate(refined, best)) {
      break;
    }
    best = refined;
  }
  return best;
}

struct CreateTrackSummary {
  std::size_t num_points = 0;
  std::size_t num_observations = 0;
};

CreateTrackSummary CreatePointsFromCorrespondences(
    const std::vector<ObservationKey>& correspondences,
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  std::vector<ObservationKey> candidates;
  candidates.reserve(correspondences.size());
  for (const ObservationKey& observation : correspondences) {
    if (!IsUsableObservation(observation, *reconstruction)) {
      continue;
    }
    const Image& image = reconstruction->GetImage(observation.first);
    if (!image.points2d[observation.second].HasPoint3D()) {
      candidates.push_back(observation);
    }
  }
  if (candidates.size() < 2 ||
      (options.ignore_two_view_tracks && candidates.size() == 2 &&
       adjacency.IsTwoViewObservation(candidates.front()))) {
    return {};
  }

  const std::optional<RobustTrackEstimate> estimate =
      EstimateRobustTrack(candidates, options, *reconstruction);
  if (!estimate) {
    return {};
  }
  Track track;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (estimate->inlier_mask[index]) {
      track.AddElement(candidates[index].first, candidates[index].second);
    }
  }
  ObservationManager observation_manager(reconstruction);
  if (!observation_manager.AddPoint3D(estimate->xyz, track).has_value()) {
    return {};
  }

  CreateTrackSummary summary{1, track.Length()};
  constexpr std::size_t kMinRecursiveTrackLength = 3;
  if (candidates.size() - track.Length() >= kMinRecursiveTrackLength) {
    const CreateTrackSummary recursive = CreatePointsFromCorrespondences(
        correspondences, adjacency, options, reconstruction);
    summary.num_points += recursive.num_points;
    summary.num_observations += recursive.num_observations;
  }
  return summary;
}

bool ContinueObservationAngular(
    const ObservationKey& reference,
    const std::vector<ObservationKey>& correspondences,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  if (!IsUsableObservation(reference, *reconstruction)) {
    return false;
  }
  Image& reference_image = reconstruction->GetImage(reference.first);
  if (reference_image.points2d[reference.second].HasPoint3D()) {
    return false;
  }

  Point3DId best_point3d_id = kInvalidPoint3DId;
  double best_error = std::numeric_limits<double>::infinity();
  std::set<Point3DId> tested_point_ids;
  for (const ObservationKey& correspondence : correspondences) {
    if (!IsUsableObservation(correspondence, *reconstruction)) {
      continue;
    }
    const Point2D& point2d = reconstruction->GetImage(correspondence.first)
                                 .points2d[correspondence.second];
    if (!point2d.HasPoint3D() ||
        !reconstruction->ExistsPoint3D(point2d.point3d_id) ||
        !tested_point_ids.insert(point2d.point3d_id).second) {
      continue;
    }
    const double error = AngularErrorDegrees(
        reference, reconstruction->GetPoint3D(point2d.point3d_id).xyz,
        *reconstruction);
    if (error < best_error ||
        (error == best_error && point2d.point3d_id < best_point3d_id)) {
      best_error = error;
      best_point3d_id = point2d.point3d_id;
    }
  }
  if (best_point3d_id == kInvalidPoint3DId ||
      best_error > options.continue_max_angle_error_deg) {
    return false;
  }
  ObservationManager observation_manager(reconstruction);
  return observation_manager.AddObservation(best_point3d_id, reference.first,
                                            reference.second);
}

struct CompletionTraversalScratch {
  explicit CompletionTraversalScratch(const CorrespondenceAdjacency& adjacency)
      : image_ranges(adjacency.by_image), visited(adjacency.observations.size(), 0U),
        observations(adjacency.observations), edge_offsets(adjacency.edge_offsets),
        edges(adjacency.edges) {}
  void NextPoint() {
    if (++epoch == 0U) {
      std::fill(visited.begin(), visited.end(), 0U);
      epoch = 1U;
    }
  }
  std::size_t Index(const ObservationKey& observation) const {
    const auto it = image_ranges.find(observation.first);
    if (it == image_ranges.end() || observation.second >= it->second.second) {
      return std::numeric_limits<std::size_t>::max();
    }
    return it->second.first + observation.second;
  }
  bool Visit(std::size_t index) {
    auto& value = visited[index];
    if (value == epoch) return false;
    value = epoch;
    return true;
  }
  const std::map<ImageId, std::pair<std::size_t, std::size_t>>& image_ranges;
  std::vector<std::uint32_t> visited;
  const std::vector<ObservationKey>& observations;
  const std::vector<std::size_t>& edge_offsets;
  const std::vector<std::size_t>& edges;
  std::uint32_t epoch = 0U;
};

std::size_t CompletePointTrack(
    Point3DId point3d_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    CompletionTraversalScratch* scratch) {
  if (!reconstruction->ExistsPoint3D(point3d_id)) {
    return 0;
  }
  const Eigen::Vector3d xyz = reconstruction->GetPoint3D(point3d_id).xyz;
  std::vector<std::size_t> frontier;
  scratch->NextPoint();
  for (const TrackElement& element :
       reconstruction->GetPoint3D(point3d_id).track.elements) {
    const ObservationKey observation{element.image_id, element.point2d_idx};
    const auto index = scratch->Index(observation);
    if (index != std::numeric_limits<std::size_t>::max()) {
      frontier.push_back(index);
      scratch->Visit(index);
    }
  }
  std::size_t num_completed = 0;
  ObservationManager observation_manager(reconstruction);
  for (std::size_t depth = 0;
       depth < options.complete_max_transitivity && !frontier.empty();
       ++depth) {
    std::vector<std::size_t> next_frontier;
    for (const auto source : frontier) {
      for (std::size_t edge = scratch->edge_offsets[source];
           edge < scratch->edge_offsets[source + 1]; ++edge) {
        const auto target_index = scratch->edges[edge];
        const auto& target = scratch->observations[target_index];
        if (!scratch->Visit(target_index) ||
            !reconstruction->IsImageRegistered(target.first) ||
            !HasPoseAndCamera(*reconstruction, target.first)) {
          continue;
        }
        Image& image = reconstruction->GetImage(target.first);
        if (target.second >= image.points2d.size()) {
          continue;
        }
        // Transitivity describes correspondence-graph traversal, not only a
        // chain of observations that can immediately be attached to this
        // point.  An intermediate feature may already belong to another point
        // or conflict with an image already present in the track, while a
        // later feature in the same verified component is still a valid new
        // observation.  Keep traversing such nodes; geometry and one-image-
        // one-observation checks below still guard every persisted update.
        next_frontier.push_back(target_index);
        Point2D& point2d = image.points2d[target.second];
        if (point2d.HasPoint3D()) {
          continue;
        }
        const Camera& camera = reconstruction->GetCamera(image.camera_id);
        const std::optional<double> error = ReprojectionError(
            camera, *image.cam_from_world, xyz, point2d.xy);
        if (!error.has_value() ||
            *error > options.complete_max_reprojection_error_px) {
          continue;
        }
        if (observation_manager.AddObservation(point3d_id, target.first,
                                               target.second)) {
          ++num_completed;
        }
      }
    }
    frontier = std::move(next_frontier);
  }
  return num_completed;
}

std::size_t CompleteUnassignedObservationsFromComponents(
    const CorrespondenceAdjacency& adjacency,
    const CompletionTraversalScratch& traversal,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    std::set<Point3DId>* dirty_point_ids) {
  struct CompletionProposal {
    ObservationKey target{kInvalidImageId, kInvalidPoint2DIdx};
    std::array<Point3DId, 4> point3d_ids{};
    std::array<double, 4> errors{};
    std::size_t num_candidates = 0;
  };

  const auto total_observations = traversal.observations.size();
  const auto flat_observation_index =
      [&traversal](const ObservationKey& observation) {
        return traversal.Index(observation);
      };

  // Share the immutable CSR graph and snapshot owners once, avoiding image-map
  // lookups and linear point deduplication on every BFS edge.
  const std::size_t no_owner = std::numeric_limits<std::size_t>::max();
  std::vector<Point3DId> point_ids;
  std::unordered_map<Point3DId, std::size_t> point_indices;
  point_ids.reserve(reconstruction->Points3D().size());
  point_indices.reserve(reconstruction->Points3D().size());
  for (const auto& [id, point] : reconstruction->Points3D()) {
    (void)point;
    point_indices.emplace(id, point_ids.size());
    point_ids.push_back(id);
  }
  std::vector<std::size_t> owners(total_observations, no_owner);
  const auto& edge_offsets = traversal.edge_offsets;
  const auto& edges = traversal.edges;
  for (const auto& [image_id, image_range] : adjacency.by_image) {
    const Image* image = reconstruction->ExistsImage(image_id)
        ? &reconstruction->GetImage(image_id) : nullptr;
    const std::size_t offset = traversal.image_ranges.at(image_id).first;
    for (std::size_t row = 0; row < image_range.second; ++row) {
      if (image != nullptr && row < image->points2d.size()) {
        const auto it = point_indices.find(image->points2d[row].point3d_id);
        if (it != point_indices.end()) owners[offset + row] = it->second;
      }
    }
  }

  std::vector<ObservationKey> targets;
  targets.reserve(total_observations / 2);
  for (const auto& [image_id, image_range] : adjacency.by_image) {
    if (!reconstruction->IsImageRegistered(image_id) ||
        !HasPoseAndCamera(*reconstruction, image_id)) {
      continue;
    }
    Image& target_image = reconstruction->GetImage(image_id);
    const std::size_t num_features =
        std::min(image_range.second, target_image.points2d.size());
    for (Point2DIdx point2d_idx = 0; point2d_idx < num_features;
         ++point2d_idx) {
      const auto index = image_range.first + point2d_idx;
      if (edge_offsets[index] == edge_offsets[index + 1] ||
          target_image.points2d[point2d_idx].HasPoint3D()) {
        continue;
      }
      targets.emplace_back(image_id, point2d_idx);
    }
  }

  std::vector<CompletionProposal> proposals(targets.size());
  const Reconstruction& read_only_reconstruction = *reconstruction;
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(targets.size())),
      [&](const parallel::Range& range) {
        std::vector<std::size_t> frontier;
        std::vector<std::size_t> next_frontier;
        std::vector<Point3DId> candidate_point_ids;
        frontier.reserve(64);
        next_frontier.reserve(64);
        candidate_point_ids.reserve(16);
        std::vector<std::uint32_t> visited_epochs(total_observations, 0U);
        std::vector<std::uint32_t> candidate_epochs(point_ids.size(), 0U);
        std::uint32_t current_epoch = 0U;
        for (int target_index = range.start; target_index < range.end;
             ++target_index) {
          const ObservationKey target = targets[target_index];
          CompletionProposal& proposal = proposals[target_index];
          proposal.target = target;
          const Image& target_image =
              read_only_reconstruction.GetImage(target.first);
          const Camera& target_camera =
              read_only_reconstruction.GetCamera(target_image.camera_id);

          frontier.clear();
          next_frontier.clear();
          candidate_point_ids.clear();
          if (++current_epoch == 0U) {
            std::fill(visited_epochs.begin(), visited_epochs.end(), 0U);
            std::fill(candidate_epochs.begin(), candidate_epochs.end(), 0U);
            current_epoch = 1U;
          }
          frontier.push_back(flat_observation_index(target));
          visited_epochs[flat_observation_index(target)] = current_epoch;
          const std::size_t parallel_component_transitivity =
              options.complete_max_transitivity;
          const auto score_candidate_points = [&]() {
            proposal.num_candidates = 0;
            for (const Point3DId candidate_id : candidate_point_ids) {
              const Point3D& candidate =
                  read_only_reconstruction.GetPoint3D(candidate_id);
              const bool already_observed_in_target =
                  std::any_of(candidate.track.elements.begin(),
                              candidate.track.elements.end(),
                              [target](const TrackElement& element) {
                                return element.image_id == target.first;
                              });
              if (already_observed_in_target) {
                continue;
              }
              const std::optional<double> error = ReprojectionError(
                  target_camera, *target_image.cam_from_world, candidate.xyz,
                  target_image.points2d[target.second].xy);
              if (!error.has_value() ||
                  *error > options.complete_max_reprojection_error_px) {
                continue;
              }
              std::size_t insert_index = proposal.num_candidates;
              while (
                  insert_index > 0 &&
                  (proposal.errors[insert_index - 1] > *error ||
                   (proposal.errors[insert_index - 1] == *error &&
                    proposal.point3d_ids[insert_index - 1] > candidate_id))) {
                if (insert_index < proposal.point3d_ids.size()) {
                  proposal.point3d_ids[insert_index] =
                      proposal.point3d_ids[insert_index - 1];
                  proposal.errors[insert_index] =
                      proposal.errors[insert_index - 1];
                }
                --insert_index;
              }
              if (insert_index < proposal.point3d_ids.size()) {
                proposal.point3d_ids[insert_index] = candidate_id;
                proposal.errors[insert_index] = *error;
              }
              proposal.num_candidates = std::min(
                  proposal.num_candidates + 1,
                  proposal.point3d_ids.size());
            }
          };
          bool candidates_scored = false;
          for (std::size_t depth = 0;
               depth < parallel_component_transitivity && !frontier.empty();
               ++depth) {
            next_frontier.clear();
            for (const std::size_t source : frontier) {
              for (std::size_t edge = edge_offsets[source];
                   edge < edge_offsets[source + 1]; ++edge) {
                const std::size_t neighbor_flat_index = edges[edge];
                if (visited_epochs[neighbor_flat_index] == current_epoch) {
                  continue;
                }
                visited_epochs[neighbor_flat_index] = current_epoch;
                next_frontier.push_back(neighbor_flat_index);
                const std::size_t owner = owners[neighbor_flat_index];
                if (owner != no_owner && candidate_epochs[owner] != current_epoch) {
                  candidate_epochs[owner] = current_epoch;
                  candidate_point_ids.push_back(point_ids[owner]);
                }
              }
            }
            frontier.swap(next_frontier);
            // Four hops cover almost every useful House proposal. Only pay
            // for deeper layers when the first four layers
            // contain no geometrically attachable point, avoiding a full
            // giant-component expansion for already solved targets.
            if (depth + 1 == 4 &&
                parallel_component_transitivity > 4) {
              score_candidate_points();
              candidates_scored = proposal.num_candidates > 0;
              if (candidates_scored) {
                break;
              }
              for (const auto id : candidate_point_ids) {
                candidate_epochs[point_indices.at(id)] = 0U;
              }
              candidate_point_ids.clear();
            }
          }
          if (!candidates_scored) {
            score_candidate_points();
          }
        }
      },
      static_cast<double>(std::max(1, parallel::getNumThreads())));

  std::vector<std::size_t> proposal_order;
  proposal_order.reserve(proposals.size());
  for (std::size_t index = 0; index < proposals.size(); ++index) {
    if (proposals[index].num_candidates > 0) {
      proposal_order.push_back(index);
    }
  }
  std::stable_sort(
      proposal_order.begin(), proposal_order.end(),
      [&proposals](std::size_t first, std::size_t second) {
        const CompletionProposal& lhs = proposals[first];
        const CompletionProposal& rhs = proposals[second];
        return lhs.errors[0] < rhs.errors[0] ||
               (lhs.errors[0] == rhs.errors[0] && lhs.target < rhs.target);
      });

  std::size_t num_completed = 0;
  ObservationManager observation_manager(reconstruction);
  for (const std::size_t proposal_index : proposal_order) {
    const CompletionProposal& proposal = proposals[proposal_index];
    Image& target_image = reconstruction->GetImage(proposal.target.first);
    if (target_image.points2d[proposal.target.second].HasPoint3D()) {
      continue;
    }
    for (std::size_t candidate_index = 0;
         candidate_index < proposal.num_candidates; ++candidate_index) {
      const Point3DId candidate_id = proposal.point3d_ids[candidate_index];
      if (!reconstruction->ExistsPoint3D(candidate_id)) {
        continue;
      }
      const Point3D& candidate = reconstruction->GetPoint3D(candidate_id);
      const bool already_observed_in_target =
          std::any_of(candidate.track.elements.begin(),
                      candidate.track.elements.end(),
                      [&proposal](const TrackElement& element) {
                        return element.image_id == proposal.target.first;
                      });
      if (already_observed_in_target) {
        continue;
      }
      if (observation_manager.AddObservation(
              candidate_id, proposal.target.first, proposal.target.second)) {
        ++num_completed;
        dirty_point_ids->insert(candidate_id);
        break;
      }
    }
  }
  return num_completed;
}

Point3DId MergePointTrack(
    Point3DId point3d_id,
    const CorrespondenceAdjacency& adjacency,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    IncrementalTriangulatorSummary* summary,
    MergeTrialCache* trials) {
  while (reconstruction->ExistsPoint3D(point3d_id)) {
    std::set<Point3DId> candidates;
    const Track track = reconstruction->GetPoint3D(point3d_id).track;
    for (const TrackElement& element : track.elements) {
      const auto neighbors =
          adjacency.Find({element.image_id, element.point2d_idx});
      if (!neighbors) {
        continue;
      }
      for (const ObservationKey& observation : *neighbors) {
        if (!reconstruction->ExistsImage(observation.first)) {
          continue;
        }
        const Image& image = reconstruction->GetImage(observation.first);
        if (observation.second >= image.points2d.size()) {
          continue;
        }
        const Point2D& point2d = image.points2d[observation.second];
        if (point2d.HasPoint3D() && point2d.point3d_id != point3d_id &&
            reconstruction->ExistsPoint3D(point2d.point3d_id)) {
          candidates.insert(point2d.point3d_id);
        }
      }
    }

    bool merged = false;
    for (const Point3DId candidate_id : candidates) {
      if (!reconstruction->ExistsPoint3D(candidate_id)) {
        continue;
      }
      const Point3DId target_id = std::min(point3d_id, candidate_id);
      const Point3DId source_id = std::max(point3d_id, candidate_id);
      if (trials->Contains(target_id, source_id, *reconstruction)) continue;
      const auto version_before = reconstruction->Point3DChangeVersion();
      const std::size_t merged_observations =
          reconstruction->GetPoint3D(target_id).track.Length() +
          reconstruction->GetPoint3D(source_id).track.Length();
      if (TryMergePoint3Ds(target_id, source_id, options,
                           reconstruction)) {
        point3d_id = target_id;
        ++summary->num_merged_points;
        summary->num_merged_observations += merged_observations;
        merged = true;
        break;
      }
      trials->Remember(target_id, source_id, version_before, *reconstruction);
    }
    if (!merged) {
      break;
    }
  }
  return reconstruction->ExistsPoint3D(point3d_id) ? point3d_id
                                                    : kInvalidPoint3DId;
}

}  // namespace

IncrementalTriangulatorCache::IncrementalTriangulatorCache(
    const MatchGraph& graph)
    : correspondence_adjacency(BuildCorrespondenceAdjacency(graph)) {}

IncrementalTriangulatorSummary TriangulateRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;
  const auto create_start = std::chrono::steady_clock::now();

  if (reconstruction == nullptr || !reconstruction->IsImageRegistered(image_id) ||
      !HasPoseAndCamera(*reconstruction, image_id)) {
    summary.message = "Image is not registered or has no pose";
    return summary;
  }

  const CorrespondenceAdjacency local_adjacency =
      cache == nullptr ? BuildCorrespondenceAdjacency(graph)
                       : CorrespondenceAdjacency{};
  const CorrespondenceAdjacency& adjacency =
      cache == nullptr ? local_adjacency : cache->correspondence_adjacency;
  const Image& reference_image = reconstruction->GetImage(image_id);
  // Registration and camera availability stay fixed during triangulation.
  // Cache these checks once instead of repeating several model lookups for
  // every direct correspondence on every local-neighbor pass.
  std::unordered_map<ImageId, const Image*> usable_images;
  usable_images.reserve(reconstruction->RegisteredImageIds().size());
  for (const ImageId id : reconstruction->RegisteredImageIds()) {
    if (HasPoseAndCamera(*reconstruction, id)) {
      usable_images.emplace(id, &reconstruction->GetImage(id));
    }
  }
  for (Point2DIdx point2d_idx = 0;
       point2d_idx < reference_image.points2d.size(); ++point2d_idx) {
    const ObservationKey reference{image_id, point2d_idx};
    std::vector<ObservationKey> correspondences =
        CollectTransitiveCorrespondences(reference, adjacency,
                                         options.max_transitivity,
                                         *reconstruction, usable_images,
                                         reference_image.points2d[point2d_idx].HasPoint3D());
    if (correspondences.empty()) {
      continue;
    }

    bool has_triangulated_correspondence = false;
    for (const ObservationKey& correspondence : correspondences) {
      const Point2D& point2d = reconstruction->GetImage(correspondence.first)
                                   .points2d[correspondence.second];
      if (point2d.HasPoint3D()) {
        has_triangulated_correspondence = true;
        break;
      }
    }
    if (has_triangulated_correspondence &&
        ContinueObservationAngular(reference, correspondences, options,
                                   reconstruction)) {
      ++summary.num_completed_observations;
    }

    correspondences.push_back(reference);
    const CreateTrackSummary created = CreatePointsFromCorrespondences(
        correspondences, adjacency, options, reconstruction);
    summary.num_created_points += created.num_points;
    summary.num_created_observations += created.num_observations;
  }

  const auto update_start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 1;
       summary.num_created_points > 0 &&
       iteration < options.max_track_update_iterations;
       ++iteration) {
    const IncrementalTriangulatorSummary completion =
        CompleteTracksForRegisteredImage(graph, image_id, options,
                                         reconstruction);
    if (!HasTrackUpdates(completion)) {
      break;
    }
    AccumulateTrackUpdates(completion, &summary);
  }

  VLOG(1) << "Triangulation timing: image=" << image_id
          << " create_seconds=" << std::chrono::duration<double>(
                 update_start - create_start).count()
          << " update_seconds=" << std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - update_start).count()
          << " created=" << summary.num_created_points;
  summary.message = "Triangulation finished";
  return summary;
}

IncrementalTriangulatorSummary CompleteTracksForRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;
  const auto start = std::chrono::steady_clock::now();
  std::size_t num_matches = 0;
  std::size_t num_merge_attempts = 0;
  std::size_t num_cached_failures = 0;

  if (reconstruction == nullptr || !reconstruction->IsImageRegistered(image_id) ||
      !HasPoseAndCamera(*reconstruction, image_id)) {
    summary.message = "Image is not registered or has no pose";
    return summary;
  }

  // Only successful updates suppress later candidates, as in the original
  // traversal. A failed merge may change a track, and must remain retryable.
  std::vector<bool> updated_observations(
      reconstruction->GetImage(image_id).points2d.size(), false);
  // COLMAP remembers merge trials within an operation. Unlike COLMAP, our
  // merge retains an existing target ID, so invalidate on ANY point mutation.
  // This cache never survives this call, BA, or camera/pose updates.
  std::unordered_set<std::pair<Point3DId, Point3DId>, PointPairHash> failed_merges;
  auto failure_version = reconstruction->Point3DChangeVersion();
  constexpr std::size_t kMaxCachedFailures = 4096;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry) ||
        !PassesPairAngleLayer(geometry, options)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction->IsImageRegistered(neighbor_id) ||
        !HasPoseAndCamera(*reconstruction, neighbor_id)) {
      continue;
    }

    num_matches += geometry.inlier_matches.size();
    Image& image1 = reconstruction->GetImage(pair.image_id1);
    Image& image2 = reconstruction->GetImage(pair.image_id2);
    for (const FeatureMatch& match : geometry.inlier_matches) {
      if (!HasValidMatchIndices(match, image1, image2)) continue;
      const Point2DIdx target_idx =
          pair.image_id1 == image_id ? match.idx1 : match.idx2;
      if (updated_observations[target_idx]) continue;
      const Point2D& first = image1.points2d[match.idx1];
      const Point2D& second = image2.points2d[match.idx2];
      const bool first_has_point = first.HasPoint3D();
      const bool second_has_point = second.HasPoint3D();

      std::size_t merged_observations = 0;
      if (first_has_point && second_has_point) {
        if (first.point3d_id == second.point3d_id) continue;
        const auto current_version = reconstruction->Point3DChangeVersion();
        if (failure_version != current_version) {
          failed_merges.clear();
          failure_version = current_version;
        }
        const auto pair_ids = std::minmax(first.point3d_id, second.point3d_id);
        const std::pair<Point3DId, Point3DId> key{pair_ids.first, pair_ids.second};
        if (failed_merges.count(key) != 0) {
          ++num_cached_failures;
          continue;
        }
        ++num_merge_attempts;
        if (TryMergeMatchedExistingPoints(first.point3d_id, second.point3d_id,
                options, reconstruction, &merged_observations)) {
          updated_observations[target_idx] = true;
          ++summary.num_merged_points;
          summary.num_merged_observations += merged_observations;
        } else if (failure_version == reconstruction->Point3DChangeVersion() &&
                   failed_merges.size() < kMaxCachedFailures) {
          // A failed attempt is reusable only when it made no partial edit.
          failed_merges.insert(key);
        }
      } else if (options.complete_max_transitivity > 0 &&
                 first_has_point != second_has_point &&
                 TryCompleteObservation(pair, image_id, match, image1, image2,
                                        options, reconstruction)) {
        updated_observations[target_idx] = true;
        ++summary.num_completed_observations;
      }
    }
  }

  VLOG(1) << "Track update timing: image=" << image_id
          << " matches=" << num_matches
          << " seconds=" << std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - start).count()
          << " completed=" << summary.num_completed_observations
          << " merged=" << summary.num_merged_points
          << " merge_attempts=" << num_merge_attempts
          << " cached_failures=" << num_cached_failures;
  summary.message = "Track completion finished";
  return summary;
}

IncrementalTriangulatorSummary CompleteAndMergeAllTracks(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache) {
  IncrementalTriangulatorSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }
  const CorrespondenceAdjacency local_adjacency =
      cache == nullptr ? BuildCorrespondenceAdjacency(graph)
                       : CorrespondenceAdjacency{};
  const CorrespondenceAdjacency& adjacency =
      cache == nullptr ? local_adjacency : cache->correspondence_adjacency;
  std::vector<Point3DId> active_point_ids;
  double completion_seconds = 0.0, merge_seconds = 0.0;
  MergeTrialCache merge_trials;
  CompletionTraversalScratch completion_scratch(adjacency);
  active_point_ids.reserve(reconstruction->Points3D().size());
  for (const auto& [point3d_id, point3d] : reconstruction->Points3D()) {
    (void)point3d;
    active_point_ids.push_back(point3d_id);
  }
  for (std::size_t iteration = 0;
       iteration < options.max_track_update_iterations &&
       !active_point_ids.empty();
       ++iteration) {
    const std::size_t updates_before =
        summary.num_completed_observations + summary.num_merged_points;
    std::set<Point3DId> dirty_point_ids;
    const auto completion_start = std::chrono::steady_clock::now();
    if (iteration == 0) {
      summary.num_completed_observations +=
          CompleteUnassignedObservationsFromComponents(
              adjacency, completion_scratch, options, reconstruction, &dirty_point_ids);
    } else {
      for (const Point3DId point3d_id : active_point_ids) {
        const std::size_t completed = CompletePointTrack(
            point3d_id, options, reconstruction, &completion_scratch);
        summary.num_completed_observations += completed;
        if (completed > 0 && reconstruction->ExistsPoint3D(point3d_id)) {
          dirty_point_ids.insert(point3d_id);
        }
      }
    }

    const double iteration_completion_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - completion_start).count();
    completion_seconds += iteration_completion_seconds;
    const auto merge_start = std::chrono::steady_clock::now();
    for (const Point3DId point3d_id : active_point_ids) {
      if (reconstruction->ExistsPoint3D(point3d_id)) {
        const std::size_t merged_before = summary.num_merged_points;
        const Point3DId merged_target = MergePointTrack(
            point3d_id, adjacency, options, reconstruction, &summary,
            &merge_trials);
        if (summary.num_merged_points > merged_before &&
            merged_target != kInvalidPoint3DId) {
          dirty_point_ids.insert(merged_target);
        }
      }
    }
    merge_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - merge_start).count();
    VLOG(1) << "Global track iteration: iteration=" << iteration
            << " active_points=" << active_point_ids.size()
            << " completion_seconds=" << iteration_completion_seconds;
    const std::size_t updates_after =
        summary.num_completed_observations + summary.num_merged_points;
    if (updates_after == updates_before) {
      break;
    }
    active_point_ids.assign(dirty_point_ids.begin(), dirty_point_ids.end());
  }
  VLOG(1) << "Global track timing: completion_seconds=" << completion_seconds
          << " merge_seconds=" << merge_seconds
          << " completed=" << summary.num_completed_observations
          << " merged=" << summary.num_merged_points;
  summary.message = "Global track completion and merging finished";
  return summary;
}

IncrementalTriangulatorSummary CompleteTracksByPoseAndDescriptors(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction) {
  IncrementalTriangulatorSummary summary;
  if (reconstruction == nullptr ||
      options.pose_guided_min_descriptor_support < 2 ||
      options.pose_guided_max_reprojection_error_px <= 0.0 ||
      options.pose_guided_max_descriptor_angle_rad <= 0.0) {
    summary.message = "Pose-guided track completion is disabled";
    return summary;
  }

  struct Candidate {
    ImageId image_id = kInvalidImageId;
    Point2DIdx point2d_idx = kInvalidPoint2DIdx;
    Point3DId point3d_id = kInvalidPoint3DId;
    double second_similarity = -1.0;
    double best_similarity = -1.0;
    double reprojection_error_px = std::numeric_limits<double>::infinity();
    std::size_t track_length = 0;
  };

  const std::vector<ImageId> image_ids(
      reconstruction->RegisteredImageIds().begin(),
      reconstruction->RegisteredImageIds().end());
  std::vector<std::vector<Candidate>> candidates_by_image(image_ids.size());
  const Reconstruction& read_only_reconstruction = *reconstruction;
  const double cell_size = options.pose_guided_max_reprojection_error_px;
  const double min_similarity =
      std::cos(options.pose_guided_max_descriptor_angle_rad);
  const auto geometry_start = std::chrono::steady_clock::now();
  parallel::parallel_for_(
      parallel::Range(0, static_cast<int>(image_ids.size())),
      [&](const parallel::Range& range) {
        for (int image_index = range.start; image_index < range.end;
             ++image_index) {
          const ImageId image_id = image_ids[image_index];
          const Image& image = read_only_reconstruction.GetImage(image_id);
          const Camera& camera =
              read_only_reconstruction.GetCamera(image.camera_id);
          const FeatureSet* target_features = FindFeatures(graph, image_id);
          if (target_features == nullptr ||
              target_features->DescriptorCols() != 128) {
            continue;
          }
          const std::size_t num_features = std::min(
              {image.points2d.size(), target_features->keypoints.size(),
               static_cast<std::size_t>(target_features->DescriptorRows())});
          std::map<std::pair<int, int>, std::vector<Point2DIdx>> grid;
          for (Point2DIdx point2d_idx = 0; point2d_idx < num_features;
               ++point2d_idx) {
            if (image.points2d[point2d_idx].HasPoint3D()) {
              continue;
            }
            const FeatureKeypoint& keypoint =
                target_features->keypoints[point2d_idx];
            grid[{static_cast<int>(std::floor(keypoint.x / cell_size)),
                  static_cast<int>(std::floor(keypoint.y / cell_size))}]
                .push_back(point2d_idx);
          }

          std::vector<Candidate>& image_candidates =
              candidates_by_image[static_cast<std::size_t>(image_index)];
          for (const auto& [point3d_id, point3d] :
               read_only_reconstruction.Points3D()) {
            if (point3d.track.Length() <
                options.pose_guided_min_descriptor_support) {
              continue;
            }
            const bool already_observed = std::any_of(
                point3d.track.elements.begin(), point3d.track.elements.end(),
                [image_id](const TrackElement& element) {
                  return element.image_id == image_id;
                });
            if (already_observed) {
              continue;
            }
            const std::optional<Eigen::Vector2d> projected =
                ProjectWorldPoint(camera, *image.cam_from_world, point3d.xyz);
            if (!projected.has_value()) {
              continue;
            }
            const int cell_x =
                static_cast<int>(std::floor(projected->x() / cell_size));
            const int cell_y =
                static_cast<int>(std::floor(projected->y() / cell_size));
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                const auto grid_it = grid.find({cell_x + dx, cell_y + dy});
                if (grid_it == grid.end()) {
                  continue;
                }
                for (const Point2DIdx target_idx : grid_it->second) {
                  const Eigen::Vector2d target_xy(
                      target_features->keypoints[target_idx].x,
                      target_features->keypoints[target_idx].y);
                  const double reprojection_error =
                      (*projected - target_xy).norm();
                  if (reprojection_error >
                      options.pose_guided_max_reprojection_error_px) {
                    continue;
                  }
                  image_candidates.push_back(
                      {image_id, target_idx, point3d_id, -1.0,
                       -1.0, reprojection_error,
                       point3d.track.Length()});
                }
              }
            }
          }
        }
      });

  std::vector<Candidate> candidates;
  for (std::vector<Candidate>& image_candidates : candidates_by_image) {
    candidates.insert(candidates.end(), image_candidates.begin(),
                      image_candidates.end());
    std::vector<Candidate>().swap(image_candidates);
  }
  // Geometric screening precedes descriptor access. Bound each batch by the
  // number of unique referenced rows, so deferred sources do
  // not turn into a second scene-sized descriptor cache during completion.
  // A single track larger than the budget is processed whole.
  struct LoadedRows {
    std::vector<Point2DIdx> indices;
    std::unordered_set<Point2DIdx> requested;
    DescriptorMatrix values;
    std::vector<double> norms;
    Eigen::Index Find(Point2DIdx index) const {
      const auto it = std::lower_bound(indices.begin(), indices.end(), index);
      return it == indices.end() || *it != index ? Eigen::Index{-1}
          : static_cast<Eigen::Index>(it - indices.begin());
    }
  };
  constexpr std::size_t kMaxReferencedRows = 128 * 1024;  // 64 MiB at 128 floats.
  const double geometry_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - geometry_start).count();
  double request_seconds = 0.0, read_seconds = 0.0, score_seconds = 0.0;
  std::size_t num_batches = 0, num_image_reads = 0, num_loaded_rows = 0;
  std::vector<Candidate> accepted;
  for (std::size_t begin = 0; begin < candidates.size();) {
    const auto request_start = std::chrono::steady_clock::now();
    std::size_t end = begin, references = 0;
    std::map<ImageId, LoadedRows> loaded;
    std::unordered_set<Point3DId> loaded_points;
    while (end < candidates.size()) {
      const Candidate& candidate = candidates[end];
      const Point3D& point = read_only_reconstruction.GetPoint3D(candidate.point3d_id);
      std::vector<ObservationKey> requests;
      requests.emplace_back(candidate.image_id, candidate.point2d_idx);
      // Track rows recur for many target candidates. Gather them once per
      // batch instead of repeatedly consuming the row budget with duplicates.
      if (!loaded_points.count(candidate.point3d_id)) {
        for (const TrackElement& element : point.track.elements) {
          const FeatureSet* features = FindFeatures(graph, element.image_id);
          if (features != nullptr && features->DescriptorCols() == 128 &&
              read_only_reconstruction.IsImageRegistered(element.image_id) &&
              element.point2d_idx < static_cast<std::uint64_t>(features->DescriptorRows())) {
            requests.emplace_back(element.image_id, element.point2d_idx);
          }
        }
      }
      std::size_t required = 0;
      for (const auto& [image_id, row] : requests) {
        const auto it = loaded.find(image_id);
        if (it == loaded.end() || !it->second.requested.count(row)) ++required;
      }
      if (end != begin && required > kMaxReferencedRows -
          std::min(references, kMaxReferencedRows)) break;
      references += required;
      for (const auto& [image_id, row] : requests) {
        auto& rows = loaded[image_id];
        if (rows.requested.insert(row).second) {
          rows.indices.push_back(row);
        }
      }
      loaded_points.insert(candidate.point3d_id);
      ++end;
    }
    request_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - request_start).count();
    const auto read_start = std::chrono::steady_clock::now();
    ++num_batches;
    std::vector<std::pair<ImageId, LoadedRows*>> image_reads;
    image_reads.reserve(loaded.size());
    for (auto& [image_id, rows] : loaded) {
      std::sort(rows.indices.begin(), rows.indices.end());
      rows.indices.erase(std::unique(rows.indices.begin(), rows.indices.end()), rows.indices.end());
      std::unordered_set<Point2DIdx>().swap(rows.requested);
      ++num_image_reads;
      num_loaded_rows += rows.indices.size();
      image_reads.emplace_back(image_id, &rows);
    }
    // Different images have independent immutable sources. Overlap file
    // validation/read latency without retaining descriptors beyond this batch.
    parallel::parallel_for_(parallel::Range(0, static_cast<int>(image_reads.size())),
        [&](const parallel::Range& range) {
      for (int index = range.start; index < range.end; ++index) {
        auto& [image_id, rows] = image_reads[static_cast<std::size_t>(index)];
        rows->values = FindFeatures(graph, image_id)->ReadDescriptorRows(rows->indices);
        rows->norms.resize(rows->indices.size());
        for (Eigen::Index row = 0; row < rows->values.rows(); ++row) {
          rows->norms[static_cast<std::size_t>(row)] = rows->values.row(row).norm();
        }
      }
    });
    read_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - read_start).count();
    const auto score_start = std::chrono::steady_clock::now();
    const auto& read_only_rows = loaded;
    std::vector<std::uint8_t> keep(end - begin, 0);
    parallel::parallel_for_(parallel::Range(0, static_cast<int>(end - begin)),
        [&](const parallel::Range& range) {
      for (int offset = range.start; offset < range.end; ++offset) {
        Candidate& candidate = candidates[begin + static_cast<std::size_t>(offset)];
        const LoadedRows& target = read_only_rows.at(candidate.image_id);
        const Eigen::Index target_row = target.Find(candidate.point2d_idx);
        const double target_norm = target.norms[static_cast<std::size_t>(target_row)];
        if (!(target_norm > 0.0)) continue;
        std::size_t support = 0;
        for (const TrackElement& element : read_only_reconstruction.GetPoint3D(
                 candidate.point3d_id).track.elements) {
          const auto source_it = read_only_rows.find(element.image_id);
          if (source_it == read_only_rows.end()) continue;
          const LoadedRows& source = source_it->second;
          const Eigen::Index source_row = source.Find(element.point2d_idx);
          if (source_row < 0) continue;
          const double source_norm = source.norms[static_cast<std::size_t>(source_row)];
          if (!(source_norm > 0.0)) continue;
          const double similarity = std::clamp(
              static_cast<double>(target.values.row(target_row).dot(source.values.row(source_row))) /
                  (target_norm * source_norm), -1.0, 1.0);
          ++support;
          if (similarity > candidate.best_similarity) {
            candidate.second_similarity = candidate.best_similarity;
            candidate.best_similarity = similarity;
          } else if (similarity > candidate.second_similarity) {
            candidate.second_similarity = similarity;
          }
        }
        keep[static_cast<std::size_t>(offset)] =
            support >= options.pose_guided_min_descriptor_support &&
            !(candidate.second_similarity < min_similarity);
      }
    });
    for (std::size_t i = begin; i < end; ++i) {
      if (keep[i - begin]) accepted.push_back(candidates[i]);
    }
    score_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - score_start).count();
    begin = end;
  }
  VLOG(1) << "Pose-guided descriptor timing: geometry_seconds=" << geometry_seconds
          << " request_seconds=" << request_seconds
          << " read_seconds=" << read_seconds << " score_seconds=" << score_seconds
          << " candidates=" << candidates.size() << " batches=" << num_batches
          << " image_reads=" << num_image_reads << " loaded_rows=" << num_loaded_rows;
  candidates = std::move(accepted);
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& lhs, const Candidate& rhs) {
                     if (lhs.second_similarity != rhs.second_similarity) {
                       return lhs.second_similarity > rhs.second_similarity;
                     }
                     if (lhs.best_similarity != rhs.best_similarity) {
                       return lhs.best_similarity > rhs.best_similarity;
                     }
                     if (lhs.reprojection_error_px !=
                         rhs.reprojection_error_px) {
                       return lhs.reprojection_error_px <
                              rhs.reprojection_error_px;
                     }
                     if (lhs.track_length != rhs.track_length) {
                       return lhs.track_length > rhs.track_length;
                     }
                     return std::tie(lhs.image_id, lhs.point2d_idx,
                                     lhs.point3d_id) <
                            std::tie(rhs.image_id, rhs.point2d_idx,
                                     rhs.point3d_id);
                   });

  ObservationManager observation_manager(reconstruction);
  for (const Candidate& candidate : candidates) {
    if (!reconstruction->ExistsPoint3D(candidate.point3d_id) ||
        !reconstruction->ExistsImage(candidate.image_id)) {
      continue;
    }
    Image& image = reconstruction->GetImage(candidate.image_id);
    if (candidate.point2d_idx >= image.points2d.size() ||
        image.points2d[candidate.point2d_idx].HasPoint3D()) {
      continue;
    }
    const Point3D& point3d =
        reconstruction->GetPoint3D(candidate.point3d_id);
    const bool already_observed = std::any_of(
        point3d.track.elements.begin(), point3d.track.elements.end(),
        [&candidate](const TrackElement& element) {
          return element.image_id == candidate.image_id;
        });
    if (!already_observed && observation_manager.AddObservation(
                                 candidate.point3d_id, candidate.image_id,
                                 candidate.point2d_idx)) {
      ++summary.num_completed_observations;
    }
  }
  summary.message = "Pose-guided track completion finished";
  return summary;
}

}  // namespace colite
