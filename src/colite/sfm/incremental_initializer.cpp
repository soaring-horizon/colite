#include "colite/sfm/incremental_initializer.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"

namespace colite {
namespace {

struct TriangulatedObservation {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  FeatureMatch match;
};

bool IsUsableInitialGeometry(const TwoViewGeometry& geometry,
                             const IncrementalInitializerOptions& options) {
  return geometry.cam2_from_cam1.has_value() && (geometry.type == TwoViewGeometry::Type::kCalibratedEssential ||
          geometry.type == TwoViewGeometry::Type::kUncalibratedFundamental ||
          geometry.type == TwoViewGeometry::Type::kPlanarHomography) &&
         geometry.cam2_from_cam1->t_cw.norm() > 1e-12 &&
         std::abs(geometry.cam2_from_cam1->t_cw.normalized().z()) <
             options.max_forward_motion &&
         geometry.inlier_matches.size() >= options.min_num_inliers &&
         geometry.median_triangulation_angle >
             options.min_triangulation_angle_deg;
}

bool IsAllowedInitialPair(
    const ImagePair& pair,
    const IncrementalInitializerOptions& options) {
  return options.allowed_image_ids.empty() ||
         (options.allowed_image_ids.count(pair.image_id1) != 0 &&
          options.allowed_image_ids.count(pair.image_id2) != 0);
}

void EnsureImageAndCameraExist(const Reconstruction& reconstruction,
                               ImageId image_id) {
  if (!reconstruction.ExistsImage(image_id)) {
    throw std::out_of_range("Initial image is missing in reconstruction");
  }
  const Image& image = reconstruction.GetImage(image_id);
  if (!reconstruction.ExistsCamera(image.camera_id)) {
    throw std::out_of_range("Initial image references a missing camera");
  }
}

void SyncImagePoints2DFromFeatures(const FeatureSet& features, Image* image) {
  if (image->points2d.size() < features.keypoints.size()) {
    image->points2d.resize(features.keypoints.size());
  }

  for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
    const FeatureKeypoint& keypoint = features.keypoints[i];
    Point2D& point2d = image->points2d[i];
    point2d.xy = Eigen::Vector2d(keypoint.x, keypoint.y);
    point2d.feature_idx = static_cast<Point2DIdx>(i);
  }
}

bool IsFinitePoint(const Eigen::Vector3d& point) {
  return std::isfinite(point.x()) && std::isfinite(point.y()) &&
         std::isfinite(point.z());
}

bool HasValidMatchIndices(const FeatureMatch& match,
                          const Image& image1,
                          const Image& image2) {
  return match.idx1 != kInvalidPoint2DIdx &&
         match.idx2 != kInvalidPoint2DIdx &&
         match.idx1 < image1.points2d.size() &&
         match.idx2 < image2.points2d.size();
}

bool IsObservationAvailable(const FeatureMatch& match,
                            const Image& image1,
                            const Image& image2) {
  return !image1.points2d[match.idx1].HasPoint3D() &&
         !image2.points2d[match.idx2].HasPoint3D();
}

bool PassesReprojectionCheck(const Camera& camera1,
                             const Pose3& cam1_from_world,
                             const Image& image1,
                             const Camera& camera2,
                             const Pose3& cam2_from_world,
                             const Image& image2,
                             const FeatureMatch& match,
                             const Eigen::Vector3d& xyz,
                             double max_reprojection_error_px) {
  const auto error1 =
      ReprojectionError(camera1, cam1_from_world, xyz,
                        image1.points2d[match.idx1].xy);
  const auto error2 =
      ReprojectionError(camera2, cam2_from_world, xyz,
                        image2.points2d[match.idx2].xy);
  return error1.has_value() && error2.has_value() &&
         *error1 <= max_reprojection_error_px &&
         *error2 <= max_reprojection_error_px;
}

bool PassesTriangulationAngleCheck(const Camera& camera1,
                                   const Pose3& cam1_from_world,
                                   const Image& image1,
                                   const Camera& camera2,
                                   const Pose3& cam2_from_world,
                                   const Image& image2,
                                   const FeatureMatch& match,
                                   double min_triangulation_angle_deg) {
  const auto bearing1 = TryPixelToBearing(camera1, image1.points2d[match.idx1].xy);
  const auto bearing2 = TryPixelToBearing(camera2, image2.points2d[match.idx2].xy);
  return bearing1 && bearing2 &&
      TriangulationAngleDegrees(cam1_from_world, *bearing1, cam2_from_world,
                                *bearing2) >= min_triangulation_angle_deg;
}

std::vector<TriangulatedObservation> CollectTriangulatedObservations(
    const TwoViewGeometry& geometry,
    const Camera& camera1,
    const Pose3& cam1_from_world,
    const Image& image1,
    const Camera& camera2,
    const Pose3& cam2_from_world,
    const Image& image2,
    const IncrementalInitializerOptions& options) {
  std::vector<TriangulatedObservation> observations;
  observations.reserve(geometry.inlier_matches.size());

  for (const FeatureMatch& match : geometry.inlier_matches) {
    if (!HasValidMatchIndices(match, image1, image2) ||
        !IsObservationAvailable(match, image1, image2) ||
        !PassesTriangulationAngleCheck(camera1, cam1_from_world, image1,
                                       camera2, cam2_from_world, image2, match,
                                       options.min_triangulation_angle_deg)) {
      continue;
    }

    const auto point = TriangulatePoint(camera1, cam1_from_world,
                                        image1.points2d[match.idx1].xy,
                                        camera2, cam2_from_world,
                                        image2.points2d[match.idx2].xy);
    if (!point.has_value() || !IsFinitePoint(*point)) {
      continue;
    }
    if (!PassesReprojectionCheck(camera1, cam1_from_world, image1, camera2,
                                 cam2_from_world, image2, match, *point,
                                 options.max_reprojection_error_px)) {
      continue;
    }

    observations.push_back({*point, match});
  }

  return observations;
}

void AddTriangulatedObservations(
    ImageId image_id1,
    ImageId image_id2,
    const std::vector<TriangulatedObservation>& observations,
    Reconstruction* reconstruction) {
  for (const TriangulatedObservation& observation : observations) {
    Track track;
    track.AddElement(image_id1, observation.match.idx1);
    track.AddElement(image_id2, observation.match.idx2);
    reconstruction->AddPoint3D(observation.xyz, track);
  }
}

}  // namespace

std::vector<ImagePair> SelectInitialImagePairCandidates(
    const MatchGraph& graph, const IncrementalInitializerOptions& options,
    const Reconstruction* reconstruction, const InitialPairTrialState* trial_state) {
  const bool specified1 = options.image_id1 != kInvalidImageId;
  const bool specified2 = options.image_id2 != kInvalidImageId;
  if (specified1 && specified2) {
    const ImagePair pair{options.image_id1, options.image_id2};
    if (pair.image_id1 == pair.image_id2 || !IsAllowedInitialPair(pair, options) ||
        (reconstruction && (!reconstruction->ExistsImage(pair.image_id1) ||
                            !reconstruction->ExistsImage(pair.image_id2)))) return {};
    return {pair};
  }
  std::map<ImageId, std::size_t> support;
  using Neighbor = std::pair<ImageId, const TwoViewGeometry*>;
  std::map<ImageId, std::vector<Neighbor>> neighbors;
  const auto prior_focal = [&](ImageId id) {
    if (!reconstruction || !reconstruction->ExistsImage(id)) return false;
    const auto camera = reconstruction->GetImage(id).camera_id;
    return reconstruction->ExistsCamera(camera) && reconstruction->GetCamera(camera).has_prior_focal_length;
  };
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!IsAllowedInitialPair(pair, options)) continue;
    if (geometry.HasUsableCorrespondences()) {
      support[pair.image_id1] += geometry.inlier_matches.size();
      support[pair.image_id2] += geometry.inlier_matches.size();
    }
    if (trial_state && trial_state->tried_pairs.count(pair)) continue;
    if (options.reestimate_geometry) {
      if (!geometry.HasUsableCorrespondences() || geometry.inlier_matches.size() < options.min_num_inliers) continue;
    } else if (!IsUsableInitialGeometry(geometry, options)) continue;
    neighbors[pair.image_id1].emplace_back(pair.image_id2, &geometry);
    neighbors[pair.image_id2].emplace_back(pair.image_id1, &geometry);
  }
  std::vector<ImageId> first_images;
  if (specified1 || specified2) first_images.push_back(specified1 ? options.image_id1 : options.image_id2);
  else for (const auto& [id, adjacent] : neighbors) {
    (void)adjacent;
    if (trial_state) {
      const auto count = trial_state->num_reg_trials.find(id);
      if (count != trial_state->num_reg_trials.end() && count->second >= options.max_num_reg_trials) continue;
    }
    first_images.push_back(id);
  }
  std::sort(first_images.begin(), first_images.end(), [&](ImageId a, ImageId b) {
    if (prior_focal(a) != prior_focal(b)) return prior_focal(a);
    if (support[a] != support[b]) return support[a] > support[b];
    return a < b;
  });
  std::vector<ImagePair> result;
  for (const auto first : first_images) {
    auto adjacent = neighbors[first];
    std::sort(adjacent.begin(), adjacent.end(), [&](const Neighbor& a, const Neighbor& b) {
      if (prior_focal(a.first) != prior_focal(b.first)) return prior_focal(a.first);
      if (a.second->inlier_matches.size() != b.second->inlier_matches.size())
        return a.second->inlier_matches.size() > b.second->inlier_matches.size();
      return a.first < b.first;
    });
    for (const auto& [second, geometry] : adjacent) {
      (void)geometry;
      result.push_back({first, second});
    }
  }
  return result;
}

std::optional<ImagePair> SelectInitialImagePair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options,
    const Reconstruction* reconstruction) {
  const std::vector<ImagePair> candidates =
      SelectInitialImagePairCandidates(graph, options, reconstruction);
  if (candidates.empty()) {
    return std::nullopt;
  }
  return candidates.front();
}

static IncrementalInitializerSummary InitializePairImpl(
    const MatchGraph& graph,
    ImagePair initial_pair,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction, const bool triangulate_points) {
  IncrementalInitializerSummary summary;
  summary.initial_pair = initial_pair;
  if (!IsAllowedInitialPair(initial_pair, options)) {
    summary.message = "Initial pair is outside the allowed image set";
    return summary;
  }

  const TwoViewGeometry* geometry =
      FindVerifiedGeometry(graph, initial_pair.image_id1, initial_pair.image_id2);
  std::optional<TwoViewGeometry> oriented;
  if (geometry && initial_pair.image_id1 > initial_pair.image_id2) {
    if (options.reestimate_geometry) {
      // Fresh estimation consumes correspondences only. Do not let an unused
      // legacy matrix (e.g. a singular cached H) block a valid calibrated fit.
      oriented.emplace();
      oriented->type = geometry->type;
      oriented->inlier_matches = geometry->inlier_matches;
      for (auto& match : oriented->inlier_matches) std::swap(match.idx1, match.idx2);
    } else oriented = ReverseTwoViewGeometry(*geometry);
    geometry = &*oriented;
  }
  if (geometry == nullptr || (options.reestimate_geometry
          ? !geometry->HasUsableCorrespondences()
          : !IsUsableInitialGeometry(*geometry, options))) {
    summary.message = "Initial pair has no usable relative geometry";
    return summary;
  }

  const FeatureSet* features1 = FindFeatures(graph, initial_pair.image_id1);
  const FeatureSet* features2 = FindFeatures(graph, initial_pair.image_id2);
  if (features1 == nullptr || features2 == nullptr) {
    summary.message = "Initial pair features are missing";
    return summary;
  }

  EnsureImageAndCameraExist(*reconstruction, initial_pair.image_id1);
  EnsureImageAndCameraExist(*reconstruction, initial_pair.image_id2);

  std::optional<TwoViewGeometry> reestimated;
  if (options.reestimate_geometry) {
    TwoViewGeometryOptions estimation;
    estimation.max_error_px = options.max_reprojection_error_px;
    estimation.min_num_iterations = 30;
    estimation.random_seed = options.random_seed;
    estimation.compute_relative_pose = true;
    reestimated = EstimateCalibratedTwoViewGeometry(
        reconstruction->GetCamera(reconstruction->GetImage(initial_pair.image_id1).camera_id), *features1,
        reconstruction->GetCamera(reconstruction->GetImage(initial_pair.image_id2).camera_id), *features2,
        geometry->inlier_matches, estimation);
    geometry = &*reestimated;
    summary.geometry_reestimated = true;
  }
  summary.num_geometry_inliers = geometry->inlier_matches.size();
  summary.median_triangulation_angle = geometry->median_triangulation_angle;
  if (!IsUsableInitialGeometry(*geometry, options)) {
    summary.message = "Initial pair has no usable relative geometry";
    return summary;
  }
  summary.geometry_accepted = true;

  Image& image1 = reconstruction->GetImage(initial_pair.image_id1);
  Image& image2 = reconstruction->GetImage(initial_pair.image_id2);
  SyncImagePoints2DFromFeatures(*features1, &image1);
  SyncImagePoints2DFromFeatures(*features2, &image2);

  Pose3 cam1_from_world;
  cam1_from_world.q_cw = Eigen::Quaterniond::Identity();
  cam1_from_world.t_cw = Eigen::Vector3d::Zero();
  const Pose3 cam2_from_world = *geometry->cam2_from_cam1;

  std::vector<TriangulatedObservation> observations;
  if (triangulate_points) {
    observations = CollectTriangulatedObservations(*geometry,
        reconstruction->GetCamera(image1.camera_id), cam1_from_world, image1,
        reconstruction->GetCamera(image2.camera_id), cam2_from_world, image2, options);
    if (observations.size() < options.min_num_triangulated_points) {
      summary.num_triangulated_points = observations.size();
      summary.message = "Initial pair produced too few triangulated points";
      return summary;
    }
  }

  image1.cam_from_world = cam1_from_world;
  image2.cam_from_world = cam2_from_world;
  reconstruction->RegisterImage(initial_pair.image_id1);
  reconstruction->RegisterImage(initial_pair.image_id2);
  AddTriangulatedObservations(initial_pair.image_id1, initial_pair.image_id2,
                              observations, reconstruction);

  summary.success = true;
  summary.num_triangulated_points = observations.size();
  summary.message = triangulate_points ? "Initial reconstruction created" : "Initial image poses registered";
  return summary;
}

IncrementalInitializerSummary RegisterInitialImagePair(
    const MatchGraph& graph, ImagePair initial_pair,
    const IncrementalInitializerOptions& options, Reconstruction* reconstruction) {
  return InitializePairImpl(graph, initial_pair, options, reconstruction, false);
}

IncrementalInitializerSummary InitializeReconstructionFromPair(
    const MatchGraph& graph, ImagePair initial_pair,
    const IncrementalInitializerOptions& options, Reconstruction* reconstruction) {
  return InitializePairImpl(graph, initial_pair, options, reconstruction, true);
}

IncrementalInitializerSummary InitializeReconstructionFromBestPair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction) {
  IncrementalInitializerSummary final_summary;
  const std::vector<ImagePair> candidates =
      SelectInitialImagePairCandidates(graph, options, reconstruction);
  if (candidates.empty()) {
    final_summary.message = "No usable initial image pair";
    return final_summary;
  }

  std::set<ImagePair> checked;
  for (const auto pair : candidates) {
    if (options.max_num_pair_trials && checked.size() >= options.max_num_pair_trials) break;
    if (!checked.insert(ImagePair::Ordered(pair.image_id1, pair.image_id2)).second) continue;
    final_summary.attempted_initial_pairs.push_back(pair);
    final_summary.num_initialization_attempts =
        final_summary.attempted_initial_pairs.size();

    IncrementalInitializerSummary attempt =
        InitializeReconstructionFromPair(graph, pair, options, reconstruction);
    attempt.attempted_initial_pairs = final_summary.attempted_initial_pairs;
    attempt.num_initialization_attempts =
        final_summary.num_initialization_attempts;
    if (attempt.success) {
      return attempt;
    }
    final_summary = std::move(attempt);
  }

  if (final_summary.message.empty()) {
    final_summary.message = "No initial image pair could be initialized";
  }
  return final_summary;
}

}  // namespace colite
