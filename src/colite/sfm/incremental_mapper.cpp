#include "colite/sfm/incremental_mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>
#include <glog/logging.h>

#include "colite/core/geometry.h"
#include "colite/core/memory.h"
#include "colite/sfm/similarity_transform.h"
#include "refinement_diagnostics.h"

namespace colite {
namespace {

class ScopedPoint3DChanges {
 public:
  explicit ScopedPoint3DChanges(Reconstruction* reconstruction)
      : reconstruction_(reconstruction) {
    if (reconstruction_) reconstruction_->BeginPoint3DChangeTracking();
  }
  ~ScopedPoint3DChanges() {
    if (reconstruction_) reconstruction_->EndPoint3DChangeTracking();
  }
  ScopedPoint3DChanges(const ScopedPoint3DChanges&) = delete;
  ScopedPoint3DChanges& operator=(const ScopedPoint3DChanges&) = delete;
  std::set<Point3DId> Finish() {
    auto* reconstruction = reconstruction_;
    reconstruction_ = nullptr;
    return reconstruction ? reconstruction->EndPoint3DChangeTracking()
                          : std::set<Point3DId>{};
  }
 private:
  Reconstruction* reconstruction_;
};

using MappingClock = std::chrono::steady_clock;

double ElapsedSeconds(const MappingClock::time_point start) {
  return std::chrono::duration<double>(MappingClock::now() - start).count();
}

struct ImageCandidate {
  ImageId image_id = kInvalidImageId;
  std::size_t num_visible_points = 0;
  std::size_t num_visible_observations = 0;
  double mean_track_length = 0.0;
  std::size_t num_registered_neighbors = 0;
  std::size_t num_verified_inliers_to_registered_images = 0;
  double score = 0.0;
  std::size_t num_registration_attempts = 0;
};

ImageId OtherImageId(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id ? pair.image_id2 : pair.image_id1;
}

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.HasUsableCorrespondences();
}

bool HasValidFeatureIndex(const FeatureSet* features, Point2DIdx idx) {
  return features != nullptr && idx != kInvalidPoint2DIdx &&
         idx < features->keypoints.size();
}

double ComputeCandidateScore(const ImageCandidate& candidate) {
  return static_cast<double>(candidate.num_visible_points) * 1000.0 +
         static_cast<double>(candidate.num_registered_neighbors) * 100.0 +
         candidate.mean_track_length * 20.0 +
         static_cast<double>(
             candidate.num_verified_inliers_to_registered_images) *
             0.01;
}

using CandidateNeighborIndex = std::map<ImageId, std::vector<MatchGraphEdge>>;

CandidateNeighborIndex BuildCandidateNeighborIndex(const MatchGraph& graph) {
  CandidateNeighborIndex neighbors;
  // Preserve verified_geometries iteration order so candidate scoring and
  // registration order remain identical to the full-graph scan.
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!IsUsableGeometry(geometry)) continue;
    neighbors[pair.image_id1].push_back({pair, &geometry});
    if (pair.image_id2 != pair.image_id1) {
      neighbors[pair.image_id2].push_back({pair, &geometry});
    }
  }
  return neighbors;
}

ImageCandidate AnalyzeImageCandidate(const MatchGraph& graph,
                                     ImageId image_id,
                                     const Reconstruction& reconstruction,
                                     const CandidateNeighborIndex& neighbors) {
  ImageCandidate candidate;
  candidate.image_id = image_id;
  if (!reconstruction.ExistsImage(image_id) ||
      reconstruction.IsImageRegistered(image_id)) {
    return candidate;
  }

  const FeatureSet* target_features = FindFeatures(graph, image_id);
  if (target_features == nullptr) {
    return candidate;
  }

  std::vector<bool> visible_target_points(target_features->keypoints.size(), false);
  std::unordered_set<Point3DId> visible_point3d_ids;
  const auto neighbors_it = neighbors.find(image_id);
  if (neighbors_it == neighbors.end()) return candidate;
  for (const auto& edge : neighbors_it->second) {
    const auto& pair = edge.pair;
    const auto& geometry = *edge.geometry;

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction.ExistsImage(neighbor_id) ||
        !reconstruction.IsImageRegistered(neighbor_id)) {
      continue;
    }

    ++candidate.num_registered_neighbors;
    candidate.num_verified_inliers_to_registered_images +=
        geometry.inlier_matches.size();

    const Image& neighbor_image = reconstruction.GetImage(neighbor_id);
    const bool target_is_first = pair.image_id1 == image_id;
    for (const FeatureMatch& match : geometry.inlier_matches) {
      const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
      const Point2DIdx neighbor_idx = target_is_first ? match.idx2 : match.idx1;
      if (!HasValidFeatureIndex(target_features, target_idx) ||
          neighbor_idx == kInvalidPoint2DIdx ||
          neighbor_idx >= neighbor_image.points2d.size()) {
        continue;
      }

      const Point2D& neighbor_point = neighbor_image.points2d[neighbor_idx];
      if (neighbor_point.HasPoint3D() &&
          reconstruction.ExistsPoint3D(neighbor_point.point3d_id)) {
        if (!visible_target_points[target_idx]) {
          visible_target_points[target_idx] = true;
          ++candidate.num_visible_points;
        }
        visible_point3d_ids.insert(neighbor_point.point3d_id);
      }
    }
  }

  for (const Point3DId point3d_id : visible_point3d_ids) {
    candidate.num_visible_observations +=
        reconstruction.GetPoint3D(point3d_id).track.Length();
  }
  if (!visible_point3d_ids.empty()) {
    candidate.mean_track_length =
        static_cast<double>(candidate.num_visible_observations) /
        static_cast<double>(visible_point3d_ids.size());
  }
  candidate.score = ComputeCandidateScore(candidate);
  return candidate;
}

std::vector<ImageCandidate> CollectImageCandidates(
    const MatchGraph& graph,
    const IncrementalMapperOptions& options,
    const Reconstruction& reconstruction,
    const std::set<ImageId>* allowed_image_ids = nullptr,
    const std::map<ImageId, std::size_t>* registration_trials = nullptr,
    std::size_t max_num_registration_trials_per_image = 0,
    const CandidateNeighborIndex* neighbors = nullptr) {
  const auto owned_neighbors = neighbors == nullptr
      ? BuildCandidateNeighborIndex(graph) : CandidateNeighborIndex{};
  if (neighbors == nullptr) neighbors = &owned_neighbors;
  std::vector<ImageCandidate> candidates;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    (void)image;
    if (allowed_image_ids != nullptr && !allowed_image_ids->empty() &&
        allowed_image_ids->count(image_id) == 0) {
      continue;
    }
    ImageCandidate candidate =
        AnalyzeImageCandidate(graph, image_id, reconstruction, *neighbors);
    if (registration_trials != nullptr) {
      const auto trial_it = registration_trials->find(image_id);
      if (trial_it != registration_trials->end()) {
        candidate.num_registration_attempts = trial_it->second;
      }
    }
    if (max_num_registration_trials_per_image > 0 &&
        candidate.num_registration_attempts >=
            max_num_registration_trials_per_image) {
      continue;
    }
    const bool pnp_ready = candidate.num_visible_points >=
                           options.registration.min_num_correspondences;
    const bool structureless_ready =
        options.use_structureless_registration &&
        candidate.num_registered_neighbors >=
            options.structureless_registration
                .min_num_registered_neighbors;
    if (pnp_ready || structureless_ready) {
      candidates.push_back(candidate);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ImageCandidate& lhs, const ImageCandidate& rhs) {
              if (lhs.num_registration_attempts !=
                  rhs.num_registration_attempts) {
                return lhs.num_registration_attempts <
                       rhs.num_registration_attempts;
              }
              if (lhs.score != rhs.score) {
                return lhs.score > rhs.score;
              }
              if (lhs.num_visible_points != rhs.num_visible_points) {
                return lhs.num_visible_points > rhs.num_visible_points;
              }
              if (lhs.num_visible_observations !=
                  rhs.num_visible_observations) {
                return lhs.num_visible_observations >
                       rhs.num_visible_observations;
              }
              if (lhs.num_registered_neighbors !=
                  rhs.num_registered_neighbors) {
                return lhs.num_registered_neighbors >
                       rhs.num_registered_neighbors;
              }
              if (lhs.num_verified_inliers_to_registered_images !=
                  rhs.num_verified_inliers_to_registered_images) {
                return lhs.num_verified_inliers_to_registered_images >
                       rhs.num_verified_inliers_to_registered_images;
              }
              return lhs.image_id < rhs.image_id;
            });
  return candidates;
}

void CopyCandidateStatsToSummary(const ImageCandidate& candidate,
                                 IncrementalMapperSummary* summary) {
  summary->candidate_num_visible_points = candidate.num_visible_points;
  summary->candidate_num_visible_observations =
      candidate.num_visible_observations;
  summary->candidate_mean_track_length = candidate.mean_track_length;
  summary->candidate_num_registered_neighbors =
      candidate.num_registered_neighbors;
  summary->candidate_num_verified_inliers_to_registered_images =
      candidate.num_verified_inliers_to_registered_images;
  summary->candidate_score = candidate.score;
  summary->registration_attempts_before_success =
      candidate.num_registration_attempts;
}

ImageRegistrationFailureSummary MakeFailureSummary(
    const ImageCandidate& candidate,
    std::size_t attempt,
    double registration_max_reprojection_error_px,
    const ImageRegistrationSummary& registration,
    const std::string& message) {
  ImageRegistrationFailureSummary failure;
  failure.image_id = candidate.image_id;
  failure.attempt = attempt;
  failure.candidate_num_visible_points = candidate.num_visible_points;
  failure.candidate_num_visible_observations =
      candidate.num_visible_observations;
  failure.candidate_mean_track_length = candidate.mean_track_length;
  failure.candidate_num_registered_neighbors =
      candidate.num_registered_neighbors;
  failure.candidate_num_verified_inliers_to_registered_images =
      candidate.num_verified_inliers_to_registered_images;
  failure.candidate_score = candidate.score;
  failure.registration_max_reprojection_error_px =
      registration_max_reprojection_error_px;
  failure.registration_num_correspondences =
      registration.num_correspondences;
  failure.registration_num_ransac_inliers =
      registration.num_ransac_inliers;
  failure.registration_num_inliers = registration.num_inliers;
  failure.message = message;
  return failure;
}

ImageRegistrarOptions RegistrationOptionsForAttempt(
    const ImageRegistrarOptions& base_options,
    std::size_t num_previous_failures,
    const IncrementalMappingOptions& mapping_options) {
  ImageRegistrarOptions options = base_options;
  if (num_previous_failures == 0) {
    return options;
  }

  const double multiplier = std::max(
      1.0, mapping_options.registration_retry_reprojection_error_multiplier);
  const double max_error = std::max(
      base_options.max_reprojection_error_px,
      mapping_options.registration_retry_max_reprojection_error_px);
  options.max_reprojection_error_px =
      std::min(max_error, base_options.max_reprojection_error_px *
                              std::pow(multiplier,
                                       static_cast<double>(
                                           num_previous_failures)));
  return options;
}

bool HasLargeReprojectionError(const Reconstruction& reconstruction,
                               const Point3D& point3d,
                               const TrackElement& element,
                               double max_error_px) {
  if (!reconstruction.ExistsImage(element.image_id)) {
    return true;
  }
  const Image& image = reconstruction.GetImage(element.image_id);
  if (!image.HasPose() || !reconstruction.ExistsCamera(image.camera_id) ||
      element.point2d_idx >= image.points2d.size()) {
    return true;
  }
  const Camera& camera = reconstruction.GetCamera(image.camera_id);
  const auto error =
      ReprojectionError(camera, *image.cam_from_world, point3d.xyz,
                        image.points2d[element.point2d_idx].xy);
  return !error.has_value() || *error > max_error_px;
}

double MaxTriangulationAngleDegrees(const Reconstruction& reconstruction,
                                    const Point3D& point3d,
                                    double required_angle) {
  double max_angle = 0.0;
  for (std::size_t i = 0; i < point3d.track.elements.size(); ++i) {
    const TrackElement& element1 = point3d.track.elements[i];
    if (!reconstruction.ExistsImage(element1.image_id)) {
      continue;
    }
    const Image& image1 = reconstruction.GetImage(element1.image_id);
    if (!image1.HasPose() || !reconstruction.ExistsCamera(image1.camera_id) ||
        element1.point2d_idx >= image1.points2d.size()) {
      continue;
    }
    const Camera& camera1 = reconstruction.GetCamera(image1.camera_id);
    const auto bearing1 = TryPixelToBearing(camera1, image1.points2d[element1.point2d_idx].xy);
    if (!bearing1) continue;

    for (std::size_t j = i + 1; j < point3d.track.elements.size(); ++j) {
      const TrackElement& element2 = point3d.track.elements[j];
      if (!reconstruction.ExistsImage(element2.image_id)) {
        continue;
      }
      const Image& image2 = reconstruction.GetImage(element2.image_id);
      if (!image2.HasPose() ||
          !reconstruction.ExistsCamera(image2.camera_id) ||
          element2.point2d_idx >= image2.points2d.size()) {
        continue;
      }
      const Camera& camera2 = reconstruction.GetCamera(image2.camera_id);
      const auto bearing2 = TryPixelToBearing(camera2, image2.points2d[element2.point2d_idx].xy);
      if (!bearing2) continue;
      max_angle = std::max(max_angle,
                           TriangulationAngleDegrees(*image1.cam_from_world,
                                                     *bearing1,
                                                     *image2.cam_from_world,
                                                     *bearing2));
      if (max_angle >= required_angle) return max_angle;
    }
  }
  return max_angle;
}

ObservationFilterSummary FilterNewlyCreatedPoints(
    Point3DId first_created_point_id,
    const ObservationFilterOptions& options,
    Reconstruction* reconstruction) {
  ObservationFilterSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  struct ObservationToDelete {
    Point3DId point3d_id;
    ImageId image_id;
    Point2DIdx point2d_idx;
  };
  std::vector<ObservationToDelete> observations_to_delete;
  const Point3DId end_point_id = reconstruction->NextPoint3DId();
  for (Point3DId point3d_id = first_created_point_id;
       point3d_id < end_point_id; ++point3d_id) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) continue;
    const Point3D& point3d = reconstruction->GetPoint3D(point3d_id);
    for (const TrackElement& element : point3d.track.elements) {
      if (HasLargeReprojectionError(*reconstruction, point3d, element,
                                    options.max_reprojection_error_px)) {
        observations_to_delete.push_back(
            {point3d_id, element.image_id, element.point2d_idx});
      }
    }
  }
  for (const ObservationToDelete& observation : observations_to_delete) {
    if (reconstruction->DeleteObservation(observation.point3d_id,
                                          observation.image_id,
                                          observation.point2d_idx)) {
      ++summary.num_removed_observations;
    }
  }

  std::vector<Point3DId> points_to_delete;
  for (Point3DId point3d_id = first_created_point_id;
       point3d_id < end_point_id; ++point3d_id) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) continue;
    const Point3D& point3d = reconstruction->GetPoint3D(point3d_id);
    if (point3d.track.Length() < options.min_track_length ||
        MaxTriangulationAngleDegrees(*reconstruction, point3d,
                                     options.min_triangulation_angle_deg) <
            options.min_triangulation_angle_deg) {
      points_to_delete.push_back(point3d_id);
    }
  }

  for (const Point3DId point3d_id : points_to_delete) {
    if (!reconstruction->ExistsPoint3D(point3d_id)) {
      continue;
    }
    const std::size_t track_length =
        reconstruction->GetPoint3D(point3d_id).track.Length();
    if (reconstruction->DeletePoint3D(point3d_id)) {
      ++summary.num_removed_points;
      summary.num_removed_observations += track_length;
    }
  }

  summary.success = true;
  summary.message = "Local retriangulation filtering finished";
  return summary;
}

void AccumulateTriangulationSummary(
    const IncrementalTriangulatorSummary& update,
    IncrementalTriangulatorSummary* summary) {
  if (summary == nullptr) {
    return;
  }
  summary->num_created_points += update.num_created_points;
  summary->num_created_observations += update.num_created_observations;
  summary->num_completed_observations += update.num_completed_observations;
  summary->num_merged_points += update.num_merged_points;
  summary->num_merged_observations += update.num_merged_observations;
  summary->num_filtered_observations += update.num_filtered_observations;
  summary->num_filtered_points += update.num_filtered_points;
  summary->message = update.message;
}

GlobalBundleAdjustmentOptions MappingGlobalBundleOptions(
    const IncrementalMappingOptions& options, const Reconstruction& model) {
  auto ba = options.global_bundle_adjustment;
  if (model.RegisteredImageIds().size() < 10) {
    if (ba.max_num_iterations > 0)
      ba.max_num_iterations = ba.max_num_iterations > std::numeric_limits<int>::max()/2
          ? std::numeric_limits<int>::max() : 2 * ba.max_num_iterations;
    ba.solver.function_tolerance /= 10;
    ba.solver.gradient_tolerance /= 10;
    ba.solver.parameter_tolerance /= 10;
    ba.solver.max_linear_solver_iterations = 200;
  }
  if (model.RegisteredImageIds().size() <= 2) ba.use_position_priors = false;
  return ba;
}

InitialPairRefinementSummary RefineInitialPair(
    const IncrementalMappingOptions& options, Reconstruction* model,
    IncrementalMappingSummary* mapping) {
  InitialPairRefinementSummary result;
  result.num_points_before = model->Points3D().size();
  if (options.run_global_bundle_adjustment) {
    result.negative_depth_filter = FilterObservationsWithNegativeDepth(model);
    if (!result.negative_depth_filter.success || model->Points3D().empty()) {
      result.message = "Initial reconstruction has no positive-depth points";
      return result;
    }
    result.bundle_adjustment_attempted = true;
    const auto start = MappingClock::now();
    result.bundle_adjustment = AdjustGlobalBundle(MappingGlobalBundleOptions(options, *model), model);
    mapping->initial_bundle_adjustment_seconds += ElapsedSeconds(start);
    mapping->num_global_ba_invalid_steps += result.bundle_adjustment.num_invalid_steps;
    if (!result.bundle_adjustment.success) {
      result.message = "Initial bundle adjustment failed: " + result.bundle_adjustment.message;
      return result;
    }
  }
  // Absolute position priors retain their target units. Their final alignment
  // remains the controller's responsibility.
  if (!options.global_bundle_adjustment.use_position_priors) {
    const auto normalization = NormalizeReconstruction(model);
    result.normalized = true;
    result.normalization_scale = normalization.scale;
  }
  if (options.run_global_observation_filter) {
    result.observation_filter = FilterObservationsAndPoints(options.global_observation_filter, model);
    if (!result.observation_filter.success) {
      result.message = result.observation_filter.message;
      return result;
    }
  }
  result.num_points_after = model->Points3D().size();
  result.success = model->RegisteredImageIds().size() >= 2 &&
      !model->Points3D().empty() &&
      model->Points3D().size() >= options.mapper.registration.min_num_inliers;
  result.message = result.success ? "Initial reconstruction refined" :
      "Initial reconstruction has too few points for absolute pose registration";
  return result;
}

IncrementalInitializerSummary InitializeMappingFromCandidates(
    const MatchGraph& graph, const IncrementalInitializerOptions& initialization,
    const IncrementalTriangulatorCache& triangulator_cache,
    const IncrementalMappingOptions& options, InitialPairTrialState* trial_state,
    std::size_t* num_model_trials,
    IncrementalMappingSummary* mapping, Reconstruction* reconstruction) {
  IncrementalInitializerSummary last;
  std::vector<ImagePair> checked;
  const auto candidates = SelectInitialImagePairCandidates(graph, initialization, reconstruction, trial_state);
  for (const auto pair : candidates) {
    if (*num_model_trials >= options.max_num_initialization_trials) {
      last.message = "Initial reconstruction trial budget exhausted";
      break;
    }
    const auto key = ImagePair::Ordered(pair.image_id1, pair.image_id2);
    if (trial_state->tried_pairs.count(key)) continue;
    if (initialization.image_id1 == kInvalidImageId && initialization.image_id2 == kInvalidImageId &&
        trial_state->num_reg_trials[pair.image_id1] >= initialization.max_num_reg_trials) continue;
    trial_state->tried_pairs.insert(key);
    // Publish only a fully refined candidate: failed BA/filtering cannot leak
    // intrinsics, poses, point IDs or links into another trial or the caller.
    Reconstruction candidate = *reconstruction;
    checked.push_back(pair);
    last = RegisterInitialImagePair(graph, pair, initialization, &candidate);
    if (!last.geometry_accepted) continue;
    ++*num_model_trials;
    ++trial_state->num_reg_trials[pair.image_id1];
    ++trial_state->num_reg_trials[pair.image_id2];
    if (last.success) {
      auto triangulation = options.mapper.triangulation;
      triangulation.min_triangulation_angle_deg = initialization.min_triangulation_angle_deg;
      for (const auto id : {pair.image_id1, pair.image_id2})
        TriangulateRegisteredImage(graph, id, triangulation, &candidate, &triangulator_cache);
      last.num_triangulated_points = candidate.Points3D().size();
      if (candidate.Points3D().empty() || candidate.Points3D().size() < initialization.min_num_triangulated_points) {
        last.success = false;
        last.message = "Initial pair produced too few triangulated points";
      }
    }
    InitialPairRefinementSummary refined;
    if (last.success) {
      try {
        refined = RefineInitialPair(options, &candidate, mapping);
      } catch (const std::invalid_argument& error) {
        refined.num_points_before = last.num_triangulated_points;
        refined.num_points_after = candidate.Points3D().size();
        refined.message = std::string("Initial refinement failed: ") + error.what();
      }
    } else {
      refined.num_points_before = last.num_triangulated_points;
      refined.message = last.message;
    }
    refined.pair = pair;
    mapping->initialization_refinements.push_back(refined);
    last.success = refined.success;
    last.message = refined.message;
    if (last.success) {
      *reconstruction = std::move(candidate);
      break;
    }
  }
  last.attempted_initial_pairs = std::move(checked);
  last.num_initialization_attempts = last.attempted_initial_pairs.size();
  if (last.message.empty()) last.message = "No usable initial image pair";
  return last;
}

}  // namespace

IncrementalMapperSummary RegisterNextImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalMapperOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* triangulator_cache) {
  IncrementalMapperSummary summary;
  summary.image_id = image_id;

  if (reconstruction == nullptr || !reconstruction->ExistsImage(image_id)) {
    summary.message = "Image is missing in reconstruction";
    return summary;
  }
  if (options.ba_local_max_refinements == 0 ||
      !std::isfinite(options.ba_local_max_refinement_change) ||
      options.ba_local_max_refinement_change < 0.0) {
    summary.message = "Invalid local refinement limit or change threshold";
    return summary;
  }
  summary.registration_max_reprojection_error_px =
      options.registration.max_reprojection_error_px;

  // 1. 通过已注册图像上的 2D-3D 关联估计新图像位姿。
  auto stage_start = MappingClock::now();
  summary.registration =
      RegisterImageWithExistingPoints(graph, image_id, options.registration,
                                      reconstruction);
  summary.registration_seconds = ElapsedSeconds(stage_start);
  if (!summary.registration.success) {
    if (!options.use_structureless_registration) {
      summary.message = summary.registration.message;
      return summary;
    }
    summary.structureless_registration = EstimateStructurelessPose(
        graph, image_id, *reconstruction,
        options.structureless_registration);
    std::size_t validated_inliers = summary.structureless_registration.success
        ? ValidateStructurelessPose(graph, image_id, *reconstruction,
            summary.structureless_registration.pose, options.structureless_registration,
            options.registration.max_reprojection_error_px) : 0;
    if (!summary.structureless_registration.success) {
      // Use match-based resection when pair poses cannot initialize the query.
      // A coherent pair-pose hypothesis that fails pixel validation must still
      // be rejected, rather than bypassing that existing consistency check.
      summary.structureless_registration = EstimateStructurelessPoseFromMatches(
          graph, image_id, *reconstruction, options.registration,
          options.structureless_registration);
      // Generalized resection already verifies the total epipolar constraint
      // count across distinct centers. Do not reapply the heuristic ray
      // initializer's fifteen-inlier gate to each contributing camera.
      auto validation_options = options.structureless_registration;
      validation_options.min_num_inliers_per_edge = 1;
      validated_inliers = summary.structureless_registration.success ? ValidateStructurelessPose(
          graph, image_id, *reconstruction, summary.structureless_registration.pose,
          validation_options,
          options.registration.max_reprojection_error_px) : 0;
    }
    if (!summary.structureless_registration.success) {
      summary.message = summary.registration.message + "; " +
                        summary.structureless_registration.message;
      return summary;
    }
    Image& image = reconstruction->GetImage(image_id);
    if (validated_inliers < options.registration.min_num_inliers) {
      summary.message = "Structure-less pose failed multi-view geometric validation";
      return summary;
    }
    image.cam_from_world = summary.structureless_registration.pose;
    reconstruction->RegisterImage(image_id);
    summary.used_structureless_registration = true;
    summary.registration.success = true;
    summary.registration.num_inliers =
        validated_inliers;
    summary.registration.message =
        "Registered from consistent relative-pose hypotheses";
  }

  // 2. 先把新图像能挂到已有 3D 点上的观测补齐，再创建新的 track。
  LocalBundleAdjustmentScope local_ba_scope;
  local_ba_scope.first_new_point_id = reconstruction->NextPoint3DId();
  ScopedPoint3DChanges point_changes(
      options.run_local_bundle_adjustment ? reconstruction : nullptr);
  stage_start = MappingClock::now();
  summary.track_completion = CompleteTracksForRegisteredImage(
      graph, image_id, options.triangulation, reconstruction);
  summary.track_completion_seconds = ElapsedSeconds(stage_start);
  stage_start = MappingClock::now();
  summary.triangulation = TriangulateRegisteredImage(
      graph, image_id, options.triangulation, reconstruction,
      triangulator_cache);
  summary.triangulation_seconds = ElapsedSeconds(stage_start);
  const ObservationFilterOptions* local_retriangulation_filter =
      options.run_local_retriangulation_filter
          ? &options.local_retriangulation_filter
          : nullptr;
  stage_start = MappingClock::now();
  const auto local_window = FindLocalBundleImages(
      *reconstruction, image_id, options.local_bundle_adjustment.max_num_images);
  summary.local_retriangulation = RetriangulateLocalNeighborhood(
      graph, image_id, options.triangulation, local_retriangulation_filter,
      reconstruction, triangulator_cache, &local_window);
  summary.local_retriangulation_seconds = ElapsedSeconds(stage_start);
  local_ba_scope.modified_point_ids = point_changes.Finish();

  // Refine the same changed-point scope until observation membership stabilizes.
  // COLMAP uses a robust loss for the first pass and quadratic loss afterwards.
  auto local_ba_options = options.local_bundle_adjustment;
  const std::size_t max_refinements = options.run_local_bundle_adjustment
      ? options.ba_local_max_refinements : 1;
  for (std::size_t pass = 0; pass < max_refinements; ++pass) {
    ScopedPoint3DChanges refinement_changes(
        options.run_local_bundle_adjustment ? reconstruction : nullptr);
    IncrementalTriangulatorSummary post_ba_tracks;
    if (options.run_local_bundle_adjustment) {
      stage_start = MappingClock::now();
      summary.local_bundle_adjustment = AdjustLocalBundle(
          image_id, local_ba_options, reconstruction, &local_ba_scope);
      summary.local_bundle_adjustment_seconds += ElapsedSeconds(stage_start);
      if (!summary.local_bundle_adjustment.success) {
        summary.message = summary.local_bundle_adjustment.message;
        return summary;
      }
      ++summary.num_local_refinements;
      stage_start = MappingClock::now();
      post_ba_tracks = CompleteTracksForRegisteredImage(
          graph, image_id, options.triangulation, reconstruction);
      summary.track_completion_seconds += ElapsedSeconds(stage_start);
      AccumulateTriangulationSummary(post_ba_tracks, &summary.track_completion);
    }

    ObservationFilterSummary filtered;
    if (options.run_observation_filter) {
      stage_start = MappingClock::now();
      auto adjusted_image_ids = summary.local_bundle_adjustment.local_image_ids;
      if (adjusted_image_ids.empty()) adjusted_image_ids.insert(image_id);
      // Completion can modify points outside the local window. Keep them in
      // the same work set for filtering and subsequent local refinement.
      const auto changed_points = refinement_changes.Finish();
      local_ba_scope.modified_point_ids.insert(changed_points.begin(), changed_points.end());
      ScopedPoint3DChanges filter_changes(reconstruction);
      filtered = FilterObservationsAndPointsInImages(
          options.observation_filter, adjusted_image_ids, reconstruction,
          &local_ba_scope.modified_point_ids);
      const auto filtered_points = filter_changes.Finish();
      local_ba_scope.modified_point_ids.insert(filtered_points.begin(), filtered_points.end());
      summary.observation_filter_seconds += ElapsedSeconds(stage_start);
      auto& total = summary.observation_filter;
      total.success = filtered.success;
      total.message = filtered.message;
      total.num_removed_observations += filtered.num_removed_observations;
      total.num_removed_points += filtered.num_removed_points;
      total.num_reprojection_observations += filtered.num_reprojection_observations;
      total.num_invalid_observations += filtered.num_invalid_observations;
      total.num_short_track_points += filtered.num_short_track_points;
      total.num_low_angle_points += filtered.num_low_angle_points;
      if (!filtered.success) {
        summary.message = filtered.message;
        return summary;
      }
    } else {
      const auto changed_points = refinement_changes.Finish();
      local_ba_scope.modified_point_ids.insert(changed_points.begin(), changed_points.end());
    }
    const auto adjusted = summary.local_bundle_adjustment.num_residuals;
    summary.last_local_refinement_change = adjusted == 0 ? 0.0 :
        static_cast<double>(post_ba_tracks.num_completed_observations +
                            post_ba_tracks.num_merged_observations +
                            filtered.num_removed_observations) / adjusted;
    if (summary.last_local_refinement_change < options.ba_local_max_refinement_change) break;
    local_ba_options.loss_scale = 0.0;
  }

  summary.success = true;
  summary.message = "Image registered and locally refined";
  return summary;
}

IncrementalTriangulatorSummary RetriangulateLocalNeighborhood(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    const ObservationFilterOptions* local_filter_options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* triangulator_cache,
    const std::set<ImageId>* local_window) {
  IncrementalTriangulatorSummary summary;
  summary.image_id = image_id;
  if (reconstruction == nullptr ||
      !reconstruction->IsImageRegistered(image_id)) {
    summary.message = "Image is not registered";
    return summary;
  }

  // This operation only allocates automatic IDs; merges retain an existing ID.
  // Follow COLMAP's scoped changed-point work sets without copying all points.
  const Point3DId first_created_point_id = reconstruction->NextPoint3DId();
  std::set<ImageId> local_image_ids;
  local_image_ids.insert(image_id);
  for (const MatchGraphEdge& edge : FindVerifiedNeighbors(graph, image_id)) {
    const ImageId neighbor_id = edge.OtherImageId(image_id);
    if (reconstruction->IsImageRegistered(neighbor_id) &&
        (local_window == nullptr || local_window->count(neighbor_id) != 0)) {
      local_image_ids.insert(neighbor_id);
    }
  }

  // COLMAP 的增量三角化会围绕新注册图像反复 complete/merge/create。这里先做
  // 保守的局部版本：新图像和强邻居都重跑一轮，使邻居-邻居遗漏 track 进入局部 BA。
  for (const ImageId local_image_id : local_image_ids) {
    // TriangulateRegisteredImage already continues observations before
    // creating points and runs completion/merge iterations after creation.
    // Calling CompleteTracksForRegisteredImage immediately beforehand scans
    // the same verified pairs twice for every local neighbor.
    const IncrementalTriangulatorSummary triangulation =
        TriangulateRegisteredImage(graph, local_image_id, options,
                                   reconstruction, triangulator_cache);
    summary.num_created_points += triangulation.num_created_points;
    summary.num_created_observations +=
        triangulation.num_created_observations;
    summary.num_completed_observations +=
        triangulation.num_completed_observations;
    summary.num_merged_points += triangulation.num_merged_points;
    summary.num_merged_observations +=
        triangulation.num_merged_observations;
  }
  const IncrementalTriangulatorSummary completion =
      CompleteTracksForRegisteredImage(graph, image_id, options,
                                       reconstruction);
  summary.num_completed_observations += completion.num_completed_observations;
  summary.num_merged_points += completion.num_merged_points;
  summary.num_merged_observations += completion.num_merged_observations;

  if (local_filter_options != nullptr) {
    const ObservationFilterSummary filter_summary = FilterNewlyCreatedPoints(
        first_created_point_id, *local_filter_options, reconstruction);
    summary.num_filtered_points = filter_summary.num_removed_points;
    summary.num_filtered_observations =
        filter_summary.num_removed_observations;
  }

  summary.message = "Local retriangulation finished";
  return summary;
}

IncrementalTriangulatorSummary RetriangulateAllRegisteredImages(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* triangulator_cache) {
  IncrementalTriangulatorSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  const std::set<ImageId> image_ids =
      reconstruction->RegisteredImageIds();
  for (const ImageId image_id : image_ids) {
    const IncrementalTriangulatorSummary update =
        TriangulateRegisteredImage(graph, image_id, options, reconstruction,
                                   triangulator_cache);
    AccumulateTriangulationSummary(update, &summary);
  }
  summary.message = "Global retriangulation finished";
  return summary;
}

std::optional<ImageId> SelectNextImage(
    const MatchGraph& graph,
    const IncrementalMapperOptions& options,
    const Reconstruction& reconstruction) {
  const std::vector<ImageCandidate> candidates =
      CollectImageCandidates(graph, options, reconstruction);
  if (candidates.empty()) {
    return std::nullopt;
  }
  return candidates.front().image_id;
}

IncrementalMappingSummary RunIncrementalMapping(
    const MatchGraph& graph,
    const IncrementalMappingOptions& options,
    Reconstruction* reconstruction, InitialPairTrialState* initial_trial_state) {
  IncrementalMappingSummary summary;
  if (reconstruction == nullptr) {
    summary.message = "Reconstruction is null";
    return summary;
  }

  if (!std::isfinite(options.ba_global_frames_ratio) || options.ba_global_frames_ratio <= 1.0 ||
      !std::isfinite(options.ba_global_points_ratio) || options.ba_global_points_ratio <= 1.0 ||
      options.ba_global_frames_freq == 0 || options.ba_global_points_freq == 0) {
    summary.message = "Global BA growth ratios must be finite and > 1; frequencies must be positive";
    return summary;
  }
  if (options.mapper.ba_local_max_refinements == 0 ||
      !std::isfinite(options.mapper.ba_local_max_refinement_change) ||
      options.mapper.ba_local_max_refinement_change < 0.0) {
    summary.message = "Invalid local refinement limit or change threshold";
    return summary;
  }
  const IncrementalTriangulatorCache triangulator_cache(graph);
  if (reconstruction->RegisteredImageIds().empty()) {
    if (options.initialization.image_id1 != kInvalidImageId &&
        options.initialization.image_id1 == options.initialization.image_id2) {
      summary.message = "Initial images must be different";
      return summary;
    }
    if (options.initialization.max_num_reg_trials == 0) {
      summary.message = "Initial per-image trial limit must be positive";
      return summary;
    }
    if (options.max_num_initialization_trials == 0) {
      summary.message = "Initial reconstruction trial budget must be positive";
      return summary;
    }
    if (options.run_global_bundle_adjustment) {
      const auto invalid = options.global_bundle_adjustment.solver.ValidationError(
          options.global_bundle_adjustment.max_num_iterations);
      if (!invalid.empty()) { summary.message = invalid; return summary; }
    }
    IncrementalInitializerOptions initialization_options =
        options.initialization;
    if (!options.allowed_image_ids.empty()) {
      initialization_options.allowed_image_ids = options.allowed_image_ids;
    }
    const auto initialization_start = MappingClock::now();
    InitialPairTrialState owned_trial_state;
    auto* trial_state = initial_trial_state ? initial_trial_state : &owned_trial_state;
    std::size_t num_model_trials = 0;
    summary.initialization = InitializeMappingFromCandidates(
        graph, initialization_options, triangulator_cache, options, trial_state, &num_model_trials, &summary, reconstruction);
    // COLMAP retries twice, halving inliers and then angle in each round.
    // Only the working copy is relaxed; the user's configured thresholds and
    // cache identity remain unchanged.
    for (int relaxation = 0; options.relax_initialization_constraints &&
         !summary.initialization.success && relaxation < 4; ++relaxation) {
      trial_state->Clear();  // COLMAP ResetInitializationStats for each relaxation.
      num_model_trials = 0;
      if (relaxation % 2 == 0) initialization_options.min_num_inliers =
          std::max<std::size_t>(1, initialization_options.min_num_inliers / 2);
      else initialization_options.min_triangulation_angle_deg *= 0.5;
      auto attempt = InitializeMappingFromCandidates(graph, initialization_options,
          triangulator_cache, options, trial_state, &num_model_trials, &summary, reconstruction);
      attempt.num_initialization_attempts += summary.initialization.num_initialization_attempts;
      attempt.attempted_initial_pairs.insert(attempt.attempted_initial_pairs.begin(),
          summary.initialization.attempted_initial_pairs.begin(), summary.initialization.attempted_initial_pairs.end());
      summary.initialization = std::move(attempt);
    }
    summary.initialization_seconds = ElapsedSeconds(initialization_start);
    if (!summary.initialization.success) {
      summary.num_registered_images =
          reconstruction->RegisteredImageIds().size();
      summary.num_points3d = reconstruction->Points3D().size();
      summary.message = summary.initialization.message;
      return summary;
    }
  }

  const auto candidate_neighbors = BuildCandidateNeighborIndex(graph);
  // Use the same public traversal budget before and after BA. A correspondence
  // outside the reprojection gate can become attachable only after refinement.
  const auto& completion_options = options.mapper.triangulation;
  std::set<refinement_internal::ObservationKey> previously_removed;
  const bool trace_refinement = options.collect_refinement_diagnostics || VLOG_IS_ON(1);
  const auto refine_global = [&]() -> bool {
    ReleaseUnusedMemory();
    if (options.run_global_track_completion &&
        reconstruction->RegisteredImageIds().size() >= 2) {
      auto stage_start = MappingClock::now();
      const auto completed_tracks = CompleteAndMergeAllTracks(
          graph, options.mapper.triangulation, reconstruction,
          &triangulator_cache);
      AccumulateTriangulationSummary(completed_tracks, &summary.global_track_completion);
      summary.global_track_completion_seconds += ElapsedSeconds(stage_start);
      stage_start = MappingClock::now();
      const auto retriangulated = RetriangulateAllRegisteredImages(
          graph, options.mapper.triangulation, reconstruction,
          &triangulator_cache);
      AccumulateTriangulationSummary(retriangulated, &summary.global_retriangulation);
      summary.global_retriangulation_seconds += ElapsedSeconds(stage_start);
      stage_start = MappingClock::now();
      const IncrementalTriangulatorSummary post_retriangulation_tracks =
          CompleteAndMergeAllTracks(graph, completion_options,
                                    reconstruction, &triangulator_cache);
      AccumulateTriangulationSummary(post_retriangulation_tracks,
                                     &summary.global_track_completion);
      summary.global_track_completion_seconds += ElapsedSeconds(stage_start);
    }

    const auto count_observations = [&]() {
      std::size_t count = 0;
      for (const auto& [id, point] : reconstruction->Points3D()) {
        count += point.track.Length();
      }
      return count;
    };
    const std::size_t max_refinements = options.run_final_global_bundle_adjustment
        ? options.max_num_global_refinements
        : std::min<std::size_t>(1, options.max_num_global_refinements);
    if (options.run_global_bundle_adjustment &&
        reconstruction->RegisteredImageIds().size() >= 2) {
      for (std::size_t pass = 0; pass < max_refinements &&
           !reconstruction->Points3D().empty(); ++pass) {
        const auto observations_before = count_observations();
        GlobalRefinementDiagnostics row;
        row.iteration = summary.num_global_refinements + 1;
        row.observations_before = observations_before;
        const auto before_state = trace_refinement ? refinement_internal::Capture(*reconstruction)
                                                  : refinement_internal::Snapshot{};
        ReleaseUnusedMemory();
        auto stage_start = MappingClock::now();
        FilterObservationsWithNegativeDepth(reconstruction);
        const auto ba = AdjustGlobalBundle(MappingGlobalBundleOptions(options, *reconstruction),
                                          reconstruction);
        if (summary.num_global_refinements == 0) {
          summary.global_bundle_adjustment = ba;
          summary.global_bundle_adjustment_seconds += ElapsedSeconds(stage_start);
        } else {
          summary.final_global_bundle_adjustment = ba;
          summary.final_global_bundle_adjustment_seconds += ElapsedSeconds(stage_start);
        }
        ++summary.num_global_refinements;
        summary.num_global_ba_invalid_steps += ba.num_invalid_steps;
        if (!ba.success) {
          summary.num_registered_images = reconstruction->RegisteredImageIds().size();
          summary.num_points3d = reconstruction->Points3D().size();
          summary.message = ba.message;
          return false;
        }
        row.initial_cost = ba.initial_cost;
        if (!options.global_bundle_adjustment.use_position_priors)
          NormalizeReconstruction(reconstruction);
        row.final_cost = ba.final_cost;
        const auto ba_state = trace_refinement ? refinement_internal::Capture(*reconstruction)
                                              : refinement_internal::Snapshot{};

        std::size_t changed_observations = 0;
        if (options.run_global_track_completion) {
          stage_start = MappingClock::now();
          const auto tracks = CompleteAndMergeAllTracks(
              graph, completion_options, reconstruction, &triangulator_cache);
          AccumulateTriangulationSummary(tracks, &summary.global_track_completion);
          changed_observations += tracks.num_completed_observations +
                                  tracks.num_merged_observations;
          row.completed_events += tracks.num_completed_observations;
          row.merged_observation_events += tracks.num_merged_observations;
          summary.global_track_completion_seconds += ElapsedSeconds(stage_start);
        }
        if (pass == 0 && options.run_pose_guided_track_completion) {
          stage_start = MappingClock::now();
          const auto guided_tracks = CompleteTracksByPoseAndDescriptors(
              graph, completion_options, reconstruction);
          const double guided_seconds = ElapsedSeconds(stage_start);
          summary.pose_guided_track_completion_seconds += guided_seconds;
          AccumulateTriangulationSummary(guided_tracks, &summary.pose_guided_track_completion);
          AccumulateTriangulationSummary(guided_tracks, &summary.global_track_completion);
          changed_observations +=
              guided_tracks.num_completed_observations + guided_tracks.num_merged_observations;
          row.completed_events += guided_tracks.num_completed_observations;
          row.merged_observation_events += guided_tracks.num_merged_observations;
          summary.global_track_completion_seconds += guided_seconds;
        }
        row.observations_after_completion = count_observations();
        const auto completed_state = trace_refinement ? refinement_internal::Capture(*reconstruction)
                                                     : refinement_internal::Snapshot{};
        if (options.run_global_observation_filter) {
          const auto observations_before_filter = count_observations();
          stage_start = MappingClock::now();
          const auto filtered = FilterObservationsAndPoints(
              options.global_observation_filter, reconstruction);
          summary.global_observation_filter_seconds += ElapsedSeconds(stage_start);
          summary.global_observation_filter.num_removed_observations += filtered.num_removed_observations;
          summary.global_observation_filter.num_removed_points += filtered.num_removed_points;
          summary.global_observation_filter.success = filtered.success;
          summary.global_observation_filter.message = filtered.message;
          changed_observations += observations_before_filter - count_observations();
          row.filtered_observations = observations_before_filter - count_observations();
          row.removed_reprojection_observations = filtered.num_reprojection_observations;
          row.removed_invalid_observations = filtered.num_invalid_observations;
          row.removed_short_track_points = filtered.num_short_track_points;
          row.removed_low_angle_points = filtered.num_low_angle_points;
          if (!filtered.success) {
            summary.message = filtered.message;
            return false;
          }
        }
        summary.last_global_refinement_change = observations_before == 0 ? 0.0 :
            static_cast<double>(changed_observations) / observations_before;
        row.observations_after_filter = count_observations();
        row.event_change_ratio = summary.last_global_refinement_change;
        if (trace_refinement) {
          refinement_internal::Record(before_state, ba_state, completed_state,
              refinement_internal::Capture(*reconstruction), &previously_removed,
              &row, &summary.global_observation_changes,
              options.global_observation_filter.max_reprojection_error_px);
        }
        summary.global_refinement_diagnostics.push_back(row);
        // Newly completed/merged tracks and removed outliers change the BA
        // objective. Refine it again until the observation set stabilizes.
        if (summary.last_global_refinement_change < options.global_refinement_change) break;
      }
    }

    // Final BA and subsequent track merges move points/intrinsics after the
    // earlier filter. Enforce the configured error/angle gates on saved output.
    if (options.run_global_observation_filter &&
        !reconstruction->Points3D().empty()) {
      const auto stage_start = MappingClock::now();
      const auto final_filter = FilterObservationsAndPoints(
          options.global_observation_filter, reconstruction);
      summary.global_observation_filter_seconds += ElapsedSeconds(stage_start);
      summary.global_observation_filter.num_removed_observations +=
          final_filter.num_removed_observations;
      summary.global_observation_filter.num_removed_points +=
          final_filter.num_removed_points;
      summary.global_observation_filter.success = final_filter.success;
      if (!final_filter.success) {
        summary.message = final_filter.message;
        return false;
      }
    }
    return true;
  };

  std::map<ImageId, std::size_t> registration_trials;
  std::size_t previous_global_images = reconstruction->RegisteredImageIds().size();
  std::size_t previous_global_points = reconstruction->Points3D().size();
  bool recovery_attempted = false;
  bool refined_since_registration = false;
  while (summary.registrations.size() < options.max_num_registrations) {
    const auto candidate_start = MappingClock::now();
    const auto candidates = CollectImageCandidates(
        graph, options.mapper, *reconstruction, &options.allowed_image_ids,
        &registration_trials, options.max_num_registration_trials_per_image,
        &candidate_neighbors);
    summary.candidate_selection_seconds += ElapsedSeconds(candidate_start);
    bool registered_any_image = false;
    for (const auto& candidate : candidates) {
      IncrementalMapperOptions attempt_options = options.mapper;
      attempt_options.registration = RegistrationOptionsForAttempt(
          options.mapper.registration, candidate.num_registration_attempts, options);
      auto registration = RegisterNextImage(graph, candidate.image_id,
          attempt_options, reconstruction, &triangulator_cache);
      if (!registration.success) {
        const auto attempt = ++registration_trials[candidate.image_id];
        summary.failed_registrations.push_back(MakeFailureSummary(
            candidate, attempt, attempt_options.registration.max_reprojection_error_px,
            registration.registration, registration.message));
        continue;
      }
      registration.registration_order = summary.registrations.size() + 1;
      CopyCandidateStatsToSummary(candidate, &registration);
      summary.registrations.push_back(std::move(registration));
      if (recovery_attempted) ++summary.num_registration_recovery_successes;
      registered_any_image = true;
      recovery_attempted = false;
      refined_since_registration = false;
      break;
    }
    if (registered_any_image) {
      const auto images = reconstruction->RegisteredImageIds().size();
      const auto points = reconstruction->Points3D().size();
      const bool grew = images >= options.ba_global_frames_ratio * previous_global_images ||
          (images >= previous_global_images && images - previous_global_images >= options.ba_global_frames_freq) ||
          points >= options.ba_global_points_ratio * previous_global_points ||
          (points >= previous_global_points && points - previous_global_points >= options.ba_global_points_freq);
      if (grew && options.run_global_bundle_adjustment && options.max_num_global_refinements > 0 && points > 0) {
        if (!refine_global()) return summary;
        ++summary.num_growth_global_refinements;
        previous_global_images = reconstruction->RegisteredImageIds().size();
        previous_global_points = reconstruction->Points3D().size();
        refined_since_registration = true;
      }
      continue;
    }

    // Like COLMAP, refine a stalled model once and retry with the improved
    // tracks/geometry. Keep configured inlier gates and per-image trial caps.
    bool has_connected_remainder = false;
    for (const auto& [image_id, image] : reconstruction->Images()) {
      if (!image.registered && (options.allowed_image_ids.empty() ||
          options.allowed_image_ids.count(image_id) != 0) &&
          AnalyzeImageCandidate(graph, image_id, *reconstruction,
                                candidate_neighbors).num_registered_neighbors > 0) {
        has_connected_remainder = true;
        break;
      }
    }
    bool can_relax_retry = false;
    for (const auto& candidate : candidates) {
      const auto failures = registration_trials[candidate.image_id];
      if (failures == 0 ||
          (options.max_num_registration_trials_per_image != 0 &&
           failures >= options.max_num_registration_trials_per_image)) continue;
      const auto previous = RegistrationOptionsForAttempt(
          options.mapper.registration, failures - 1, options);
      const auto next = RegistrationOptionsForAttempt(
          options.mapper.registration, failures, options);
      can_relax_retry |= next.max_reprojection_error_px > previous.max_reprojection_error_px;
    }
    if (recovery_attempted || !has_connected_remainder ||
        !options.run_global_bundle_adjustment || options.max_num_global_refinements == 0 ||
        reconstruction->Points3D().empty()) {
      // An explicitly enabled threshold schedule changes the next attempt,
      // even without a global geometry update. Preserve that bounded path.
      if (can_relax_retry) continue;
      break;
    }
    ++summary.num_registration_recovery_attempts;
    LOG(INFO) << "Registration stalled: refining "
              << reconstruction->RegisteredImageIds().size()
              << " registered images before retry";
    if (!refine_global()) return summary;
    recovery_attempted = true;
    refined_since_registration = true;
  }
  if (!refined_since_registration && !refine_global()) return summary;
  summary.num_registered_images = reconstruction->RegisteredImageIds().size();
  summary.num_points3d = reconstruction->Points3D().size();
  summary.success = summary.num_registered_images >= 2;
  summary.message = summary.success ? "Incremental mapping finished"
                                    : "Incremental mapping failed";
  return summary;
}

}  // namespace colite
