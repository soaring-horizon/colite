#include "colite/sfm/image_registrar.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <vector>
#include <memory>

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <Eigen/Geometry>

#include "colite/core/geometry.h"
#include "colite/core/ransac.h"
#include "colmap/epnp.h"
#include "ceres_manifold_compat.h"
#include "reprojection_cost.h"
#include "colite/sfm/observation_manager.h"

#ifdef COLITE_HAS_POSELIB
#include <PoseLib/solvers/p3p.h>
#else
#include "poselib/p3p.h"
#endif

namespace colite {
namespace {

constexpr std::size_t kMinEpnpPoints = 4;
constexpr std::size_t kMinPnPPoints = 3;

struct PnPCorrespondence {
  Point3DId point3d_id = kInvalidPoint3DId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
};

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

bool PairContainsImage(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id || pair.image_id2 == image_id;
}

ImageId OtherImageId(const ImagePair& pair, ImageId image_id) {
  return pair.image_id1 == image_id ? pair.image_id2 : pair.image_id1;
}

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.HasUsableCorrespondences();
}

void AddCorrespondenceIfValid(const FeatureMatch& match,
                              bool target_is_first,
                              Image& target_image,
                              const Image& neighbor_image,
                              const Reconstruction& reconstruction,
                              std::map<Point2DIdx, PnPCorrespondence>*
                                  correspondences_by_target_idx) {
  const Point2DIdx target_idx = target_is_first ? match.idx1 : match.idx2;
  const Point2DIdx neighbor_idx = target_is_first ? match.idx2 : match.idx1;
  if (target_idx == kInvalidPoint2DIdx ||
      neighbor_idx == kInvalidPoint2DIdx ||
      target_idx >= target_image.points2d.size() ||
      neighbor_idx >= neighbor_image.points2d.size()) {
    return;
  }

  const Point2D& neighbor_point2d = neighbor_image.points2d[neighbor_idx];
  if (!neighbor_point2d.HasPoint3D() ||
      !reconstruction.ExistsPoint3D(neighbor_point2d.point3d_id)) {
    return;
  }
  if (target_image.points2d[target_idx].HasPoint3D() ||
      correspondences_by_target_idx->find(target_idx) !=
          correspondences_by_target_idx->end()) {
    return;
  }

  const Point3D& point3d =
      reconstruction.GetPoint3D(neighbor_point2d.point3d_id);
  correspondences_by_target_idx->emplace(
      target_idx,
      PnPCorrespondence{point3d.id, target_idx, point3d.xyz,
                        target_image.points2d[target_idx].xy});
}

std::vector<PnPCorrespondence> CollectPnPCorrespondences(
    const MatchGraph& graph,
    ImageId image_id,
    Reconstruction* reconstruction) {
  std::vector<PnPCorrespondence> correspondences;
  if (!reconstruction->ExistsImage(image_id) ||
      reconstruction->IsImageRegistered(image_id)) {
    return correspondences;
  }

  const FeatureSet* features = FindFeatures(graph, image_id);
  if (features == nullptr) {
    return correspondences;
  }

  Image& target_image = reconstruction->GetImage(image_id);
  SyncImagePoints2DFromFeatures(*features, &target_image);

  std::map<Point2DIdx, PnPCorrespondence> unique_correspondences;
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    if (!PairContainsImage(pair, image_id) || !IsUsableGeometry(geometry)) {
      continue;
    }

    const ImageId neighbor_id = OtherImageId(pair, image_id);
    if (!reconstruction->IsImageRegistered(neighbor_id)) {
      continue;
    }

    const Image& neighbor_image = reconstruction->GetImage(neighbor_id);
    const bool target_is_first = pair.image_id1 == image_id;
    for (const FeatureMatch& match : geometry.inlier_matches) {
      AddCorrespondenceIfValid(match, target_is_first, target_image,
                               neighbor_image, *reconstruction,
                               &unique_correspondences);
    }
  }

  correspondences.reserve(unique_correspondences.size());
  for (const auto& [target_idx, correspondence] : unique_correspondences) {
    (void)target_idx;
    if (!TryUnprojectPixel(reconstruction->GetCamera(target_image.camera_id),
                           correspondence.xy, 1.0)) continue;
    correspondences.push_back(correspondence);
  }
  return correspondences;
}

std::vector<std::size_t> AllIndices(const std::size_t count) {
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

std::vector<std::size_t> SampleUniqueIndices(std::mt19937* rng,
                                             const std::size_t count,
                                             const std::size_t sample_size) {
  std::vector<std::size_t> indices = AllIndices(count);
  std::shuffle(indices.begin(), indices.end(), *rng);
  indices.resize(sample_size);
  return indices;
}

Eigen::Vector2d NormalizedImagePoint(const Camera& camera,
                                     const Eigen::Vector2d& pixel) {
  const Eigen::Vector3d ray = UnprojectPixel(camera, pixel, 1.0);
  return {ray.x() / ray.z(), ray.y() / ray.z()};
}

std::vector<std::size_t> InliersForPose(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const Pose3& pose,
    const double max_reprojection_error_px,
    double* inlier_squared_error) {
  std::vector<std::size_t> inliers;
  inliers.reserve(correspondences.size());
  *inlier_squared_error = 0;
  for (std::size_t idx = 0; idx < correspondences.size(); ++idx) {
    const std::optional<double> error =
        ReprojectionError(camera, pose, correspondences[idx].xyz,
                          correspondences[idx].xy);
    if (error && *error <= max_reprojection_error_px) {
      inliers.push_back(idx);
      *inlier_squared_error += *error * *error;
    }
  }
  return inliers;
}

std::optional<Pose3> EstimatePoseEpnp(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& indices) {
  std::vector<Eigen::Vector2d> pixels;
  std::vector<Eigen::Vector3d> points;
  pixels.reserve(indices.size());
  points.reserve(indices.size());
  for (const auto index : indices) {
    pixels.push_back(correspondences[index].xy);
    points.push_back(correspondences[index].xyz);
  }
  return EstimateAbsolutePoseEpnp(camera, pixels, points);
}

struct PnPRansacResult {
  Pose3 pose;
  std::vector<std::size_t> inlier_indices;
  std::size_t num_inliers_before_final_refinement = 0;
  std::size_t num_iterations = 0;
  double inlier_squared_error = std::numeric_limits<double>::infinity();
};

bool BetterSupport(std::size_t count, double squared_error,
                   const PnPRansacResult& best) {
  return count > best.inlier_indices.size() ||
         (count == best.inlier_indices.size() &&
          squared_error < best.inlier_squared_error);
}

std::vector<Pose3> EstimateMinimalPoses(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& indices) {
  std::vector<Pose3> poses;
  if (indices.size() != 3) {
    return poses;
  }
  std::vector<Eigen::Vector3d> rays;
  std::vector<Eigen::Vector3d> points3d;
  rays.reserve(3);
  points3d.reserve(3);
  for (const std::size_t idx : indices) {
    rays.push_back(
        NormalizedImagePoint(camera, correspondences[idx].xy)
            .homogeneous()
            .normalized());
    points3d.push_back(correspondences[idx].xyz);
  }
#ifdef COLITE_HAS_POSELIB
  namespace solver = poselib;
#else
  namespace solver = third_party::poselib;
#endif
  std::vector<solver::CameraPose> solutions;
  solver::p3p(rays, points3d, &solutions);
  poses.reserve(solutions.size());
  for (const solver::CameraPose& solution : solutions) {
    const Eigen::Matrix<double, 3, 4> matrix = solution.Rt();
    Pose3 pose;
    pose.q_cw = Eigen::Quaterniond(matrix.leftCols<3>()).normalized();
    pose.t_cw = matrix.col(3);
    poses.push_back(pose);
  }
  return poses;
}

PnPRansacResult EstimatePoseRansac(
    const Camera& camera,
    const std::vector<PnPCorrespondence>& correspondences,
    const ImageRegistrarOptions& options) {
  PnPRansacResult best;
  if (correspondences.size() < kMinPnPPoints) {
    return best;
  }

  std::mt19937 rng(options.random_seed < 0 ? std::random_device{}()
                                         : static_cast<unsigned>(options.random_seed));
  const int min_iterations = std::max(1, options.min_num_iterations);
  // Use the same prior and finite-population stopping rule as two-view
  // RANSAC. A perfect hypothesis must still satisfy the minimum trial budget.
  const int max_iterations = std::clamp(
      RansacTrials(static_cast<std::size_t>(options.min_inlier_ratio * 100000),
                   100000, kMinPnPPoints, options.confidence,
                   std::max(min_iterations, options.max_num_iterations)),
      min_iterations, std::max(min_iterations, options.max_num_iterations));
  int required_iterations = max_iterations;
  for (int iteration = 0; iteration < required_iterations; ++iteration) {
    best.num_iterations = static_cast<std::size_t>(iteration + 1);
    const std::vector<std::size_t> sample =
        correspondences.size() == kMinPnPPoints
            ? AllIndices(correspondences.size())
            : SampleUniqueIndices(&rng, correspondences.size(),
                                  kMinPnPPoints);
    for (const Pose3& pose :
         EstimateMinimalPoses(camera, correspondences, sample)) {
      double squared_error;
      std::vector<std::size_t> inliers = InliersForPose(
          camera, correspondences, pose, options.max_reprojection_error_px,
          &squared_error);
      if (BetterSupport(inliers.size(), squared_error, best)) {
        best.pose = pose;
        best.inlier_indices = std::move(inliers);
        best.inlier_squared_error = squared_error;

        // As in COLMAP's LO-RANSAC, EPnP fits all current inliers and can
        // expand support recursively; a tied support must lower pixel error.
        for (int local_trial = 0;
             local_trial < 10 &&
             best.inlier_indices.size() >= kMinEpnpPoints;
             ++local_trial) {
          const auto previous_count = best.inlier_indices.size();
          const std::optional<Pose3> local =
              EstimatePoseEpnp(camera, correspondences, best.inlier_indices);
          if (local) {
            double local_error;
            std::vector<std::size_t> local_inliers = InliersForPose(
                camera, correspondences, *local,
                options.max_reprojection_error_px, &local_error);
            if (BetterSupport(local_inliers.size(), local_error, best)) {
              best.pose = *local;
              best.inlier_indices = std::move(local_inliers);
              best.inlier_squared_error = local_error;
            }
          }
          if (best.inlier_indices.size() <= previous_count) break;
        }

        required_iterations = std::max(
            iteration + 1,
            std::clamp(RansacTrials(best.inlier_indices.size(),
                                   correspondences.size(), kMinPnPPoints,
                                   options.confidence, max_iterations),
                       min_iterations, max_iterations));
      }
    }
  }

  best.num_inliers_before_final_refinement = best.inlier_indices.size();
  return best;
}

void LinkInlierCorrespondencesToImage(
    ImageId image_id,
    const std::vector<PnPCorrespondence>& correspondences,
    const std::vector<std::size_t>& inlier_indices,
    Reconstruction* reconstruction) {
  const Image& image = reconstruction->GetImage(image_id);
  const Camera& camera = reconstruction->GetCamera(image.camera_id);
  std::map<Point3DId, std::pair<std::size_t, double>> best_by_point;
  for (const std::size_t correspondence_idx : inlier_indices) {
    const PnPCorrespondence& correspondence =
        correspondences[correspondence_idx];
    if (correspondence.point2d_idx >= image.points2d.size() ||
        !reconstruction->ExistsPoint3D(correspondence.point3d_id)) {
      continue;
    }
    const double error =
        ReprojectionError(
            camera, *image.cam_from_world,
            reconstruction->GetPoint3D(correspondence.point3d_id).xyz,
            image.points2d[correspondence.point2d_idx].xy)
            .value_or(std::numeric_limits<double>::infinity());
    const auto best_it = best_by_point.find(correspondence.point3d_id);
    if (best_it == best_by_point.end() || error < best_it->second.second ||
        (error == best_it->second.second &&
         correspondence.point2d_idx <
             correspondences[best_it->second.first].point2d_idx)) {
      best_by_point[correspondence.point3d_id] = {correspondence_idx, error};
    }
  }

  ObservationManager observation_manager(reconstruction);
  for (const auto& [point3d_id, selected] : best_by_point) {
    const PnPCorrespondence& correspondence =
        correspondences[selected.first];
    observation_manager.AddObservation(point3d_id, image_id,
                                       correspondence.point2d_idx);
  }
}

}  // namespace

std::optional<Pose3> EstimateAbsolutePoseEpnp(
    const Camera& camera, const std::vector<Eigen::Vector2d>& points2d,
    const std::vector<Eigen::Vector3d>& points3d) {
  if (points2d.size() < kMinEpnpPoints || points2d.size() != points3d.size() ||
      !camera.VerifyParams()) return std::nullopt;
  std::vector<third_party::colmap::Point2DWithRay> observations;
  observations.reserve(points2d.size());
  for (std::size_t i = 0; i < points2d.size(); ++i) {
    if (!points2d[i].allFinite() || !points3d[i].allFinite()) return std::nullopt;
    const auto ray = TryPixelToBearing(camera, points2d[i]);
    if (!ray) return std::nullopt;
    observations.push_back({points2d[i], *ray});
  }
  third_party::colmap::EPNPEstimator estimator(
      [&camera](const Eigen::Vector3d& point) {
        return ProjectWorldPoint(camera, Pose3{}, point);
      });
  std::vector<third_party::colmap::Matrix3x4d> models;
  estimator.Estimate(observations, points3d, &models);
  if (models.empty() || !models[0].allFinite()) return std::nullopt;
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond(models[0].leftCols<3>()).normalized();
  pose.t_cw = models[0].col(3);
  if (!pose.q_cw.coeffs().allFinite()) return std::nullopt;
  return pose;
}

AbsolutePoseRefinementSummary RefineAbsolutePose(
    const AbsolutePoseRefinementOptions& options, const Camera& camera,
    const std::vector<Eigen::Vector2d>& points2d,
    const std::vector<Eigen::Vector3d>& points3d,
    const std::vector<char>& inlier_mask, Pose3* pose) {
  AbsolutePoseRefinementSummary summary;
  if (pose == nullptr || points2d.size() != points3d.size() ||
      points2d.size() != inlier_mask.size() || !camera.VerifyParams() ||
      options.max_num_iterations < 0 || !std::isfinite(options.gradient_tolerance) ||
      options.gradient_tolerance < 0 || !std::isfinite(options.loss_function_scale) ||
      options.loss_function_scale <= 0) {
    summary.message = "Invalid absolute pose refinement inputs; pose unchanged";
    return summary;
  }
  if (!pose->q_cw.coeffs().allFinite() || !pose->t_cw.allFinite() ||
      !std::isfinite(pose->q_cw.norm()) || pose->q_cw.norm() <= 0) {
    summary.message = "Invalid initial absolute pose; pose unchanged";
    return summary;
  }
  for (std::size_t i = 0; i < inlier_mask.size(); ++i) {
    if (!inlier_mask[i]) continue;
    if (!points2d[i].allFinite() || !points3d[i].allFinite()) {
      summary.message = "Non-finite absolute pose inlier; pose unchanged";
      return summary;
    }
    ++summary.num_observations;
  }
  if (summary.num_observations == 0) {
    summary.message = "Absolute pose refinement has no inliers; pose unchanged";
    return summary;
  }
  Eigen::Quaterniond q = pose->q_cw.normalized();
  Eigen::Vector3d translation = pose->t_cw;
  auto fixed_points = points3d;
  auto fixed_camera = camera.params;
  auto loss = std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  ceres_compat::AddEigenQuaternionParameterBlock(&problem, q.coeffs().data());
  problem.AddParameterBlock(translation.data(), 3);
  for (std::size_t i = 0; i < inlier_mask.size(); ++i) {
    if (!inlier_mask[i]) continue;
    problem.AddResidualBlock(bundle_internal::CreateReprojectionCost(
        camera.model_id, fixed_camera.size(), points2d[i]), loss.get(),
        q.coeffs().data(), translation.data(), fixed_points[i].data(), fixed_camera.data());
    problem.SetParameterBlockConstant(fixed_points[i].data());
  }
  problem.SetParameterBlockConstant(fixed_camera.data());
  ceres::Solver::Options solver;
  solver.max_num_iterations = options.max_num_iterations;
  solver.gradient_tolerance = options.gradient_tolerance;
  solver.linear_solver_type = ceres::DENSE_QR;
  solver.logging_type = ceres::SILENT;
  solver.minimizer_progress_to_stdout = false;
  solver.num_threads = 1;
  ceres::Solver::Summary solved;
  ceres::Solve(solver, &problem, &solved);
  summary.num_iterations = static_cast<int>(solved.iterations.size());
  summary.initial_cost = solved.initial_cost;
  summary.final_cost = solved.final_cost;
  summary.message = solved.BriefReport();
  summary.success = solved.IsSolutionUsable() && std::isfinite(solved.initial_cost) &&
      std::isfinite(solved.final_cost) && solved.final_cost >= 0 &&
      solved.final_cost <= solved.initial_cost + 1e-10 * std::max(1.0, solved.initial_cost) &&
      q.coeffs().allFinite() && std::isfinite(q.norm()) && q.norm() > 0 && translation.allFinite();
  if (summary.success) {
    pose->q_cw = q.normalized();
    pose->t_cw = translation;
  } else {
    summary.message += "; pose unchanged (unusable or non-finite solution)";
  }
  VLOG(1) << "Absolute pose refinement: " << summary.message;
  return summary;
}

ImageRegistrationSummary RegisterImageWithExistingPoints(
    const MatchGraph& graph,
    ImageId image_id,
    const ImageRegistrarOptions& options,
    Reconstruction* reconstruction) {
  ImageRegistrationSummary summary;
  summary.image_id = image_id;

  if (!std::isfinite(options.min_inlier_ratio) || options.min_inlier_ratio < 0.0 ||
      options.min_inlier_ratio > 1.0) {
    summary.message = "Invalid PnP minimum inlier ratio";
    return summary;
  }

  if (reconstruction == nullptr || !reconstruction->ExistsImage(image_id)) {
    summary.message = "Image is missing in reconstruction";
    return summary;
  }
  if (reconstruction->IsImageRegistered(image_id)) {
    summary.message = "Image is already registered";
    return summary;
  }

  const std::vector<PnPCorrespondence> correspondences =
      CollectPnPCorrespondences(graph, image_id, reconstruction);
  summary.num_correspondences = correspondences.size();
  if (correspondences.size() < options.min_num_correspondences) {
    summary.message = "Too few 2D-3D correspondences";
    return summary;
  }

  const Image& image = reconstruction->GetImage(image_id);
  const Camera& camera = reconstruction->GetCamera(image.camera_id);
  const PnPRansacResult result =
      EstimatePoseRansac(camera, correspondences, options);
  summary.num_ransac_inliers = result.num_inliers_before_final_refinement;
  summary.num_ransac_iterations = result.num_iterations;
  summary.num_inliers = result.inlier_indices.size();
  if (summary.num_inliers < options.min_num_inliers ||
      static_cast<double>(summary.num_inliers) <
          options.min_inlier_ratio * correspondences.size()) {
    summary.message = "PnP RANSAC failed or found too few inliers";
    return summary;
  }

  std::vector<Eigen::Vector2d> pixels;
  std::vector<Eigen::Vector3d> points;
  std::vector<char> inlier_mask(correspondences.size(), false);
  pixels.reserve(correspondences.size()); points.reserve(correspondences.size());
  for (const auto& correspondence : correspondences) {
    pixels.push_back(correspondence.xy); points.push_back(correspondence.xyz);
  }
  for (const auto index : result.inlier_indices) inlier_mask[index] = true;
  Pose3 refined_pose = result.pose;
  summary.refinement = RefineAbsolutePose(
      options.refinement, camera, pixels, points, inlier_mask, &refined_pose);
  if (!summary.refinement.success) {
    summary.message = "PnP pose refinement failed: " + summary.refinement.message;
    return summary;
  }
  Image& mutable_image = reconstruction->GetImage(image_id);
  mutable_image.cam_from_world = refined_pose;
  reconstruction->RegisterImage(image_id);
  LinkInlierCorrespondencesToImage(image_id, correspondences,
                                   result.inlier_indices, reconstruction);

  summary.success = true;
  summary.message = "Image registered";
  return summary;
}

}  // namespace colite
