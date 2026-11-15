#include "colite/sfm/structureless_registrar.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include <random>
#include <glog/logging.h>

#include "colite/core/geometry.h"
#include "colite/sfm/image_registrar.h"

#ifdef COLITE_HAS_POSELIB
#include <PoseLib/robust.h>
#endif

namespace colite {

StructurelessRegistrationSummary EstimateStructurelessPoseFromMatches(
    const MatchGraph& graph, const ImageId image_id,
    const Reconstruction& reconstruction,
    const ImageRegistrarOptions& registration_options,
    const StructurelessRegistrationOptions& options) {
  StructurelessRegistrationSummary summary;
#ifndef COLITE_HAS_POSELIB
  (void)graph;
  (void)image_id;
  (void)reconstruction;
  (void)registration_options;
  (void)options;
  summary.message = "Generalized structure-less resection requires PoseLib";
  return summary;
#else
  const auto* features = FindFeatures(graph, image_id);
  if (!features || !reconstruction.ExistsImage(image_id) ||
      reconstruction.IsImageRegistered(image_id)) {
    summary.message = "Missing or already registered query image";
    return summary;
  }
  const auto& camera = reconstruction.GetCamera(reconstruction.GetImage(image_id).camera_id);
  if (!camera.VerifyParams() || !std::isfinite(registration_options.max_reprojection_error_px) ||
      registration_options.max_reprojection_error_px <= 0 ||
      !std::isfinite(registration_options.min_inlier_ratio) ||
      registration_options.min_inlier_ratio < 0 || registration_options.min_inlier_ratio > 1 ||
      !std::isfinite(registration_options.confidence) || registration_options.confidence <= 0 ||
      registration_options.confidence >= 1 ||
      options.min_num_registered_neighbors < 2 || options.min_num_inliers_per_edge == 0) {
    summary.message = "Invalid generalized resection options or camera";
    return summary;
  }
  std::vector<poselib::PairwiseMatches> pairs;
  std::vector<poselib::CameraPose> extrinsics;
  std::vector<std::vector<Point2DIdx>> target_indices;
  std::vector<Eigen::Vector3d> centers;
  std::size_t total = 0;
  for (const auto& edge : FindVerifiedNeighbors(graph, image_id)) {
    const auto neighbor_id = edge.OtherImageId(image_id);
    if (!edge.geometry || edge.geometry->type == TwoViewGeometry::Type::kUndefined ||
        edge.geometry->type == TwoViewGeometry::Type::kDegenerate ||
        !reconstruction.IsImageRegistered(neighbor_id)) continue;
    const auto* neighbor_features = FindFeatures(graph, neighbor_id);
    const auto& neighbor = reconstruction.GetImage(neighbor_id);
    if (!neighbor_features || !neighbor.HasPose()) continue;
    const auto& nc = reconstruction.GetCamera(neighbor.camera_id);
    if (!nc.VerifyParams() || !neighbor.cam_from_world->t_cw.allFinite() ||
        !neighbor.cam_from_world->q_cw.coeffs().allFinite() ||
        std::abs(neighbor.cam_from_world->q_cw.norm() - 1) > 1e-6) continue;
    poselib::PairwiseMatches pair;
    pair.cam_id1 = pairs.size();
    pair.cam_id2 = 0;
    std::vector<Point2DIdx> indices;
    std::set<Point2DIdx> used_target, used_neighbor;
    for (const auto& match : edge.geometry->inlier_matches) {
      const bool target_first = edge.pair.image_id1 == image_id;
      const auto ti = target_first ? match.idx1 : match.idx2;
      const auto ni = target_first ? match.idx2 : match.idx1;
      if (ti >= features->keypoints.size() || ni >= neighbor_features->keypoints.size() ||
          used_target.count(ti) || used_neighbor.count(ni)) continue;
      const auto& tk = features->keypoints[ti];
      const auto& nk = neighbor_features->keypoints[ni];
      Eigen::Vector3d qr, nr;
      if (!UnprojectPixelWithCameraParams(camera.model_id, camera.params.data(), {tk.x, tk.y}, 1, &qr) ||
          !UnprojectPixelWithCameraParams(nc.model_id, nc.params.data(), {nk.x, nk.y}, 1, &nr) ||
          !qr.allFinite() || !nr.allFinite() || qr.z() <= 1e-12 || nr.z() <= 1e-12) continue;
      pair.x1.emplace_back(nr.x() / nr.z(), nr.y() / nr.z());
      pair.x2.emplace_back(qr.x() / qr.z(), qr.y() / qr.z());
      indices.push_back(ti);
      used_target.insert(ti);
      used_neighbor.insert(ni);
    }
    // The generalized solver samples constraints across cameras. A weak
    // camera can fix scale even when it has fewer than the ray initializer's
    // per-edge minimum. Total robust support is checked after estimation.
    if (indices.empty()) continue;
    const auto& pose = *neighbor.cam_from_world;
    extrinsics.emplace_back(pose.q_cw.toRotationMatrix(), pose.t_cw);
    centers.push_back(pose.CameraCenter());
    total += indices.size();
    pairs.push_back(std::move(pair));
    target_indices.push_back(std::move(indices));
  }
  summary.num_pose_hypotheses = pairs.size();
  if (pairs.size() < options.min_num_registered_neighbors || total < 6 ||
      total / 2 < registration_options.min_num_inliers) {
    summary.message = "Too few registered-neighbor 2D-2D correspondences";
    return summary;
  }
  const Eigen::Vector3d origin = centers.front();
  double baseline = 0;
  for (const auto& center : centers) baseline = std::max(baseline, (center - origin).norm());
  if (!std::isfinite(baseline) || baseline <= 1e-12) {
    summary.message = "Registered neighbors have no usable baseline";
    return summary;
  }
  for (auto& pose : extrinsics) pose.t = (pose.R() * origin + pose.t) / baseline;
  // Undistort with Colite's full camera-model support, then estimate with unit
  // pinhole cameras. Preserve the world scale by normalizing neighbor centers.
  const poselib::Camera unit_camera("SIMPLE_PINHOLE", {1, 0, 0}, 1, 1);
  const double threshold = .5 * registration_options.max_reprojection_error_px / camera.MeanFocalLength();
  if (!std::isfinite(threshold) || threshold <= 0) {
    summary.message = "Invalid normalized epipolar threshold";
    return summary;
  }
  poselib::RansacOptions ransac;
  ransac.min_iterations = std::max(1, registration_options.min_num_iterations);
  ransac.max_iterations = std::max<int>(ransac.min_iterations, registration_options.max_num_iterations);
  ransac.success_prob = registration_options.confidence;
  ransac.max_epipolar_error = threshold;
  ransac.seed = registration_options.random_seed < 0 ? std::random_device{}()
      : static_cast<unsigned>(registration_options.random_seed);
  poselib::BundleOptions refinement;
  refinement.max_iterations = 25;
  refinement.loss_scale = threshold;
  poselib::CameraPose estimated;
  std::vector<std::vector<char>> inliers;
  const auto stats = poselib::estimate_generalized_relative_pose(
      pairs, extrinsics, std::vector<poselib::Camera>(pairs.size(), unit_camera),
      {poselib::CameraPose{}}, {unit_camera}, ransac, refinement, &estimated, &inliers);
  summary.used_generalized_pose = true;
  summary.num_ransac_iterations = stats.iterations;
  std::set<Point2DIdx> unique_inliers;
  std::vector<Eigen::Vector3d> inlier_centers;
  for (std::size_t edge = 0; edge < inliers.size() && edge < pairs.size(); ++edge) {
    std::size_t count = 0;
    for (std::size_t idx = 0; idx < inliers[edge].size() && idx < target_indices[edge].size(); ++idx) {
      if (!inliers[edge][idx]) continue;
      ++count;
      unique_inliers.insert(target_indices[edge][idx]);
    }
    summary.num_inlier_correspondences += count;
    VLOG(1) << "Generalized resection neighbor: image=" << image_id
            << " neighbor=" << edge << " inliers=" << count;
    if (count > 0) {
      ++summary.num_consistent_hypotheses;
      inlier_centers.push_back(centers[edge]);
    }
  }
  summary.num_unique_inlier_points = unique_inliers.size();
  VLOG(1) << "Generalized resection: image=" << image_id
          << " correspondences=" << total
          << " inliers=" << summary.num_inlier_correspondences
          << " unique_inliers=" << unique_inliers.size()
          << " supported_neighbors=" << summary.num_consistent_hypotheses
          << " iterations=" << stats.iterations;
  // Robust support must span distinct camera centers, rather than two
  // coincident views plus outliers in a third camera.
  const bool supported_baseline = !inlier_centers.empty() && std::any_of(
      inlier_centers.begin(), inlier_centers.end(), [&](const auto& center) {
        return (center - inlier_centers.front()).norm() > 1e-9 * baseline;
      });
  if (summary.num_inlier_correspondences / 2 < registration_options.min_num_inliers ||
      summary.num_inlier_correspondences < registration_options.min_inlier_ratio * total ||
      unique_inliers.size() < registration_options.min_num_inliers ||
      summary.num_consistent_hypotheses < options.min_num_registered_neighbors ||
      !supported_baseline ||
      !estimated.q.allFinite() || !estimated.t.allFinite() || estimated.q.norm() <= 1e-12) {
    summary.message = "Generalized resection has insufficient independent support";
    return summary;
  }
  summary.pose.q_cw = Eigen::Quaterniond(estimated.q[0], estimated.q[1], estimated.q[2], estimated.q[3]).normalized();
  summary.pose.t_cw = baseline * estimated.t - summary.pose.q_cw * origin;
  summary.success = summary.pose.t_cw.allFinite();
  summary.message = "Generalized structure-less pose estimated from verified matches";
  return summary;
#endif
}

}  // namespace colite
