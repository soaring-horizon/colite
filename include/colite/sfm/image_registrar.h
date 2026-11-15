#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// Non-robust, calibrated EPnP fit. Requires at least four correspondences;
// callers must select inliers first. Returns nullopt for invalid rays,
// non-finite data, rank-deficient world points or no finite solution.
std::optional<Pose3> EstimateAbsolutePoseEpnp(
    const Camera& camera, const std::vector<Eigen::Vector2d>& points2d,
    const std::vector<Eigen::Vector3d>& points3d);

struct AbsolutePoseRefinementOptions {
  int max_num_iterations = 100;
  double gradient_tolerance = 1.0;
  double loss_function_scale = 1.0;
};

struct AbsolutePoseRefinementSummary {
  bool success = false;
  std::size_t num_observations = 0;
  int num_iterations = 0;
  double initial_cost = -1.0;
  double final_cost = -1.0;
  std::string message;
};

// Refine only pose, with fixed camera and world points, using Cauchy pixel
// residuals. Inlier mask keeps original row indices. Failure leaves pose intact.
AbsolutePoseRefinementSummary RefineAbsolutePose(
    const AbsolutePoseRefinementOptions& options, const Camera& camera,
    const std::vector<Eigen::Vector2d>& points2d,
    const std::vector<Eigen::Vector3d>& points3d,
    const std::vector<char>& inlier_mask, Pose3* pose);

// 新图像注册参数。P3P RANSAC、EPnP局部重估和非线性位姿细化。
struct ImageRegistrarOptions {
  std::size_t min_num_correspondences = 15;
  std::size_t min_num_inliers = 12;
  double min_inlier_ratio = 0.25;
  int min_num_iterations = 100;
  int max_num_iterations = 10000;
  double max_reprojection_error_px = 4.0;
  double confidence = 0.99999;
  int random_seed = 0;
  AbsolutePoseRefinementOptions refinement;
};

struct ImageRegistrationSummary {
  bool success = false;
  ImageId image_id = kInvalidImageId;
  std::size_t num_correspondences = 0;
  std::size_t num_ransac_iterations = 0;
  std::size_t num_ransac_inliers = 0;
  std::size_t num_inliers = 0;
  AbsolutePoseRefinementSummary refinement;
  std::string message;
};

// 使用已注册图像的 2D-3D 观测注册一张新图像。
// 函数会从 MatchGraph::features 同步目标图像 2D 点，并通过 verified_geometries
// 找到与已注册邻居之间的内点匹配，估计 world -> camera 位姿。
ImageRegistrationSummary RegisterImageWithExistingPoints(
    const MatchGraph& graph,
    ImageId image_id,
    const ImageRegistrarOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
