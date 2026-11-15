#pragma once

#include <cstddef>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/reconstruction.h"

namespace colite {

enum class BundleGaugeFixing {
  kNone,
  kFirstImage,
  kFirstTwoImages,
};

// CPU Ceres controls shared by local/global BA and the standalone command.
// Defaults follow COLMAP 4.0 rather than Ceres' own stopping criteria.
struct BundleAdjustmentSolverOptions {
  int max_linear_solver_iterations = 200;
  double function_tolerance = 0.0;
  double gradient_tolerance = 1e-4;
  double parameter_tolerance = 0.0;
  int min_num_residuals_for_cpu_multi_threading = 50000;
  int max_num_images_direct_dense_cpu_solver = 50;
  int max_num_images_direct_sparse_cpu_solver = 1000;

  std::string ValidationError(int max_num_iterations) const {
    if (max_num_iterations < 0) return "max_num_iterations must be non-negative";
    if (max_linear_solver_iterations <= 0)
      return "max_linear_solver_iterations must be positive";
    for (const double value : {function_tolerance, gradient_tolerance, parameter_tolerance}) {
      if (!std::isfinite(value) || value < 0)
        return "BA tolerances must be finite and non-negative";
    }
    if (min_num_residuals_for_cpu_multi_threading < 0 ||
        max_num_images_direct_dense_cpu_solver < 0 ||
        max_num_images_direct_sparse_cpu_solver <= max_num_images_direct_dense_cpu_solver)
      return "BA solver thresholds require non-negative values and dense < sparse";
    return {};
  }
};

// 全局 BA 参数。默认仍固定相机内参；需要优化内参时通过 refine_camera_params
// 指定焦距、主点、畸变等参数组，语义参考 COLMAP 的 BA refine 选项。
struct GlobalBundleAdjustmentOptions {
  int max_num_iterations = 50;
  BundleAdjustmentSolverOptions solver;
  std::size_t min_track_length = 0;
  double loss_scale = 0.0;
  bool refine_poses = true;
  bool constant_pose_rotation = false;
  bool refine_points = true;
  CameraParameterGroup refine_camera_params = CameraParameterGroup::kNone;
  std::set<std::string> refine_camera_param_names;
  std::set<ImageId> constant_image_ids;
  // Keep only the rotations of these images fixed while translations and
  // structure remain variable. Used by staged global SfM.
  std::set<ImageId> constant_rotation_image_ids;
  std::map<ImageId, std::vector<int>> constant_image_translation_idxs;
  BundleGaugeFixing gauge_fixing = BundleGaugeFixing::kFirstTwoImages;
  bool use_position_priors = false;
  double position_prior_stddev = 5.0;
  double position_prior_loss_scale = 2.0;
};

struct GlobalBundleAdjustmentSummary {
  bool success = false;
  std::string linear_solver_type;
  std::string requested_linear_solver_type;
  int solver_num_threads = 0;
  int solver_max_num_iterations = 0;
  double solver_gradient_tolerance = -1.0;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_variable_cameras = 0;
  std::size_t num_gauge_constant_translation_images = 0;
  std::size_t num_residuals = 0;
  std::size_t num_scalar_residuals = 0;
  std::size_t num_position_priors = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  double initial_mean_position_error = -1.0;
  double final_mean_position_error = -1.0;
  double initial_cost = -1.0;
  double final_cost = -1.0;
  int num_unsuccessful_steps = 0;
  int num_invalid_steps = 0;
  std::set<ImageId> gauge_constant_pose_image_ids;
  std::map<ImageId, std::vector<int>> gauge_constant_translation_idxs;
  std::string message;
};

GlobalBundleAdjustmentSummary AdjustGlobalBundle(
    const GlobalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
