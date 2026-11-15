#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "colite/core/camera.h"
#include "colite/core/reconstruction.h"
#include "colite/sfm/global_bundle_adjuster.h"

namespace colite {

// 局部 BA 选项。共享内参仅在局部目标覆盖该相机全部已注册图像时开放。
struct LocalBundleAdjustmentOptions {
  int max_num_iterations = 25;
  BundleAdjustmentSolverOptions solver;
  double loss_scale = 1.0;
  std::size_t max_num_images = 6;
  // 未提供显式修改点 scope 时，自动收集外部固定图像的数量上限。
  // Mapper 的 scoped BA 会纳入所选可变点的完整外部观测，不受此上限截断。
  std::size_t max_num_fixed_images = 8;
  bool refine_poses = true;
  bool refine_points = true;
  CameraParameterGroup refine_camera_params = CameraParameterGroup::kNone;
  std::set<std::string> refine_camera_param_names;
  std::map<ImageId, std::vector<int>> constant_image_translation_idxs;
  BundleGaugeFixing gauge_fixing = BundleGaugeFixing::kFirstTwoImages;

  // 用于固定 gauge 的图像。若为空，局部窗口中 ID 最小的图像会被固定。
  std::set<ImageId> constant_image_ids;
};

struct LocalBundleAdjustmentSummary {
  bool success = false;
  std::string linear_solver_type;
  std::string requested_linear_solver_type;
  int solver_num_threads = 0;
  ImageId root_image_id = kInvalidImageId;
  std::set<ImageId> local_image_ids;
  std::set<ImageId> fixed_image_ids;
  std::size_t num_local_images = 0;
  std::size_t num_constant_images = 0;
  std::size_t num_variable_images = 0;
  std::size_t num_variable_points = 0;
  std::size_t num_constant_points = 0;
  std::size_t num_constant_cameras = 0;
  std::size_t num_variable_cameras = 0;
  std::size_t num_gauge_constant_translation_images = 0;
  std::size_t num_residuals = 0;
  std::size_t num_scalar_residuals = 0;
  double initial_mean_reprojection_error = -1.0;
  double final_mean_reprojection_error = -1.0;
  double initial_cost = -1.0;
  double final_cost = -1.0;
  int num_unsuccessful_steps = 0;
  int num_invalid_steps = 0;
  std::set<ImageId> gauge_constant_pose_image_ids;
  std::map<ImageId, std::vector<int>> gauge_constant_translation_idxs;
  std::string message;
};

// Explicit mapper scope: newly allocated points and modified short tracks
// may be optimized with their complete external observations. Other points
// remain variable only when fully contained in the local image window.
struct LocalBundleAdjustmentScope {
  std::set<Point3DId> modified_point_ids;
  Point3DId first_new_point_id = kInvalidPoint3DId;
};

// Select the covisibility window shared by local triangulation and BA.
std::set<ImageId> FindLocalBundleImages(
    const Reconstruction& reconstruction,
    ImageId root_image_id,
    std::size_t max_num_images);

// 围绕 root_image_id 构建一个小的局部 BA 窗口并优化。
// 稳定点约束局部位姿；提供 scope 时，仅对完整局部点及修改的新点 / 短轨迹
// 开放点坐标优化，并纳入这些点的全部必要外部固定观测。
LocalBundleAdjustmentSummary AdjustLocalBundle(
    ImageId root_image_id,
    const LocalBundleAdjustmentOptions& options,
    Reconstruction* reconstruction,
    const LocalBundleAdjustmentScope* scope = nullptr);

}  // namespace colite
