#pragma once

#include <vector>

#include "colite/core/camera.h"
#include "colite/core/feature.h"
#include "colite/core/match_graph.h"
#include "colite/core/two_view_geometry.h"

namespace colite {

// 两视图几何验证参数。启用 PoseLib 时最小采样使用七点 F、五点 E，
// 否则均回退八点法；F/E 非最小细化使用八点法，H 使用 DLT。
struct TwoViewGeometryOptions {
  // RANSAC 内点阈值，单位为像素。用于 F/H；E 会换算到归一化平面。
  double max_error_px = 4.0;

  // RANSAC 置信度，用于自适应迭代次数。
  double confidence = 0.999;
  // RANSAC prior for the trial budget, not an acceptance threshold.
  double min_inlier_ratio = 0.25;
  int random_seed = -1;

  // 内部 RANSAC 最大迭代次数。
  // Adaptive stopping is bounded by the prior, confidence and trial cap.
  // Perfect support and a strong prior can stop before min_num_iterations.
  int min_num_iterations = 100;
  int max_num_iterations = 10000;

  // 判定几何关系有效所需的最少内点数。
  int min_num_inliers = 15;

  // E 的内点数需要接近 F 的内点数，才认为标定内参在该图像对上可信。
  double min_E_F_inlier_ratio = 0.95;

  // H 的内点数接近 E/F 时，认为该图像对更可能是平面或纯旋转退化。
  double max_H_inlier_ratio = 0.8;

  bool compute_relative_pose = false;
  bool detect_watermark = true;
  bool multiple_models = false;
  bool multiple_ignore_watermark = true;
  double watermark_min_inlier_ratio = 0.7;
  double watermark_border_size = 0.1;
  double watermark_detection_max_error = 4.0;
  bool filter_stationary_matches = false;
  double stationary_matches_max_error = 4.0;
  bool force_H_use = false;  // Core API only, as in COLMAP 4.0.

  void Check() const;
};

// 根据两张图像的特征匹配估计两视图几何关系。
// 输入匹配索引必须指向对应 FeatureSet 的 keypoints；无效索引会抛出异常。
TwoViewGeometry EstimateTwoViewGeometry(
    const Camera& camera1,
    const FeatureSet& features1,
    const Camera& camera2,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& matches,
    const TwoViewGeometryOptions& options = {});

// Test the supplied calibration even if no focal-length prior is recorded.
// E/F/H still compete as in the calibrated COLMAP estimator.
TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const Camera& camera1, const FeatureSet& features1,
    const Camera& camera2, const FeatureSet& features2,
    const std::vector<FeatureMatch>& matches,
    const TwoViewGeometryOptions& options = {});

bool EstimateTwoViewGeometryPose(
    const Camera& camera1, const FeatureSet& features1,
    const Camera& camera2, const FeatureSet& features2, TwoViewGeometry* geometry);

// Recover missing poses/parallax in memory. PLANAR_OR_PANORAMIC can resolve to
// PLANAR/PANORAMIC; matrices and inlier sets are unchanged.
std::size_t RecoverMatchGraphRelativePoses(
    const std::map<ImageId, Camera>& cameras_by_image, MatchGraph* graph);

}  // namespace colite
