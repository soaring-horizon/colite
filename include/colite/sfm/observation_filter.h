#pragma once

#include <cstddef>
#include <set>
#include <string>

#include "colite/core/reconstruction.h"

namespace colite {

struct NegativeDepthFilterSummary {
  bool success = false;
  // Counts directly rejected observations, excluding other links removed when
  // deleting a point whose track would otherwise have fewer than two views.
  std::size_t num_filtered_observations = 0;
  std::size_t num_removed_points = 0;
  std::size_t num_remaining_observations = 0;
  std::string message;
};

// COLMAP standalone BA preprocessing: only registered images are considered;
// valid camera depth is >= double epsilon. No reprojection/angle filtering.
NegativeDepthFilterSummary FilterObservationsWithNegativeDepth(
    Reconstruction* reconstruction);

// 局部/增量建图后的坏观测过滤参数。第一版参考 COLMAP 的核心规则：
// 重投影误差过大删观测，track 太短或三角化角过小删 3D 点。
struct ObservationFilterOptions {
  double max_reprojection_error_px = 4.0;
  double min_triangulation_angle_deg = 0.5;
  std::size_t min_track_length = 2;
};

struct ObservationFilterSummary {
  bool success = false;
  std::size_t num_removed_observations = 0;
  std::size_t num_removed_points = 0;
  std::size_t num_reprojection_observations = 0;
  std::size_t num_invalid_observations = 0;
  std::size_t num_short_track_points = 0;
  std::size_t num_low_angle_points = 0;
  std::string message;
};

ObservationFilterSummary FilterObservationsAndPoints(
    const ObservationFilterOptions& options,
    Reconstruction* reconstruction);

// 仅过滤指定图像所观测到的 3D 点。局部 BA 后使用该接口，避免每次注册
// 新图像都重复过滤与本次局部优化无关的整个模型。
// additional_point_ids 可补充局部窗口外的修改点，与窗口中的点去重后一起处理。
ObservationFilterSummary FilterObservationsAndPointsInImages(
    const ObservationFilterOptions& options,
    const std::set<ImageId>& image_ids,
    Reconstruction* reconstruction,
    const std::set<Point3DId>* additional_point_ids = nullptr);

}  // namespace colite
