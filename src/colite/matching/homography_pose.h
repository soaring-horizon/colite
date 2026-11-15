#pragma once
#include <optional>
#include <vector>
#include <Eigen/Core>
#include "colite/core/pose.h"
namespace colite::matching_internal {
std::optional<Pose3> PoseFromHomography(
    const Eigen::Matrix3d& normalized_h,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2);
}
