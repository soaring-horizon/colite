#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>

#include "colite/core/feature.h"
#include "colite/core/pose.h"

namespace colite {

// 两视图几何验证结果，用于初始化、建图连通性判断和相对位姿恢复。
struct TwoViewGeometry {
  enum class Type {
    kUndefined = 0,
    kDegenerate = 1,
    kCalibratedEssential = 2,
    kUncalibratedFundamental = 3,
    kPlanarHomography = 4,
    kPanoramic = 5,
    kPlanarOrPanoramic = 6,
    kWatermark = 7,
    kMultiple = 8,
    kRigCalibrated = 9,
  };

  Type type = Type::kUndefined;
  // Presence is independent of the numerical value, as in COLMAP 4.
  std::optional<Eigen::Matrix3d> E, F, H;
  std::optional<Pose3> cam2_from_cam1;
  std::vector<FeatureMatch> inlier_matches;
  double median_triangulation_angle = -1.0;

  bool HasUsableCorrespondences() const {
    return type != Type::kUndefined && type != Type::kDegenerate &&
           type != Type::kWatermark && type != Type::kRigCalibrated &&
           !inlier_matches.empty();
  }
};

}  // namespace colite

