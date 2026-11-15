#pragma once

#include <vector>

#include <Eigen/Core>

#include "colite/core/feature.h"
#include "colite/core/camera.h"
#include "colite/core/two_view_geometry.h"
#include "colite/matching/descriptor_matcher.h"

namespace colite {

struct GuidedMatchingOptions {
  // Pixel threshold: Sampson error for E/F, forward transfer error for H.
  // E uses undistorted normalized coordinates and the cameras' focal scales.
  double max_sampson_error_px = 4.0;
};

bool HasGuidedMatchingModel(const TwoViewGeometry& geometry);

// Dispatch the verified model: calibrated E on normalized rays, uncalibrated F
// on pixels, planar/panoramic H on pixels. Descriptor storage is never copied.
std::vector<FeatureMatch> MatchPreparedFeaturePairGuided(
    const Camera& camera1, const FeatureSet& features1,
    const PreparedFeatureMatcher& prepared1,
    const Camera& camera2, const FeatureSet& features2,
    const PreparedFeatureMatcher& prepared2,
    const TwoViewGeometry& geometry, double max_error_px,
    const DescriptorMatchingOptions& options,
    GuidedDescriptorMatchingStats* stats = nullptr);

// 使用已有基础矩阵对宽松匹配候选做 guided filtering。
// 该函数不重新搜索描述子，只负责把不满足极线几何的候选匹配剔除。
std::vector<FeatureMatch> FilterMatchesByGuidedFundamental(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& candidate_matches,
    const Eigen::Matrix3d& fundamental_matrix,
    const GuidedMatchingOptions& options = {});

}  // namespace colite
