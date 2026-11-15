#pragma once

#include <cstddef>
#include <map>
#include <memory>

#include "colite/core/camera.h"
#include "colite/core/match_graph.h"
#include "colite/matching/descriptor_matcher.h"
#include "colite/matching/guided_matching.h"
#include "colite/matching/two_view_geometry_verifier.h"

namespace colite {

using PreparedFeatureMatcherCache =
    std::map<ImageId, std::unique_ptr<PreparedFeatureMatcher>>;

struct WeakPairRematchingOptions {
  // Zero applies guided re-matching to every pair; positive values restrict
  // it to neighbors in file order.
  // 只对文件顺序近邻图像对启用二阶段重匹配；0 表示不限制邻域。
  std::size_t nearby_image_window = 2;

  // 新几何至少需要比旧几何多出的内点数，避免用噪声候选替换稳定结果。
  std::size_t min_inlier_gain = 5;

  DescriptorMatchingOptions relaxed_matching;
  GuidedMatchingOptions guided_matching;
  TwoViewGeometryOptions verification;
};

struct WeakPairRematchingSummary {
  std::size_t num_checked_pairs = 0;
  std::size_t num_rematched_pairs = 0;
  std::size_t num_improved_pairs = 0;
  double candidate_generation_work_seconds = 0.0;
  double descriptor_search_work_seconds = 0.0;
  double geometry_verification_work_seconds = 0.0;
};

// Reuse the verified geometry for guided descriptor re-search. An optional
// complete cache must use relaxed_matching.backend; with it, graph features
// may contain keypoints only. Only pairs with a clear inlier gain are updated.
WeakPairRematchingSummary RematchWeakImagePairs(
    const std::map<ImageId, Camera>& cameras_by_image,
    const WeakPairRematchingOptions& options,
    MatchGraph* graph,
    const PreparedFeatureMatcherCache* prepared_cache = nullptr);

}  // namespace colite
