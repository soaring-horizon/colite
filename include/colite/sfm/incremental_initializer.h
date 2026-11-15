#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// 增量 SfM 初始化参数。第一版只负责选择两张初始图像并三角化初始点云，
// 后续图像注册、局部 BA 和全局 BA 会在单独模块中继续推进。
struct IncrementalInitializerOptions {
  ImageId image_id1 = kInvalidImageId;
  ImageId image_id2 = kInvalidImageId;
  // Applies only to automatically selected first seed images. A specified
  // single seed bypasses this search limit, as in COLMAP.
  std::size_t max_num_reg_trials = 2;
  std::size_t min_num_inliers = 15;
  std::size_t min_num_triangulated_points = 15;
  double min_triangulation_angle_deg = 0.5;
  double max_reprojection_error_px = 4.0;
  // Reject initial pairs whose unit translation has |z| >= this threshold.
  double max_forward_motion = 0.95;
  // The standard mapper re-estimates calibration from verified matches. Core
  // callers can also initialize from an already estimated, trusted geometry.
  bool reestimate_geometry = false;
  int random_seed = 0;
  // Pair-check budget for the low-level geometry-only BestPair API. The full
  // mapper has a separate reconstruction-trial budget (Mapper.init_num_trials).
  std::size_t max_num_pair_trials = 3;
  // Empty means all images. Multi-model mapping uses this set to initialize
  // only from the current remaining verified component.
  std::set<ImageId> allowed_image_ids;
};

struct InitialPairTrialState {
  std::map<ImageId, std::size_t> num_reg_trials;
  // Canonical graph keys, even when the selected initial pair is oriented.
  std::set<ImagePair> tried_pairs;
  void Clear() { num_reg_trials.clear(); tried_pairs.clear(); }
};

struct IncrementalInitializerSummary {
  bool success = false;
  ImagePair initial_pair;
  std::size_t num_initialization_attempts = 0;
  std::vector<ImagePair> attempted_initial_pairs;
  std::size_t num_triangulated_points = 0;
  bool geometry_reestimated = false;
  bool geometry_accepted = false;
  std::size_t num_geometry_inliers = 0;
  double median_triangulation_angle = -1.0;
  std::string message;
};

// 从已验证两视图图中选择最适合初始化的图像对。
// 先按焦距先验和全图对应支撑选择首图，再按焦距先验和边内点数选择邻居。
// reconstruction 可提供各图像的相机先验；为空时不区分焦距先验。
std::optional<ImagePair> SelectInitialImagePair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options = {},
    const Reconstruction* reconstruction = nullptr);

std::vector<ImagePair> SelectInitialImagePairCandidates(
    const MatchGraph& graph, const IncrementalInitializerOptions& options,
    const Reconstruction* reconstruction = nullptr,
    const InitialPairTrialState* trial_state = nullptr);

// 使用指定图像对初始化 Reconstruction。Reconstruction 需要已经包含对应 Image
// 和 Camera；图像的 points2d 可以为空，函数会从 MatchGraph::features 同步。
IncrementalInitializerSummary InitializeReconstructionFromPair(
    const MatchGraph& graph,
    ImagePair initial_pair,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction);

// Estimate/validate the initial geometry and register its two image poses,
// without creating points. The standard mapper then uses the shared robust
// triangulator with its full creation/continuation/two-view-track options.
IncrementalInitializerSummary RegisterInitialImagePair(
    const MatchGraph& graph, ImagePair initial_pair,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction);

// 自动选择最佳初始图像对并初始化 Reconstruction。
IncrementalInitializerSummary InitializeReconstructionFromBestPair(
    const MatchGraph& graph,
    const IncrementalInitializerOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
