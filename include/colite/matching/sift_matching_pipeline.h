#pragma once

#include <vector>

#include "colite/matching/two_view_geometry_batch.h"
#include "colite/matching/weak_pair_rematcher.h"

namespace colite {

struct SiftMatchingPipelineOptions {
  DescriptorMatchingOptions matching;
  TwoViewGeometryBatchOptions geometry;
  WeakPairRematchingOptions guided;
  bool guided_matching = false;
  // Standalone matching persists initial_matches separately and only needs
  // verified geometries afterwards. Integrated reconstruction retains both.
  bool retain_matches = true;
  std::size_t nearby_image_window = 0;
  double nearby_max_ratio = 0.9;
};

struct SiftMatchingPipelineSummary {
  double wall_seconds = 0.0;
  double matching_work_seconds = 0.0;
  double geometry_work_seconds = 0.0;
  double guided_work_seconds = 0.0;
  std::size_t num_initial_matches = 0;
  WeakPairRematchingSummary guided;
};

// Overlap descriptor matching, independent pair verification, and guided search.
// The pair list contains each unordered image pair at most once.
// Features/cameras/prepared descriptors remain immutable until workers join;
// results are committed in pair order. Optional raw output precedes guided
// updates, for the standalone matcher's separate raw/geometry checkpoints.
SiftMatchingPipelineSummary RunSiftMatchingPipeline(
    const std::vector<ImageId>& image_ids,
    const std::vector<ImagePair>& pairs,
    const std::map<ImageId, Camera>& cameras_by_image,
    const PreparedFeatureMatcherCache& prepared,
    const SiftMatchingPipelineOptions& options,
    MatchGraph* graph,
    std::vector<std::vector<FeatureMatch>>* initial_matches = nullptr);

}  // namespace colite
