#pragma once

#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "colite/core/match_graph.h"
#include "colite/core/reconstruction.h"

namespace colite {

// SfM 映射期间 MatchGraph 保持只读。预先构建一次观测对应关系，避免每次局部
// 三角化都重新扫描并分配完整的 verified correspondence graph。
struct IncrementalTriangulatorCache {
  using ObservationKey = std::pair<ImageId, Point2DIdx>;
  struct CorrespondenceAdjacency {
    // One immutable CSR graph is shared by local triangulation and global
    // completion. Edges store dense observation indices, in input order.
    struct Neighbors {
      struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type = ObservationKey;
        using difference_type = std::ptrdiff_t;
        using pointer = const ObservationKey*;
        using reference = const ObservationKey&;
        const CorrespondenceAdjacency* graph = nullptr;
        std::size_t position = 0;
        reference operator*() const {
          return graph->observations[graph->edges[position]];
        }
        pointer operator->() const { return &operator*(); }
        Iterator& operator++() { ++position; return *this; }
        Iterator operator++(int) { auto previous = *this; ++*this; return previous; }
        bool operator==(const Iterator& other) const {
          return graph == other.graph && position == other.position;
        }
        bool operator!=(const Iterator& other) const { return !(*this == other); }
      };
      const CorrespondenceAdjacency* graph;
      std::size_t first, last;
      Iterator begin() const { return {graph, first}; }
      Iterator end() const { return {graph, last}; }
      std::size_t size() const { return last - first; }
      bool empty() const { return first == last; }
      const ObservationKey& front() const { return *begin(); }
    };

    std::size_t Index(const ObservationKey& observation) const {
      const auto image_it = by_image.find(observation.first);
      if (image_it == by_image.end() ||
          observation.second >= image_it->second.second) {
        return std::numeric_limits<std::size_t>::max();
      }
      return image_it->second.first + observation.second;
    }
    std::optional<Neighbors> Find(const ObservationKey& observation) const {
      const auto index = Index(observation);
      if (index == std::numeric_limits<std::size_t>::max()) return std::nullopt;
      return Neighbors{this, edge_offsets[index], edge_offsets[index + 1]};
    }
    bool IsTwoViewObservation(const ObservationKey& observation) const {
      const auto neighbors = Find(observation);
      if (!neighbors || neighbors->size() != 1) return false;
      const auto other_neighbors = Find(neighbors->front());
      return other_neighbors && other_neighbors->size() == 1;
    }
    // image -> (first dense observation index, number of feature rows).
    std::map<ImageId, std::pair<std::size_t, std::size_t>> by_image;
    std::vector<ObservationKey> observations;
    std::vector<std::size_t> edge_offsets, edges;
  };

  explicit IncrementalTriangulatorCache(const MatchGraph& graph);

  CorrespondenceAdjacency correspondence_adjacency;
};

// 增量三角化参数。命名和默认值参考 COLMAP
// IncrementalTriangulator::Options。
struct IncrementalTriangulatorOptions {
  // Match COLMAP's one-hop creation default. Robust creation still keeps only
  // the best geometrically consistent observation from each image when a
  // caller explicitly enables transitive collection.
  std::size_t max_transitivity = 1;
  double min_triangulation_angle_deg = 1.5;
  double min_pair_median_triangulation_angle_deg = 0.0;
  double create_max_angle_error_deg = 2.0;
  double continue_max_angle_error_deg = 2.0;
  double merge_max_reprojection_error_px = 4.0;
  double complete_max_reprojection_error_px = 4.0;
  // Zero disables correspondence propagation; merging remains independent.
  std::size_t complete_max_transitivity = 5;
  // Final-pose descriptor-guided completion. These conservative defaults
  // require a target feature to agree with at least two existing track
  // descriptors and to lie within two pixels of the projected 3D point.
  double pose_guided_max_reprojection_error_px = 2.0;
  double pose_guided_max_descriptor_angle_rad = 0.301;
  std::size_t pose_guided_min_descriptor_support = 2;
  // 新建点后追加执行的 track completion / merge 轮数上限，接近 COLMAP 的 complete-merge-create 局部循环。
  std::size_t max_track_update_iterations = 5;
  // During incremental mapping, only complete the image being processed so
  // future PnP visibility remains stable. Global completion may enable this to
  // propagate tracks to either endpoint of a verified correspondence.
  bool complete_neighbor_observations = false;
  bool ignore_two_view_tracks = true;
};

struct IncrementalTriangulatorSummary {
  ImageId image_id = kInvalidImageId;
  std::size_t num_created_points = 0;
  std::size_t num_created_observations = 0;
  std::size_t num_completed_observations = 0;
  std::size_t num_merged_points = 0;
  std::size_t num_merged_observations = 0;
  std::size_t num_filtered_observations = 0;
  std::size_t num_filtered_points = 0;
  std::string message;
};

// 为一张已注册图像创建新的 3D 点。
// 函数会遍历该图像和已注册邻居之间的 verified matches，只对两端都尚未关联 3D
// 的观测进行三角化，并将通过几何检查的新 track 写入 Reconstruction。
IncrementalTriangulatorSummary TriangulateRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache = nullptr);

// 为一张已注册图像补全已有 3D 点的观测。
// 若目标图像的 2D 点尚未关联 3D，但它通过 verified match 连接到已注册邻居
// 的已有 3D 点，则在重投影误差检查通过后将该观测加入对应 track。
IncrementalTriangulatorSummary CompleteTracksForRegisteredImage(
    const MatchGraph& graph,
    ImageId image_id,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction);

// COLMAP-style global track refinement: recursively complete tracks through
// the correspondence graph and then merge connected compatible 3D points.
IncrementalTriangulatorSummary CompleteAndMergeAllTracks(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction,
    const IncrementalTriangulatorCache* cache = nullptr);

// Extend existing tracks after camera poses stabilize. Unlike correspondence
// graph traversal, this searches unassigned features near each projected 3D
// point and requires descriptor agreement with multiple track observations.
IncrementalTriangulatorSummary CompleteTracksByPoseAndDescriptors(
    const MatchGraph& graph,
    const IncrementalTriangulatorOptions& options,
    Reconstruction* reconstruction);

}  // namespace colite
