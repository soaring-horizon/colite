#include "colite/matching/match_graph_consistency.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace colite {
namespace {

struct Observation {
  ImageId image_id = kInvalidImageId;
  Point2DIdx point2d_idx = kInvalidPoint2DIdx;

  bool operator<(const Observation& other) const {
    return std::tie(image_id, point2d_idx) <
           std::tie(other.image_id, other.point2d_idx);
  }
};

struct VerifiedEdge {
  ImagePair pair;
  FeatureMatch match;
  std::size_t geometry_match_idx = 0;
  std::size_t node1 = 0;
  std::size_t node2 = 0;
  std::size_t triangle_support = 0;
  std::size_t min_support = 0;
  std::size_t total_support = 0;
  double squared_sampson_error = std::numeric_limits<double>::infinity();
};

class DisjointSets {
 public:
  explicit DisjointSets(const std::vector<Observation>& observations)
      : DisjointSets(observations.size()) {}

  explicit DisjointSets(std::size_t size) : parent_(size), rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), std::size_t{0});
  }

  std::size_t Find(std::size_t node) {
    if (parent_[node] != node) {
      parent_[node] = Find(parent_[node]);
    }
    return parent_[node];
  }

  void Union(std::size_t node1, std::size_t node2) {
    std::size_t root1 = Find(node1);
    std::size_t root2 = Find(node2);
    if (root1 == root2) {
      return;
    }
    if (rank_[root1] < rank_[root2]) {
      std::swap(root1, root2);
    }
    parent_[root2] = root1;
    if (rank_[root1] == rank_[root2]) {
      ++rank_[root1];
    }
  }

 private:
  std::vector<std::size_t> parent_;
  // Union-by-rank cannot exceed log2(number of nodes), even on 64-bit hosts.
  std::vector<std::uint8_t> rank_;
};

bool IsUsableGeometry(const TwoViewGeometry& geometry) {
  return geometry.HasUsableCorrespondences();
}

bool IsValidMatch(const MatchGraph& graph,
                  const ImagePair& pair,
                  const FeatureMatch& match) {
  if (match.idx1 == kInvalidPoint2DIdx ||
      match.idx2 == kInvalidPoint2DIdx) {
    return false;
  }
  const FeatureSet* features1 = FindFeatures(graph, pair.image_id1);
  const FeatureSet* features2 = FindFeatures(graph, pair.image_id2);
  return (features1 == nullptr ||
          match.idx1 < static_cast<Point2DIdx>(features1->keypoints.size())) &&
         (features2 == nullptr ||
          match.idx2 < static_cast<Point2DIdx>(features2->keypoints.size()));
}

double SquaredSampsonError(const MatchGraph& graph,
                           const VerifiedEdge& edge,
                           const TwoViewGeometry& geometry) {
  const FeatureSet* features1 = FindFeatures(graph, edge.pair.image_id1);
  const FeatureSet* features2 = FindFeatures(graph, edge.pair.image_id2);
  if (features1 == nullptr || features2 == nullptr ||
      (!geometry.F || geometry.F->norm() <= 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  const FeatureKeypoint& keypoint1 = features1->keypoints[edge.match.idx1];
  const FeatureKeypoint& keypoint2 = features2->keypoints[edge.match.idx2];
  const Eigen::Vector3d x1(keypoint1.x, keypoint1.y, 1.0);
  const Eigen::Vector3d x2(keypoint2.x, keypoint2.y, 1.0);
  const Eigen::Vector3d fx1 = *geometry.F * x1;
  const Eigen::Vector3d ftx2 = geometry.F->transpose() * x2;
  const double residual = x2.transpose() * *geometry.F * x1;
  const double denominator =
      fx1.head<2>().squaredNorm() + ftx2.head<2>().squaredNorm();
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    return std::numeric_limits<double>::infinity();
  }
  return residual * residual / denominator;
}

bool BetterEdge(const VerifiedEdge& lhs, const VerifiedEdge& rhs) {
  if (lhs.triangle_support != rhs.triangle_support) {
    return lhs.triangle_support > rhs.triangle_support;
  }
  if (lhs.min_support != rhs.min_support) {
    return lhs.min_support > rhs.min_support;
  }
  if (lhs.total_support != rhs.total_support) {
    return lhs.total_support > rhs.total_support;
  }
  if (lhs.squared_sampson_error != rhs.squared_sampson_error) {
    return lhs.squared_sampson_error < rhs.squared_sampson_error;
  }
  return std::tie(lhs.pair.image_id1, lhs.pair.image_id2, lhs.match.idx1,
                  lhs.match.idx2) <
         std::tie(rhs.pair.image_id1, rhs.pair.image_id2, rhs.match.idx1,
                  rhs.match.idx2);
}

struct CompactEdge {
  std::size_t node1;
  std::size_t node2;
  std::size_t match_idx;
};

void MeasureInputComponents(
    const std::vector<Observation>& observations,
    const std::vector<CompactEdge>& edges,
    MatchGraphConsistencySummary* summary) {
  DisjointSets sets(observations);
  for (const auto& edge : edges) sets.Union(edge.node1, edge.node2);
  std::vector<std::pair<std::size_t, ImageId>> memberships;
  memberships.reserve(observations.size());
  for (std::size_t node = 0; node < observations.size(); ++node) {
    memberships.emplace_back(sets.Find(node), observations[node].image_id);
  }
  std::sort(memberships.begin(), memberships.end());
  std::size_t previous_root = observations.size();
  bool conflicted = false;
  for (std::size_t begin = 0; begin < memberships.size();) {
    std::size_t end = begin + 1;
    while (end < memberships.size() && memberships[end] == memberships[begin]) ++end;
    if (memberships[begin].first != previous_root) {
      ++summary->num_input_components;
      previous_root = memberships[begin].first;
      conflicted = false;
    }
    if (end - begin > 1) {
      summary->num_input_conflicting_observations += end - begin - 1;
      if (!conflicted) ++summary->num_input_conflicted_components;
      conflicted = true;
    }
    begin = end;
  }
}

// Mutual descriptor search normally produces valid one-to-one pairs already.
// In that case normalization changes no edges: stream connectivity directly
// instead of retaining a second copy of every edge and two disjoint sets.
// Ambiguous, invalid, sparse external graphs retain the general ranking path.
std::optional<MatchGraphConsistencySummary> NormalizeOneToOneGraph(
    MatchGraph* graph) {
  std::size_t edge_count = 0, observation_count = 0;
  for (const auto& [pair, geometry] : graph->verified_geometries) {
    if (!IsUsableGeometry(geometry)) continue;
    if (!graph->features.count(pair.image_id1) || !graph->features.count(pair.image_id2)) {
      return std::nullopt;
    }
    edge_count += geometry.inlier_matches.size();
  }
  std::map<ImageId, std::pair<std::size_t, std::size_t>> image_ranges;
  for (const auto& [image_id, features] : graph->features) {
    image_ranges.emplace(image_id,
                         std::make_pair(observation_count, features.keypoints.size()));
    observation_count += features.keypoints.size();
  }
  // Avoid allocating dense scratch for mostly unmatched feature collections.
  if (edge_count < observation_count / 2) return std::nullopt;
  DisjointSets sets(observation_count);
  std::vector<bool> active(observation_count, false);
  for (const auto& [pair, geometry] : graph->verified_geometries) {
    if (!IsUsableGeometry(geometry)) continue;
    const auto first = image_ranges.find(pair.image_id1);
    const auto second = image_ranges.find(pair.image_id2);
    if (first == image_ranges.end() || second == image_ranges.end()) return std::nullopt;
    const auto [offset1, size1] = first->second;
    const auto [offset2, size2] = second->second;
    std::vector<bool> seen1(size1, false), seen2(size2, false);
    for (const auto& match : geometry.inlier_matches) {
      if (match.idx1 == kInvalidPoint2DIdx || match.idx2 == kInvalidPoint2DIdx ||
          match.idx1 >= size1 || match.idx2 >= size2 ||
          seen1[match.idx1] || seen2[match.idx2]) return std::nullopt;
      seen1[match.idx1] = seen2[match.idx2] = true;
      const auto node1 = offset1 + match.idx1, node2 = offset2 + match.idx2;
      sets.Union(node1, node2);
      active[node1] = active[node2] = true;
    }
  }
  std::vector<std::pair<std::size_t, ImageId>> memberships;
  memberships.reserve(std::count(active.begin(), active.end(), true));
  for (const auto& [image_id, range] : image_ranges) {
    for (std::size_t node = range.first; node < range.first + range.second; ++node) {
      if (active[node]) memberships.emplace_back(sets.Find(node), image_id);
    }
  }
  std::sort(memberships.begin(), memberships.end());
  MatchGraphConsistencySummary summary;
  summary.num_input_edges = summary.num_output_edges = edge_count;
  std::size_t previous_root = observation_count;
  bool conflicted = false;
  for (std::size_t begin = 0; begin < memberships.size();) {
    std::size_t end = begin + 1;
    while (end < memberships.size() && memberships[end] == memberships[begin]) ++end;
    if (memberships[begin].first != previous_root) {
      ++summary.num_input_components;
      previous_root = memberships[begin].first;
      conflicted = false;
    }
    if (end - begin > 1) {
      summary.num_input_conflicting_observations += end - begin - 1;
      if (!conflicted) ++summary.num_input_conflicted_components;
      conflicted = true;
    }
    begin = end;
  }
  summary.num_output_components = summary.num_input_components;
  for (auto& [pair, geometry] : graph->verified_geometries) {
    if (!IsUsableGeometry(geometry)) geometry.inlier_matches.clear();
  }
  return summary;
}

}  // namespace

MatchGraphConsistencySummary EnforceVerifiedMatchGraphConsistency(
    MatchGraph* graph) {
  MatchGraphConsistencySummary summary;
  if (graph == nullptr) {
    return summary;
  }
  if (const auto normalized = NormalizeOneToOneGraph(graph)) return *normalized;

  // Store observation connectivity once; ranking and one-to-one bookkeeping
  // are pair-local. Cross-pair ordering cannot affect pair-local decisions.
  struct ImageNodeIndex {
    std::vector<std::size_t> dense;
    std::unordered_map<Point2DIdx, std::size_t> sparse;
  };
  std::map<ImageId, ImageNodeIndex> node_indices;
  std::vector<Observation> observations;
  std::vector<CompactEdge> edges;
  struct PairRange {
    ImagePair pair;
    TwoViewGeometry* geometry;
    std::size_t begin;
    std::size_t end;
    bool conflicting;
  };
  std::vector<PairRange> ranges;
  std::size_t edge_count = 0;
  for (const auto& [pair, geometry] : graph->verified_geometries) {
    if (IsUsableGeometry(geometry)) edge_count += geometry.inlier_matches.size();
  }
  edges.reserve(edge_count);
  const auto missing = std::numeric_limits<std::size_t>::max();
  auto get_image_index = [&](ImageId image_id) -> ImageNodeIndex& {
    const auto [it, inserted] = node_indices.try_emplace(image_id);
    if (inserted) {
      if (const auto* features = FindFeatures(*graph, image_id)) {
        it->second.dense.assign(features->keypoints.size(), missing);
      }
    }
    return it->second;
  };
  auto get_node = [&](ImageNodeIndex& index, ImageId image_id, Point2DIdx idx) {
    if (!index.dense.empty()) {
      auto& node = index.dense[idx];
      if (node == missing) {
        node = observations.size();
        observations.push_back({image_id, idx});
      }
      return node;
    }
    // Missing feature metadata is supported by the graph utility. Keep a
    // sparse fallback so a large external feature ID never sizes an array.
    const auto [it, inserted] = index.sparse.emplace(idx, observations.size());
    if (inserted) observations.push_back({image_id, idx});
    return it->second;
  };
  bool needs_ranking = false;
  for (auto& [pair, geometry] : graph->verified_geometries) {
    if (!IsUsableGeometry(geometry)) {
      geometry.inlier_matches.clear();
      continue;
    }
    summary.num_input_edges += geometry.inlier_matches.size();
    const std::size_t begin = edges.size();
    std::unordered_set<Point2DIdx> first, second;
    ImageNodeIndex* index1 = nullptr;
    ImageNodeIndex* index2 = nullptr;
    bool conflicting = false;
    for (std::size_t idx = 0; idx < geometry.inlier_matches.size(); ++idx) {
      const auto& match = geometry.inlier_matches[idx];
      if (!IsValidMatch(*graph, pair, match)) {
        ++summary.num_invalid_edges;
        continue;
      }
      if (index1 == nullptr) {
        index1 = &get_image_index(pair.image_id1);
        index2 = &get_image_index(pair.image_id2);
      }
      const bool unique_first = first.insert(match.idx1).second;
      const bool unique_second = second.insert(match.idx2).second;
      conflicting |= !unique_first || !unique_second;
      edges.push_back({get_node(*index1, pair.image_id1, match.idx1),
                       get_node(*index2, pair.image_id2, match.idx2), idx});
    }
    ranges.push_back({pair, &geometry, begin, edges.size(), conflicting});
    needs_ranking |= conflicting;
  }
  node_indices.clear();
  if (edges.empty()) {
    for (auto& range : ranges) range.geometry->inlier_matches.clear();
    return summary;
  }
  MeasureInputComponents(observations, edges, &summary);

  // Already one-to-one pairs need neither triangle scoring nor sorting.
  // For ambiguous pairs, use the same global, unique adjacency as before.
  std::vector<std::vector<std::size_t>> neighbors;
  if (needs_ranking) {
    neighbors.resize(observations.size());
    for (const auto& edge : edges) {
      neighbors[edge.node1].push_back(edge.node2);
      neighbors[edge.node2].push_back(edge.node1);
    }
    for (auto& adjacent : neighbors) {
      std::sort(adjacent.begin(), adjacent.end());
      adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    }
  }
  DisjointSets output_sets(observations);
  std::vector<bool> active_nodes(observations.size(), false);
  for (auto& range : ranges) {
    auto& matches = range.geometry->inlier_matches;
    std::vector<bool> retained(matches.size(), false);
    if (!range.conflicting) {
      for (std::size_t idx = range.begin; idx < range.end; ++idx) {
        retained[edges[idx].match_idx] = true;
      }
    } else {
      std::vector<VerifiedEdge> ranked;
      ranked.reserve(range.end - range.begin);
      for (std::size_t idx = range.begin; idx < range.end; ++idx) {
        const auto& compact = edges[idx];
        VerifiedEdge edge;
        edge.pair = range.pair;
        edge.match = matches[compact.match_idx];
        edge.geometry_match_idx = compact.match_idx;
        edge.node1 = compact.node1;
        edge.node2 = compact.node2;
        edge.squared_sampson_error = SquaredSampsonError(*graph, edge, *range.geometry);
        const auto& first = neighbors[edge.node1];
        const auto& second = neighbors[edge.node2];
        auto it1 = first.begin(), it2 = second.begin();
        while (it1 != first.end() && it2 != second.end()) {
          if (*it1 < *it2) ++it1;
          else if (*it2 < *it1) ++it2;
          else { ++edge.triangle_support; ++it1; ++it2; }
        }
        edge.min_support = std::min(first.size(), second.size());
        edge.total_support = first.size() + second.size();
        ranked.push_back(edge);
      }
      std::stable_sort(ranked.begin(), ranked.end(), BetterEdge);
      std::unordered_set<Point2DIdx> first, second;
      std::unordered_set<std::uint64_t> exact;
      for (const auto& edge : ranked) {
        const std::uint64_t key = (std::uint64_t{edge.match.idx1} << 32) | edge.match.idx2;
        if (!exact.insert(key).second) {
          ++summary.num_exact_duplicate_edges;
        } else if (first.count(edge.match.idx1) || second.count(edge.match.idx2)) {
          ++summary.num_pair_conflict_edges;
        } else {
          first.insert(edge.match.idx1);
          second.insert(edge.match.idx2);
          retained[edge.geometry_match_idx] = true;
        }
      }
    }
    for (std::size_t idx = range.begin; idx < range.end; ++idx) {
      const auto& edge = edges[idx];
      if (!retained[edge.match_idx]) continue;
      output_sets.Union(edge.node1, edge.node2);
      active_nodes[edge.node1] = active_nodes[edge.node2] = true;
      ++summary.num_output_edges;
    }
    // Keep the original file order (including the first exact duplicate).
    std::size_t count = 0;
    for (std::size_t idx = 0; idx < matches.size(); ++idx) {
      if (retained[idx]) matches[count++] = matches[idx];
    }
    matches.resize(count);
  }
  std::vector<bool> roots(observations.size(), false);
  for (std::size_t node = 0; node < observations.size(); ++node) {
    if (active_nodes[node]) roots[output_sets.Find(node)] = true;
  }
  summary.num_output_components = std::count(roots.begin(), roots.end(), true);
  return summary;
}

}  // namespace colite
