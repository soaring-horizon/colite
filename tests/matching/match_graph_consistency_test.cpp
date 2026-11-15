#include "colite/matching/match_graph_consistency.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

namespace colite {
namespace {

TEST(MatchGraphConsistencyTest, HandlesMixedDenseAndSparseExternalFeatureIds) {
  MatchGraph graph;
  graph.features[3].keypoints.resize(2);
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = {
      {0, kInvalidPoint2DIdx - 1}, {1, 8}, {2, 9}, {kInvalidPoint2DIdx, 8}};
  SetVerifiedGeometry(&graph, 3, 1001, geometry);
  const auto summary = EnforceVerifiedMatchGraphConsistency(&graph);
  EXPECT_EQ(summary.num_invalid_edges, 2U);
  EXPECT_EQ(summary.num_output_edges, 2U);
  const auto& matches = graph.verified_geometries.at({3, 1001}).inlier_matches;
  ASSERT_EQ(matches.size(), 2U);
  EXPECT_EQ(matches[0].idx1, 0U);
  EXPECT_EQ(matches[0].idx2, kInvalidPoint2DIdx - 1);
  EXPECT_EQ(matches[1].idx1, 1U);
  EXPECT_EQ(matches[1].idx2, 8U);
}

FeatureSet MakeFeatures(std::size_t count) {
  FeatureSet features;
  features.keypoints.resize(count);
  return features;
}

TwoViewGeometry MakeGeometry(
    std::initializer_list<FeatureMatch> matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kUncalibratedFundamental;
  geometry.inlier_matches = matches;
  return geometry;
}

TEST(MatchGraphConsistencyTest, DenseOneToOnePathMatchesSparseGraphDiagnostics) {
  MatchGraph dense;
  std::mt19937 random(17);
  for (ImageId id : {3U, 19U, 101U, 1001U}) {
    SetFeatures(&dense, id, MakeFeatures(30));
  }
  for (ImageId first : {3U, 19U, 101U}) {
    for (ImageId second : {19U, 101U, 1001U}) {
      if (first >= second) continue;
      std::vector<Point2DIdx> permutation(30);
      std::iota(permutation.begin(), permutation.end(), 0U);
      std::shuffle(permutation.begin(), permutation.end(), random);
      auto geometry = MakeGeometry({});
      for (Point2DIdx idx = 0; idx < permutation.size(); ++idx) {
        geometry.inlier_matches.push_back({idx, permutation[idx]});
      }
      SetVerifiedGeometry(&dense, first, second, geometry);
    }
  }
  MatchGraph sparse = dense;
  sparse.features.clear();  // Force the metadata-independent general path.
  const auto fast = EnforceVerifiedMatchGraphConsistency(&dense);
  const auto general = EnforceVerifiedMatchGraphConsistency(&sparse);
  EXPECT_EQ(fast.num_input_edges, general.num_input_edges);
  EXPECT_EQ(fast.num_output_edges, general.num_output_edges);
  EXPECT_EQ(fast.num_input_components, general.num_input_components);
  EXPECT_EQ(fast.num_output_components, general.num_output_components);
  EXPECT_EQ(fast.num_input_conflicted_components, general.num_input_conflicted_components);
  EXPECT_EQ(fast.num_input_conflicting_observations, general.num_input_conflicting_observations);
  EXPECT_EQ(fast.num_invalid_edges, 0U);
  EXPECT_EQ(fast.num_pair_conflict_edges, 0U);
  EXPECT_EQ(fast.num_exact_duplicate_edges, 0U);
  for (const auto& [pair, geometry] : dense.verified_geometries) {
    const auto& other = sparse.verified_geometries.at(pair);
    ASSERT_EQ(geometry.inlier_matches.size(), other.inlier_matches.size());
    for (std::size_t i = 0; i < geometry.inlier_matches.size(); ++i) {
      EXPECT_EQ(geometry.inlier_matches[i].idx1, other.inlier_matches[i].idx1);
      EXPECT_EQ(geometry.inlier_matches[i].idx2, other.inlier_matches[i].idx2);
    }
  }
}

TEST(MatchGraphConsistencyTest, RemovesPairOneToManyConflicts) {
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeFeatures(3));
  SetFeatures(&graph, 2, MakeFeatures(3));
  SetVerifiedGeometry(
      &graph, 1, 2, MakeGeometry({{0, 0}, {0, 1}, {1, 1}, {2, 2}}));

  const MatchGraphConsistencySummary summary =
      EnforceVerifiedMatchGraphConsistency(&graph);

  EXPECT_EQ(summary.num_input_edges, 4);
  EXPECT_EQ(summary.num_pair_conflict_edges, 2);
  EXPECT_EQ(summary.num_output_edges, 2);
  const TwoViewGeometry* geometry = FindVerifiedGeometry(graph, 1, 2);
  ASSERT_NE(geometry, nullptr);
  EXPECT_EQ(geometry->inlier_matches.size(), 2);
}

TEST(MatchGraphConsistencyTest, ReportsButPreservesCrossPairConflict) {
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    SetFeatures(&graph, image_id, MakeFeatures(2));
  }
  SetVerifiedGeometry(&graph, 1, 2, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeGeometry({{1, 0}}));

  const MatchGraphConsistencySummary summary =
      EnforceVerifiedMatchGraphConsistency(&graph);

  EXPECT_EQ(summary.num_input_conflicted_components, 1);
  EXPECT_EQ(summary.num_input_conflicting_observations, 1);
  EXPECT_EQ(summary.num_cross_component_conflict_edges, 0);
  EXPECT_EQ(summary.num_output_edges, 3);
  EXPECT_EQ(summary.num_output_components, 1);
}

TEST(MatchGraphConsistencyTest, KeepsConsistentCycleEdges) {
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    SetFeatures(&graph, image_id, MakeFeatures(1));
  }
  SetVerifiedGeometry(&graph, 1, 2, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeGeometry({{0, 0}}));

  const MatchGraphConsistencySummary summary =
      EnforceVerifiedMatchGraphConsistency(&graph);

  EXPECT_EQ(summary.num_cross_component_conflict_edges, 0);
  EXPECT_EQ(summary.num_output_edges, 3);
  EXPECT_EQ(summary.num_output_components, 1);
}

TEST(MatchGraphConsistencyTest, TriangleSupportWinsAndPreservesFileOrder) {
  MatchGraph graph;
  for (ImageId id = 1; id <= 3; ++id) SetFeatures(&graph, id, MakeFeatures(4));
  SetVerifiedGeometry(&graph, 1, 2,
                      MakeGeometry({{3, 3}, {0, 1}, {0, 0}, {0, 0}, {9, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeGeometry({{0, 0}}));
  const auto summary = EnforceVerifiedMatchGraphConsistency(&graph);
  const auto& matches = graph.verified_geometries.at({1, 2}).inlier_matches;
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].idx1, 3);
  EXPECT_EQ(matches[1].idx1, 0);
  EXPECT_EQ(matches[1].idx2, 0);
  EXPECT_EQ(summary.num_invalid_edges, 1);
  EXPECT_EQ(summary.num_exact_duplicate_edges, 1);
  EXPECT_EQ(summary.num_pair_conflict_edges, 1);
  EXPECT_EQ(summary.num_input_components, 2);
  EXPECT_EQ(summary.num_input_conflicted_components, 1);
  EXPECT_EQ(summary.num_input_conflicting_observations, 1);
  EXPECT_EQ(summary.num_output_components, 2);
}

TEST(MatchGraphConsistencyTest, DiscardsAllInvalidAndUnusableGeometries) {
  MatchGraph graph;
  SetFeatures(&graph, 1, MakeFeatures(1));
  SetFeatures(&graph, 2, MakeFeatures(1));
  SetVerifiedGeometry(&graph, 1, 2, MakeGeometry({{1, 0}, {0, kInvalidPoint2DIdx}}));
  auto unusable = MakeGeometry({{0, 0}});
  unusable.type = TwoViewGeometry::Type::kDegenerate;
  SetVerifiedGeometry(&graph, 1, 3, unusable);
  const auto summary = EnforceVerifiedMatchGraphConsistency(&graph);
  EXPECT_EQ(summary.num_invalid_edges, 2);
  EXPECT_EQ(summary.num_input_edges, 2);
  EXPECT_EQ(summary.num_output_edges, 0);
  for (const auto& [pair, geometry] : graph.verified_geometries) {
    EXPECT_TRUE(geometry.inlier_matches.empty());
  }
}

}  // namespace
}  // namespace colite
