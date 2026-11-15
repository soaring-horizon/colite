#include "colite/core/match_graph.h"

#include <vector>

#include <gtest/gtest.h>

namespace colite {
namespace {

TEST(MatchGraphTest, StoresMatchesWithOrderedImagePairAndSwapsReversedInput) {
  MatchGraph graph;
  SetMatches(&graph, 7, 3, {{2, 5}, {4, 9}});

  const std::vector<FeatureMatch>* matches = FindMatches(graph, 3, 7);

  ASSERT_NE(matches, nullptr);
  ASSERT_EQ(matches->size(), 2U);
  EXPECT_EQ((*matches)[0].idx1, 5U);
  EXPECT_EQ((*matches)[0].idx2, 2U);
  EXPECT_EQ((*matches)[1].idx1, 9U);
  EXPECT_EQ((*matches)[1].idx2, 4U);
}

TEST(MatchGraphTest, FindsVerifiedNeighborsSortedByInlierCount) {
  MatchGraph graph;

  TwoViewGeometry weak_geometry;
  weak_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  weak_geometry.inlier_matches = {{0, 0}, {1, 1}};
  SetVerifiedGeometry(&graph, 1, 2, weak_geometry);

  TwoViewGeometry strong_geometry;
  strong_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  strong_geometry.inlier_matches = {{0, 0}, {1, 1}, {2, 2}};
  SetVerifiedGeometry(&graph, 3, 1, strong_geometry);

  const std::vector<MatchGraphEdge> neighbors =
      FindVerifiedNeighbors(graph, 1);

  ASSERT_EQ(neighbors.size(), 2U);
  EXPECT_EQ(neighbors[0].OtherImageId(1), 3U);
  EXPECT_EQ(neighbors[0].NumInliers(), 3U);
  EXPECT_EQ(neighbors[1].OtherImageId(1), 2U);
  EXPECT_EQ(neighbors[1].NumInliers(), 2U);
}

TEST(MatchGraphTest, ReversesGeometryAndPreservesAbsentFields) {
  MatchGraph graph;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  Eigen::Matrix3d f;
  f << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  geometry.F = f;
  geometry.H = (Eigen::Vector3d(2, 3, 1)).asDiagonal();
  geometry.cam2_from_cam1.emplace();
  geometry.cam2_from_cam1->q_cw = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitY()));
  geometry.cam2_from_cam1->t_cw = {1, 2, 3};
  geometry.inlier_matches = {{2, 5}};
  SetVerifiedGeometry(&graph, 7, 3, geometry);
  const auto& result = graph.verified_geometries.at(ImagePair::Ordered(3, 7));
  EXPECT_FALSE(result.E);
  EXPECT_TRUE(result.F->isApprox(f.transpose(), 0.0));
  EXPECT_TRUE(result.H->isApprox((Eigen::Vector3d(0.5, 1.0 / 3, 1)).asDiagonal().toDenseMatrix()));
  EXPECT_TRUE(result.cam2_from_cam1->t_cw.isApprox(geometry.cam2_from_cam1->Inverse().t_cw));
  EXPECT_EQ(result.inlier_matches[0].idx1, 5U);
  EXPECT_EQ(result.inlier_matches[0].idx2, 2U);
  geometry.H = Eigen::Matrix3d::Zero();
  EXPECT_THROW(SetVerifiedGeometry(&graph, 7, 3, geometry), std::invalid_argument);
}

}  // namespace
}  // namespace colite
