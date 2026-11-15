#include "colite/matching/weak_pair_rematcher.h"

#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"

namespace colite {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  camera.has_prior_focal_length = true;
  return camera;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

void AppendProjectedPoint(const Camera& camera,
                          const Pose3& cam1_from_world,
                          const Pose3& cam2_from_world,
                          const Eigen::Vector3d& point_world,
                          FeatureSet* features1,
                          FeatureSet* features2) {
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());
  features1->keypoints.push_back(
      {static_cast<float>(pixel1->x()), static_cast<float>(pixel1->y()), 1.0F,
       0.0F});
  features2->keypoints.push_back(
      {static_cast<float>(pixel2->x()), static_cast<float>(pixel2->y()), 1.0F,
       0.0F});
}

void FillDescriptorsForRelaxedRecovery(FeatureSet* features1,
                                       FeatureSet* features2) {
  constexpr int kBlockSize = 8;
  constexpr float kDescriptorValue = 181.0F;
  features1->descriptors.resize(10, 128);
  features2->descriptors.resize(12, 128);
  features1->descriptors.setZero();
  features2->descriptors.setZero();

  for (int i = 0; i < 10; ++i) {
    for (int offset = 0; offset < kBlockSize; ++offset) {
      const int col = i * kBlockSize + offset;
      features1->descriptors(i, col) = kDescriptorValue;
      features2->descriptors(i, col) = kDescriptorValue;
    }
  }

  // 第 8/9 个点的第二近邻较近：ratio=0.8 会拒绝，ratio=0.95 会接受。
  for (int offset = 0; offset < kBlockSize; ++offset) {
    features2->descriptors(10, 8 * kBlockSize + offset) = kDescriptorValue;
    features2->descriptors(11, 9 * kBlockSize + offset) = kDescriptorValue;
  }
  features2->descriptors(8, 8 * kBlockSize + kBlockSize - 1) = 170.0F;
  features2->descriptors(9, 9 * kBlockSize + kBlockSize - 1) = 170.0F;
  features2->descriptors(10, 8 * kBlockSize + kBlockSize - 1) = 166.0F;
  features2->descriptors(11, 9 * kBlockSize + kBlockSize - 1) = 166.0F;

  features2->keypoints.push_back(features2->keypoints[8]);
  features2->keypoints.push_back(features2->keypoints[9]);
}

TEST(WeakPairRematcherTest, ImprovesNearbyWeakPairWithGuidedCandidates) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  for (int i = 0; i < 10; ++i) {
    const Eigen::Vector3d point_world(-1.0 + 0.25 * i, -0.4 + 0.08 * i,
                                      4.0 + 0.13 * i + 0.01 * i * i);
    AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                         point_world, &features1, &features2);
  }
  FillDescriptorsForRelaxedRecovery(&features1, &features2);

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);

  std::vector<FeatureMatch> initial_matches;
  for (Point2DIdx idx = 0; idx < 8; ++idx) {
    initial_matches.push_back({idx, idx});
  }
  SetMatches(&graph, 1, 2, initial_matches);

  TwoViewGeometryOptions geometry_options;
  geometry_options.compute_relative_pose = true;
  geometry_options.random_seed = 0;
  geometry_options.min_num_inliers = 8;
  geometry_options.max_error_px = 1.0;
  const TwoViewGeometry initial_geometry = EstimateTwoViewGeometry(
      camera, features1, camera, features2, initial_matches, geometry_options);
  ASSERT_NE(initial_geometry.type, TwoViewGeometry::Type::kDegenerate);
  ASSERT_FALSE(initial_geometry.inlier_matches.empty());
  SetVerifiedGeometry(&graph, 1, 2, initial_geometry);

  WeakPairRematchingOptions options;
  options.nearby_image_window = 2;
  options.min_inlier_gain = 1;
  options.relaxed_matching.max_ratio = 0.95;
  options.relaxed_matching.mutual_check = true;
  options.guided_matching.max_sampson_error_px = 2.0;
  options.verification = geometry_options;

  MatchGraph cached_graph = graph;
  PreparedFeatureMatcherCache prepared;
  for (auto& [image_id, features] : cached_graph.features) {
    prepared.emplace(image_id, std::make_unique<PreparedFeatureMatcher>(
        features, options.relaxed_matching.backend));
    features.descriptors.resize(0, 0);
  }

  const WeakPairRematchingSummary summary =
      RematchWeakImagePairs({{1, camera}, {2, camera}}, options, &graph);

  const std::vector<FeatureMatch>* rematched = FindMatches(graph, 1, 2);
  const TwoViewGeometry* geometry = FindVerifiedGeometry(graph, 1, 2);
  ASSERT_NE(rematched, nullptr);
  ASSERT_NE(geometry, nullptr);
  EXPECT_EQ(summary.num_checked_pairs, 1U);
  EXPECT_EQ(summary.num_improved_pairs, 1U);
  EXPECT_GE(rematched->size(), 10U);
  EXPECT_GE(geometry->inlier_matches.size(), 10U);
  const auto cached_summary = RematchWeakImagePairs(
      {{1, camera}, {2, camera}}, options, &cached_graph, &prepared);
  EXPECT_EQ(cached_summary.num_improved_pairs, summary.num_improved_pairs);
  const auto& cached_matches = cached_graph.verified_geometries.at({1, 2}).inlier_matches;
  ASSERT_EQ(cached_matches.size(), geometry->inlier_matches.size());
  for (std::size_t idx = 0; idx < cached_matches.size(); ++idx) {
    EXPECT_EQ(cached_matches[idx].idx1, geometry->inlier_matches[idx].idx1);
    EXPECT_EQ(cached_matches[idx].idx2, geometry->inlier_matches[idx].idx2);
  }
}

void SetDistinctDescriptors(FeatureSet* first, FeatureSet* second) {
  first->descriptors = DescriptorMatrix::Zero(first->keypoints.size(), 128);
  for (Eigen::Index i = 0; i < first->descriptors.rows(); ++i) {
    for (int j = 0; j < 8; ++j) first->descriptors(i, 8 * i + j) = 181.0F;
  }
  second->descriptors = first->descriptors;
}

MatchGraph MakeGuidedRecoveryGraph(const FeatureSet& first, const FeatureSet& second,
                                  TwoViewGeometry geometry) {
  MatchGraph graph;
  SetFeatures(&graph, 1, first);
  SetFeatures(&graph, 2, second);
  for (Point2DIdx i = 0; i < 8; ++i) geometry.inlier_matches.push_back({i, i});
  SetMatches(&graph, 1, 2, geometry.inlier_matches);
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  return graph;
}

TEST(WeakPairRematcherTest, CalibratedGuidanceUsesEssentialInsteadOfFundamental) {
  for (const bool include_wrong_f : {false, true}) {
    SCOPED_TRACE(include_wrong_f);
    Camera camera = MakeCamera();
    // Distortion must be removed before applying E; pixel F is not equivalent.
    camera.model_id = CameraModelId::kSimpleRadial;
    camera.params = {700, 500, 400, 0.3};
    FeatureSet first, second;
    for (int i = 0; i < 10; ++i) {
      AppendProjectedPoint(camera, CameraAt({0, 0, 0}), CameraAt({1, 0, 0}),
          Eigen::Vector3d(-2.0 + .4 * i, .6 + .05 * i, 4.0 + .1 * i), &first, &second);
    }
    SetDistinctDescriptors(&first, &second);
    TwoViewGeometry geometry;
    geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
    Eigen::Matrix3d essential;
    essential << 0, 0, 0, 0, 0, 1, 0, -1, 0;
    geometry.E = essential;
    if (include_wrong_f) {
      Eigen::Matrix3d wrong;
      wrong << 0, 0, 1, 0, 0, 0, -1, 0, 0;
      geometry.F = wrong;
    }
    auto graph = MakeGuidedRecoveryGraph(first, second, geometry);
    WeakPairRematchingOptions options;
    options.nearby_image_window = 0;
    options.min_inlier_gain = 1;
    options.guided_matching.max_sampson_error_px = 2;
    const auto summary = RematchWeakImagePairs({{1, camera}, {2, camera}}, options, &graph);
    EXPECT_EQ(summary.num_rematched_pairs, 1U);
    EXPECT_EQ(summary.num_improved_pairs, 1U);
    const auto* result = FindVerifiedGeometry(graph, 1, 2);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->inlier_matches.size(), 10U);
    EXPECT_EQ(result->F.has_value(), include_wrong_f);
    EXPECT_TRUE(result->E->isApprox(essential));
  }
}

TEST(WeakPairRematcherTest, PlanarAndPanoramicGuidanceUsesHomographyWithoutFundamental) {
  for (const auto type : {TwoViewGeometry::Type::kPlanarHomography,
                          TwoViewGeometry::Type::kPanoramic,
                          TwoViewGeometry::Type::kPlanarOrPanoramic}) {
    FeatureSet first, second;
    for (int i = 0; i < 10; ++i) {
      first.keypoints.push_back({100.0F + 40 * i, 200.0F + 20 * i, 1, 0});
      // A 3 px forward error remains within 4 px. With scale .25 its
      // inverse-transfer error is 12 px; mutual search must use one joint gate.
      second.keypoints.push_back({.25F * first.keypoints.back().x + 3,
                                  .25F * first.keypoints.back().y, 1, 0});
    }
    SetDistinctDescriptors(&first, &second);
    TwoViewGeometry geometry;
    geometry.type = type;
    Eigen::Matrix3d homography = Eigen::Matrix3d::Identity();
    homography(0, 0) = homography(1, 1) = .25;
    geometry.H = homography;
    auto graph = MakeGuidedRecoveryGraph(first, second, geometry);
    WeakPairRematchingOptions options;
    options.nearby_image_window = 0;
    options.min_inlier_gain = 1;
    options.guided_matching.max_sampson_error_px = 4;
    const auto summary = RematchWeakImagePairs({{1, MakeCamera()}, {2, MakeCamera()}}, options, &graph);
    EXPECT_EQ(summary.num_rematched_pairs, 1U);
    EXPECT_EQ(summary.num_improved_pairs, 1U);
    const auto* result = FindVerifiedGeometry(graph, 1, 2);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->inlier_matches.size(), 10U);
    EXPECT_EQ(result->type, type);
    EXPECT_FALSE(result->F.has_value());
    EXPECT_TRUE(result->H->isApprox(homography));
  }
}

}  // namespace
}  // namespace colite
