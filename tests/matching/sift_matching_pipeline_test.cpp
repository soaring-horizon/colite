#include "colite/matching/sift_matching_pipeline.h"

#include <random>
#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/core/parallel.h"

namespace colite {
namespace {

struct Fixture {
  MatchGraph graph;
  std::map<ImageId, Camera> cameras;
  PreparedFeatureMatcherCache prepared;
  const std::vector<ImageId> ids{3, 12, 20};
  const std::vector<ImagePair> pairs{{3, 12}, {3, 20}, {12, 20}};
  Fixture() {
    std::mt19937 rng(37);
    DescriptorMatrix descriptors(40, 128);
    for (int row = 0; row < 40; ++row) {
      for (int col = 0; col < 128; ++col) descriptors(row, col) = 1.0F + rng() % 1000;
      descriptors.row(row).normalize();
    }
    for (std::size_t i = 0; i < ids.size(); ++i) {
      Camera camera;
      camera.id = ids[i];
      camera.model_id = CameraModelId::kPinhole;
      camera.width = 1000; camera.height = 800;
      camera.params = {700, 700, 500, 400};
      cameras.emplace(ids[i], camera);
      Pose3 pose;
      pose.t_cw = Eigen::Vector3d(-0.4 * i, 0.03 * i, 0);
      FeatureSet features;
      features.descriptors = descriptors;
      for (int row = 0; row < 40; ++row) {
        const int x = row % 8, y = row / 8;
        const auto pixel = *ProjectWorldPoint(camera, pose,
            Eigen::Vector3d(-1.0 + 0.3*x, -0.6 + 0.3*y, 4.0 + 0.11*x + 0.17*y + 0.03*x*y));
        features.keypoints.push_back({static_cast<float>(pixel.x()), static_cast<float>(pixel.y()), 1, 0});
      }
      prepared.emplace(ids[i], std::make_unique<PreparedFeatureMatcher>(features, DescriptorMatcherBackend::kBruteForce));
      graph.features.emplace(ids[i], std::move(features));
    }
  }
};

void ExpectMatches(const std::vector<FeatureMatch>& a, const std::vector<FeatureMatch>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].idx1, b[i].idx1); EXPECT_EQ(a[i].idx2, b[i].idx2);
  }
}

TEST(SiftMatchingPipelineTest, PreservesSerialRawGeometryAndGuidedResults) {
  Fixture fixture;
  for (bool guided : {false, true}) {
    SiftMatchingPipelineOptions options;
    options.geometry.geometry_options.random_seed = 0;
    options.guided.verification.random_seed = 0;
    options.geometry.keep_degenerate = true;
    options.guided_matching = guided;
    options.guided.nearby_image_window = 0;
    options.guided.relaxed_matching = options.matching;
    options.guided.min_inlier_gain = 0;
    MatchGraph serial = fixture.graph, pipelined = fixture.graph;
    std::vector<std::vector<FeatureMatch>> expected_raw;
    for (const auto& pair : fixture.pairs) {
      auto raw = MatchPreparedFeatureSets(*fixture.prepared.at(pair.image_id1),
          *fixture.prepared.at(pair.image_id2), options.matching);
      ASSERT_EQ(raw.size(), 40U);
      expected_raw.push_back(raw);
      SetMatches(&serial, pair.image_id1, pair.image_id2, raw);
    }
    VerifyMatchGraphGeometries(fixture.cameras, &serial, options.geometry);
    WeakPairRematchingSummary serial_guided;
    if (guided) serial_guided = RematchWeakImagePairs(fixture.cameras, options.guided, &serial, &fixture.prepared);
    std::vector<std::vector<FeatureMatch>> raw;
    auto summary = RunSiftMatchingPipeline(fixture.ids, fixture.pairs, fixture.cameras,
        fixture.prepared, options, &pipelined, &raw);
    ASSERT_EQ(raw.size(), expected_raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) ExpectMatches(raw[i], expected_raw[i]);
    EXPECT_EQ(summary.num_initial_matches, 120U);
    EXPECT_EQ(summary.guided.num_rematched_pairs, serial_guided.num_rematched_pairs);
    EXPECT_EQ(summary.guided.num_improved_pairs, serial_guided.num_improved_pairs);
    ASSERT_EQ(pipelined.matches.size(), serial.matches.size());
    ASSERT_EQ(pipelined.verified_geometries.size(), serial.verified_geometries.size());
    for (const auto& [pair, geometry] : serial.verified_geometries) {
      const auto& other = pipelined.verified_geometries.at(pair);
      EXPECT_EQ(geometry.type, other.type);
      ASSERT_EQ(geometry.F.has_value(), other.F.has_value());
      if (geometry.F) EXPECT_TRUE(geometry.F->isApprox(*other.F, 0.0));
      ASSERT_EQ(geometry.E.has_value(), other.E.has_value());
      if (geometry.E) EXPECT_TRUE(geometry.E->isApprox(*other.E, 0.0));
      ASSERT_EQ(geometry.H.has_value(), other.H.has_value());
      if (geometry.H) EXPECT_TRUE(geometry.H->isApprox(*other.H, 0.0));
      ExpectMatches(geometry.inlier_matches, other.inlier_matches);
      ExpectMatches(serial.matches.at(pair), pipelined.matches.at(pair));
    }
  }
}

TEST(SiftMatchingPipelineTest, PropagatesWorkerAndProducerExceptionsAfterJoining) {
  Fixture fixture;
  SiftMatchingPipelineOptions options;
    options.geometry.geometry_options.random_seed = 0;
    options.guided.verification.random_seed = 0;
  options.matching.backend = DescriptorMatcherBackend::kFlann;
  EXPECT_THROW(RunSiftMatchingPipeline(fixture.ids, fixture.pairs, fixture.cameras,
      fixture.prepared, options, &fixture.graph), std::invalid_argument);
  options.matching.backend = DescriptorMatcherBackend::kBruteForce;
  fixture.graph.features.at(12).keypoints.clear();
  EXPECT_THROW(RunSiftMatchingPipeline(fixture.ids, fixture.pairs, fixture.cameras,
      fixture.prepared, options, &fixture.graph), std::out_of_range);
  EXPECT_TRUE(fixture.graph.matches.empty());
}

TEST(SiftMatchingPipelineTest, VerifiedOnlyRetainsRawCheckpointWithoutGraphCopies) {
  Fixture fixture;
  SiftMatchingPipelineOptions options;
    options.geometry.geometry_options.random_seed = 0;
    options.guided.verification.random_seed = 0;
  options.geometry.keep_degenerate = true;
  options.guided_matching = true;
  options.guided.nearby_image_window = 0;
  options.guided.relaxed_matching = options.matching;
  MatchGraph both = fixture.graph, verified_only = fixture.graph;
  std::vector<std::vector<FeatureMatch>> expected_raw, raw;
  RunSiftMatchingPipeline(fixture.ids, fixture.pairs, fixture.cameras,
      fixture.prepared, options, &both, &expected_raw);
  options.retain_matches = false;
  RunSiftMatchingPipeline(fixture.ids, fixture.pairs, fixture.cameras,
      fixture.prepared, options, &verified_only, &raw);
  EXPECT_TRUE(verified_only.matches.empty());
  ASSERT_EQ(raw.size(), expected_raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) ExpectMatches(raw[i], expected_raw[i]);
  ASSERT_EQ(verified_only.verified_geometries.size(), both.verified_geometries.size());
  for (const auto& [pair, geometry] : both.verified_geometries) {
    const auto& other = verified_only.verified_geometries.at(pair);
    EXPECT_EQ(geometry.type, other.type);
    ASSERT_EQ(geometry.F.has_value(), other.F.has_value());
      if (geometry.F) EXPECT_TRUE(geometry.F->isApprox(*other.F, 0.0));
    ExpectMatches(geometry.inlier_matches, other.inlier_matches);
  }
}

}  // namespace
}  // namespace colite
