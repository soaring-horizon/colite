#include "colite/matching/descriptor_matcher.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>
#include <random>

#include <gtest/gtest.h>

namespace colite {
namespace {

DescriptorMatrix MakeBlockDescriptors(std::initializer_list<int> blocks) {
  DescriptorMatrix descriptors(static_cast<Eigen::Index>(blocks.size()), 128);
  descriptors.setZero();
  Eigen::Index row_idx = 0;
  for (const int block : blocks) {
    for (Eigen::Index offset = 0; offset < 16; ++offset) {
      descriptors(row_idx, static_cast<Eigen::Index>(block * 16) + offset) =
          0.25F;
    }
    ++row_idx;
  }
  return descriptors;
}

DescriptorMatrix MakeUnitSiftDescriptors(std::initializer_list<int> blocks) {
  DescriptorMatrix descriptors(static_cast<Eigen::Index>(blocks.size()), 128);
  descriptors.setZero();
  Eigen::Index row_idx = 0;
  for (const int block : blocks) {
    for (Eigen::Index offset = 0; offset < 16; ++offset) {
      descriptors(row_idx, static_cast<Eigen::Index>(block * 16) + offset) =
          0.25F;
    }
    ++row_idx;
  }
  return descriptors;
}

std::vector<std::pair<Point2DIdx, Point2DIdx>> ToPairs(
    const std::vector<FeatureMatch>& matches) {
  std::vector<std::pair<Point2DIdx, Point2DIdx>> pairs;
  for (const FeatureMatch& match : matches) {
    pairs.emplace_back(match.idx1, match.idx2);
  }
  std::sort(pairs.begin(), pairs.end());
  return pairs;
}

TEST(DescriptorMatcherTest, BruteForceMatchesWithRatioAndMutualCheck) {
  const DescriptorMatrix descriptors1 = MakeBlockDescriptors({0, 1, 2});
  const DescriptorMatrix descriptors2 = MakeBlockDescriptors({0, 1, 2, 3});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 0.8;
  options.mutual_check = true;

  const std::vector<FeatureMatch> matches =
      MatchDescriptors(descriptors1, descriptors2, options);

  EXPECT_EQ(ToPairs(matches),
            (std::vector<std::pair<Point2DIdx, Point2DIdx>>{
                {0, 0}, {1, 1}, {2, 2}}));
}

TEST(DescriptorMatcherTest, FlannMatchesSamePairsAsBruteForceBranch) {
  const DescriptorMatrix descriptors1 = MakeBlockDescriptors({0, 1, 2});
  const DescriptorMatrix descriptors2 = MakeBlockDescriptors({0, 1, 2, 3});

  DescriptorMatchingOptions bf_options;
  bf_options.backend = DescriptorMatcherBackend::kBruteForce;
  bf_options.max_ratio = 0.8;
  bf_options.mutual_check = true;

  DescriptorMatchingOptions flann_options = bf_options;
  flann_options.backend = DescriptorMatcherBackend::kFlann;

  const std::vector<FeatureMatch> bf_matches =
      MatchDescriptors(descriptors1, descriptors2, bf_options);
  const std::vector<FeatureMatch> flann_matches =
      MatchDescriptors(descriptors1, descriptors2, flann_options);

  EXPECT_EQ(ToPairs(flann_matches), ToPairs(bf_matches));
}

TEST(DescriptorMatcherTest, PreparedFlannMatchesOneShotMatcher) {
  FeatureSet features1;
  FeatureSet features2;
  features1.descriptors = MakeBlockDescriptors({0, 1, 2});
  features2.descriptors = MakeBlockDescriptors({0, 1, 2, 3});
  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kFlann;
  const PreparedFeatureMatcher prepared1(features1, options.backend);
  const PreparedFeatureMatcher prepared2(features2, options.backend);
  EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2, options)),
            ToPairs(MatchFeatureSets(features1, features2, options)));
}

TEST(DescriptorMatcherTest, RejectsIncompatibleLearnedFeatureContract) {
  FeatureSet sift;
  FeatureSet aliked;
  sift.descriptors = MakeBlockDescriptors({0});
  aliked.extractor_type = FeatureExtractorType::kAlikedN16Rot;
  aliked.descriptor_metric = DescriptorMetric::kCosine;
  aliked.descriptors = MakeBlockDescriptors({0});

  DescriptorMatchingOptions options;
  options.matcher_type = FeatureMatcherType::kSiftBruteForce;
  EXPECT_THROW(MatchFeatureSets(sift, aliked, options), std::invalid_argument);

  options.matcher_type = FeatureMatcherType::kAlikedBruteForce;
  EXPECT_THROW(MatchFeatureSets(aliked, aliked, options),
               std::invalid_argument);
}

TEST(DescriptorMatcherTest, CudaMatchesCpuBruteForceExactly) {
  if (!IsCudaDescriptorMatcherAvailable()) {
    GTEST_SKIP() << "CUDA descriptor matcher is unavailable";
  }
  DescriptorMatrix descriptors1(41, 128);
  DescriptorMatrix descriptors2(47, 128);
  for (Eigen::Index row = 0; row < descriptors1.rows(); ++row) {
    for (Eigen::Index col = 0; col < descriptors1.cols(); ++col) {
      descriptors1(row, col) = static_cast<float>(
          ((row * 37 + col * 17 + row * col) % 256));
    }
  }
  for (Eigen::Index row = 0; row < descriptors2.rows(); ++row) {
    for (Eigen::Index col = 0; col < descriptors2.cols(); ++col) {
      descriptors2(row, col) = static_cast<float>(
          ((row * 29 + col * 13 + row * col * 3) % 256));
    }
  }
  descriptors2.topRows(20) = descriptors1.topRows(20);
  descriptors1.row(39).setConstant(255.0F);
  descriptors2.row(40).setZero();
  descriptors2.row(41).setConstant(1.0F);
  descriptors2.row(42).setConstant(254.0F);
  descriptors2.row(43).setConstant(255.0F);

  DescriptorMatchingOptions cpu_options;
  cpu_options.backend = DescriptorMatcherBackend::kBruteForce;
  DescriptorMatchingOptions cuda_options = cpu_options;
  cuda_options.backend = DescriptorMatcherBackend::kCudaBruteForce;

  EXPECT_EQ(ToPairs(MatchDescriptors(descriptors1, descriptors2, cuda_options)),
            ToPairs(MatchDescriptors(descriptors1, descriptors2, cpu_options)));

  FeatureSet features1;
  FeatureSet features2;
  features1.descriptors = descriptors1;
  features2.descriptors = descriptors2;
  const PreparedFeatureMatcher prepared1(features1, cuda_options.backend);
  const PreparedFeatureMatcher prepared2(features2, cuda_options.backend);
  EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2,
                                             cuda_options)),
            ToPairs(MatchDescriptors(descriptors1, descriptors2,
                                     cpu_options)));
  ReleaseDescriptorMatcherWorkspace();
  ReleaseDescriptorMatcherWorkspace();
  EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2,
                                             cuda_options)),
            ToPairs(MatchDescriptors(descriptors1, descriptors2,
                                     cpu_options)));
}

TEST(DescriptorMatcherTest, RejectsAmbiguousMatchByRatio) {
  const DescriptorMatrix descriptors1 = MakeBlockDescriptors({0});
  const DescriptorMatrix descriptors2 = MakeBlockDescriptors({0, 0});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 0.4;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

TEST(DescriptorMatcherTest, PreparedCudaHandlesTensorTileBoundariesAndFallbackDimensions) {
  if (!IsCudaDescriptorMatcherAvailable()) GTEST_SKIP();
  std::mt19937 random(29);
  DescriptorMatchingOptions cpu;
  cpu.backend = DescriptorMatcherBackend::kBruteForce;
  DescriptorMatchingOptions gpu = cpu;
  gpu.backend = DescriptorMatcherBackend::kCudaBruteForce;
  for (int dimensions : {4, 12, 16, 128, 256}) {
    for (int rows : {1, 15, 16, 17, 31, 32, 33, 65}) {
      SCOPED_TRACE(::testing::Message() << dimensions << " dimensions, " << rows << " rows");
      FeatureSet first, second;
      first.descriptors.resize(rows, dimensions);
      second.descriptors.resize(rows + 3, dimensions);
      for (auto* descriptors : {&first.descriptors, &second.descriptors}) {
        for (Eigen::Index row = 0; row < descriptors->rows(); ++row) {
          for (int column = 0; column < dimensions; ++column) {
            (*descriptors)(row, column) = 1.0F + random() % 1000;
          }
          descriptors->row(row).normalize();
        }
      }
      second.descriptors.topRows(rows) = first.descriptors;
      const auto expected = ToPairs(MatchFeatureSets(first, second, cpu));
      const PreparedFeatureMatcher prepared1(first, gpu.backend), prepared2(second, gpu.backend);
      EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2, gpu)), expected);
      ReleaseDescriptorMatcherWorkspace();
      EXPECT_EQ(ToPairs(MatchPreparedFeatureSets(prepared1, prepared2, gpu)), expected);
    }
  }
}

TEST(DescriptorMatcherTest, CudaRowReductionHandlesTiesAcrossWarpLanes) {
  if (!IsCudaDescriptorMatcherAvailable()) GTEST_SKIP();
  DescriptorMatrix descriptors1 = MakeUnitSiftDescriptors({0, 1});
  DescriptorMatrix descriptors2(65, 128);
  for (Eigen::Index row = 0; row < descriptors2.rows(); ++row) {
    descriptors2.row(row) = MakeUnitSiftDescriptors({2}).row(0);
  }
  descriptors2.row(1) = descriptors1.row(0);
  descriptors2.row(32) = descriptors1.row(0);
  descriptors2.row(64) = descriptors1.row(1);
  DescriptorMatchingOptions cpu;
  cpu.backend = DescriptorMatcherBackend::kBruteForce;
  cpu.max_ratio = 1.0;
  DescriptorMatchingOptions gpu = cpu;
  gpu.backend = DescriptorMatcherBackend::kCudaBruteForce;
  EXPECT_EQ(ToPairs(MatchDescriptors(descriptors1, descriptors2, gpu)),
            ToPairs(MatchDescriptors(descriptors1, descriptors2, cpu)));
  EXPECT_EQ(ToPairs(MatchDescriptors(descriptors1.bottomRows(1), descriptors2.bottomRows(1), gpu)),
            ToPairs(MatchDescriptors(descriptors1.bottomRows(1), descriptors2.bottomRows(1), cpu)));
}

TEST(DescriptorMatcherTest, CudaHandlesOddRowsAndTiesAcrossColumnTiles) {
  if (!IsCudaDescriptorMatcherAvailable()) GTEST_SKIP();
  FeatureSet first, second;
  first.descriptors.resize(259, 128);
  second.descriptors.resize(263, 128);
  std::mt19937 random(19);
  std::uniform_real_distribution<float> sample(0, 1);
  for (int i = 0; i < 259; ++i) {
    for (int j = 0; j < 128; ++j) first.descriptors(i, j) = sample(random);
    first.descriptors.row(i).normalize();
    first.keypoints.push_back({static_cast<float>(i), static_cast<float>(i % 3), 1, 0});
  }
  first.descriptors.row(257) = first.descriptors.row(0);
  for (int i = 0; i < 263; ++i) {
    const int source = i < 259 ? i : 0;
    second.descriptors.row(i) = first.descriptors.row(source);
    second.keypoints.push_back(first.keypoints[source]);
  }
  DescriptorMatchingOptions cpu, gpu;
  gpu.backend = DescriptorMatcherBackend::kCudaBruteForce;
  const auto expected = MatchFeatureSets(first, second, cpu);
  ASSERT_GT(expected.size(), 250U);
  EXPECT_EQ(ToPairs(MatchFeatureSets(first, second, gpu)), ToPairs(expected));
  Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
  fundamental(1, 2) = -1;
  fundamental(2, 1) = 1;
  EXPECT_EQ(ToPairs(MatchFeatureSetsGuidedFundamental(first, second, fundamental, 0.5, gpu)),
            ToPairs(MatchFeatureSetsGuidedFundamental(first, second, fundamental, 0.5, cpu)));
}

TEST(DescriptorMatcherTest, RejectsMatchBeyondColmapMaxDistance) {
  const DescriptorMatrix descriptors1 = MakeUnitSiftDescriptors({0});
  const DescriptorMatrix descriptors2 = MakeUnitSiftDescriptors({1, 2});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;
  options.max_distance = 0.7;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

TEST(DescriptorMatcherTest, RejectsEqualBestAndSecondBestByRatio) {
  const DescriptorMatrix descriptors1 = MakeUnitSiftDescriptors({0});
  const DescriptorMatrix descriptors2 = MakeUnitSiftDescriptors({0, 0});

  DescriptorMatchingOptions options;
  options.backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;
  options.max_distance = 1.0;
  options.mutual_check = false;

  EXPECT_TRUE(MatchDescriptors(descriptors1, descriptors2, options).empty());
}

TEST(DescriptorMatcherTest, GuidedSearchRecoversMatchBehindOffEpipolarDistractor) {
  FeatureSet features1;
  FeatureSet features2;
  features1.keypoints.push_back({10.0F, 10.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({20.0F, 10.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({20.0F, 100.0F, 1.0F, 0.0F});
  features1.descriptors = MakeUnitSiftDescriptors({0});
  features2.descriptors = MakeUnitSiftDescriptors({0, 0});
  for (Eigen::Index col = 0; col < 16; ++col) {
    features2.descriptors(0, col) = 0.24F;
  }

  DescriptorMatchingOptions options;
  options.max_ratio = 0.8;
  options.max_distance = 0.7;
  options.mutual_check = true;
  const std::vector<FeatureMatch> global =
      MatchFeatureSets(features1, features2, options);
  ASSERT_EQ(global.size(), 1U);
  EXPECT_EQ(global.front().idx2, 1U);

  Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
  fundamental(1, 2) = -1.0;
  fundamental(2, 1) = 1.0;
  const std::vector<FeatureMatch> guided =
      MatchFeatureSetsGuidedFundamental(features1, features2, fundamental,
                                        2.0, options);
  ASSERT_EQ(guided.size(), 1U);
  EXPECT_EQ(guided.front().idx1, 0U);
  EXPECT_EQ(guided.front().idx2, 0U);

  const PreparedFeatureMatcher prepared1(
      features1, DescriptorMatcherBackend::kBruteForce);
  const PreparedFeatureMatcher prepared2(
      features2, DescriptorMatcherBackend::kBruteForce);
  const std::vector<FeatureMatch> prepared_guided =
      MatchPreparedFeatureSetsGuidedFundamental(
          prepared1, prepared2, fundamental, 2.0, options);
  ASSERT_EQ(prepared_guided.size(), guided.size());
  EXPECT_EQ(prepared_guided.front().idx1, guided.front().idx1);
  EXPECT_EQ(prepared_guided.front().idx2, guided.front().idx2);

  if (IsCudaDescriptorMatcherAvailable()) {
    DescriptorMatchingOptions cuda_options = options;
    cuda_options.backend = DescriptorMatcherBackend::kCudaBruteForce;
    const std::vector<FeatureMatch> cuda_guided =
        MatchFeatureSetsGuidedFundamental(
            features1, features2, fundamental, 2.0, cuda_options);
    ASSERT_EQ(cuda_guided.size(), guided.size());
    EXPECT_EQ(cuda_guided.front().idx1, guided.front().idx1);
    EXPECT_EQ(cuda_guided.front().idx2, guided.front().idx2);
  }
}

TEST(DescriptorMatcherTest, GenericGuidanceSharesGeometryGateAcrossBackends) {
  std::vector<DescriptorMatcherBackend> backends = {
      DescriptorMatcherBackend::kBruteForce, DescriptorMatcherBackend::kFlann};
  if (IsCudaDescriptorMatcherAvailable()) backends.push_back(DescriptorMatcherBackend::kCudaBruteForce);
  for (const auto backend : backends) {
    SCOPED_TRACE(static_cast<int>(backend));
    FeatureSet first, second;
    first.descriptors = MakeUnitSiftDescriptors({0, 1, 2, 3});
    second.descriptors = first.descriptors;
    for (int i = 0; i < 4; ++i) {
      first.keypoints.push_back({100.0F + 100 * i, 200.0F + 100 * i, 1, 0});
      second.keypoints.push_back(first.keypoints.back());
    }
    PreparedFeatureMatcher prepared1(first, backend), prepared2(second, backend);
    for (const auto type : {GuidedGeometryType::kEpipolar, GuidedGeometryType::kHomography}) {
      Eigen::Matrix3d matrix = Eigen::Matrix3d::Zero();
      std::vector<Eigen::Vector2d> xy1, xy2;
      double threshold = 0;
      if (type == GuidedGeometryType::kEpipolar) {
        matrix(1, 2) = 1;
        matrix(2, 1) = -1;
        threshold = .001;
        for (int i = 0; i < 4; ++i) {
          xy1.emplace_back(-.4 + .2 * i, .1 * i);
          xy2.emplace_back(xy1.back().x() - .2, xy1.back().y());
        }
      } else {
        matrix.setIdentity();
        matrix(0, 0) = matrix(1, 1) = .25;
        threshold = 4;
        for (const auto& key : first.keypoints) {
          xy1.emplace_back(key.x, key.y);
          xy2.emplace_back(.25 * key.x + 3, .25 * key.y);
        }
      }
      // An invalid inverse-projection row is excluded, with original indices
      // preserved. No second descriptor matrix is built for normalized rays.
      xy1[2].setConstant(std::numeric_limits<double>::quiet_NaN());
      for (const bool mutual : {false, true}) {
        DescriptorMatchingOptions options;
        options.backend = backend;
        options.mutual_check = mutual;
        for (const double scale : {1e-120, 1.0, 1e120}) {
          const auto matches = MatchPreparedFeatureSetsGuidedGeometry(
              prepared1, prepared2, type, matrix * scale, xy1, xy2, threshold, options);
          const std::vector<std::pair<Point2DIdx, Point2DIdx>> expected{{0, 0}, {1, 1}, {3, 3}};
          EXPECT_EQ(ToPairs(matches), expected);
        }
      }
    }
  }
}

}  // namespace
}  // namespace colite
