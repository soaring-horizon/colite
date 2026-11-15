#include "colite/features/feature_mask.h"

#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include "colite/exe/command.h"
#include "colite/exe/feature_extractor.h"
#include "colite/io/image_reader.h"
#include "colite/io/sfm_workspace.h"

namespace colite {
namespace {

class FeatureMaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root = std::filesystem::temp_directory_path() /
        ("colite_masks_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root / "images/sub");
  }
  void TearDown() override { std::filesystem::remove_all(root); }
  std::filesystem::path root;
};

TEST_F(FeatureMaskTest, NoMasksLeaveFeaturesUnchangedWithoutInputIo) {
  FeatureMasks masks("missing", {}, {});
  FeatureSet features;
  features.keypoints.push_back({2, 3, 1, 0});
  masks.Apply("missing.jpg", &features);
  EXPECT_TRUE(masks.Fingerprint().empty());
  EXPECT_EQ(features.keypoints.size(), 1U);
}

TEST_F(FeatureMaskTest, IntersectsMasksPreservingDescriptorsAndOriginalShapes) {
  const auto image = root / "images/sub/view.jpg";
  ASSERT_TRUE(cv::imwrite(image.string(), cv::Mat(4, 8, CV_8UC1, cv::Scalar(128))));
  std::filesystem::create_directories(root / "masks/sub");
  cv::Mat camera(4, 8, CV_8UC1, cv::Scalar(255));
  camera.colRange(0, 2).setTo(0);
  cv::Mat individual(4, 7, CV_8UC1, cv::Scalar(1));
  individual.row(0).setTo(0);
  ASSERT_TRUE(cv::imwrite((root / "global.png").string(), camera));
  ASSERT_TRUE(cv::imwrite((root / "masks/sub/view.jpg.png").string(), individual));
  ASSERT_TRUE(cv::imwrite((root / "masks/sub/view.png").string(), cv::Mat(4, 8, CV_8UC1, cv::Scalar(0))));
  const ImageMaskOptions options{root / "masks", root / "global.png"};
  FeatureMasks masks(root / "images", {image}, options);
  FeatureSet features;
  features.keypoints = {{1.9f, 1.2f}, {2.9f, .5f}, {2.9f, 1.2f}, {6.9f, 3.9f},
      {7.1f, 2}, {8.1f, 2}, {3, 4}, {std::numeric_limits<float>::quiet_NaN(), 0}};
  features.descriptors.resize(8, 4);
  features.source_keypoint_columns = 6;
  features.source_keypoint_shapes.resize(8, 4);
  for (int i = 0; i < 8; ++i) {
    features.descriptors.row(i).setConstant(10 + i);
    features.source_keypoint_shapes.row(i).setConstant(20 + i);
  }
  masks.Apply(image, &features);
  ASSERT_EQ(features.keypoints.size(), 2U);
  EXPECT_FLOAT_EQ(features.keypoints[0].x, 2.9f);
  EXPECT_FLOAT_EQ(features.keypoints[1].x, 6.9f);
  EXPECT_FLOAT_EQ(features.descriptors(0, 0), 12);
  EXPECT_FLOAT_EQ(features.descriptors(1, 0), 13);
  EXPECT_FLOAT_EQ(features.source_keypoint_shapes(0, 0), 22);
  EXPECT_FLOAT_EQ(features.source_keypoint_shapes(1, 0), 23);
  WriteWorkspaceFeatureSet(root / "workspace", 1, features);
  EXPECT_EQ(ReadWorkspaceFeatureSet(root / "workspace", 1).source_keypoint_columns, 6U);
  // The appended-extension name wins; deleting it selects the stem fallback.
  std::filesystem::remove(root / "masks/sub/view.jpg.png");
  FeatureMasks fallback(root / "images", {image}, options);
  EXPECT_NE(fallback.Fingerprint(), masks.Fingerprint());
  fallback.Apply(image, &features);
  EXPECT_TRUE(features.keypoints.empty());
  EXPECT_EQ(features.descriptors.rows(), 0);
}

TEST_F(FeatureMaskTest, ContentIdentityDetectsSameSizePreservedMtimeEditsAndRejectsInvalidMasks) {
  const auto image = root / "images/view.png", mask = root / "global.bmp";
  ASSERT_TRUE(cv::imwrite(image.string(), cv::Mat(16, 16, CV_8UC1, cv::Scalar(128))));
  cv::Mat pixels(16, 16, CV_8UC1, cv::Scalar(255));
  ASSERT_TRUE(cv::imwrite(mask.string(), pixels));
  const ImageMaskOptions options{{}, mask};
  const auto first = FeatureMasks(root / "images", {image}, options).Fingerprint();
  const auto time = std::filesystem::last_write_time(mask);
  const auto size = std::filesystem::file_size(mask);
  pixels.at<std::uint8_t>(0, 0) = 0;
  ASSERT_TRUE(cv::imwrite(mask.string(), pixels));
  ASSERT_EQ(std::filesystem::file_size(mask), size);
  std::filesystem::last_write_time(mask, time);
  EXPECT_NE(FeatureMasks(root / "images", {image}, options).Fingerprint(), first);
  EXPECT_THROW(FeatureMasks(root / "images", {image}, {root / "absent", {}}), std::invalid_argument);
  { std::ofstream corrupt(mask); corrupt << "invalid"; }
  EXPECT_THROW(FeatureMasks(root / "images", {image}, options), std::invalid_argument);
}

TEST_F(FeatureMaskTest, ResolvesMasksByLogicalNameForSymlinkedImages) {
  const auto target = root / "external.png", image = root / "images/sub/link.png";
  ASSERT_TRUE(cv::imwrite(target.string(), cv::Mat(4, 8, CV_8UC1, cv::Scalar(128))));
  std::error_code error;
  std::filesystem::create_symlink(target, image, error);
  if (error) GTEST_SKIP() << "Symlinks unavailable: " << error.message();
  std::filesystem::create_directories(root / "masks/sub");
  ASSERT_TRUE(cv::imwrite((root / "masks/sub/link.png.png").string(), cv::Mat(4, 8, CV_8UC1, cv::Scalar(255))));
  FeatureSet features;
  features.keypoints = {{2.5f, 1.5f}};
  features.descriptors = DescriptorMatrix::Ones(1, 4);
  EXPECT_NO_THROW(FeatureMasks(root / "images", {image}, {root / "masks", {}}).Apply(image, &features));
  EXPECT_EQ(features.keypoints.size(), 1U);
}

TEST_F(FeatureMaskTest, EncodedMaskCoordinatesFollowAllExifOrientations) {
  std::vector<unsigned char> jpeg;
  ASSERT_TRUE(cv::imencode(".jpg", cv::Mat(2, 3, CV_8UC3, cv::Scalar(128, 128, 128)), jpeg));
  cv::Mat mask(2, 3, CV_8UC1, cv::Scalar(0));
  mask.at<std::uint8_t>(0, 1) = 1;
  const auto path = root / "global.png";
  ASSERT_TRUE(cv::imwrite(path.string(), mask));
  for (int orientation = 1; orientation <= 8; ++orientation) {
    const std::vector<unsigned char> header = {0xff, 0xe1, 0, 34, 'E', 'x', 'i', 'f', 0, 0,
        'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0,
        static_cast<unsigned char>(orientation), 0, 0, 0, 0, 0, 0, 0};
    const auto image = root / "images" / (std::to_string(orientation) + ".jpg");
    {
      std::ofstream out(image, std::ios::binary);
      out.write(reinterpret_cast<const char*>(jpeg.data()), 2);
      out.write(reinterpret_cast<const char*>(header.data()), header.size());
      out.write(reinterpret_cast<const char*>(jpeg.data() + 2), jpeg.size() - 2);
    }
    const auto oriented = ApplyExifOrientation(mask, orientation);
    FeatureSet features;
    for (int y = 0; y < oriented.rows; ++y)
      for (int x = 0; x < oriented.cols; ++x) features.keypoints.push_back({x + .5f, y + .5f});
    features.descriptors = DescriptorMatrix::Ones(6, 4);
    FeatureMasks(root / "images", {image}, {{}, path}).Apply(image, &features);
    ASSERT_EQ(features.keypoints.size(), 1U) << orientation;
    const auto& point = features.keypoints[0];
    EXPECT_EQ(oriented.at<std::uint8_t>(static_cast<int>(point.y), static_cast<int>(point.x)), 1) << orientation;
  }
}

TEST_F(FeatureMaskTest, StandaloneExtractorInvalidatesChangedMaskBeforeCacheHit) {
  const auto image = root / "images/view.png", mask = root / "global.bmp";
  cv::Mat texture(128, 128, CV_8UC1);
  cv::RNG(42).fill(texture, cv::RNG::UNIFORM, 0, 256);
  ASSERT_TRUE(cv::imwrite(image.string(), texture));
  ASSERT_TRUE(cv::imwrite(mask.string(), cv::Mat(128, 128, CV_8UC1, cv::Scalar(255))));
  const auto work = root / "workspace";
  const std::vector<std::string> args{"feature_extractor", "--image_path", (root / "images").string(),
      "--Colite.workspace_path", work.string(), "--FeatureExtraction.use_gpu", "0", "--FeatureExtraction.num_threads", "2",
      "--SiftExtraction.max_num_features", "256", "--ImageReader.camera_mask_path", mask.string()};
  auto run = [&](bool hit) {
    std::ostringstream out, err;
    EXPECT_EQ(exe::RunFeatureExtractor(args, out, err), 0) << err.str();
    EXPECT_NE(out.str().find("Feature cache hit: " + std::to_string(hit)), std::string::npos) << out.str();
  };
  run(false);
  ASSERT_GT(ReadWorkspaceFeatureSet(work, 1).keypoints.size(), 0U);
  run(true);
  const auto time = std::filesystem::last_write_time(mask);
  const auto size = std::filesystem::file_size(mask);
  ASSERT_TRUE(cv::imwrite(mask.string(), cv::Mat(128, 128, CV_8UC1, cv::Scalar(0))));
  ASSERT_EQ(std::filesystem::file_size(mask), size);
  std::filesystem::last_write_time(mask, time);
  run(false);
  EXPECT_TRUE(ReadWorkspaceFeatureSet(work, 1).keypoints.empty());
  run(true);
  { std::ofstream corrupt(mask); corrupt << "invalid"; }
  std::ostringstream out, err;
  EXPECT_EQ(exe::RunFeatureExtractor(args, out, err), 1);
  EXPECT_NE(err.str().find("Cannot decode image mask"), std::string::npos);
}

TEST_F(FeatureMaskTest, AutoValidatesMasksBeforeDryRunWithoutWritingOutputs) {
  ASSERT_TRUE(cv::imwrite((root / "images/view.png").string(), cv::Mat(16, 16, CV_8UC1, cv::Scalar(128))));
  const auto mask = root / "global.png", output = root / "pipeline";
  ASSERT_TRUE(cv::imwrite(mask.string(), cv::Mat(16, 16, CV_8UC1, cv::Scalar(255))));
  const std::vector<std::string> args{"colite", "recon_auto", "--image_path", (root / "images").string(),
      "--output_path", output.string(), "--stop_after", "sfm", "--dry_run", "--sfm.ImageReader.camera_mask_path", mask.string(),
      "--sfm.FeatureExtraction.use_gpu", "0", "--sfm.FeatureMatching.use_gpu", "0"};
  std::ostringstream out, err;
  EXPECT_EQ(exe::RunColite(args, out, err), 0) << err.str();
  EXPECT_NE(out.str().find("PLAN recon_sfm"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(output));
  std::filesystem::remove(mask);
  out.str(""); err.str("");
  EXPECT_EQ(exe::RunColite(args, out, err), 1);
  EXPECT_EQ(out.str().find("PLAN recon_sfm"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(output));
}

}  // namespace
}  // namespace colite
