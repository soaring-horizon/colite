#include "colite/mvs/dense_map.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace colite::mvs {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

TEST(DenseDepthMapTest, StoresDepthValuesAndValidity) {
  DenseDepthMap depth_map(3, 2, 0.5F, 8.0F);

  EXPECT_EQ(depth_map.Width(), 3U);
  EXPECT_EQ(depth_map.Height(), 2U);
  EXPECT_EQ(depth_map.NumPixels(), 6U);
  EXPECT_FLOAT_EQ(depth_map.DepthMin(), 0.5F);
  EXPECT_FLOAT_EQ(depth_map.DepthMax(), 8.0F);
  EXPECT_FALSE(depth_map.IsValid(0, 0));

  depth_map.Set(1, 2, 4.5F);

  EXPECT_TRUE(depth_map.IsValid(1, 2));
  EXPECT_FLOAT_EQ(depth_map.Get(1, 2), 4.5F);
  EXPECT_EQ(depth_map.ValidPixelCount(), 1U);
  EXPECT_THROW(depth_map.Get(2, 0), std::out_of_range);
}

TEST(DenseDepthMapTest, RecomputesRangeFromPositiveFiniteDepths) {
  DenseDepthMap depth_map(2, 2);
  depth_map.Set(0, 0, 3.0F);
  depth_map.Set(0, 1, -1.0F);
  depth_map.Set(1, 0, 5.0F);
  depth_map.Set(1, 1, std::numeric_limits<float>::infinity());

  EXPECT_TRUE(depth_map.RecomputeDepthRange());

  EXPECT_FLOAT_EQ(depth_map.DepthMin(), 3.0F);
  EXPECT_FLOAT_EQ(depth_map.DepthMax(), 5.0F);
  EXPECT_EQ(depth_map.ValidPixelCount(), 2U);
}

TEST(DenseDepthMapTest, MutableDataUpdatesDepthValues) {
  DenseDepthMap depth_map(2, 2);

  std::vector<float>& data = depth_map.MutableData();
  ASSERT_EQ(data.size(), 4U);
  data[3] = 7.5F;

  EXPECT_TRUE(depth_map.IsValid(1, 1));
  EXPECT_FLOAT_EQ(depth_map.Get(1, 1), 7.5F);
  EXPECT_EQ(depth_map.ValidPixelCount(), 1U);
}

TEST(DenseMapIoTest, WritesColmapHeadersAndPlanarNormalChannels) {
  const auto root = std::filesystem::temp_directory_path() / "colite_dense_colmap_writer_test";
  std::filesystem::remove_all(root);
  DenseDepthMap depth(2, 1); depth.Set(0, 0, 3); depth.Set(0, 1, 5);
  NormalMap normal(2, 1);
  normal.Set(0, 0, Eigen::Vector3f(0.6F, 0, -0.8F));
  normal.Set(0, 1, Eigen::Vector3f(0, 0.8F, -0.6F));
  WriteDenseDepthMapColmapBinary(root / "depth.bin", depth);
  WriteNormalMapColmapBinary(root / "normal.bin", normal);
  EXPECT_EQ(ReadTextFile(root / "depth.bin").substr(0, 6), "2&1&1&");
  std::ifstream file(root / "normal.bin", std::ios::binary);
  file.seekg(6);
  float data[6]{}; file.read(reinterpret_cast<char*>(data), sizeof(data));
  const float expected[] = {0.6F, 0, 0, 0.8F, -0.8F, -0.6F};
  for (int i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(data[i], expected[i]);
  file.close();
  const auto loaded = ReadNormalMapBinary(root / "normal.bin");
  EXPECT_TRUE(loaded.Get(0, 0).isApprox(normal.Get(0, 0)));
  EXPECT_TRUE(loaded.Get(0, 1).isApprox(normal.Get(0, 1)));
  EXPECT_EQ(ReadDenseDepthMapSize(root / "depth.bin", true), std::make_pair(std::size_t{2}, std::size_t{1}));
  std::filesystem::resize_file(root / "depth.bin", 7);
  EXPECT_THROW(ReadDenseDepthMapSize(root / "depth.bin", true), std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST(DenseDepthMapTest, DownsizePreservesZAndNormalDirectionAndNeverUpsamples) {
  DenseDepthMap depth(4, 4); depth.Fill(3);
  NormalMap normal(4, 4); normal.Fill(Eigen::Vector3f(0.6F, 0, -0.8F));
  DownsizeDepthNormalMaps(2, 2, &depth, &normal);
  ASSERT_EQ(depth.Width(), 2); ASSERT_EQ(normal.Height(), 2);
  for (std::size_t r = 0; r < 2; ++r) for (std::size_t c = 0; c < 2; ++c) {
    EXPECT_NEAR(depth.Get(r, c), 3, 1e-6);
    EXPECT_TRUE(normal.Get(r, c).isApprox(Eigen::Vector3f(0.6F, 0, -0.8F), 1e-6F));
  }
  DownsizeDepthNormalMaps(16, 16, &depth, &normal);
  EXPECT_EQ(depth.Width(), 2);
  EXPECT_EQ(DenseMapDownsizeShape(7, 5, 4, 4), std::make_pair(std::size_t{4}, std::size_t{3}));
  EXPECT_THROW(DownsizeDepthNormalMaps(0, 0, &depth, &normal), std::invalid_argument);
}

TEST(DenseDepthMapTest, DownsizeUsesPrefilteredPixelCenters) {
  DenseDepthMap depth(4, 4);
  NormalMap normal(4, 4); normal.Fill(Eigen::Vector3f(0, 0, -1));
  for (std::size_t r = 0; r < 4; ++r) for (std::size_t c = 0; c < 4; ++c) depth.Set(r, c, 1 + r * 4 + c);
  DownsizeDepthNormalMaps(2, 2, &depth, &normal);
  // COLMAP 4.0's independent image downsampling fixture, shifted by +1 Z.
  const float expected[] = {3.76810598F, 5.66086388F, 11.3391361F, 13.2318935F};
  for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(depth.Data()[i], expected[i], 1e-3);
}

TEST(NormalMapTest, StoresNormalizedNormalsAndValidity) {
  NormalMap normal_map(2, 2);

  EXPECT_EQ(normal_map.Width(), 2U);
  EXPECT_EQ(normal_map.Height(), 2U);
  EXPECT_FALSE(normal_map.IsValid(0, 0));

  normal_map.SetNormalized(0, 1, Eigen::Vector3f(0.0F, 0.0F, -2.0F));

  EXPECT_TRUE(normal_map.IsValid(0, 1));
  EXPECT_NEAR(normal_map.Get(0, 1).norm(), 1.0F, 1e-6F);
  EXPECT_FLOAT_EQ(normal_map.Get(0, 1).z(), -1.0F);
  EXPECT_EQ(normal_map.ValidPixelCount(), 1U);
  EXPECT_THROW(normal_map.Set(3, 0, Eigen::Vector3f::UnitX()),
               std::out_of_range);
}

TEST(NormalMapTest, MutableDataUpdatesNormalValues) {
  NormalMap normal_map(2, 2);

  std::vector<Eigen::Vector3f>& data = normal_map.MutableData();
  ASSERT_EQ(data.size(), 4U);
  data[2] = Eigen::Vector3f(0.0F, 0.0F, -1.0F);

  EXPECT_TRUE(normal_map.IsValid(1, 0));
  EXPECT_FLOAT_EQ(normal_map.Get(1, 0).z(), -1.0F);
  EXPECT_EQ(normal_map.ValidPixelCount(), 1U);
}

TEST(DenseCostMapTest, StoresZeroAsValidCostAndInfinityAsInvalid) {
  DenseCostMap cost_map(3, 2);

  EXPECT_EQ(cost_map.Width(), 3U);
  EXPECT_EQ(cost_map.Height(), 2U);
  EXPECT_FALSE(cost_map.IsValid(0, 0));

  cost_map.Set(1, 2, 0.0F);

  EXPECT_TRUE(cost_map.IsValid(1, 2));
  EXPECT_FLOAT_EQ(cost_map.Get(1, 2), 0.0F);
  EXPECT_EQ(cost_map.ValidPixelCount(), 1U);
  EXPECT_THROW(cost_map.Get(2, 0), std::out_of_range);
}

TEST(DenseCostMapTest, MutableDataUpdatesCostValues) {
  DenseCostMap cost_map(2, 2);

  std::vector<float>& data = cost_map.MutableData();
  ASSERT_EQ(data.size(), 4U);
  data[1] = 0.125F;

  EXPECT_TRUE(cost_map.IsValid(0, 1));
  EXPECT_FLOAT_EQ(cost_map.Get(0, 1), 0.125F);
  EXPECT_EQ(cost_map.ValidPixelCount(), 1U);
}

TEST(DenseMapIoTest, RoundTripsDepthAndNormalMapsAsDebugText) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_dense_map_io_test";
  std::filesystem::create_directories(root);

  DenseDepthMap depth_map(2, 1, 0.1F, 10.0F);
  depth_map.Set(0, 0, 2.5F);
  depth_map.Set(0, 1, 4.5F);
  WriteDenseDepthMapText(root / "depth.txt", depth_map);

  const DenseDepthMap loaded_depth =
      ReadDenseDepthMapText(root / "depth.txt");
  EXPECT_EQ(loaded_depth.Width(), 2U);
  EXPECT_EQ(loaded_depth.Height(), 1U);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMin(), 0.1F);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMax(), 10.0F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(0, 1), 4.5F);

  NormalMap normal_map(1, 2);
  normal_map.SetNormalized(0, 0, Eigen::Vector3f(0.0F, 0.0F, -1.0F));
  normal_map.SetNormalized(1, 0, Eigen::Vector3f(1.0F, 0.0F, 0.0F));
  WriteNormalMapText(root / "normal.txt", normal_map);

  const NormalMap loaded_normal = ReadNormalMapText(root / "normal.txt");
  EXPECT_EQ(loaded_normal.Width(), 1U);
  EXPECT_EQ(loaded_normal.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded_normal.Get(0, 0).z(), -1.0F);
  EXPECT_FLOAT_EQ(loaded_normal.Get(1, 0).x(), 1.0F);
}

TEST(DenseMapIoTest, RoundTripsDepthAndNormalMapsAsBinary) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_dense_map_binary_test";
  std::filesystem::create_directories(root);

  DenseDepthMap depth_map(3, 2, 0.5F, 8.0F);
  depth_map.Set(0, 0, 1.5F);
  depth_map.Set(0, 1, 0.0F);
  depth_map.Set(1, 2, 6.25F);
  WriteDenseDepthMapBinary(root / "depth.bin", depth_map);

  const DenseDepthMap loaded_depth =
      ReadDenseDepthMapBinary(root / "depth.bin");
  EXPECT_EQ(loaded_depth.Width(), 3U);
  EXPECT_EQ(loaded_depth.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMin(), 0.5F);
  EXPECT_FLOAT_EQ(loaded_depth.DepthMax(), 8.0F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(0, 0), 1.5F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(0, 1), 0.0F);
  EXPECT_FLOAT_EQ(loaded_depth.Get(1, 2), 6.25F);
  EXPECT_EQ(loaded_depth.ValidPixelCount(), 2U);

  NormalMap normal_map(2, 2);
  normal_map.SetNormalized(0, 0, Eigen::Vector3f(0.0F, 0.0F, -1.0F));
  normal_map.SetNormalized(1, 1, Eigen::Vector3f(1.0F, 2.0F, 2.0F));
  WriteNormalMapBinary(root / "normal.bin", normal_map);

  const NormalMap loaded_normal = ReadNormalMapBinary(root / "normal.bin");
  EXPECT_EQ(loaded_normal.Width(), 2U);
  EXPECT_EQ(loaded_normal.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded_normal.Get(0, 0).z(), -1.0F);
  EXPECT_NEAR(loaded_normal.Get(1, 1).norm(), 1.0F, 1e-6F);
  EXPECT_FALSE(loaded_normal.IsValid(0, 1));
}

TEST(DenseMapIoTest, ReadsColmapDepthAndNormalBinaryMaps) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_colmap_dense_map_binary_test";
  std::filesystem::create_directories(root);

  {
    std::ofstream file(root / "depth.geometric.bin", std::ios::binary);
    file << "2&1&1&";
    const float values[] = {2.5F, 4.5F};
    file.write(reinterpret_cast<const char*>(values), sizeof(values));
  }
  const DenseDepthMap depth =
      ReadDenseDepthMapBinary(root / "depth.geometric.bin");
  EXPECT_EQ(depth.Width(), 2U);
  EXPECT_EQ(depth.Height(), 1U);
  EXPECT_FLOAT_EQ(depth.Get(0, 0), 2.5F);
  EXPECT_FLOAT_EQ(depth.Get(0, 1), 4.5F);
  EXPECT_FLOAT_EQ(depth.DepthMin(), 2.5F);
  EXPECT_FLOAT_EQ(depth.DepthMax(), 4.5F);

  {
    std::ofstream file(root / "normal.geometric.bin", std::ios::binary);
    file << "1&2&3&";
    // COLMAP serializes the complete x, y, then z channel.
    const float values[] = {0.0F, 1.0F, 0.0F, 0.0F, -1.0F, 0.0F};
    file.write(reinterpret_cast<const char*>(values), sizeof(values));
  }
  const NormalMap normal =
      ReadNormalMapBinary(root / "normal.geometric.bin");
  EXPECT_EQ(normal.Width(), 1U);
  EXPECT_EQ(normal.Height(), 2U);
  EXPECT_FLOAT_EQ(normal.Get(0, 0).z(), -1.0F);
  EXPECT_FLOAT_EQ(normal.Get(1, 0).x(), 1.0F);
}

TEST(DenseMapIoTest, WritesCostMapAsDebugText) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_cost_map_io_test";
  std::filesystem::create_directories(root);

  DenseCostMap cost_map(2, 1);
  cost_map.Set(0, 0, 0.25F);
  WriteDenseCostMapText(root / "cost.txt", cost_map);

  const std::string text = ReadTextFile(root / "cost.txt");
  EXPECT_NE(text.find("COLITE_DENSE_COST_MAP_TEXT"), std::string::npos);
  EXPECT_NE(text.find("2 1"), std::string::npos);
  EXPECT_NE(text.find("0.25"), std::string::npos);

  const DenseCostMap loaded = ReadDenseCostMapText(root / "cost.txt");
  EXPECT_EQ(loaded.Width(), 2U);
  EXPECT_EQ(loaded.Height(), 1U);
  EXPECT_FLOAT_EQ(loaded.Get(0, 0), 0.25F);
  EXPECT_FALSE(loaded.IsValid(0, 1));
}

TEST(DenseMapIoTest, RoundTripsCostMapAsBinaryFloatData) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_cost_map_binary_io_test";
  std::filesystem::create_directories(root);

  DenseCostMap cost_map(3, 2);
  cost_map.Set(0, 0, 0.25F);
  cost_map.Set(1, 2, 0.75F);
  WriteDenseCostMapBinary(root / "cost.bin", cost_map);

  const DenseCostMap loaded = ReadDenseCostMapBinary(root / "cost.bin");
  EXPECT_EQ(loaded.Width(), 3U);
  EXPECT_EQ(loaded.Height(), 2U);
  EXPECT_FLOAT_EQ(loaded.Get(0, 0), 0.25F);
  EXPECT_FLOAT_EQ(loaded.Get(1, 2), 0.75F);
  EXPECT_FALSE(loaded.IsValid(0, 1));
}

}  // namespace
}  // namespace colite::mvs
