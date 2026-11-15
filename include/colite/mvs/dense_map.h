#pragma once

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

#include <Eigen/Core>

namespace colite::mvs {

// 稠密深度图。像素值采用相机坐标系下的正深度，0/负数/非有限值都视为无效深度。
class DenseDepthMap {
 public:
  DenseDepthMap() = default;
  DenseDepthMap(std::size_t width,
                std::size_t height,
                float depth_min = -1.0F,
                float depth_max = -1.0F);

  std::size_t Width() const {
    return width_;
  }
  std::size_t Height() const {
    return height_;
  }
  std::size_t NumPixels() const {
    return data_.size();
  }
  bool Empty() const {
    return data_.empty();
  }

  float DepthMin() const;
  float DepthMax() const;
  void SetDepthRange(float depth_min, float depth_max);

  void Resize(std::size_t width, std::size_t height, float value = 0.0F);
  void Fill(float value);

  float Get(std::size_t row, std::size_t col) const {
    return data_.at(Offset(row, col));
  }
  void Set(std::size_t row, std::size_t col, float depth);
  bool IsValid(std::size_t row, std::size_t col) const {
    const float depth = Get(row, col);
    return std::isfinite(depth) && depth > 0.0F;
  }
  std::size_t ValidPixelCount() const;

  // 从当前有效深度重新计算深度范围。没有有效深度时返回 false。
  bool RecomputeDepthRange();

  const std::vector<float>& Data() const {
    return data_;
  }
  std::vector<float>& MutableData();

 private:
  std::size_t Offset(std::size_t row, std::size_t col) const {
    if (row >= height_ || col >= width_) {
      throw std::out_of_range("DenseDepthMap pixel index is out of range");
    }
    return row * width_ + col;
  }

  std::size_t width_ = 0;
  std::size_t height_ = 0;
  float depth_min_ = -1.0F;
  float depth_max_ = -1.0F;
  std::vector<float> data_;
};

// 稠密法线图。每个像素保存相机坐标系下的单位法线；零向量或非有限向量视为无效。
class NormalMap {
 public:
  NormalMap() = default;
  NormalMap(std::size_t width, std::size_t height);

  std::size_t Width() const {
    return width_;
  }
  std::size_t Height() const {
    return height_;
  }
  std::size_t NumPixels() const {
    return data_.size();
  }
  bool Empty() const {
    return data_.empty();
  }

  void Resize(std::size_t width, std::size_t height);
  void Fill(const Eigen::Vector3f& normal);

  Eigen::Vector3f Get(std::size_t row, std::size_t col) const {
    return data_.at(Offset(row, col));
  }
  void Set(std::size_t row, std::size_t col, const Eigen::Vector3f& normal);
  void SetNormalized(std::size_t row,
                     std::size_t col,
                     const Eigen::Vector3f& normal);
  bool IsValid(std::size_t row, std::size_t col) const {
    const Eigen::Vector3f normal = Get(row, col);
    return normal.allFinite() && normal.squaredNorm() > 0.0F;
  }
  std::size_t ValidPixelCount() const;

  const std::vector<Eigen::Vector3f>& Data() const {
    return data_;
  }
  std::vector<Eigen::Vector3f>& MutableData();

 private:
  std::size_t Offset(std::size_t row, std::size_t col) const {
    if (row >= height_ || col >= width_) {
      throw std::out_of_range("NormalMap pixel index is out of range");
    }
    return row * width_ + col;
  }

  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::vector<Eigen::Vector3f> data_;
};

// 稠密代价图。0 是合法的最好代价，因此使用正无穷作为默认无效值。
class DenseCostMap {
 public:
  DenseCostMap() = default;
  DenseCostMap(std::size_t width,
               std::size_t height,
               float invalid_cost = std::numeric_limits<float>::infinity());

  std::size_t Width() const {
    return width_;
  }
  std::size_t Height() const {
    return height_;
  }
  std::size_t NumPixels() const {
    return data_.size();
  }
  bool Empty() const {
    return data_.empty();
  }

  void Resize(std::size_t width,
              std::size_t height,
              float invalid_cost = std::numeric_limits<float>::infinity());
  void Fill(float cost);

  float Get(std::size_t row, std::size_t col) const {
    return data_.at(Offset(row, col));
  }
  void Set(std::size_t row, std::size_t col, float cost);
  bool IsValid(std::size_t row, std::size_t col) const {
    const float cost = Get(row, col);
    return std::isfinite(cost) && cost >= 0.0F;
  }
  std::size_t ValidPixelCount() const;

  const std::vector<float>& Data() const {
    return data_;
  }
  std::vector<float>& MutableData();

 private:
  std::size_t Offset(std::size_t row, std::size_t col) const {
    if (row >= height_ || col >= width_) {
      throw std::out_of_range("DenseCostMap pixel index is out of range");
    }
    return row * width_ + col;
  }

  std::size_t width_ = 0;
  std::size_t height_ = 0;
  std::vector<float> data_;
};

void WriteDenseDepthMapText(const std::filesystem::path& path,
                            const DenseDepthMap& depth_map);
DenseDepthMap ReadDenseDepthMapText(const std::filesystem::path& path);

void WriteDenseDepthMapBinary(const std::filesystem::path& path,
                              const DenseDepthMap& depth_map);
DenseDepthMap ReadDenseDepthMapBinary(const std::filesystem::path& path);
// COLMAP header (width&height&channels&) and planar float32 payload.
void WriteDenseDepthMapColmapBinary(const std::filesystem::path& path,
                                   const DenseDepthMap& depth_map);
void WriteNormalMapColmapBinary(const std::filesystem::path& path,
                               const NormalMap& normal_map);
std::pair<std::size_t, std::size_t> ReadDenseDepthMapSize(
    const std::filesystem::path& path, bool binary);
std::pair<std::size_t, std::size_t> DenseMapDownsizeShape(
    std::size_t width, std::size_t height,
    std::size_t max_width, std::size_t max_height);
// Gaussian prefilter and pixel-center bilinear resampling; never upsamples.
// Camera Z remains in the same units. Normals are renormalized afterwards.
void DownsizeDepthNormalMaps(std::size_t max_width, std::size_t max_height,
                             DenseDepthMap* depth_map, NormalMap* normal_map);

void WriteNormalMapText(const std::filesystem::path& path,
                        const NormalMap& normal_map);
NormalMap ReadNormalMapText(const std::filesystem::path& path);

void WriteNormalMapBinary(const std::filesystem::path& path,
                          const NormalMap& normal_map);
NormalMap ReadNormalMapBinary(const std::filesystem::path& path);

void WriteDenseCostMapText(const std::filesystem::path& path,
                           const DenseCostMap& cost_map);
DenseCostMap ReadDenseCostMapText(const std::filesystem::path& path);
void WriteDenseCostMapBinary(const std::filesystem::path& path,
                             const DenseCostMap& cost_map);
DenseCostMap ReadDenseCostMapBinary(const std::filesystem::path& path);

}  // namespace colite::mvs
