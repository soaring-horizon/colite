#include "colite/mvs/dense_map.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <opencv2/imgproc.hpp>

namespace colite::mvs {
namespace {

constexpr char kDepthMapMagic[] = "COLITE_DENSE_DEPTH_MAP_TEXT";
constexpr char kNormalMapMagic[] = "COLITE_NORMAL_MAP_TEXT";
constexpr char kCostMapMagic[] = "COLITE_DENSE_COST_MAP_TEXT";
constexpr char kDepthMapBinaryMagic[] = "COLITE_DEPTH_MAP_BINARY";
constexpr char kNormalMapBinaryMagic[] = "COLITE_NORMAL_MAP_BINARY";
constexpr int kTextMapVersion = 1;
constexpr int kBinaryMapVersion = 1;

bool IsValidDepthValue(float depth) {
  return std::isfinite(depth) && depth > 0.0F;
}

bool IsValidNormalValue(const Eigen::Vector3f& normal) {
  return normal.allFinite() && normal.squaredNorm() > 0.0F;
}

void RequireStream(bool condition, const std::filesystem::path& path) {
  if (!condition) {
    throw std::runtime_error("Invalid dense map file: " + path.string());
  }
}

void RequireBinaryStream(bool condition, const std::filesystem::path& path) {
  if (!condition) {
    throw std::runtime_error("Invalid binary dense map file: " +
                             path.string());
  }
}

std::pair<std::size_t, std::size_t> ReadColmapDenseMapHeader(
    std::ifstream& file,
    std::size_t expected_channels,
    const std::filesystem::path& path) {
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t channels = 0;
  char delimiter = '\0';
  file >> width >> delimiter;
  RequireBinaryStream(delimiter == '&', path);
  file >> height >> delimiter;
  RequireBinaryStream(delimiter == '&', path);
  file >> channels >> delimiter;
  RequireBinaryStream(
      static_cast<bool>(file) && delimiter == '&' && width > 0 &&
          height > 0 && channels == expected_channels &&
          width <= std::numeric_limits<std::size_t>::max() / height,
      path);
  return {width, height};
}

}  // namespace

void WriteDenseDepthMapColmapBinary(const std::filesystem::path& path,
                                   const DenseDepthMap& map) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  file << map.Width() << '&' << map.Height() << "&1&";
  file.write(reinterpret_cast<const char*>(map.Data().data()),
             static_cast<std::streamsize>(map.NumPixels() * sizeof(float)));
  RequireBinaryStream(static_cast<bool>(file), path);
}

void WriteNormalMapColmapBinary(const std::filesystem::path& path,
                               const NormalMap& map) {
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  file << map.Width() << '&' << map.Height() << "&3&";
  std::vector<float> channel(map.NumPixels());
  for (int component = 0; component < 3; ++component) {
    for (std::size_t i = 0; i < channel.size(); ++i) channel[i] = map.Data()[i][component];
    file.write(reinterpret_cast<const char*>(channel.data()),
               static_cast<std::streamsize>(channel.size() * sizeof(float)));
  }
  RequireBinaryStream(static_cast<bool>(file), path);
}

std::pair<std::size_t, std::size_t> ReadDenseDepthMapSize(
    const std::filesystem::path& path, bool binary) {
  std::ifstream file(path, std::ios::binary);
  RequireStream(static_cast<bool>(file), path);
  std::size_t width = 0, height = 0;
  if (binary && std::isdigit(static_cast<unsigned char>(file.peek()))) {
    std::tie(width, height) = ReadColmapDenseMapHeader(file, 1, path);
  } else {
    std::string magic;
    int version = 0;
    float min_depth = 0, max_depth = 0;
    file >> magic >> version >> width >> height >> min_depth >> max_depth;
    RequireStream(static_cast<bool>(file) && version == 1 &&
        magic == (binary ? kDepthMapBinaryMagic : kDepthMapMagic), path);
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  RequireStream(width > 0 && height > 0 &&
      width <= std::numeric_limits<std::size_t>::max() / height / sizeof(float), path);
  if (binary) {
    const auto offset = file.tellg();
    RequireBinaryStream(offset >= 0 && std::filesystem::file_size(path) >=
        static_cast<std::uintmax_t>(offset) && std::filesystem::file_size(path) -
        static_cast<std::uintmax_t>(offset) == width * height * sizeof(float), path);
  }
  return {width, height};
}

std::pair<std::size_t, std::size_t> DenseMapDownsizeShape(
    std::size_t width, std::size_t height,
    std::size_t max_width, std::size_t max_height) {
  if (width == 0 || height == 0 || max_width == 0 || max_height == 0) {
    throw std::invalid_argument("Dense map dimensions must be positive");
  }
  const double scale = std::min({1.0, double(max_width) / width, double(max_height) / height});
  return {std::max<std::size_t>(1, std::lround(width * scale)),
          std::max<std::size_t>(1, std::lround(height * scale))};
}

void DownsizeDepthNormalMaps(std::size_t max_width, std::size_t max_height,
                             DenseDepthMap* depth, NormalMap* normals) {
  if (depth == nullptr || normals == nullptr || depth->Empty() ||
      depth->Width() != normals->Width() || depth->Height() != normals->Height()) {
    throw std::invalid_argument("Depth and normal maps must have matching nonempty dimensions");
  }
  const auto [width, height] = DenseMapDownsizeShape(
      depth->Width(), depth->Height(), max_width, max_height);
  if (width == depth->Width() && height == depth->Height()) return;
  const auto kernel = [](double scale) {
    const double sigma = std::max(1e-8, 0.5 * (scale - 1.0));
    const int radius = static_cast<int>(std::ceil(3.0 * sigma));
    return cv::getGaussianKernel(2 * radius + 1, sigma, CV_32F);
  };
  const auto kx = kernel(double(depth->Width()) / width);
  const auto ky = kernel(double(depth->Height()) / height);
  const auto downsample = [&](const cv::Mat& input) {
    cv::Mat smooth, result;
    cv::sepFilter2D(input, smooth, -1, kx, ky, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::resize(smooth, result, cv::Size(static_cast<int>(width), static_cast<int>(height)),
               0, 0, cv::INTER_LINEAR);
    return result;
  };
  const cv::Mat depth_pixels(static_cast<int>(depth->Height()),
      static_cast<int>(depth->Width()), CV_32F, depth->MutableData().data());
  cv::Mat normal_pixels(depth_pixels.size(), CV_32FC3);
  for (std::size_t row = 0; row < depth->Height(); ++row) {
    for (std::size_t col = 0; col < depth->Width(); ++col) {
      const auto normal = normals->Get(row, col);
      normal_pixels.at<cv::Vec3f>(row, col) = {normal.x(), normal.y(), normal.z()};
    }
  }
  const auto resized_depth = downsample(depth_pixels);
  const auto resized_normal = downsample(normal_pixels);
  DenseDepthMap output_depth(width, height);
  std::copy_n(resized_depth.ptr<float>(), width * height, output_depth.MutableData().begin());
  output_depth.RecomputeDepthRange();
  NormalMap output_normal(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      const auto value = resized_normal.at<cv::Vec3f>(row, col);
      output_normal.SetNormalized(row, col, Eigen::Vector3f(value[0], value[1], value[2]));
    }
  }
  *depth = std::move(output_depth);
  *normals = std::move(output_normal);
}

DenseDepthMap::DenseDepthMap(std::size_t width,
                             std::size_t height,
                             float depth_min,
                             float depth_max)
    : width_(width),
      height_(height),
      depth_min_(depth_min),
      depth_max_(depth_max),
      data_(width * height, 0.0F) {}

float DenseDepthMap::DepthMin() const {
  return depth_min_;
}

float DenseDepthMap::DepthMax() const {
  return depth_max_;
}

void DenseDepthMap::SetDepthRange(float depth_min, float depth_max) {
  depth_min_ = depth_min;
  depth_max_ = depth_max;
}

void DenseDepthMap::Resize(std::size_t width,
                           std::size_t height,
                           float value) {
  width_ = width;
  height_ = height;
  data_.assign(width_ * height_, value);
}

void DenseDepthMap::Fill(float value) {
  std::fill(data_.begin(), data_.end(), value);
}

void DenseDepthMap::Set(std::size_t row, std::size_t col, float depth) {
  data_.at(Offset(row, col)) = depth;
}

std::size_t DenseDepthMap::ValidPixelCount() const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(), IsValidDepthValue));
}

bool DenseDepthMap::RecomputeDepthRange() {
  float min_depth = std::numeric_limits<float>::infinity();
  float max_depth = 0.0F;
  for (const float depth : data_) {
    if (!IsValidDepthValue(depth)) {
      continue;
    }
    min_depth = std::min(min_depth, depth);
    max_depth = std::max(max_depth, depth);
  }
  if (!std::isfinite(min_depth)) {
    depth_min_ = -1.0F;
    depth_max_ = -1.0F;
    return false;
  }
  depth_min_ = min_depth;
  depth_max_ = max_depth;
  return true;
}

std::vector<float>& DenseDepthMap::MutableData() {
  return data_;
}

NormalMap::NormalMap(std::size_t width, std::size_t height)
    : width_(width),
      height_(height),
      data_(width * height, Eigen::Vector3f::Zero()) {}

void NormalMap::Resize(std::size_t width, std::size_t height) {
  width_ = width;
  height_ = height;
  data_.assign(width_ * height_, Eigen::Vector3f::Zero());
}

void NormalMap::Fill(const Eigen::Vector3f& normal) {
  std::fill(data_.begin(), data_.end(), normal);
}

void NormalMap::Set(std::size_t row,
                    std::size_t col,
                    const Eigen::Vector3f& normal) {
  data_.at(Offset(row, col)) = normal;
}

void NormalMap::SetNormalized(std::size_t row,
                              std::size_t col,
                              const Eigen::Vector3f& normal) {
  if (!IsValidNormalValue(normal)) {
    Set(row, col, Eigen::Vector3f::Zero());
    return;
  }
  Set(row, col, normal.normalized());
}

std::size_t NormalMap::ValidPixelCount() const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(), IsValidNormalValue));
}

std::vector<Eigen::Vector3f>& NormalMap::MutableData() {
  return data_;
}

DenseCostMap::DenseCostMap(std::size_t width,
                           std::size_t height,
                           float invalid_cost)
    : width_(width), height_(height), data_(width * height, invalid_cost) {}

void DenseCostMap::Resize(std::size_t width,
                          std::size_t height,
                          float invalid_cost) {
  width_ = width;
  height_ = height;
  data_.assign(width_ * height_, invalid_cost);
}

void DenseCostMap::Fill(float cost) {
  std::fill(data_.begin(), data_.end(), cost);
}

void DenseCostMap::Set(std::size_t row, std::size_t col, float cost) {
  data_.at(Offset(row, col)) = cost;
}

std::size_t DenseCostMap::ValidPixelCount() const {
  return static_cast<std::size_t>(
      std::count_if(data_.begin(), data_.end(), [](float cost) {
        return std::isfinite(cost) && cost >= 0.0F;
      }));
}

std::vector<float>& DenseCostMap::MutableData() {
  return data_;
}

void WriteDenseDepthMapText(const std::filesystem::path& path,
                            const DenseDepthMap& depth_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense depth map: " +
                             path.string());
  }

  file << kDepthMapMagic << " " << kTextMapVersion << "\n"
       << depth_map.Width() << " " << depth_map.Height() << "\n"
       << std::setprecision(9) << depth_map.DepthMin() << " "
       << depth_map.DepthMax() << "\n";
  for (std::size_t row = 0; row < depth_map.Height(); ++row) {
    for (std::size_t col = 0; col < depth_map.Width(); ++col) {
      if (col > 0) {
        file << " ";
      }
      file << std::setprecision(9) << depth_map.Get(row, col);
    }
    file << "\n";
  }
}

DenseDepthMap ReadDenseDepthMapText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to read dense depth map: " +
                             path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireStream(magic == kDepthMapMagic && version == kTextMapVersion, path);

  std::size_t width = 0;
  std::size_t height = 0;
  float depth_min = -1.0F;
  float depth_max = -1.0F;
  file >> width >> height >> depth_min >> depth_max;
  RequireStream(static_cast<bool>(file), path);

  DenseDepthMap depth_map(width, height, depth_min, depth_max);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      float depth = 0.0F;
      file >> depth;
      RequireStream(static_cast<bool>(file), path);
      depth_map.Set(row, col, depth);
    }
  }
  return depth_map;
}

void WriteDenseDepthMapBinary(const std::filesystem::path& path,
                              const DenseDepthMap& depth_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write binary dense depth map: " +
                             path.string());
  }

  file << kDepthMapBinaryMagic << " " << kBinaryMapVersion << "\n"
       << depth_map.Width() << " " << depth_map.Height() << " "
       << std::setprecision(9) << depth_map.DepthMin() << " "
       << depth_map.DepthMax() << "\n";
  if (!depth_map.Data().empty()) {
    file.write(reinterpret_cast<const char*>(depth_map.Data().data()),
               static_cast<std::streamsize>(depth_map.Data().size() *
                                            sizeof(float)));
  }
  RequireBinaryStream(static_cast<bool>(file), path);
}

DenseDepthMap ReadDenseDepthMapBinary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read binary dense depth map: " +
                             path.string());
  }

  if (std::isdigit(static_cast<unsigned char>(file.peek())) != 0) {
    const auto [width, height] = ReadColmapDenseMapHeader(file, 1, path);
    DenseDepthMap depth_map(width, height);
    std::vector<float>& values = depth_map.MutableData();
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    RequireBinaryStream(static_cast<bool>(file), path);
    depth_map.RecomputeDepthRange();
    return depth_map;
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireBinaryStream(magic == kDepthMapBinaryMagic &&
                          version == kBinaryMapVersion,
                      path);

  std::size_t width = 0;
  std::size_t height = 0;
  float depth_min = -1.0F;
  float depth_max = -1.0F;
  file >> width >> height >> depth_min >> depth_max;
  RequireBinaryStream(static_cast<bool>(file), path);
  file.get();

  DenseDepthMap depth_map(width, height, depth_min, depth_max);
  if (!depth_map.Data().empty()) {
    std::vector<float>& values = depth_map.MutableData();
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    RequireBinaryStream(static_cast<bool>(file), path);
  }
  return depth_map;
}

void WriteNormalMapText(const std::filesystem::path& path,
                        const NormalMap& normal_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write normal map: " + path.string());
  }

  file << kNormalMapMagic << " " << kTextMapVersion << "\n"
       << normal_map.Width() << " " << normal_map.Height() << "\n";
  for (std::size_t row = 0; row < normal_map.Height(); ++row) {
    for (std::size_t col = 0; col < normal_map.Width(); ++col) {
      const Eigen::Vector3f normal = normal_map.Get(row, col);
      if (col > 0) {
        file << " ";
      }
      file << std::setprecision(9) << normal.x() << " " << normal.y() << " "
           << normal.z();
    }
    file << "\n";
  }
}

NormalMap ReadNormalMapText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to read normal map: " + path.string());
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireStream(magic == kNormalMapMagic && version == kTextMapVersion, path);

  std::size_t width = 0;
  std::size_t height = 0;
  file >> width >> height;
  RequireStream(static_cast<bool>(file), path);

  NormalMap normal_map(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      Eigen::Vector3f normal;
      file >> normal.x() >> normal.y() >> normal.z();
      RequireStream(static_cast<bool>(file), path);
      normal_map.Set(row, col, normal);
    }
  }
  return normal_map;
}

void WriteNormalMapBinary(const std::filesystem::path& path,
                          const NormalMap& normal_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write binary normal map: " +
                             path.string());
  }

  file << kNormalMapBinaryMagic << " " << kBinaryMapVersion << "\n"
       << normal_map.Width() << " " << normal_map.Height() << "\n";
  static_assert(sizeof(Eigen::Vector3f) == 3 * sizeof(float),
                "NormalMap binary layout requires packed Vector3f values.");
  if (!normal_map.Data().empty()) {
    file.write(
        reinterpret_cast<const char*>(normal_map.Data().data()),
        static_cast<std::streamsize>(normal_map.Data().size() *
                                     sizeof(Eigen::Vector3f)));
  }
  RequireBinaryStream(static_cast<bool>(file), path);
}

NormalMap ReadNormalMapBinary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read binary normal map: " +
                             path.string());
  }

  if (std::isdigit(static_cast<unsigned char>(file.peek())) != 0) {
    const auto [width, height] = ReadColmapDenseMapHeader(file, 3, path);
    NormalMap normal_map(width, height);
    std::vector<Eigen::Vector3f>& values = normal_map.MutableData();
    // COLMAP stores multi-channel Mat data one complete channel at a time,
    // whereas NormalMap keeps xyz interleaved per pixel.
    std::vector<float> channel(values.size());
    for (Eigen::Index component = 0; component < 3; ++component) {
      file.read(reinterpret_cast<char*>(channel.data()),
                static_cast<std::streamsize>(channel.size() * sizeof(float)));
      RequireBinaryStream(static_cast<bool>(file), path);
      for (std::size_t pixel = 0; pixel < values.size(); ++pixel) {
        values[pixel][component] = channel[pixel];
      }
    }
    return normal_map;
  }

  std::string magic;
  int version = 0;
  file >> magic >> version;
  RequireBinaryStream(magic == kNormalMapBinaryMagic &&
                          version == kBinaryMapVersion,
                      path);

  std::size_t width = 0;
  std::size_t height = 0;
  file >> width >> height;
  RequireBinaryStream(static_cast<bool>(file), path);
  file.get();

  NormalMap normal_map(width, height);
  static_assert(sizeof(Eigen::Vector3f) == 3 * sizeof(float),
                "NormalMap binary layout requires packed Vector3f values.");
  if (!normal_map.Data().empty()) {
    std::vector<Eigen::Vector3f>& values = normal_map.MutableData();
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() *
                                     sizeof(Eigen::Vector3f)));
    RequireBinaryStream(static_cast<bool>(file), path);
  }
  return normal_map;
}

void WriteDenseCostMapText(const std::filesystem::path& path,
                           const DenseCostMap& cost_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense cost map: " +
                             path.string());
  }

  file << kCostMapMagic << " " << kTextMapVersion << "\n"
       << cost_map.Width() << " " << cost_map.Height() << "\n";
  for (std::size_t row = 0; row < cost_map.Height(); ++row) {
    for (std::size_t col = 0; col < cost_map.Width(); ++col) {
      if (col > 0) {
        file << " ";
      }
      file << std::setprecision(9) << cost_map.Get(row, col);
    }
    file << "\n";
  }
}

DenseCostMap ReadDenseCostMapText(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to read dense cost map: " + path.string());
  }
  std::string magic;
  int version = 0;
  std::size_t width = 0;
  std::size_t height = 0;
  file >> magic >> version >> width >> height;
  RequireStream(magic == kCostMapMagic && version == kTextMapVersion, path);
  DenseCostMap cost_map(width, height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      std::string token;
      file >> token;
      RequireStream(static_cast<bool>(file), path);
      float value = 0.0F;
      try {
        value = std::stof(token);
      } catch (const std::exception&) {
        throw std::runtime_error("Invalid dense map file: " + path.string());
      }
      cost_map.Set(row, col, value);
    }
  }
  return cost_map;
}

void WriteDenseCostMapBinary(const std::filesystem::path& path,
                             const DenseCostMap& cost_map) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write binary dense cost map: " +
                             path.string());
  }
  file << kCostMapMagic << " " << kBinaryMapVersion << "\n"
       << cost_map.Width() << " " << cost_map.Height() << "\n";
  if (!cost_map.Data().empty()) {
    file.write(reinterpret_cast<const char*>(cost_map.Data().data()),
               static_cast<std::streamsize>(cost_map.Data().size() *
                                            sizeof(float)));
  }
  RequireBinaryStream(static_cast<bool>(file), path);
}

DenseCostMap ReadDenseCostMapBinary(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to read binary dense cost map: " +
                             path.string());
  }
  std::string magic;
  int version = 0;
  std::size_t width = 0;
  std::size_t height = 0;
  file >> magic >> version >> width >> height;
  RequireBinaryStream(
      magic == kCostMapMagic && version == kBinaryMapVersion, path);
  file.get();
  DenseCostMap cost_map(width, height);
  if (!cost_map.Data().empty()) {
    std::vector<float>& values = cost_map.MutableData();
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    RequireBinaryStream(static_cast<bool>(file), path);
  }
  return cost_map;
}

}  // namespace colite::mvs
