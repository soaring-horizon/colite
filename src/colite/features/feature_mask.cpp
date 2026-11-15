#include "colite/features/feature_mask.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <opencv2/imgcodecs.hpp>
#include "colite/io/camera_prior.h"
#include "colite/io/image_reader.h"
#include "colite/io/sfm_workspace.h"

namespace colite {
namespace {

cv::Mat ReadMask(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("Image mask file does not exist: " + path.string());
  auto mask = cv::imread(path.string(), cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION);
  if (mask.empty()) throw std::invalid_argument("Cannot decode image mask: " + path.string());
  return mask;
}

std::string FileKey(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot read image mask: " + path.string());
  std::uint64_t hash = 1469598103934665603ULL;
  std::array<char, 65536> buffer;
  while (stream) {
    stream.read(buffer.data(), buffer.size());
    for (std::streamsize i = 0; i < stream.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  if (!stream.eof()) throw std::runtime_error("Failed to fingerprint image mask: " + path.string());
  std::ostringstream key;
  key << std::hex << hash;
  return key.str();
}

std::filesystem::path ResolveMask(const std::filesystem::path& root,
    const std::filesystem::path& image, const std::filesystem::path& masks) {
  // Image names are logical paths below image_path, including symlinked
  // image files. Resolving symlink targets here changes the COLMAP mask name.
  const auto relative = std::filesystem::absolute(image).lexically_normal().lexically_relative(
      std::filesystem::absolute(root).lexically_normal());
  if (relative.empty() || *relative.begin() == "..")
    throw std::invalid_argument("Image mask requires an image below image_path: " + image.string());
  auto path = masks / (relative.generic_string() + ".png");
  if (std::filesystem::is_regular_file(path)) return path;
  auto alternate = relative;
  alternate.replace_extension(".png");
  if (std::filesystem::is_regular_file(masks / alternate)) return masks / alternate;
  throw std::invalid_argument("Image mask file does not exist: " + path.string());
}

bool Allows(const cv::Mat& mask, std::int64_t x, std::int64_t y) {
  return mask.empty() || (x >= 0 && y >= 0 && x < mask.cols && y < mask.rows &&
      mask.at<std::uint8_t>(static_cast<int>(y), static_cast<int>(x)) != 0);
}

}  // namespace

struct FeatureMasks::Impl {
  struct ImageMask {
    std::filesystem::path path;
    int orientation = 1;
    cv::Size encoded_size;
  };
  cv::Mat camera_mask;
  std::map<std::filesystem::path, ImageMask> images;
  std::string fingerprint;
};

FeatureMasks::FeatureMasks(const std::filesystem::path& image_root,
    const std::vector<std::filesystem::path>& images, const ImageMaskOptions& options) {
  if (options.Empty()) return;
  if (images.empty()) throw std::invalid_argument("Image masks require input images");
  auto impl = std::make_shared<Impl>();
  std::vector<std::pair<std::string, std::string>> keys{{"algorithm", "colmap_image_masks_v1"},
      {"mask_path", options.mask_path.generic_string()},
      {"camera_mask_path", options.camera_mask_path.generic_string()}};
  if (!options.camera_mask_path.empty()) {
    impl->camera_mask = ReadMask(options.camera_mask_path);
    keys.emplace_back("camera_mask_content", FileKey(options.camera_mask_path));
  }
  for (const auto& image : images) {
    Impl::ImageMask entry;
    entry.orientation = ReadExifOrientation(image);
    entry.encoded_size = ReadImageSizeWithExifOrientation(image);
    if (entry.encoded_size.empty()) throw std::invalid_argument("Cannot read masked image: " + image.string());
    if (entry.orientation >= 5) std::swap(entry.encoded_size.width, entry.encoded_size.height);
    if (!options.mask_path.empty()) {
      entry.path = ResolveMask(image_root, image, options.mask_path);
      ReadMask(entry.path);
      keys.emplace_back(entry.path.generic_string(), FileKey(entry.path));
    }
    impl->images.emplace(image.lexically_normal(), std::move(entry));
  }
  impl->fingerprint = StableConfigHash(keys);
  impl_ = std::move(impl);
}

std::string FeatureMasks::Fingerprint() const {
  return impl_ ? impl_->fingerprint : std::string{};
}

void FeatureMasks::Apply(const std::filesystem::path& image, FeatureSet* features) const {
  if (!impl_) return;
  if (!features) throw std::invalid_argument("Null features for image mask");
  const auto& entry = impl_->images.at(image.lexically_normal());
  const auto mask = entry.path.empty() ? cv::Mat{} : ReadMask(entry.path);
  std::vector<Point2DIdx> keep;
  keep.reserve(features->keypoints.size());
  for (std::size_t i = 0; i < features->keypoints.size(); ++i) {
    const auto& point = features->keypoints[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
        point.x < std::numeric_limits<int>::min() || double(point.x) > std::numeric_limits<int>::max() ||
        point.y < std::numeric_limits<int>::min() || double(point.y) > std::numeric_limits<int>::max()) continue;
    // COLMAP samples at integer truncation, after keypoints return to the
    // original camera resolution. Undo EXIF orientation on pixel indices.
    std::int64_t x = static_cast<int>(point.x), y = static_cast<int>(point.y);
    const std::int64_t w = entry.encoded_size.width, h = entry.encoded_size.height;
    const auto ox = x, oy = y;
    switch (entry.orientation) {
      case 2: x = w - 1 - ox; break;
      case 3: x = w - 1 - ox; y = h - 1 - oy; break;
      case 4: y = h - 1 - oy; break;
      case 5: x = oy; y = ox; break;
      case 6: x = oy; y = h - 1 - ox; break;
      case 7: x = w - 1 - oy; y = h - 1 - ox; break;
      case 8: x = w - 1 - oy; y = ox; break;
      default: break;
    }
    if (Allows(impl_->camera_mask, x, y) && Allows(mask, x, y))
      keep.push_back(static_cast<Point2DIdx>(i));
  }
  if (keep.size() == features->keypoints.size()) return;
  if (features->DescriptorRows() != static_cast<Eigen::Index>(features->keypoints.size()))
    throw std::invalid_argument("Image mask requires one descriptor per keypoint");
  if (features->source_keypoint_shapes.cols() > 0 &&
      features->source_keypoint_shapes.rows() != static_cast<Eigen::Index>(features->keypoints.size()))
    throw std::invalid_argument("Image mask requires one source shape per keypoint");
  if (features->descriptor_source) {
    features->descriptors = features->ReadDescriptorRows(keep);
    features->descriptor_source.reset();
  } else {
    for (std::size_t i = 0; i < keep.size(); ++i)
      if (keep[i] != i) features->descriptors.row(i) = features->descriptors.row(keep[i]);
    features->descriptors.conservativeResize(keep.size(), Eigen::NoChange);
  }
  for (std::size_t i = 0; i < keep.size(); ++i) {
    if (keep[i] == i) continue;
    features->keypoints[i] = features->keypoints[keep[i]];
    if (features->source_keypoint_shapes.cols() > 0)
      features->source_keypoint_shapes.row(i) = features->source_keypoint_shapes.row(keep[i]);
  }
  features->keypoints.resize(keep.size());
  if (features->source_keypoint_shapes.cols() > 0)
    features->source_keypoint_shapes.conservativeResize(keep.size(), Eigen::NoChange);
}

}  // namespace colite
