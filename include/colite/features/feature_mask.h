#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include "colite/core/feature.h"

namespace colite {

struct ImageMaskOptions {
  std::filesystem::path mask_path;
  std::filesystem::path camera_mask_path;
  bool Empty() const { return mask_path.empty() && camera_mask_path.empty(); }
};

// Immutable mask plan. Masks use the encoded image's pixel coordinates,
// matching COLMAP. Apply maps Colite's EXIF-oriented keypoints back to those
// coordinates. Zero and out-of-mask pixels are excluded; masks are not resized.
// Construction validates every requested mask before any cache can be reused.
class FeatureMasks {
 public:
  FeatureMasks(const std::filesystem::path& image_root,
               const std::vector<std::filesystem::path>& images,
               const ImageMaskOptions& options);
  std::string Fingerprint() const;
  void Apply(const std::filesystem::path& image, FeatureSet* features) const;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

}  // namespace colite
