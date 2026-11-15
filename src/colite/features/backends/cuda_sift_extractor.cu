#include "backends/cuda_sift_extractor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <map>
#include <tuple>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>

#include "cuda_sift/cuda_sift.hpp"
#include "../sift_descriptor_conversion.h"

namespace colite {
namespace {

constexpr float kRadiansToDegrees = 57.295779513082320876F;
cv::Mat ToContinuousGray(const cv::Mat& image) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty.");
  }
  cv::Mat gray;
  if (image.channels() == 1) {
    gray = image;
  } else {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  }
  if (gray.type() != CV_8UC1) {
    gray.convertTo(gray, CV_8UC1);
  }
  return gray.isContinuous() ? gray : gray.clone();
}

void TransformVlFeatDescriptorToUbc(float* descriptor) {
  std::array<float, 128> original{};
  std::copy_n(descriptor, original.size(), original.data());
  constexpr std::array<int, 8> permutation{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 8; ++k) {
        descriptor[8 * (j + 4 * i) + permutation[k]] =
            original[8 * (j + 4 * i) + k];
      }
    }
  }
}

features::cuda::CudaSiftMode CudaMode(const SiftBackend backend) {
  return backend == SiftBackend::kCudaVlSift
             ? features::cuda::CudaSiftMode::VLFeat
             : features::cuda::CudaSiftMode::OpenCV;
}

// Several discrete OpenCV extrema can converge to the same localized keypoint.
// Limit the canonical point's distinct orientations, rather than retaining
// duplicate seeds that bypass the per-seed device limit.
void CanonicalizeCudaOpenCvFeatures(const int maximum_orientations,
                                   features::cuda::CudaSiftResult* result) {
  if (result->keypoints.empty()) return;
  std::vector<std::size_t> order(result->keypoints.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](const auto a, const auto b) {
    return std::abs(result->keypoints[a].response) > std::abs(result->keypoints[b].response);
  });
  std::map<std::tuple<float, float, float>, std::vector<float>> orientations;
  std::vector<std::size_t> selected;
  for (const auto index : order) {
    const auto& point = result->keypoints[index];
    auto& angles = orientations[{point.x, point.y, point.size}];
    if (angles.size() >= static_cast<std::size_t>(maximum_orientations) ||
        std::any_of(angles.begin(), angles.end(), [&](const float angle) {
          return std::abs(std::remainder(point.angle - angle, 360.0F)) < 1e-4F;
        })) continue;
    angles.push_back(point.angle);
    selected.push_back(index);
  }
  if (selected.size() == result->keypoints.size()) return;
  std::sort(selected.begin(), selected.end());
  std::vector<features::cuda::CudaSiftKeypoint> points;
  std::vector<float> descriptors;
  points.reserve(selected.size()); descriptors.reserve(selected.size() * 128);
  for (const auto index : selected) {
    points.push_back(result->keypoints[index]);
    descriptors.insert(descriptors.end(), result->descriptors.begin() + index * 128,
                       result->descriptors.begin() + (index + 1) * 128);
  }
  result->keypoints = std::move(points);
  result->descriptors = std::move(descriptors);
}

void RetainBestCudaFeatures(const std::size_t maximum,
                            features::cuda::CudaSiftResult* result) {
  if (maximum == 0 || result->keypoints.size() <= maximum) return;
  std::vector<std::size_t> order(result->keypoints.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](const std::size_t lhs,
                                                    const std::size_t rhs) {
    return std::abs(result->keypoints[lhs].response) >
           std::abs(result->keypoints[rhs].response);
  });

  std::vector<features::cuda::CudaSiftKeypoint> keypoints;
  std::vector<float> descriptors;
  keypoints.reserve(maximum);
  descriptors.reserve(maximum * 128);
  for (std::size_t rank = 0; rank < maximum; ++rank) {
    const std::size_t source = order[rank];
    keypoints.push_back(result->keypoints[source]);
    descriptors.insert(descriptors.end(),
                       result->descriptors.begin() + source * 128,
                       result->descriptors.begin() + (source + 1) * 128);
  }
  result->keypoints = std::move(keypoints);
  result->descriptors = std::move(descriptors);
}

void RetainCudaVlFeatLevels(const std::size_t maximum,
                           features::cuda::CudaSiftResult* result) {
  if (maximum == 0) return;
  using Level = std::pair<int, int>;
  std::map<Level, std::size_t> levels;
  for (const auto& point : result->keypoints) ++levels[{point.octave, point.layer}];
  if (levels.empty()) return;
  auto first = levels.begin()->first;
  std::size_t count = 0;
  // COLMAP SiftGPU (-tc2) repeats complete-level culling after expanding
  // multiple orientations. Count oriented rows here, not base detections.
  // Keeping the boundary level intact never splits a base's orientations.
  for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
    count += it->second;
    first = it->first;
    if (count > maximum) break;
  }
  std::vector<features::cuda::CudaSiftKeypoint> keypoints;
  std::vector<float> descriptors;
  for (std::size_t i = 0; i < result->keypoints.size(); ++i) {
    const auto& point = result->keypoints[i];
    if (Level{point.octave, point.layer} < first) continue;
    keypoints.push_back(point);
    descriptors.insert(descriptors.end(), result->descriptors.begin() + i * 128,
                        result->descriptors.begin() + (i + 1) * 128);
  }
  result->keypoints = std::move(keypoints);
  result->descriptors = std::move(descriptors);
}

}  // namespace

bool IsCudaSiftAvailable() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

FeatureSet ExtractCudaSiftFeatures(const cv::Mat& image,
                                   const SiftExtractionOptions& options) {
  if (!IsCudaSiftAvailable()) {
    throw std::runtime_error("CUDA SIFT was built, but no CUDA device is available.");
  }

  const SiftBackend backend = ResolveSiftBackend(options.backend);
  const bool vlfeat_mode = backend == SiftBackend::kCudaVlSift;
  features::cuda::CudaSiftConfig config;
  config.mode = CudaMode(backend);
  config.firstOctave = options.upsample_first_octave ? -1 : 0;
  config.octaveLayers = options.num_octave_layers;
  config.maxOctaves = options.num_octaves;
  config.sigma = static_cast<float>(options.sigma);
  config.contrastThreshold = static_cast<float>(options.peak_threshold * options.num_octave_layers);
  // GPU pyramid intensities are [0, 255]; VLFeat's public peak is normalized.
  config.peakThreshold = static_cast<float>(options.peak_threshold * 255.0);
  config.edgeThreshold = static_cast<float>(options.edge_threshold);
  // Budget base features by scale and keep all selected orientations.
  config.maxFeatures = vlfeat_mode ? 0 : options.max_num_features;
  config.maxOrientations = options.max_num_orientations;

  const cv::Mat gray = ToContinuousGray(image);
  features::cuda::CudaSiftResult cuda_result =
      features::cuda::CudaSift(config, options.cuda_device)
          .detectAndCompute(gray.ptr<std::uint8_t>(), gray.cols, gray.rows);
  if (vlfeat_mode) RetainCudaVlFeatLevels(static_cast<std::size_t>(std::max(0, options.max_num_features)), &cuda_result);
  else {
    CanonicalizeCudaOpenCvFeatures(options.max_num_orientations, &cuda_result);
    RetainBestCudaFeatures(static_cast<std::size_t>(options.max_num_features), &cuda_result);
  }

  FeatureSet result;
  result.keypoints.reserve(cuda_result.keypoints.size());
  for (const features::cuda::CudaSiftKeypoint& keypoint :
       cuda_result.keypoints) {
    FeatureKeypoint converted;
    // All public SIFT features use COLMAP pixel centers and Gaussian sigma.
    // OpenCV's internal size is a diameter; VLFeat already reports sigma.
    converted.x = keypoint.x + 0.5F;
    converted.y = keypoint.y + 0.5F;
    converted.scale = vlfeat_mode ? keypoint.size : keypoint.size * 0.5F;
    converted.orientation =
        vlfeat_mode ? keypoint.angle * kRadiansToDegrees : keypoint.angle;
    if (converted.orientation < 0.0F) converted.orientation += 360.0F;
    converted.response = std::abs(keypoint.response);
    result.keypoints.push_back(converted);
  }

  auto& descriptors = cuda_result.descriptors;
  if (descriptors.size() != result.keypoints.size() * 128) {
    throw std::runtime_error("CUDA SIFT descriptor buffer has unexpected size.");
  }
  if (vlfeat_mode) {
    for (std::size_t i = 0; i < result.keypoints.size(); ++i) {
      TransformVlFeatDescriptorToUbc(descriptors.data() + i * 128);
    }
  }
  result.descriptors = features::ConvertSiftDescriptors(
      descriptors.data(), result.keypoints.size(), options.use_root_sift);
  result.descriptor_normalization = options.use_root_sift ? DescriptorNormalization::kRootL1 : DescriptorNormalization::kL2;
  return result;
}

}  // namespace colite
