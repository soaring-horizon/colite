#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace colite {

// Dynamic trial budget for sampling without replacement. The conservative
// multiplier matches the COLMAP 4.0 RANSAC default. Callers enforce their own
// minimum trial count, including when every observation is an inlier.
inline int RansacTrials(std::size_t inliers, std::size_t total,
                        std::size_t sample, double confidence, int cap) {
  if (confidence >= 1 || total < sample || inliers < sample) return cap;
  if (confidence <= 0) return 0;
  double probability = 1;
  for (std::size_t i = 0; i < sample; ++i) {
    probability *= static_cast<double>(inliers - i) / (total - i);
  }
  if (probability >= 1) return std::min(1, cap);
  if (probability <= 0) return cap;
  const double trials = 3 * std::log1p(-confidence) / std::log1p(-probability);
  return std::isfinite(trials) && trials < cap
             ? static_cast<int>(std::ceil(trials))
             : cap;
}

}  // namespace colite
