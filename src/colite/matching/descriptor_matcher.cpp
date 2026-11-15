#include "colite/matching/descriptor_matcher.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#if defined(COLITE_ENABLE_AVX2)
#include <immintrin.h>
#endif

#include <Eigen/Core>
#include <flann/flann.hpp>

#include "descriptor_matcher_cuda.h"

namespace colite {
namespace {

// COLMAP 将 SIFT 描述子量化到长度约为 512 的 uint8 空间中，匹配阈值也在该
// 空间下定义。
constexpr int kSqSiftDescriptorNorm = 512 * 512;
constexpr int kFlannNumTrees = 4;
constexpr int kFlannNumLeavesToVisit = 128;

using DescriptorByteMatrix =
    Eigen::Matrix<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using DistanceMatrix =
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FlannIndex = flann::Index<flann::L2<std::uint8_t>>;

void ValidateSiftFeatureSet(const FeatureSet& features,
                            const FeatureMatcherType matcher_type) {
  if (!IsFeatureMatcherCompatible(features.extractor_type, matcher_type)) {
    throw std::invalid_argument(
        "Feature extractor type is incompatible with matcher type.");
  }
  if (matcher_type != FeatureMatcherType::kSiftBruteForce ||
      features.extractor_type != FeatureExtractorType::kSift ||
      features.descriptor_scalar_type != DescriptorScalarType::kFloat32 ||
      features.descriptor_metric != DescriptorMetric::kSiftAngular) {
    throw std::invalid_argument(
        "The selected matcher is not implemented for this descriptor contract.");
  }
  if (!features.keypoints.empty() && features.descriptors.size() != 0 &&
      features.descriptors.rows() !=
          static_cast<Eigen::Index>(features.keypoints.size())) {
    throw std::invalid_argument(
        "Feature keypoint and descriptor row counts do not match.");
  }
}

std::uint8_t ClampToByte(const float value) {
  if (!std::isfinite(value) || value <= 0.0F) {
    return 0;
  }
  if (value >= 255.0F) {
    return 255;
  }
  return static_cast<std::uint8_t>(std::round(value));
}

DescriptorByteMatrix QuantizeDescriptors(const DescriptorMatrix& descriptors) {
  DescriptorByteMatrix quantized(descriptors.rows(), descriptors.cols());
  if (descriptors.size() == 0) {
    return quantized;
  }

  bool unit_normalized = true;
  for (Eigen::Index idx = 0; idx < descriptors.size(); ++idx) {
    const float value = descriptors.data()[idx];
    if (std::isfinite(value) && value > 1.0001F) {
      unit_normalized = false;
      break;
    }
  }

  const float scale = unit_normalized ? 512.0F : 1.0F;
  for (Eigen::Index row = 0; row < descriptors.rows(); ++row) {
    for (Eigen::Index col = 0; col < descriptors.cols(); ++col) {
      quantized(row, col) = ClampToByte(scale * descriptors(row, col));
    }
  }
  return quantized;
}

#if defined(COLITE_ENABLE_AVX2) && defined(__GNUC__)
__attribute__((target("avx2")))
#endif
int DotProduct(const DescriptorByteMatrix& descriptors1,
               const Eigen::Index row1,
               const DescriptorByteMatrix& descriptors2,
               const Eigen::Index row2) {
#if defined(COLITE_ENABLE_AVX2)
  const std::uint8_t* lhs = descriptors1.row(row1).data();
  const std::uint8_t* rhs = descriptors2.row(row2).data();
  const Eigen::Index dimensions = descriptors1.cols();
  Eigen::Index col = 0;
  __m256i accumulator = _mm256_setzero_si256();
  for (; col + 32 <= dimensions; col += 32) {
    const __m256i lhs_bytes =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs + col));
    const __m256i rhs_bytes =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs + col));
    const __m256i lhs_low =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lhs_bytes));
    const __m256i rhs_low =
        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rhs_bytes));
    const __m256i lhs_high =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(lhs_bytes, 1));
    const __m256i rhs_high =
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rhs_bytes, 1));
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(lhs_low, rhs_low));
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(lhs_high, rhs_high));
  }
  const __m128i low = _mm256_castsi256_si128(accumulator);
  const __m128i high = _mm256_extracti128_si256(accumulator, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  int dot_product = _mm_cvtsi128_si32(sum);
  for (; col < dimensions; ++col) {
    dot_product += static_cast<int>(lhs[col]) * static_cast<int>(rhs[col]);
  }
  return dot_product;
#else
  int dot_product = 0;
  for (Eigen::Index col = 0; col < descriptors1.cols(); ++col) {
    dot_product += static_cast<int>(descriptors1(row1, col)) *
                   static_cast<int>(descriptors2(row2, col));
  }
  return dot_product;
#endif
}

std::vector<int> FindBestMatchesOneWayBruteForce(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const double max_ratio,
    const double max_distance) {
  const float inv_sq_descriptor_norm =
      static_cast<float>(1.0 / kSqSiftDescriptorNorm);

  std::vector<int> matches(static_cast<std::size_t>(query_descriptors.rows()),
                           -1);

  for (Eigen::Index query_idx = 0; query_idx < query_descriptors.rows();
       ++query_idx) {
    int best_train_idx = -1;
    int best_dot_product = -1;
    int second_best_dot_product = -1;

    for (Eigen::Index train_idx = 0; train_idx < train_descriptors.rows();
         ++train_idx) {
      const int dot_product =
          DotProduct(query_descriptors, query_idx, train_descriptors,
                     train_idx);
      if (dot_product > best_dot_product) {
        best_train_idx = static_cast<int>(train_idx);
        second_best_dot_product = best_dot_product;
        best_dot_product = dot_product;
      } else if (dot_product > second_best_dot_product) {
        second_best_dot_product = dot_product;
      }
    }

    if (best_train_idx < 0) {
      continue;
    }

    const float best_distance = std::acos(std::min(
        inv_sq_descriptor_norm * static_cast<float>(best_dot_product), 1.0F));
    if (best_distance > max_distance) {
      continue;
    }

    const float second_best_distance = second_best_dot_product < 0
                                           ? std::numeric_limits<float>::max()
                                           : std::acos(std::min(
                                                 inv_sq_descriptor_norm *
                                                     static_cast<float>(
                                                         second_best_dot_product),
                                                 1.0F));
    // 与 COLMAP 保持一致：best == second_best 时必须拒绝，避免歧义匹配通过。
    if (best_distance >= max_ratio * second_best_distance) {
      continue;
    }

    matches[static_cast<std::size_t>(query_idx)] = best_train_idx;
  }

  return matches;
}

struct FeatureGrid {
  static constexpr double kCellSize = 16.0;
  int num_columns = 1;
  int num_rows = 1;
  std::vector<std::vector<int>> cells;
};

FeatureGrid BuildFeatureGrid(const FeatureSet& features) {
  double max_x = 0.0;
  double max_y = 0.0;
  for (const FeatureKeypoint& keypoint : features.keypoints) {
    max_x = std::max(max_x, static_cast<double>(keypoint.x));
    max_y = std::max(max_y, static_cast<double>(keypoint.y));
  }
  FeatureGrid grid;
  grid.num_columns =
      std::max(1, static_cast<int>(max_x / FeatureGrid::kCellSize) + 1);
  grid.num_rows =
      std::max(1, static_cast<int>(max_y / FeatureGrid::kCellSize) + 1);
  grid.cells.resize(static_cast<std::size_t>(grid.num_columns * grid.num_rows));
  for (int idx = 0; idx < static_cast<int>(features.keypoints.size()); ++idx) {
    const FeatureKeypoint& keypoint = features.keypoints[idx];
    const int column = std::clamp(
        static_cast<int>(keypoint.x / FeatureGrid::kCellSize), 0,
        grid.num_columns - 1);
    const int row = std::clamp(
        static_cast<int>(keypoint.y / FeatureGrid::kCellSize), 0,
        grid.num_rows - 1);
    grid.cells[static_cast<std::size_t>(row * grid.num_columns + column)]
        .push_back(idx);
  }
  return grid;
}

struct GuidedCandidateIndex {
  std::vector<int> offsets;
  std::vector<int> indices;
};

GuidedCandidateIndex BuildGuidedCandidateIndex(
    const std::vector<FeatureKeypoint>& query_keypoints,
    const std::vector<FeatureKeypoint>& train_keypoints,
    const FeatureGrid& grid,
    const Eigen::Matrix3d& fundamental_matrix,
    const double max_sampson_error_px) {
  const double max_squared_error =
      max_sampson_error_px * max_sampson_error_px;
  GuidedCandidateIndex index;
  index.offsets.resize(query_keypoints.size() + 1, 0);
  index.indices.reserve(query_keypoints.size() * 32);

  for (int query_idx = 0;
       query_idx < static_cast<int>(query_keypoints.size());
       ++query_idx) {
    const FeatureKeypoint& query = query_keypoints[query_idx];
    const double line_x = fundamental_matrix(0, 0) * query.x +
                          fundamental_matrix(0, 1) * query.y +
                          fundamental_matrix(0, 2);
    const double line_y = fundamental_matrix(1, 0) * query.x +
                          fundamental_matrix(1, 1) * query.y +
                          fundamental_matrix(1, 2);
    const double line_z = fundamental_matrix(2, 0) * query.x +
                          fundamental_matrix(2, 1) * query.y +
                          fundamental_matrix(2, 2);
    const double line_normal_squared =
        line_x * line_x + line_y * line_y;
    if (!std::isfinite(line_x) || !std::isfinite(line_y) ||
        !std::isfinite(line_z) || line_normal_squared <= 0.0) {
      index.offsets[static_cast<std::size_t>(query_idx) + 1] =
          static_cast<int>(index.indices.size());
      continue;
    }
    const auto append_cell_candidates = [&](const std::vector<int>& cell) {
      for (const int train_idx : cell) {
        const FeatureKeypoint& train = train_keypoints[train_idx];
        const double ftx2_x = fundamental_matrix(0, 0) * train.x +
                              fundamental_matrix(1, 0) * train.y +
                              fundamental_matrix(2, 0);
        const double ftx2_y = fundamental_matrix(0, 1) * train.x +
                              fundamental_matrix(1, 1) * train.y +
                              fundamental_matrix(2, 1);
        const double residual =
            train.x * line_x + train.y * line_y + line_z;
        const double denominator =
            line_normal_squared + ftx2_x * ftx2_x + ftx2_y * ftx2_y;
        if (denominator > 0.0 && std::isfinite(denominator) &&
            residual * residual <= max_squared_error * denominator) {
          index.indices.push_back(train_idx);
        }
      }
    };
    constexpr int kCellRadius = 4;
    if (std::abs(line_y) >= std::abs(line_x)) {
      for (int column = 0; column < grid.num_columns; ++column) {
        const double x = (column + 0.5) * FeatureGrid::kCellSize;
        const int center_row = static_cast<int>(
            (-(line_x * x + line_z) / line_y) /
            FeatureGrid::kCellSize);
        for (int row = std::max(0, center_row - kCellRadius);
             row <= std::min(grid.num_rows - 1, center_row + kCellRadius);
             ++row) {
          const auto& cell = grid.cells[static_cast<std::size_t>(
              row * grid.num_columns + column)];
          append_cell_candidates(cell);
        }
      }
    } else {
      for (int row = 0; row < grid.num_rows; ++row) {
        const double y = (row + 0.5) * FeatureGrid::kCellSize;
        const int center_column = static_cast<int>(
            (-(line_y * y + line_z) / line_x) /
            FeatureGrid::kCellSize);
        for (int column = std::max(0, center_column - kCellRadius);
             column <= std::min(grid.num_columns - 1,
                               center_column + kCellRadius);
             ++column) {
          const auto& cell = grid.cells[static_cast<std::size_t>(
              row * grid.num_columns + column)];
          append_cell_candidates(cell);
        }
      }
    }
    index.offsets[static_cast<std::size_t>(query_idx) + 1] =
        static_cast<int>(index.indices.size());
  }
  return index;
}

std::vector<int> FindBestMatchesOneWayGuided(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const GuidedCandidateIndex& candidates,
    const DescriptorMatchingOptions& options) {
  const float inv_sq_descriptor_norm =
      static_cast<float>(1.0 / kSqSiftDescriptorNorm);
  std::vector<int> matches(
      static_cast<std::size_t>(query_descriptors.rows()), -1);
  for (int query_idx = 0; query_idx < query_descriptors.rows(); ++query_idx) {
    int best_train_idx = -1;
    int best_dot_product = -1;
    int second_best_dot_product = -1;
    for (int candidate_idx =
             candidates.offsets[static_cast<std::size_t>(query_idx)];
         candidate_idx <
         candidates.offsets[static_cast<std::size_t>(query_idx) + 1];
         ++candidate_idx) {
      const int train_idx =
          candidates.indices[static_cast<std::size_t>(candidate_idx)];
      const int dot_product = DotProduct(query_descriptors, query_idx,
                                         train_descriptors, train_idx);
      if (dot_product > best_dot_product) {
        best_train_idx = train_idx;
        second_best_dot_product = best_dot_product;
        best_dot_product = dot_product;
      } else if (dot_product > second_best_dot_product) {
        second_best_dot_product = dot_product;
      }
    }
    if (best_train_idx < 0) continue;
    const float best_distance = std::acos(std::min(
        inv_sq_descriptor_norm * static_cast<float>(best_dot_product), 1.0F));
    const float second_best_distance = second_best_dot_product < 0
        ? std::numeric_limits<float>::max()
        : std::acos(std::min(inv_sq_descriptor_norm *
                                 static_cast<float>(second_best_dot_product),
                             1.0F));
    if (best_distance <= options.max_distance &&
        best_distance < options.max_ratio * second_best_distance) {
      matches[static_cast<std::size_t>(query_idx)] = best_train_idx;
    }
  }
  return matches;
}

std::vector<int> FindBestMatchesOneWayIndex(const DistanceMatrix& indices,
                                            const DistanceMatrix& l2_dists,
                                            const double max_ratio,
                                            const double max_distance) {
  const int max_l2_dist = static_cast<int>(
      kSqSiftDescriptorNorm * max_distance * max_distance);

  std::vector<int> matches(static_cast<std::size_t>(indices.rows()), -1);
  for (int query_idx = 0; query_idx < indices.rows(); ++query_idx) {
    int best_train_idx = -1;
    int best_l2_dist = std::numeric_limits<int>::max();
    int second_best_l2_dist = std::numeric_limits<int>::max();

    for (int neighbor_idx = 0; neighbor_idx < indices.cols(); ++neighbor_idx) {
      const int train_idx = indices(query_idx, neighbor_idx);
      const int l2_dist = l2_dists(query_idx, neighbor_idx);
      if (l2_dist < best_l2_dist) {
        best_train_idx = train_idx;
        second_best_l2_dist = best_l2_dist;
        best_l2_dist = l2_dist;
      } else if (l2_dist < second_best_l2_dist) {
        second_best_l2_dist = l2_dist;
      }
    }

    if (best_train_idx < 0) {
      continue;
    }
    if (best_l2_dist > max_l2_dist) {
      continue;
    }
    if (std::sqrt(static_cast<float>(best_l2_dist)) >=
        max_ratio * std::sqrt(static_cast<float>(second_best_l2_dist))) {
      continue;
    }

    matches[static_cast<std::size_t>(query_idx)] = best_train_idx;
  }
  return matches;
}

std::vector<int> BestMatchesBruteForce(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const DescriptorMatchingOptions& options) {
  if (query_descriptors.size() == 0 || train_descriptors.size() == 0) {
    return std::vector<int>(static_cast<std::size_t>(query_descriptors.rows()),
                            -1);
  }
  return FindBestMatchesOneWayBruteForce(query_descriptors, train_descriptors,
                                         options.max_ratio,
                                         options.max_distance);
}

std::vector<int> FilterCudaBestMatches(
    const std::vector<cuda_matcher::BestMatch>& candidates,
    const DescriptorMatchingOptions& options) {
  const float inv_sq_descriptor_norm =
      static_cast<float>(1.0 / kSqSiftDescriptorNorm);
  std::vector<int> matches(candidates.size(), -1);
  for (std::size_t query_idx = 0; query_idx < candidates.size(); ++query_idx) {
    const cuda_matcher::BestMatch& candidate = candidates[query_idx];
    if (candidate.index < 0) continue;
    const float best_distance = std::acos(std::min(
        inv_sq_descriptor_norm * static_cast<float>(candidate.best_dot),
        1.0F));
    if (best_distance > options.max_distance) continue;
    const float second_best_distance =
        candidate.second_best_dot < 0
            ? std::numeric_limits<float>::max()
            : std::acos(std::min(
                  inv_sq_descriptor_norm *
                      static_cast<float>(candidate.second_best_dot),
                  1.0F));
    if (best_distance >= options.max_ratio * second_best_distance) continue;
    matches[query_idx] = candidate.index;
  }
  return matches;
}

std::vector<int> BestMatchesCuda(
    const cuda_matcher::DescriptorBuffer& query,
    const cuda_matcher::DescriptorBuffer& train,
    const DescriptorMatchingOptions& options) {
  return FilterCudaBestMatches(cuda_matcher::FindBestMatches(query, train),
                               options);
}

std::vector<int> BestMatchesFlann(
    const DescriptorByteMatrix& query_descriptors,
    const DescriptorByteMatrix& train_descriptors,
    const DescriptorMatchingOptions& options,
    FlannIndex* prepared_index = nullptr,
    std::mutex* prepared_index_mutex = nullptr) {
  std::vector<int> best(static_cast<std::size_t>(query_descriptors.rows()), -1);
  if (query_descriptors.size() == 0 || train_descriptors.size() == 0) {
    return best;
  }

  const int neighbor_count = std::min<int>(2, train_descriptors.rows());
  if (neighbor_count <= 0) {
    return best;
  }

  flann::Matrix<std::uint8_t> train_matrix(
      const_cast<std::uint8_t*>(train_descriptors.data()),
      static_cast<std::size_t>(train_descriptors.rows()),
      static_cast<std::size_t>(train_descriptors.cols()));
  std::unique_ptr<FlannIndex> owned_index;
  if (prepared_index == nullptr) {
    owned_index = std::make_unique<FlannIndex>(
        train_matrix, flann::KDTreeIndexParams(kFlannNumTrees));
    owned_index->buildIndex();
    prepared_index = owned_index.get();
  }

  flann::Matrix<std::uint8_t> query_matrix(
      const_cast<std::uint8_t*>(query_descriptors.data()),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(query_descriptors.cols()));

  DistanceMatrix indices(query_descriptors.rows(), neighbor_count);
  DistanceMatrix l2_dists(query_descriptors.rows(), neighbor_count);
  std::vector<float> l2_dist_storage(
      static_cast<std::size_t>(query_descriptors.rows()) *
      static_cast<std::size_t>(neighbor_count));

  flann::Matrix<int> indices_matrix(
      indices.data(), static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(neighbor_count));
  flann::Matrix<float> l2_dist_matrix(
      l2_dist_storage.data(),
      static_cast<std::size_t>(query_descriptors.rows()),
      static_cast<std::size_t>(neighbor_count));

  const auto search = [&]() {
    prepared_index->knnSearch(query_matrix,
                              indices_matrix,
                              l2_dist_matrix,
                              neighbor_count,
                              flann::SearchParams(kFlannNumLeavesToVisit));
  };
  if (prepared_index_mutex != nullptr) {
    const std::lock_guard<std::mutex> lock(*prepared_index_mutex);
    search();
  } else {
    search();
  }

  for (int row = 0; row < query_descriptors.rows(); ++row) {
    for (int col = 0; col < neighbor_count; ++col) {
      const std::size_t offset = static_cast<std::size_t>(row) *
                                     static_cast<std::size_t>(neighbor_count) +
                                 static_cast<std::size_t>(col);
      l2_dists(row, col) =
          static_cast<int>(std::round(l2_dist_storage[offset]));
    }
  }

  return FindBestMatchesOneWayIndex(indices, l2_dists, options.max_ratio,
                                    options.max_distance);
}

std::vector<int> BestMatches(const DescriptorByteMatrix& query_descriptors,
                             const DescriptorByteMatrix& train_descriptors,
                             const DescriptorMatchingOptions& options) {
  if (options.backend == DescriptorMatcherBackend::kBruteForce) {
    return BestMatchesBruteForce(query_descriptors, train_descriptors, options);
  }
  if (options.backend == DescriptorMatcherBackend::kFlann) {
    return BestMatchesFlann(query_descriptors, train_descriptors, options);
  }
  if (options.backend == DescriptorMatcherBackend::kCudaBruteForce) {
    const auto query = cuda_matcher::CreateBuffer(
        query_descriptors.data(), static_cast<int>(query_descriptors.rows()),
        static_cast<int>(query_descriptors.cols()));
    const auto train = cuda_matcher::CreateBuffer(
        train_descriptors.data(), static_cast<int>(train_descriptors.rows()),
        static_cast<int>(train_descriptors.cols()));
    return BestMatchesCuda(*query, *train, options);
  }
  throw std::invalid_argument("Unknown descriptor matcher backend.");
}

}  // namespace

struct PreparedFeatureMatcher::Impl {
  explicit Impl(const FeatureSet& features,
                const DescriptorMatcherBackend matcher_backend,
                const FeatureMatcherType matcher_type)
      : keypoints(matcher_backend == DescriptorMatcherBackend::kCudaBruteForce
                      ? std::vector<FeatureKeypoint>{} : features.keypoints),
        num_keypoints(features.keypoints.size()),
        descriptors(QuantizeDescriptors(features.descriptors)),
        descriptor_rows(descriptors.rows()),
        descriptor_cols(descriptors.cols()),
        grid(matcher_backend == DescriptorMatcherBackend::kCudaBruteForce
                 ? FeatureGrid{} : BuildFeatureGrid(features)),
        backend(matcher_backend),
        matcher_type(matcher_type) {
    ValidateSiftFeatureSet(features, matcher_type);
    if (backend == DescriptorMatcherBackend::kFlann &&
        descriptors.size() != 0) {
      flann::Matrix<std::uint8_t> matrix(
          descriptors.data(), static_cast<std::size_t>(descriptors.rows()),
          static_cast<std::size_t>(descriptors.cols()));
      index = std::make_unique<FlannIndex>(
          matrix, flann::KDTreeIndexParams(kFlannNumTrees));
      index->buildIndex();
    }
    if (backend == DescriptorMatcherBackend::kCudaBruteForce &&
        descriptors.size() != 0) {
      std::vector<float> keypoints_xy;
      keypoints_xy.reserve(features.keypoints.size() * 2);
      for (const FeatureKeypoint& keypoint : features.keypoints) {
        keypoints_xy.push_back(keypoint.x);
        keypoints_xy.push_back(keypoint.y);
      }
      cuda_buffer = cuda_matcher::CreateBuffer(
          descriptors.data(), static_cast<int>(descriptors.rows()),
          static_cast<int>(descriptors.cols()),
          num_keypoints == static_cast<std::size_t>(descriptors.rows())
              ? keypoints_xy.data()
              : nullptr);
      // CUDA raw and guided matching use the uploaded descriptors/keypoints.
      // Keep dimensions independently so validation does not require a second
      // full copy of the byte descriptors on the host.
      descriptors.resize(0, 0);
    }
  }

  std::vector<FeatureKeypoint> keypoints;
  std::size_t num_keypoints;
  DescriptorByteMatrix descriptors;
  Eigen::Index descriptor_rows;
  Eigen::Index descriptor_cols;
  FeatureGrid grid;
  DescriptorMatcherBackend backend;
  FeatureMatcherType matcher_type;
  std::unique_ptr<FlannIndex> index;
  std::unique_ptr<cuda_matcher::DescriptorBuffer> cuda_buffer;
  mutable std::mutex index_mutex;

  bool Empty() const { return descriptor_rows == 0 || descriptor_cols == 0; }
};

PreparedFeatureMatcher::PreparedFeatureMatcher(
    const FeatureSet& features,
    const DescriptorMatcherBackend backend)
    : impl_(std::make_unique<Impl>(features, backend,
                                   FeatureMatcherType::kSiftBruteForce)) {}

PreparedFeatureMatcher::~PreparedFeatureMatcher() = default;
void ReleaseDescriptorMatcherWorkspace() { cuda_matcher::ReleaseWorkspace(); }
PreparedFeatureMatcher::PreparedFeatureMatcher(
    PreparedFeatureMatcher&&) noexcept = default;
PreparedFeatureMatcher& PreparedFeatureMatcher::operator=(
    PreparedFeatureMatcher&&) noexcept = default;

std::vector<FeatureMatch> MatchPreparedFeatureSets(
    const PreparedFeatureMatcher& features1,
    const PreparedFeatureMatcher& features2,
    const DescriptorMatchingOptions& options) {
  if (features1.impl_->backend != options.backend ||
      features2.impl_->backend != options.backend) {
    throw std::invalid_argument(
        "Prepared matcher backend does not match matching options.");
  }
  if (features1.impl_->matcher_type != options.matcher_type ||
      features2.impl_->matcher_type != options.matcher_type) {
    throw std::invalid_argument(
        "Prepared matcher type does not match matching options.");
  }
  if (options.max_ratio <= 0.0 || options.max_distance <= 0.0) {
    throw std::invalid_argument("Matching thresholds must be positive.");
  }
  if (features1.impl_->descriptor_cols != features2.impl_->descriptor_cols &&
      !features1.impl_->Empty() && !features2.impl_->Empty()) {
    throw std::invalid_argument("Descriptor dimensions do not match.");
  }
  if (features1.impl_->Empty() || features2.impl_->Empty()) {
    return {};
  }

  const auto best_matches = [&](const PreparedFeatureMatcher::Impl& query,
                                const PreparedFeatureMatcher::Impl& train) {
    if (options.backend == DescriptorMatcherBackend::kBruteForce) {
      return BestMatchesBruteForce(query.descriptors, train.descriptors,
                                   options);
    }
    if (options.backend == DescriptorMatcherBackend::kCudaBruteForce) {
      return BestMatchesCuda(*query.cuda_buffer, *train.cuda_buffer, options);
    }
    return BestMatchesFlann(query.descriptors, train.descriptors, options,
                            train.index.get(), &train.index_mutex);
  };
  std::vector<int> forward;
  std::vector<int> backward;
  if (options.backend == DescriptorMatcherBackend::kCudaBruteForce &&
      options.mutual_check) {
    const cuda_matcher::MutualBestMatches candidates =
        cuda_matcher::FindBestMatchesMutual(*features1.impl_->cuda_buffer,
                                            *features2.impl_->cuda_buffer);
    forward = FilterCudaBestMatches(candidates.forward, options);
    backward = FilterCudaBestMatches(candidates.backward, options);
  } else {
    forward = best_matches(*features1.impl_, *features2.impl_);
  }
  if (options.mutual_check && backward.empty()) {
    backward = best_matches(*features2.impl_, *features1.impl_);
  }
  std::vector<FeatureMatch> matches;
  for (int idx1 = 0; idx1 < static_cast<int>(forward.size()); ++idx1) {
    const int idx2 = forward[static_cast<std::size_t>(idx1)];
    if (idx2 >= 0 &&
        (!options.mutual_check ||
         backward[static_cast<std::size_t>(idx2)] == idx1)) {
      matches.push_back({static_cast<Point2DIdx>(idx1),
                         static_cast<Point2DIdx>(idx2)});
    }
  }
  return matches;
}

std::vector<FeatureMatch> MatchPreparedFeatureSetsGuidedFundamental(
    const PreparedFeatureMatcher& features1,
    const PreparedFeatureMatcher& features2,
    const Eigen::Matrix3d& fundamental_matrix,
    const double max_sampson_error_px,
    const DescriptorMatchingOptions& options,
    GuidedDescriptorMatchingStats* stats) {
  if (fundamental_matrix.norm() <= 0.0 || max_sampson_error_px < 0.0 ||
      options.max_ratio <= 0.0 || options.max_distance <= 0.0) {
    return {};
  }
  const auto& impl1 = *features1.impl_;
  const auto& impl2 = *features2.impl_;
  if (impl1.backend != options.backend || impl2.backend != options.backend) {
    throw std::invalid_argument(
        "Prepared matcher backend does not match guided matching options.");
  }
  if (impl1.matcher_type != options.matcher_type ||
      impl2.matcher_type != options.matcher_type) {
    throw std::invalid_argument(
        "Prepared matcher type does not match guided matching options.");
  }
  if (impl1.descriptor_rows != static_cast<Eigen::Index>(impl1.num_keypoints) ||
      impl2.descriptor_rows != static_cast<Eigen::Index>(impl2.num_keypoints) ||
      (impl1.descriptor_cols != impl2.descriptor_cols &&
       !impl1.Empty() && !impl2.Empty())) {
    return {};
  }
  if (impl1.Empty() || impl2.Empty()) {
    return {};
  }
  std::vector<int> forward;
  std::vector<int> backward;
  if (options.backend == DescriptorMatcherBackend::kCudaBruteForce) {
    const auto search_start = std::chrono::steady_clock::now();
    const cuda_matcher::MutualBestMatches candidates =
        cuda_matcher::FindBestMatchesGuidedFundamental(
            *impl1.cuda_buffer, *impl2.cuda_buffer,
            fundamental_matrix.data(), max_sampson_error_px);
    forward = FilterCudaBestMatches(candidates.forward, options);
    if (options.mutual_check) {
      backward = FilterCudaBestMatches(candidates.backward, options);
    }
    if (stats != nullptr) {
      stats->descriptor_search_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        search_start)
              .count();
    }
  } else {
    const auto best_matches = [&](const PreparedFeatureMatcher::Impl& query,
                                  const PreparedFeatureMatcher::Impl& train,
                                  const Eigen::Matrix3d& geometry) {
    auto search_start = std::chrono::steady_clock::now();
    std::vector<int> matches;
    const auto candidate_start = std::chrono::steady_clock::now();
    const GuidedCandidateIndex candidates = BuildGuidedCandidateIndex(
        query.keypoints, train.keypoints, train.grid, geometry,
        max_sampson_error_px);
    if (stats != nullptr) {
      stats->candidate_generation_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - candidate_start)
              .count();
    }
    search_start = std::chrono::steady_clock::now();
    matches = FindBestMatchesOneWayGuided(
        query.descriptors, train.descriptors, candidates, options);
    if (stats != nullptr) {
      stats->descriptor_search_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        search_start)
              .count();
    }
    return matches;
    };
    forward = best_matches(impl1, impl2, fundamental_matrix);
    if (options.mutual_check) {
      backward = best_matches(impl2, impl1, fundamental_matrix.transpose());
    }
  }
  std::vector<FeatureMatch> matches;
  for (int idx1 = 0; idx1 < static_cast<int>(forward.size()); ++idx1) {
    const int idx2 = forward[static_cast<std::size_t>(idx1)];
    if (idx2 >= 0 &&
        (!options.mutual_check ||
         backward[static_cast<std::size_t>(idx2)] == idx1)) {
      matches.push_back({static_cast<Point2DIdx>(idx1),
                         static_cast<Point2DIdx>(idx2)});
    }
  }
  return matches;
}

std::vector<FeatureMatch> MatchPreparedFeatureSetsGuidedGeometry(
    const PreparedFeatureMatcher& features1,
    const PreparedFeatureMatcher& features2,
    const GuidedGeometryType type, const Eigen::Matrix3d& matrix,
    const std::vector<Eigen::Vector2d>& coordinates1,
    const std::vector<Eigen::Vector2d>& coordinates2,
    const double max_error, const DescriptorMatchingOptions& options,
    GuidedDescriptorMatchingStats* stats) {
  if (!matrix.allFinite() || matrix.cwiseAbs().maxCoeff() <= 0 ||
      !std::isfinite(max_error) || max_error < 0 ||
      !std::isfinite(options.max_ratio) || options.max_ratio <= 0 ||
      !std::isfinite(options.max_distance) || options.max_distance <= 0) return {};
  const auto& a = *features1.impl_;
  const auto& b = *features2.impl_;
  if (a.backend != options.backend || b.backend != options.backend ||
      a.matcher_type != options.matcher_type || b.matcher_type != options.matcher_type)
    throw std::invalid_argument("Prepared matcher backend/type does not match guided options");
  if (coordinates1.size() != static_cast<std::size_t>(a.descriptor_rows) ||
      coordinates2.size() != static_cast<std::size_t>(b.descriptor_rows))
    throw std::invalid_argument("Guided coordinates must preserve descriptor row indices");
  if (a.Empty() || b.Empty()) return {};
  if (a.descriptor_cols != b.descriptor_cols) return {};
  if (coordinates1.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      coordinates2.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("Too many guided feature coordinates");

  const auto started = std::chrono::steady_clock::now();
  const Eigen::Matrix3d model = matrix / matrix.cwiseAbs().maxCoeff();
  const double limit = max_error * max_error;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<Eigen::Vector3d> first_lines(coordinates1.size(), Eigen::Vector3d::Constant(nan));
  std::vector<Eigen::Vector3d> second_lines(coordinates2.size(), Eigen::Vector3d::Constant(nan));
  for (std::size_t i = 0; i < coordinates1.size(); ++i) {
    if (coordinates1[i].allFinite()) first_lines[i] = model * Eigen::Vector3d(coordinates1[i].x(), coordinates1[i].y(), 1);
  }
  if (type == GuidedGeometryType::kEpipolar) {
    for (std::size_t j = 0; j < coordinates2.size(); ++j) {
      if (coordinates2[j].allFinite()) second_lines[j] = model.transpose() * Eigen::Vector3d(coordinates2[j].x(), coordinates2[j].y(), 1);
    }
  }
  GuidedCandidateIndex forward_candidates, backward_candidates;
  forward_candidates.offsets.resize(coordinates1.size() + 1, 0);
  backward_candidates.offsets.resize(coordinates2.size() + 1, 0);
  for (std::size_t i = 0; i < coordinates1.size(); ++i) {
    const auto& line = first_lines[i];
    if (line.allFinite() && (type != GuidedGeometryType::kHomography || line.z() != 0)) {
      for (std::size_t j = 0; j < coordinates2.size(); ++j) {
        if (!coordinates2[j].allFinite()) continue;
        bool accepted = false;
        if (type == GuidedGeometryType::kHomography) {
          const Eigen::Vector2d projected = line.head<2>() / line.z();
          const double error = (projected - coordinates2[j]).squaredNorm();
          accepted = std::isfinite(error) && error <= limit;
        } else {
          const double denominator = line.head<2>().squaredNorm() + second_lines[j].head<2>().squaredNorm();
          const double residual = coordinates2[j].x() * line.x() + coordinates2[j].y() * line.y() + line.z();
          accepted = denominator > 0 && std::isfinite(denominator) && std::isfinite(residual) &&
                     residual * residual <= limit * denominator;
        }
        if (accepted) {
          if (forward_candidates.indices.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("Too many guided descriptor candidates");
          forward_candidates.indices.push_back(static_cast<int>(j));
          if (options.mutual_check) ++backward_candidates.offsets[j + 1];
        }
      }
    }
    forward_candidates.offsets[i + 1] = static_cast<int>(forward_candidates.indices.size());
  }
  if (options.mutual_check) {
    for (std::size_t j = 1; j < backward_candidates.offsets.size(); ++j)
      backward_candidates.offsets[j] += backward_candidates.offsets[j - 1];
    backward_candidates.indices.resize(forward_candidates.indices.size());
    auto cursor = backward_candidates.offsets;
    for (std::size_t i = 0; i < coordinates1.size(); ++i) {
      for (int k = forward_candidates.offsets[i]; k < forward_candidates.offsets[i + 1]; ++k) {
        const int j = forward_candidates.indices[k];
        backward_candidates.indices[cursor[j]++] = static_cast<int>(i);
      }
    }
  }
  if (stats) stats->candidate_generation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const auto search_started = std::chrono::steady_clock::now();
  std::vector<int> forward, backward;
  if (options.backend == DescriptorMatcherBackend::kCudaBruteForce) {
    forward = FilterCudaBestMatches(cuda_matcher::FindBestMatchesGuided(
        *a.cuda_buffer, *b.cuda_buffer, forward_candidates.offsets, forward_candidates.indices), options);
    if (options.mutual_check) backward = FilterCudaBestMatches(cuda_matcher::FindBestMatchesGuided(
        *b.cuda_buffer, *a.cuda_buffer, backward_candidates.offsets, backward_candidates.indices), options);
  } else {
    forward = FindBestMatchesOneWayGuided(a.descriptors, b.descriptors, forward_candidates, options);
    if (options.mutual_check) backward = FindBestMatchesOneWayGuided(b.descriptors, a.descriptors, backward_candidates, options);
  }
  if (stats) stats->descriptor_search_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - search_started).count();
  std::vector<FeatureMatch> matches;
  for (std::size_t i = 0; i < forward.size(); ++i) {
    const int j = forward[i];
    if (j >= 0 && (!options.mutual_check || backward[j] == static_cast<int>(i)))
      matches.push_back({static_cast<Point2DIdx>(i), static_cast<Point2DIdx>(j)});
  }
  return matches;
}

bool IsCudaDescriptorMatcherAvailable() { return cuda_matcher::IsAvailable(); }
void SetCudaDescriptorMatcherDevice(int device) {
#if defined(COLITE_ENABLE_CUDA_MATCHER)
  cuda_matcher::SetDevice(device);
#else
  (void)device;
  throw std::runtime_error("CUDA matching is not enabled in this build");
#endif
}

std::vector<FeatureMatch> MatchFeatureSets(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const DescriptorMatchingOptions& options) {
  ValidateSiftFeatureSet(features1, options.matcher_type);
  ValidateSiftFeatureSet(features2, options.matcher_type);
  if (features1.extractor_type != features2.extractor_type) {
    throw std::invalid_argument("Feature extractor types do not match.");
  }
  return MatchDescriptors(features1.descriptors, features2.descriptors, options);
}

std::vector<FeatureMatch> MatchFeatureSetsGuidedFundamental(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const Eigen::Matrix3d& fundamental_matrix,
    const double max_sampson_error_px,
  const DescriptorMatchingOptions& options) {
  const PreparedFeatureMatcher prepared1(
      features1, options.backend);
  const PreparedFeatureMatcher prepared2(
      features2, options.backend);
  return MatchPreparedFeatureSetsGuidedFundamental(
      prepared1, prepared2, fundamental_matrix, max_sampson_error_px,
      options, nullptr);
}

std::vector<FeatureMatch> MatchDescriptors(
    const DescriptorMatrix& descriptors1,
    const DescriptorMatrix& descriptors2,
    const DescriptorMatchingOptions& options) {
  if (options.max_ratio <= 0.0) {
    throw std::invalid_argument("max_ratio must be positive.");
  }
  if (options.max_distance <= 0.0) {
    throw std::invalid_argument("max_distance must be positive.");
  }
  if (descriptors1.cols() != descriptors2.cols() &&
      descriptors1.size() != 0 && descriptors2.size() != 0) {
    throw std::invalid_argument("Descriptor dimensions do not match.");
  }
  if (descriptors1.size() == 0 || descriptors2.size() == 0) {
    return {};
  }

  const DescriptorByteMatrix quantized1 = QuantizeDescriptors(descriptors1);
  const DescriptorByteMatrix quantized2 = QuantizeDescriptors(descriptors2);

  const std::vector<int> forward = BestMatches(quantized1, quantized2, options);
  std::vector<int> backward;
  if (options.mutual_check) {
    backward = BestMatches(quantized2, quantized1, options);
  }

  std::vector<FeatureMatch> matches;
  for (int idx1 = 0; idx1 < static_cast<int>(forward.size()); ++idx1) {
    const int idx2 = forward[static_cast<std::size_t>(idx1)];
    if (idx2 < 0) {
      continue;
    }
    if (options.mutual_check &&
        backward[static_cast<std::size_t>(idx2)] != idx1) {
      continue;
    }
    matches.push_back({static_cast<Point2DIdx>(idx1),
                       static_cast<Point2DIdx>(idx2)});
  }
  return matches;
}

}  // namespace colite
