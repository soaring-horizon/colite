#include "colite/matching/two_view_geometry_verifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>
#include <unordered_set>

#include "homography_pose.h"

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "colite/core/geometry.h"
#include "colite/core/ransac.h"

#ifdef COLITE_HAS_POSELIB
#include <PoseLib/solvers/relpose_5pt.h>
#include <PoseLib/solvers/relpose_7pt.h>
#endif

namespace colite {
namespace {

constexpr double kMinHomogeneousScale = 1e-12;

struct MatchedPoints {
  std::vector<Eigen::Vector2d> pixels1;
  std::vector<Eigen::Vector2d> pixels2;
  std::vector<Eigen::Vector2d> normalized1;
  std::vector<Eigen::Vector2d> normalized2;
  std::vector<std::size_t> normalized_indices;
  std::vector<FeatureMatch> matches;
};

struct NormalizedPoints2D {
  std::vector<Eigen::Vector2d> points;
  Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
};

struct MatrixRansacResult {
  Eigen::Matrix3d model = Eigen::Matrix3d::Zero();
  std::vector<std::size_t> inlier_indices;
};

bool IsValidFeatureIndex(Point2DIdx idx, const FeatureSet& features) {
  return idx != kInvalidPoint2DIdx &&
         idx < static_cast<Point2DIdx>(features.keypoints.size());
}

std::optional<Eigen::Vector2d> NormalizedPointFromPixel(const Camera& camera,
                                         const FeatureKeypoint& keypoint) {
  const auto ray = TryUnprojectPixel(camera, Eigen::Vector2d(keypoint.x, keypoint.y), 1.0);
  if (!ray) return std::nullopt;
  return Eigen::Vector2d(ray->x() / ray->z(), ray->y() / ray->z());
}

MatchedPoints ExtractMatchedPoints(const Camera& camera1,
                                   const FeatureSet& features1,
                                   const Camera& camera2,
                                   const FeatureSet& features2,
                                   const std::vector<FeatureMatch>& matches) {
  MatchedPoints points;
  points.pixels1.reserve(matches.size());
  points.pixels2.reserve(matches.size());
  points.normalized1.reserve(matches.size());
  points.normalized2.reserve(matches.size());
  points.matches.reserve(matches.size());
  points.normalized_indices.reserve(matches.size());

  for (const FeatureMatch& match : matches) {
    if (!IsValidFeatureIndex(match.idx1, features1) ||
        !IsValidFeatureIndex(match.idx2, features2)) {
      throw std::out_of_range("Feature match index is outside keypoint range");
    }

    const FeatureKeypoint& keypoint1 = features1.keypoints[match.idx1];
    const FeatureKeypoint& keypoint2 = features2.keypoints[match.idx2];
    points.pixels1.emplace_back(keypoint1.x, keypoint1.y);
    points.pixels2.emplace_back(keypoint2.x, keypoint2.y);
    const auto normalized1 = NormalizedPointFromPixel(camera1, keypoint1);
    const auto normalized2 = NormalizedPointFromPixel(camera2, keypoint2);
    if (normalized1 && normalized2) {
      points.normalized_indices.push_back(points.matches.size());
      points.normalized1.push_back(*normalized1);
      points.normalized2.push_back(*normalized2);
    }
    points.matches.push_back(match);
  }

  return points;
}

NormalizedPoints2D NormalizePoints(const std::vector<Eigen::Vector2d>& points,
                                   const std::vector<std::size_t>& indices) {
  NormalizedPoints2D result;
  result.points.reserve(indices.size());
  if (indices.empty()) {
    return result;
  }

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const std::size_t idx : indices) {
    centroid += points[idx];
  }
  centroid /= static_cast<double>(indices.size());

  double mean_distance = 0.0;
  for (const std::size_t idx : indices) {
    mean_distance += (points[idx] - centroid).norm();
  }
  mean_distance /= static_cast<double>(indices.size());
  const double scale =
      mean_distance > 0.0 ? std::sqrt(2.0) / mean_distance : 1.0;

  result.transform << scale, 0.0, -scale * centroid.x(), 0.0, scale,
      -scale * centroid.y(), 0.0, 0.0, 1.0;
  for (const std::size_t idx : indices) {
    result.points.push_back((result.transform *
                             points[idx].homogeneous())
                                .hnormalized());
  }
  return result;
}

std::vector<std::size_t> AllIndices(const std::size_t count) {
  std::vector<std::size_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

std::vector<std::size_t> SampleUniqueIndices(std::mt19937* rng,
                                             const std::size_t count,
                                             const std::size_t sample_size) {
  std::vector<std::size_t> indices = AllIndices(count);
  std::shuffle(indices.begin(), indices.end(), *rng);
  indices.resize(sample_size);
  return indices;
}

std::optional<Eigen::Matrix3d> EstimateFundamentalMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& indices) {
  if (indices.size() < 8) {
    return std::nullopt;
  }

  const NormalizedPoints2D norm1 = NormalizePoints(points1, indices);
  const NormalizedPoints2D norm2 = NormalizePoints(points2, indices);

  Eigen::MatrixXd a(indices.size(), 9);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const double x1 = norm1.points[i].x();
    const double y1 = norm1.points[i].y();
    const double x2 = norm2.points[i].x();
    const double y2 = norm2.points[i].y();
    a.row(static_cast<Eigen::Index>(i)) << x2 * x1, x2 * y1, x2, y2 * x1,
        y2 * y1, y2, x1, y1, 1.0;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
  Eigen::Matrix3d f =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          svd.matrixV().col(8).data());

  const Eigen::JacobiSVD<Eigen::Matrix3d> f_svd(
      f, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = f_svd.singularValues();
  singular_values.z() = 0.0;
  f = f_svd.matrixU() * singular_values.asDiagonal() *
      f_svd.matrixV().transpose();

  Eigen::Matrix3d denormalized =
      norm2.transform.transpose() * f * norm1.transform;
  const double norm = denormalized.norm();
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return std::nullopt;
  }
  return denormalized / norm;
}

std::optional<Eigen::Matrix3d> EstimateEssentialMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& indices) {
  auto essential = EstimateFundamentalMatrix(points1, points2, indices);
  if (!essential) {
    return std::nullopt;
  }

  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      *essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const double sigma =
      0.5 * (svd.singularValues().x() + svd.singularValues().y());
  Eigen::Matrix3d corrected = svd.matrixU() *
                              Eigen::Vector3d(sigma, sigma, 0.0).asDiagonal() *
                              svd.matrixV().transpose();
  const double norm = corrected.norm();
  if (norm <= 0.0 || !std::isfinite(norm)) {
    return std::nullopt;
  }
  return corrected / norm;
}

std::optional<Eigen::Matrix3d> EstimateHomographyMatrix(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& indices) {
  if (indices.size() < 4) {
    return std::nullopt;
  }

  const NormalizedPoints2D norm1 = NormalizePoints(points1, indices);
  const NormalizedPoints2D norm2 = NormalizePoints(points2, indices);

  Eigen::MatrixXd a(static_cast<Eigen::Index>(2 * indices.size()), 9);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const double x = norm1.points[i].x();
    const double y = norm1.points[i].y();
    const double xp = norm2.points[i].x();
    const double yp = norm2.points[i].y();
    a.row(static_cast<Eigen::Index>(2 * i))
        << -x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp;
    a.row(static_cast<Eigen::Index>(2 * i + 1))
        << 0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp;
  }

  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
  Eigen::Matrix3d h =
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          svd.matrixV().col(8).data());
  h = norm2.transform.inverse() * h * norm1.transform;

  const double scale =
      std::abs(h(2, 2)) > kMinHomogeneousScale ? h(2, 2) : h.norm();
  if (std::abs(scale) <= kMinHomogeneousScale || !std::isfinite(scale)) {
    return std::nullopt;
  }
  return h / scale;
}

double SampsonError(const Eigen::Matrix3d& matrix,
                    const Eigen::Vector2d& point1,
                    const Eigen::Vector2d& point2) {
  const Eigen::Vector3d x1 = point1.homogeneous();
  const Eigen::Vector3d x2 = point2.homogeneous();
  const Eigen::Vector3d mx1 = matrix * x1;
  const Eigen::Vector3d mtx2 = matrix.transpose() * x2;
  const double x2tmx1 = x2.dot(mx1);
  const double denom =
      mx1.x() * mx1.x() + mx1.y() * mx1.y() + mtx2.x() * mtx2.x() +
      mtx2.y() * mtx2.y();
  if (denom <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::abs(x2tmx1) / std::sqrt(denom);
}

double HomographyTransferError(const Eigen::Matrix3d& homography,
                               const Eigen::Vector2d& point1,
                               const Eigen::Vector2d& point2) {
  const Eigen::Vector3d transferred = homography * point1.homogeneous();
  if (std::abs(transferred.z()) <= kMinHomogeneousScale) {
    return std::numeric_limits<double>::infinity();
  }
  return (transferred.hnormalized() - point2).norm();
}

std::mt19937 MakeRng(int seed) {
  return std::mt19937(seed < 0 ? std::random_device{}() : static_cast<unsigned>(seed));
}

template <typename Estimator, typename Residual, typename MinimalEstimator>
MatrixRansacResult RunMatrixRansacWithHypotheses(const std::size_t num_points,
                                   const std::size_t sample_size,
                                   const int min_num_iterations,
                                   const int max_num_iterations,
                                   const double confidence,
                                   const double max_error,
                                   Estimator estimator,
                                   Residual residual,
                                   MinimalEstimator minimal_estimator,
                                   const std::size_t local_sample_size,
                                   const double min_inlier_ratio, const int random_seed) {
  MatrixRansacResult best;
  if (num_points < sample_size) {
    return best;
  }

  auto rng = MakeRng(random_seed);
  const int min_iterations = std::max(0, min_num_iterations);
  const int max_iterations = RansacTrials(static_cast<std::size_t>(min_inlier_ratio * 100000),
      100000, sample_size, confidence, max_num_iterations);
  int required_iterations = max_iterations;
  auto score_model = [&](const Eigen::Matrix3d& model) {
    std::vector<std::size_t> inliers;
    inliers.reserve(num_points);
    for (std::size_t idx = 0; idx < num_points; ++idx) {
      const double error = residual(model, idx);
      if (std::isfinite(error) && error <= max_error) {
        inliers.push_back(idx);
      }
    }
    return inliers;
  };

  for (int iteration = 0; iteration < required_iterations; ++iteration) {
    const std::vector<std::size_t> sample =
        num_points == sample_size ? AllIndices(num_points)
                                  : SampleUniqueIndices(&rng, num_points,
                                                        sample_size);
    for (const Eigen::Matrix3d& hypothesis : minimal_estimator(sample)) {
      std::optional<Eigen::Matrix3d> model = hypothesis;
      std::vector<std::size_t> inliers = score_model(*model);

      // Only locally optimize hypotheses that improve the current support.
      // Seven-point F can produce up to three candidates; evaluate each one,
      // but use the eight-point estimator for nonminimal F refinement.
      if (inliers.size() <= best.inlier_indices.size()) continue;
      if (inliers.size() >= local_sample_size) {
        const std::optional<Eigen::Matrix3d> local_model = estimator(inliers);
        if (local_model) {
          std::vector<std::size_t> local_inliers = score_model(*local_model);
          if (local_inliers.size() >= inliers.size()) {
            model = local_model;
            inliers = std::move(local_inliers);
          }
        }
      }

      if (inliers.size() > best.inlier_indices.size()) {
        best.model = *model;
        best.inlier_indices = std::move(inliers);
        if (best.inlier_indices.size() == num_points) break;
        required_iterations = std::min(max_iterations, std::max(min_iterations,
            RansacTrials(best.inlier_indices.size(), num_points, sample_size, confidence, max_iterations)));
        required_iterations = std::max(required_iterations, iteration + 1);
      }
    }
    if (best.inlier_indices.size() == num_points) break;
  }

  if (best.inlier_indices.size() >= local_sample_size) {
    const std::optional<Eigen::Matrix3d> refined =
        estimator(best.inlier_indices);
    if (refined) {
      std::vector<std::size_t> refined_inliers = score_model(*refined);
      if (refined_inliers.size() >= best.inlier_indices.size()) {
        best.model = *refined;
        best.inlier_indices = std::move(refined_inliers);
      }
    }
  }

  return best;
}

template <typename Estimator, typename Residual>
MatrixRansacResult RunMatrixRansac(
    const std::size_t num_points, const std::size_t sample_size,
    const int min_num_iterations, const int max_num_iterations,
    const double confidence, const double max_error,
    Estimator estimator, Residual residual, double min_inlier_ratio, int random_seed) {
  return RunMatrixRansacWithHypotheses(
      num_points, sample_size, min_num_iterations, max_num_iterations,
      confidence, max_error, estimator, residual,
      [&](const std::vector<std::size_t>& sample) {
        std::vector<Eigen::Matrix3d> models;
        if (const auto model = estimator(sample)) models.push_back(*model);
        return models;
      }, sample_size, min_inlier_ratio, random_seed);
}

MatrixRansacResult RunFundamentalRansac(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometryOptions& options) {
  const auto nonminimal_estimator = [&](const std::vector<std::size_t>& sample) {
    return EstimateFundamentalMatrix(points1, points2, sample);
  };
  const auto residual = [&](const Eigen::Matrix3d& model, std::size_t idx) {
    return SampsonError(model, points1[idx], points2[idx]);
  };
#ifndef COLITE_HAS_POSELIB
  return RunMatrixRansac(
      points1.size(), 8, options.min_num_iterations, options.max_num_iterations,
      options.confidence, options.max_error_px, nonminimal_estimator, residual,
      options.min_inlier_ratio, options.random_seed);
#else
  return RunMatrixRansacWithHypotheses(
      points1.size(), 7, options.min_num_iterations, options.max_num_iterations,
      options.confidence, options.max_error_px, nonminimal_estimator, residual,
      [&](const std::vector<std::size_t>& sample) {
        const auto norm1 = NormalizePoints(points1, sample);
        const auto norm2 = NormalizePoints(points2, sample);
        std::vector<Eigen::Vector3d> rays1, rays2;
        for (std::size_t i = 0; i < sample.size(); ++i) {
          rays1.push_back(norm1.points[i].homogeneous());
          rays2.push_back(norm2.points[i].homogeneous());
        }
        std::vector<Eigen::Matrix3d> candidates, models;
        poselib::relpose_7pt(rays1, rays2, &candidates);
        for (const auto& candidate : candidates) {
          Eigen::Matrix3d model =
              norm2.transform.transpose() * candidate * norm1.transform;
          const double norm = model.norm();
          if (norm > 0.0 && std::isfinite(norm)) models.push_back(model / norm);
        }
        return models;
      }, 8, options.min_inlier_ratio, options.random_seed);
#endif
}

MatrixRansacResult RunEssentialRansac(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const TwoViewGeometryOptions& options,
    const double max_error) {
#ifndef COLITE_HAS_POSELIB
  return RunMatrixRansac(
      points1.size(), 8, options.min_num_iterations,
      options.max_num_iterations, options.confidence, max_error,
      [&](const std::vector<std::size_t>& sample) {
        return EstimateEssentialMatrix(points1, points2, sample);
      },
      [&](const Eigen::Matrix3d& model, const std::size_t idx) {
        return SampsonError(model, points1[idx], points2[idx]);
      }, options.min_inlier_ratio, options.random_seed);
#else
  constexpr std::size_t kSampleSize = 5;
  MatrixRansacResult best;
  if (points1.size() < kSampleSize || points1.size() != points2.size()) {
    return best;
  }
  auto score_model = [&](const Eigen::Matrix3d& model) {
    std::vector<std::size_t> inliers;
    inliers.reserve(points1.size());
    for (std::size_t idx = 0; idx < points1.size(); ++idx) {
      const double error = SampsonError(model, points1[idx], points2[idx]);
      if (std::isfinite(error) && error <= max_error) {
        inliers.push_back(idx);
      }
    }
    return inliers;
  };

  auto rng = MakeRng(options.random_seed);
  const int min_iterations = std::max(0, options.min_num_iterations);
  const int max_iterations = RansacTrials(static_cast<std::size_t>(options.min_inlier_ratio * 100000),
      100000, kSampleSize, options.confidence, options.max_num_iterations);
  int required_iterations = max_iterations;
  for (int iteration = 0; iteration < required_iterations; ++iteration) {
    const std::vector<std::size_t> sample =
        SampleUniqueIndices(&rng, points1.size(), kSampleSize);
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    rays1.reserve(kSampleSize);
    rays2.reserve(kSampleSize);
    for (const std::size_t idx : sample) {
      rays1.push_back(points1[idx].homogeneous().normalized());
      rays2.push_back(points2[idx].homogeneous().normalized());
    }
    std::vector<Eigen::Matrix3d> models;
    poselib::relpose_5pt(rays1, rays2, &models);
    for (Eigen::Matrix3d model : models) {
      const double norm = model.norm();
      if (!(norm > 0.0) || !std::isfinite(norm)) {
        continue;
      }
      model /= norm;
      std::vector<std::size_t> inliers = score_model(model);
      if (inliers.size() <= best.inlier_indices.size()) {
        continue;
      }
      if (inliers.size() >= 8) {
        const std::optional<Eigen::Matrix3d> local =
            EstimateEssentialMatrix(points1, points2, inliers);
        if (local) {
          std::vector<std::size_t> local_inliers = score_model(*local);
          if (local_inliers.size() >= inliers.size()) {
            model = *local;
            inliers = std::move(local_inliers);
          }
        }
      }
      best.model = model;
      best.inlier_indices = std::move(inliers);
      if (best.inlier_indices.size() == points1.size()) {
        return best;
      }
      required_iterations = std::min(max_iterations, std::max(min_iterations,
          RansacTrials(best.inlier_indices.size(), points1.size(), kSampleSize, options.confidence, max_iterations)));
      required_iterations = std::max(required_iterations, iteration + 1);
    }
  }
  return best;
#endif
}

std::vector<FeatureMatch> MatchesFromInlierIndices(
    const std::vector<FeatureMatch>& matches,
    const std::vector<std::size_t>& inlier_indices) {
  std::vector<FeatureMatch> inliers;
  inliers.reserve(inlier_indices.size());
  for (const std::size_t idx : inlier_indices) {
    inliers.push_back(matches[idx]);
  }
  return inliers;
}

double NormalizedRansacThreshold(const Camera& camera1,
                                 const Camera& camera2,
                                 double max_error_px) {
  const double focal1 = camera1.MeanFocalLength(), focal2 = camera2.MeanFocalLength();
  if (focal1 <= 0 || focal2 <= 0 || !std::isfinite(focal1) || !std::isfinite(focal2)) return max_error_px;
  return 0.5 * max_error_px * (1.0 / focal1 + 1.0 / focal2);
}

double InlierRatio(const std::size_t numerator,
                   const std::size_t denominator) {
  if (denominator == 0) {
    return numerator == 0 ? 0.0 : std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(numerator) / static_cast<double>(denominator);
}

bool HasEnoughInliers(const MatrixRansacResult& result,
                      const TwoViewGeometryOptions& options) {
  return !result.inlier_indices.empty() && result.inlier_indices.size() >=
         static_cast<std::size_t>(std::max(options.min_num_inliers, 0));
}

const std::vector<std::size_t>& PreferMoreInliers(
    const std::vector<std::size_t>& lhs,
    const std::vector<std::size_t>& rhs) {
  return rhs.size() > lhs.size() ? rhs : lhs;
}

std::optional<Eigen::Vector3d> TriangulateNormalizedPoint(
    const Eigen::Vector2d& point1,
    const Eigen::Vector2d& point2,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation) {
  Eigen::Matrix<double, 3, 4> p1;
  p1 << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 4> p2;
  p2 << rotation, translation;

  Eigen::Matrix4d a;
  a.row(0) = point1.x() * p1.row(2) - p1.row(0);
  a.row(1) = point1.y() * p1.row(2) - p1.row(1);
  a.row(2) = point2.x() * p2.row(2) - p2.row(0);
  a.row(3) = point2.y() * p2.row(2) - p2.row(1);

  const Eigen::JacobiSVD<Eigen::Matrix4d> svd(a, Eigen::ComputeFullV);
  const Eigen::Vector4d homogeneous = svd.matrixV().col(3);
  if (std::abs(homogeneous.w()) <= kMinHomogeneousScale) {
    return std::nullopt;
  }
  return homogeneous.hnormalized();
}

bool HasPositiveDepthInBothViews(const Eigen::Vector3d& point_cam1,
                                 const Eigen::Matrix3d& rotation,
                                 const Eigen::Vector3d& translation) {
  const Eigen::Vector3d point_cam2 = rotation * point_cam1 + translation;
  return point_cam1.z() > 0.0 && point_cam2.z() > 0.0;
}

Pose3 ChoosePoseFromEssentialMatrix(
    const Eigen::Matrix3d& essential,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    const std::vector<std::size_t>& inlier_indices) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d u = svd.matrixU();
  Eigen::Matrix3d vt = svd.matrixV().transpose();
  if (u.determinant() < 0.0) {
    u.col(2) *= -1.0;
  }
  if (vt.determinant() < 0.0) {
    vt.row(2) *= -1.0;
  }

  Eigen::Matrix3d w;
  w << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> candidates;
  candidates.emplace_back(u * w * vt, u.col(2));
  candidates.emplace_back(u * w * vt, -u.col(2));
  candidates.emplace_back(u * w.transpose() * vt, u.col(2));
  candidates.emplace_back(u * w.transpose() * vt, -u.col(2));

  std::size_t best_positive_depths = 0;
  Pose3 best_pose;
  for (auto& [rotation, translation] : candidates) {
    if (rotation.determinant() < 0.0) {
      rotation = -rotation;
      translation = -translation;
    }

    std::size_t positive_depths = 0;
    for (const std::size_t idx : inlier_indices) {
      const auto point_cam1 =
          TriangulateNormalizedPoint(points1[idx], points2[idx], rotation,
                                     translation);
      if (point_cam1 &&
          HasPositiveDepthInBothViews(*point_cam1, rotation, translation)) {
        ++positive_depths;
      }
    }

    if (positive_depths > best_positive_depths) {
      best_positive_depths = positive_depths;
      best_pose.q_cw = Eigen::Quaterniond(rotation).normalized();
      best_pose.t_cw = translation.normalized();
    }
  }
  return best_pose;
}

double ComputeMedianTriangulationAngle(
    const Camera& camera1,
    const FeatureSet& features1,
    const Camera& camera2,
    const FeatureSet& features2,
    const Pose3& cam2_from_cam1,
    const std::vector<FeatureMatch>& inlier_matches) {
  if (inlier_matches.empty()) {
    return -1.0;
  }

  Pose3 cam1_from_world;
  cam1_from_world.q_cw = Eigen::Quaterniond::Identity();
  cam1_from_world.t_cw = Eigen::Vector3d::Zero();

  std::vector<double> angles;
  angles.reserve(inlier_matches.size());
  for (const FeatureMatch& match : inlier_matches) {
    const FeatureKeypoint& keypoint1 = features1.keypoints[match.idx1];
    const FeatureKeypoint& keypoint2 = features2.keypoints[match.idx2];
    const auto bearing1 = TryPixelToBearing(camera1, Eigen::Vector2d(keypoint1.x, keypoint1.y));
    const auto bearing2 = TryPixelToBearing(camera2, Eigen::Vector2d(keypoint2.x, keypoint2.y));
    if (!bearing1 || !bearing2) continue;
    angles.push_back(TriangulationAngleDegrees(
        cam1_from_world, *bearing1, cam2_from_cam1, *bearing2));
  }

  std::sort(angles.begin(), angles.end());
  if (angles.empty()) return -1.0;
  return angles[angles.size() / 2];
}

bool DetectWatermark(const Camera& camera1, const Camera& camera2,
                     const FeatureSet& features1, const FeatureSet& features2,
                     const std::vector<FeatureMatch>& matches, const TwoViewGeometryOptions& options) {
  if (matches.empty()) return false;
  const auto interior = [&](const Camera& camera, const FeatureKeypoint& point) {
    const double border = options.watermark_border_size * std::hypot(camera.width, camera.height);
    return point.x >= border && point.y >= border &&
           point.x <= camera.width - border && point.y <= camera.height - border;
  };
  std::vector<Eigen::Vector2d> translations;
  std::size_t border_count = 0;
  for (const auto& match : matches) {
    const auto& a = features1.keypoints[match.idx1];
    const auto& b = features2.keypoints[match.idx2];
    if (!interior(camera1, a) && !interior(camera2, b)) ++border_count;
    translations.emplace_back(double(b.x) - a.x, double(b.y) - a.y);
  }
  const double required = options.watermark_min_inlier_ratio * matches.size();
  if (border_count < required) return false;
  auto rng = MakeRng(options.random_seed);
  std::uniform_int_distribution<std::size_t> sample(0, translations.size() - 1);
  const double max_error2 = options.watermark_detection_max_error * options.watermark_detection_max_error;
  const int trials = RansacTrials(static_cast<std::size_t>(options.watermark_min_inlier_ratio * 100000),
      100000, 1, options.confidence, options.max_num_iterations);
  std::size_t best = 0;
  for (int trial = 0; trial < trials; ++trial) {
    const auto estimate = translations[sample(rng)];
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    std::size_t count = 0;
    for (const auto& value : translations) if ((value - estimate).squaredNorm() <= max_error2) { mean += value; ++count; }
    if (count > best && count > 0) {
      mean /= count;
      std::size_t refined = 0;
      for (const auto& value : translations) if ((value - mean).squaredNorm() <= max_error2) ++refined;
      best = std::max({best, count, refined});
      if (best >= required) return true;
    }
  }
  return false;
}

}  // namespace

void TwoViewGeometryOptions::Check() const {
  const auto ratio = [](double value) { return std::isfinite(value) && value >= 0 && value <= 1; };
  if (min_num_inliers < 0 || min_num_iterations < 0 || max_num_iterations < min_num_iterations ||
      !ratio(confidence) || !ratio(min_inlier_ratio) || !ratio(min_E_F_inlier_ratio) ||
      !ratio(max_H_inlier_ratio) || !ratio(watermark_min_inlier_ratio) || !ratio(watermark_border_size) ||
      random_seed < -1 || !std::isfinite(max_error_px) || max_error_px <= 0 ||
      !std::isfinite(watermark_detection_max_error) || watermark_detection_max_error <= 0 ||
      !std::isfinite(stationary_matches_max_error) || stationary_matches_max_error < 0) {
    throw std::invalid_argument("Invalid TwoViewGeometry options");
  }
}

bool EstimateTwoViewGeometryPose(
    const Camera& camera1, const FeatureSet& features1,
    const Camera& camera2, const FeatureSet& features2, TwoViewGeometry* geometry) {
  using Type = TwoViewGeometry::Type;
  if (geometry->inlier_matches.empty() || geometry->type < Type::kCalibratedEssential ||
      geometry->type > Type::kPlanarOrPanoramic) return false;
  const auto points = ExtractMatchedPoints(camera1, features1, camera2, features2, geometry->inlier_matches);
  if (points.normalized1.empty()) return false;
  if (!geometry->cam2_from_cam1) {
    if (geometry->type == Type::kCalibratedEssential || geometry->type == Type::kUncalibratedFundamental) {
      std::optional<Eigen::Matrix3d> essential = geometry->type == Type::kCalibratedEssential ? geometry->E : std::nullopt;
      if (!essential && geometry->F) essential = camera2.CalibrationMatrix().transpose() * *geometry->F * camera1.CalibrationMatrix();
      if (!essential || !essential->allFinite() || essential->norm() <= 1e-12) return false;
      std::vector<std::size_t> indices(points.normalized1.size());
      std::iota(indices.begin(), indices.end(), 0);
      const auto pose = ChoosePoseFromEssentialMatrix(*essential, points.normalized1, points.normalized2, indices);
      if (!pose.q_cw.coeffs().allFinite() || pose.t_cw.squaredNorm() < 1e-12) return false;
      geometry->cam2_from_cam1 = pose;
    } else {
      if (!geometry->H) return false;
      geometry->cam2_from_cam1 = matching_internal::PoseFromHomography(
          camera2.CalibrationMatrix().inverse() * *geometry->H * camera1.CalibrationMatrix(), points.normalized1, points.normalized2);
      if (!geometry->cam2_from_cam1) return false;
    }
  }
  if (geometry->type == Type::kPlanarOrPanoramic) {
    geometry->type = geometry->cam2_from_cam1->t_cw.squaredNorm() < 1e-12 ? Type::kPanoramic : Type::kPlanarHomography;
  }
  geometry->median_triangulation_angle = geometry->type == Type::kPanoramic ? 0.0 :
      ComputeMedianTriangulationAngle(camera1, features1, camera2, features2, *geometry->cam2_from_cam1, geometry->inlier_matches);
  return true;
}

TwoViewGeometry EstimateTwoViewGeometry(
    const Camera& camera1, const FeatureSet& features1,
    const Camera& camera2, const FeatureSet& features2,
    const std::vector<FeatureMatch>& input_matches, const TwoViewGeometryOptions& options) {
  options.Check();
  std::vector<FeatureMatch> filtered;
  const auto* matches = &input_matches;
  for (const auto& match : input_matches) {
    if (!IsValidFeatureIndex(match.idx1, features1) || !IsValidFeatureIndex(match.idx2, features2)) {
      throw std::out_of_range("Feature match index is outside keypoint range");
    }
  }
  if (options.filter_stationary_matches) {
    for (const auto& match : input_matches) {
      const auto& a = features1.keypoints[match.idx1]; const auto& b = features2.keypoints[match.idx2];
      if (std::hypot(double(a.x) - b.x, double(a.y) - b.y) > options.stationary_matches_max_error) filtered.push_back(match);
    }
    matches = &filtered;
  }
  if (options.multiple_models) {
    auto remaining = *matches;
    std::vector<TwoViewGeometry> models;
    auto single = options;
    single.multiple_models = false; single.filter_stationary_matches = false;
    while (!remaining.empty()) {
      auto geometry = EstimateTwoViewGeometry(camera1, features1, camera2, features2, remaining, single);
      if (geometry.type == TwoViewGeometry::Type::kDegenerate || geometry.inlier_matches.empty()) break;
      std::unordered_set<std::uint64_t> inliers;
      for (const auto& match : geometry.inlier_matches) inliers.insert((std::uint64_t(match.idx1) << 32) | match.idx2);
      remaining.erase(std::remove_if(remaining.begin(), remaining.end(), [&](const auto& match) {
        return inliers.count((std::uint64_t(match.idx1) << 32) | match.idx2);
      }), remaining.end());
      if (!options.multiple_ignore_watermark || geometry.type != TwoViewGeometry::Type::kWatermark) models.push_back(std::move(geometry));
    }
    if (models.size() == 1) return std::move(models.front());
    TwoViewGeometry result;
    result.type = models.empty() ? TwoViewGeometry::Type::kDegenerate : TwoViewGeometry::Type::kMultiple;
    for (auto& model : models) result.inlier_matches.insert(result.inlier_matches.end(), model.inlier_matches.begin(), model.inlier_matches.end());
    return result;
  }
  TwoViewGeometry geometry;
  if (matches->size() < std::max<std::size_t>(4, options.min_num_inliers)) {
    geometry.type = TwoViewGeometry::Type::kDegenerate; return geometry;
  }
  const auto points = ExtractMatchedPoints(camera1, features1, camera2, features2, *matches);
  const auto homography = RunMatrixRansac(points.matches.size(), 4, options.min_num_iterations,
      options.max_num_iterations, options.confidence, options.max_error_px,
      [&](const auto& sample) { return EstimateHomographyMatrix(points.pixels1, points.pixels2, sample); },
      [&](const auto& model, std::size_t idx) { return HomographyTransferError(model, points.pixels1[idx], points.pixels2[idx]); },
      options.min_inlier_ratio, options.random_seed);
  if (!homography.inlier_indices.empty()) geometry.H = homography.model;
  MatrixRansacResult fundamental, essential;
  if (!options.force_H_use) {
    fundamental = RunFundamentalRansac(points.pixels1, points.pixels2, options);
    if (!fundamental.inlier_indices.empty()) geometry.F = fundamental.model;
    if (camera1.has_prior_focal_length && camera2.has_prior_focal_length) {
      essential = RunEssentialRansac(points.normalized1, points.normalized2, options,
                                     NormalizedRansacThreshold(camera1, camera2, options.max_error_px));
      for (auto& index : essential.inlier_indices) index = points.normalized_indices[index];
      if (!essential.inlier_indices.empty()) geometry.E = essential.model;
    }
  }
  const auto* inliers = &fundamental.inlier_indices;
  if (HasEnoughInliers(essential, options) && InlierRatio(essential.inlier_indices.size(), fundamental.inlier_indices.size()) > options.min_E_F_inlier_ratio) {
    geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
    inliers = &PreferMoreInliers(essential.inlier_indices, fundamental.inlier_indices);
    if (HasEnoughInliers(homography, options) && InlierRatio(homography.inlier_indices.size(), essential.inlier_indices.size()) > options.max_H_inlier_ratio) {
      geometry.type = TwoViewGeometry::Type::kPlanarOrPanoramic;
      inliers = &PreferMoreInliers(*inliers, homography.inlier_indices);
    }
  } else if (HasEnoughInliers(fundamental, options)) {
    geometry.type = TwoViewGeometry::Type::kUncalibratedFundamental;
    if (HasEnoughInliers(homography, options) && InlierRatio(homography.inlier_indices.size(), fundamental.inlier_indices.size()) > options.max_H_inlier_ratio) {
      geometry.type = TwoViewGeometry::Type::kPlanarOrPanoramic;
      if (homography.inlier_indices.size() >= inliers->size()) inliers = &homography.inlier_indices;
    }
  } else if (HasEnoughInliers(homography, options)) {
    geometry.type = TwoViewGeometry::Type::kPlanarOrPanoramic;
    inliers = &homography.inlier_indices;
  } else {
    geometry.type = TwoViewGeometry::Type::kDegenerate; return geometry;
  }
  geometry.inlier_matches = MatchesFromInlierIndices(points.matches, *inliers);
  if (options.detect_watermark && DetectWatermark(camera1, camera2, features1, features2, geometry.inlier_matches, options)) {
    geometry.type = TwoViewGeometry::Type::kWatermark;
  }
  if (options.compute_relative_pose) EstimateTwoViewGeometryPose(camera1, features1, camera2, features2, &geometry);
  return geometry;
}

TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const Camera& camera1, const FeatureSet& features1,
    const Camera& camera2, const FeatureSet& features2,
    const std::vector<FeatureMatch>& matches,
    const TwoViewGeometryOptions& options) {
  Camera calibrated1 = camera1, calibrated2 = camera2;
  calibrated1.has_prior_focal_length = true;
  calibrated2.has_prior_focal_length = true;
  return EstimateTwoViewGeometry(calibrated1, features1, calibrated2, features2,
                                matches, options);
}

std::size_t RecoverMatchGraphRelativePoses(
    const std::map<ImageId, Camera>& cameras_by_image, MatchGraph* graph) {
  std::size_t recovered = 0;
  for (auto& [pair, geometry] : graph->verified_geometries) {
    if (geometry.type == TwoViewGeometry::Type::kRigCalibrated) throw std::invalid_argument("RIG_CALIBRATED mapping requires multi-sensor rig support");
    if (geometry.cam2_from_cam1 && geometry.median_triangulation_angle >= 0.0) continue;
    const bool had_pose = geometry.cam2_from_cam1.has_value();
    if (geometry.type < TwoViewGeometry::Type::kCalibratedEssential || geometry.type > TwoViewGeometry::Type::kPlanarOrPanoramic || geometry.inlier_matches.empty()) continue;
    if (EstimateTwoViewGeometryPose(cameras_by_image.at(pair.image_id1), graph->features.at(pair.image_id1),
        cameras_by_image.at(pair.image_id2), graph->features.at(pair.image_id2), &geometry) && !had_pose) ++recovered;
  }
  return recovered;
}

}  // namespace colite
