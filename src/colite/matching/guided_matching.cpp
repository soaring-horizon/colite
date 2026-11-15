#include "colite/matching/guided_matching.h"

#include <cmath>
#include <chrono>
#include <limits>

namespace colite {
namespace {

bool IsValidFeatureIndex(Point2DIdx idx, const FeatureSet& features) {
  return idx != kInvalidPoint2DIdx &&
         idx < static_cast<Point2DIdx>(features.keypoints.size());
}

double SquaredSampsonError(const Eigen::Matrix3d& fundamental_matrix,
                           const FeatureKeypoint& keypoint1,
                           const FeatureKeypoint& keypoint2) {
  const Eigen::Vector3d x1(keypoint1.x, keypoint1.y, 1.0);
  const Eigen::Vector3d x2(keypoint2.x, keypoint2.y, 1.0);
  const Eigen::Vector3d fx1 = fundamental_matrix * x1;
  const Eigen::Vector3d ftx2 = fundamental_matrix.transpose() * x2;
  const double residual = x2.transpose() * fundamental_matrix * x1;
  const double denominator =
      fx1.x() * fx1.x() + fx1.y() * fx1.y() +
      ftx2.x() * ftx2.x() + ftx2.y() * ftx2.y();
  if (denominator <= 0.0 || !std::isfinite(denominator)) {
    return std::numeric_limits<double>::infinity();
  }
  return residual * residual / denominator;
}

}  // namespace

bool HasGuidedMatchingModel(const TwoViewGeometry& geometry) {
  const auto valid = [](const std::optional<Eigen::Matrix3d>& matrix) {
    return matrix && matrix->allFinite() && matrix->cwiseAbs().maxCoeff() > 0;
  };
  switch (geometry.type) {
    case TwoViewGeometry::Type::kCalibratedEssential: return valid(geometry.E);
    case TwoViewGeometry::Type::kUncalibratedFundamental: return valid(geometry.F);
    case TwoViewGeometry::Type::kPlanarHomography:
    case TwoViewGeometry::Type::kPanoramic:
    case TwoViewGeometry::Type::kPlanarOrPanoramic: return valid(geometry.H);
    default: return false;
  }
}

std::vector<FeatureMatch> MatchPreparedFeaturePairGuided(
    const Camera& camera1, const FeatureSet& features1,
    const PreparedFeatureMatcher& prepared1,
    const Camera& camera2, const FeatureSet& features2,
    const PreparedFeatureMatcher& prepared2,
    const TwoViewGeometry& geometry, const double max_error_px,
    const DescriptorMatchingOptions& options, GuidedDescriptorMatchingStats* stats) {
  if (!HasGuidedMatchingModel(geometry) || !std::isfinite(max_error_px) || max_error_px < 0) return {};
  if (geometry.type == TwoViewGeometry::Type::kUncalibratedFundamental) {
    return MatchPreparedFeatureSetsGuidedFundamental(prepared1, prepared2, *geometry.F,
                                                    max_error_px, options, stats);
  }
  const auto started = std::chrono::steady_clock::now();
  const bool calibrated = geometry.type == TwoViewGeometry::Type::kCalibratedEssential;
  const auto coordinates = [calibrated](const Camera& camera, const FeatureSet& features) {
    std::vector<Eigen::Vector2d> result;
    result.reserve(features.keypoints.size());
    for (const auto& point : features.keypoints) {
      const Eigen::Vector2d pixel(point.x, point.y);
      if (!calibrated) {
        result.push_back(pixel);
      } else {
        const auto ray = TryUnprojectPixel(camera, pixel, 1.0);
        result.push_back(ray ? Eigen::Vector2d(ray->x() / ray->z(), ray->y() / ray->z())
                             : Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()));
      }
    }
    return result;
  };
  const auto points1 = coordinates(camera1, features1);
  const auto points2 = coordinates(camera2, features2);
  double threshold = max_error_px;
  if (calibrated) {
    const double first = max_error_px / camera1.MeanFocalLength();
    const double second = max_error_px / camera2.MeanFocalLength();
    threshold = std::sqrt(0.5 * (first * first + second * second));
  }
  if (stats) stats->candidate_generation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return MatchPreparedFeatureSetsGuidedGeometry(prepared1, prepared2,
      calibrated ? GuidedGeometryType::kEpipolar : GuidedGeometryType::kHomography,
      calibrated ? *geometry.E : *geometry.H, points1, points2, threshold, options, stats);
}

std::vector<FeatureMatch> FilterMatchesByGuidedFundamental(
    const FeatureSet& features1,
    const FeatureSet& features2,
    const std::vector<FeatureMatch>& candidate_matches,
    const Eigen::Matrix3d& fundamental_matrix,
    const GuidedMatchingOptions& options) {
  std::vector<FeatureMatch> guided_matches;
  if (candidate_matches.empty() || fundamental_matrix.norm() <= 0.0 ||
      options.max_sampson_error_px < 0.0) {
    return guided_matches;
  }

  const double max_squared_error =
      options.max_sampson_error_px * options.max_sampson_error_px;
  guided_matches.reserve(candidate_matches.size());
  for (const FeatureMatch& match : candidate_matches) {
    if (!IsValidFeatureIndex(match.idx1, features1) ||
        !IsValidFeatureIndex(match.idx2, features2)) {
      continue;
    }

    const double squared_error = SquaredSampsonError(
        fundamental_matrix, features1.keypoints[match.idx1],
        features2.keypoints[match.idx2]);
    if (squared_error <= max_squared_error) {
      guided_matches.push_back(match);
    }
  }
  return guided_matches;
}

}  // namespace colite
