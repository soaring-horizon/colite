#include "colite/matching/two_view_geometry_verifier.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"

namespace colite {
namespace {

Camera MakeCamera() {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  camera.has_prior_focal_length = true;
  return camera;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world,
               const Eigen::Matrix3d& cam_from_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond(cam_from_world).normalized();
  pose.t_cw = -cam_from_world * center_world;
  return pose;
}

void AppendProjectedPoint(const Camera& camera,
                          const Pose3& cam1_from_world,
                          const Pose3& cam2_from_world,
                          const Eigen::Vector3d& point_world,
                          FeatureSet* features1,
                          FeatureSet* features2,
                          std::vector<FeatureMatch>* matches) {
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  const Point2DIdx idx =
      static_cast<Point2DIdx>(features1->keypoints.size());
  features1->keypoints.push_back(
      {static_cast<float>(pixel1->x()), static_cast<float>(pixel1->y()), 1.0F,
       0.0F});
  features2->keypoints.push_back(
      {static_cast<float>(pixel2->x()), static_cast<float>(pixel2->y()), 1.0F,
       0.0F});
  matches->push_back({idx, idx});
}

TEST(TwoViewGeometryVerifierTest, RecoversCalibratedGeometryFromInlierMatches) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      const Eigen::Vector3d point_world(
          -1.0 + 0.4 * x, -0.6 + 0.3 * y,
          4.0 + 0.07 * x + 0.11 * y + 0.03 * x * y);
      AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                           point_world, &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.random_seed = 0;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kCalibratedEssential);
  EXPECT_GE(geometry.inlier_matches.size(), 20U);
  EXPECT_GT(geometry.E.value().norm(), 0.0);
  EXPECT_GT(geometry.F.value().norm(), 0.0);
  EXPECT_GT(geometry.median_triangulation_angle, 0.0);
  MatchGraph graph;
  graph.features[1] = features1;
  graph.features[2] = features2;
  auto missing_pose = geometry;
  missing_pose.cam2_from_cam1.reset();
  missing_pose.median_triangulation_angle = -1;
  SetVerifiedGeometry(&graph, 1, 2, missing_pose);
  const std::map<ImageId, Camera> cameras{{1, camera}, {2, camera}};
  EXPECT_EQ(RecoverMatchGraphRelativePoses(cameras, &graph), 1U);
  const auto& recovered = graph.verified_geometries.begin()->second;
  ASSERT_TRUE(recovered.cam2_from_cam1);
  EXPECT_GT(recovered.cam2_from_cam1->t_cw.dot(cam2_from_world.t_cw), 0.999);
  EXPECT_NEAR(recovered.median_triangulation_angle, geometry.median_triangulation_angle, 1e-10);
  EXPECT_TRUE(recovered.E->isApprox(*geometry.E, 0));
  EXPECT_EQ(recovered.inlier_matches.size(), geometry.inlier_matches.size());
  EXPECT_EQ(RecoverMatchGraphRelativePoses(cameras, &graph), 0U);

}

TEST(TwoViewGeometryVerifierTest, FundamentalFitRejectsDistractorsAndIsRepeatable) {
  Camera camera = MakeCamera();
  camera.has_prior_focal_length = false;
  const Pose3 pose1 = CameraAt(Eigen::Vector3d::Zero());
  const Pose3 pose2 = CameraAt(
      Eigen::Vector3d(0.9, 0.1, 0.05),
      Eigen::AngleAxisd(0.13, Eigen::Vector3d::UnitY()).toRotationMatrix());
  std::mt19937 rng(47);
  std::uniform_real_distribution<double> xy(-1.3, 1.3), z(3.5, 7.5);
  std::uniform_real_distribution<float> pixel_x(50.0F, 950.0F),
      pixel_y(50.0F, 750.0F), noise(-0.2F, 0.2F);
  FeatureSet features1, features2;
  std::vector<FeatureMatch> matches;
  for (int i = 0; i < 100; ++i) {
    const double x = xy(rng), y = xy(rng), depth = z(rng);
    AppendProjectedPoint(camera, pose1, pose2, {x, y, depth},
                         &features1, &features2, &matches);
    features2.keypoints.back().x += noise(rng);
    features2.keypoints.back().y += noise(rng);
  }
  for (int i = 0; i < 100; ++i) {
    const auto index = static_cast<Point2DIdx>(features1.keypoints.size());
    features1.keypoints.push_back({pixel_x(rng), pixel_y(rng), 1.0F, 0.0F});
    features2.keypoints.push_back({pixel_x(rng), pixel_y(rng), 1.0F, 0.0F});
    matches.push_back({index, index});
  }
  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.random_seed = 0;
  options.max_error_px = 0.75;
  options.min_num_inliers = 80;
  options.max_num_iterations = 2000;
  options.max_H_inlier_ratio = 1.0;
  const auto geometry = EstimateTwoViewGeometry(
      camera, features1, camera, features2, matches, options);
  ASSERT_EQ(geometry.type, TwoViewGeometry::Type::kUncalibratedFundamental);
  std::size_t true_inliers = 0;
  for (const auto& match : geometry.inlier_matches) {
    if (match.idx1 < 100) ++true_inliers;
  }
  EXPECT_GE(true_inliers, 95U);
  EXPECT_LE(geometry.inlier_matches.size() - true_inliers, 3U);
  EXPECT_NEAR(geometry.F.value().determinant(), 0.0, 1e-10);
  const auto repeat = EstimateTwoViewGeometry(
      camera, features1, camera, features2, matches, options);
  EXPECT_EQ((geometry.F.value() - repeat.F.value()).norm(), 0.0);
  ASSERT_EQ(geometry.inlier_matches.size(), repeat.inlier_matches.size());
  for (std::size_t i = 0; i < geometry.inlier_matches.size(); ++i) {
    EXPECT_EQ(geometry.inlier_matches[i].idx1, repeat.inlier_matches[i].idx1);
    EXPECT_EQ(geometry.inlier_matches[i].idx2, repeat.inlier_matches[i].idx2);
  }
}

TEST(TwoViewGeometryVerifierTest, ClassifiesPlanarSceneAsHomography) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(0.8, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      const Eigen::Vector3d point_world(-1.2 + 0.45 * x, -0.7 + 0.35 * y,
                                        4.5);
      AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                           point_world, &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.random_seed = 0;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPlanarHomography);
  EXPECT_GE(geometry.inlier_matches.size(), 20U);
  EXPECT_GT(geometry.H.value().norm(), 0.0);
  EXPECT_GT(geometry.median_triangulation_angle, 0.0);
}

TEST(TwoViewGeometryVerifierTest, ClassifiesPureRotationAsPanoramic) {
  const Camera camera = MakeCamera();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const double angle_rad = 8.0 * M_PI / 180.0;
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(angle_rad, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Pose3 cam2_from_world =
      CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0), rotation);

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      const Eigen::Vector3d point_world(-1.0 + 0.4 * x, -0.6 + 0.3 * y,
                                        5.0 + 0.2 * x + 0.15 * y);
      AppendProjectedPoint(camera, cam1_from_world, cam2_from_world,
                           point_world, &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.random_seed = 0;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPanoramic);
  EXPECT_GE(geometry.inlier_matches.size(), 20U);
  EXPECT_GT(geometry.H.value().norm(), 0.0);
  EXPECT_EQ(geometry.median_triangulation_angle, 0.0);
}

TEST(TwoViewGeometryVerifierTest, ReturnsDegenerateWhenMatchesAreInsufficient) {
  const Camera camera = MakeCamera();
  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;

  features1.keypoints.push_back({500.0F, 400.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({490.0F, 400.0F, 1.0F, 0.0F});
  matches.push_back({0, 0});

  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.random_seed = 0;
  options.min_num_inliers = 8;

  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              options);

  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kDegenerate);
  EXPECT_TRUE(geometry.inlier_matches.empty());
}

TEST(TwoViewGeometryVerifierTest, PartialCameraDomainPreservesPixelModelsAndPoseMatches) {
  auto camera = MakeCamera();
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700, 500, 400, -1};
  FeatureSet a, b;
  std::vector<FeatureMatch> matches;
  std::mt19937 generator(19);
  std::uniform_real_distribution<double> xs(-.5, .5), ys(-.4, .4), zs(3, 6);
  for (int i = 0; i < 30; ++i) {
    AppendProjectedPoint(camera, CameraAt({0, 0, 0}), CameraAt({.4, .05, 0}),
        {xs(generator), ys(generator), zs(generator)}, &a, &b, &matches);
  }
  // Deliberately place invalid rays before the valid set: E's compact inlier
  // indices must be mapped back to the original feature/match rows.
  a.keypoints.insert(a.keypoints.begin(), {0, 0, 1, 0});
  b.keypoints.insert(b.keypoints.begin(), {990, 790, 1, 0});
  for (auto& match : matches) { ++match.idx1; ++match.idx2; }
  matches.insert(matches.begin(), {0, 0});
  TwoViewGeometryOptions options;
  options.random_seed = 0; options.compute_relative_pose = true;
  options.detect_watermark = false; options.max_error_px = .001;
  options.max_H_inlier_ratio = 1;
  const auto geometry = EstimateTwoViewGeometry(camera, a, camera, b, matches, options);
  ASSERT_TRUE(geometry.E);
  ASSERT_TRUE(geometry.cam2_from_cam1);
  EXPECT_GT(geometry.median_triangulation_angle, 1);
  EXPECT_GT(geometry.cam2_from_cam1->t_cw.normalized().dot(
      Eigen::Vector3d(-.4, -.05, 0).normalized()), .999);
  ASSERT_EQ(geometry.inlier_matches.size(), 30U);
  for (const auto& match : geometry.inlier_matches) {
    EXPECT_GT(match.idx1, 0U);
    EXPECT_LT(match.idx1, a.keypoints.size());
    EXPECT_EQ(match.idx1, match.idx2);
  }

  // A pixel-only H is still estimable when all rays are outside the domain.
  FeatureSet outside1, outside2;
  std::vector<FeatureMatch> outside_matches;
  for (Point2DIdx i = 0; i < 20; ++i) {
    const float x = 10 + 5 * (i % 5), y = 10 + 5 * (i / 5);
    outside1.keypoints.push_back({x, y, 1, 0});
    outside2.keypoints.push_back({x + 2, y + 1, 1, 0});
    outside_matches.push_back({i, i});
  }
  options.force_H_use = true;
  const auto h = EstimateTwoViewGeometry(camera, outside1, camera, outside2, outside_matches, options);
  EXPECT_TRUE(h.H);
  EXPECT_EQ(h.inlier_matches.size(), 20U);
  EXPECT_FALSE(h.cam2_from_cam1);
}

TEST(TwoViewGeometryVerifierTest, ThrowsWhenMatchIndexIsInvalid) {
  const Camera camera = MakeCamera();
  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;

  features1.keypoints.push_back({500.0F, 400.0F, 1.0F, 0.0F});
  features2.keypoints.push_back({490.0F, 400.0F, 1.0F, 0.0F});
  matches.push_back({0, 7});

  EXPECT_THROW(EstimateTwoViewGeometry(camera, features1, camera, features2,
                                       matches),
               std::out_of_range);
}

TEST(TwoViewGeometryVerifierTest, DefaultsOmitPoseAndHonorFocalPrior) {
  auto camera1 = MakeCamera(), camera2 = MakeCamera();
  FeatureSet first, second;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 6; ++y) for (int x = 0; x < 8; ++x) {
    AppendProjectedPoint(camera1, CameraAt({0, 0, 0}), CameraAt({1, 0.1, 0}),
        {-1.2 + 0.3*x, -0.8 + 0.3*y, 4 + 0.15*x*x + 0.11*y*y}, &first, &second, &matches);
  }
  TwoViewGeometryOptions options;
  options.random_seed = 0;
  options.max_H_inlier_ratio = 1;
  auto geometry = EstimateTwoViewGeometry(camera1, first, camera2, second, matches, options);
  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kCalibratedEssential);
  EXPECT_TRUE(geometry.E);
  EXPECT_FALSE(geometry.cam2_from_cam1);
  EXPECT_EQ(geometry.median_triangulation_angle, -1);
  camera2.has_prior_focal_length = false;
  geometry = EstimateTwoViewGeometry(camera1, first, camera2, second, matches, options);
  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kUncalibratedFundamental);
  EXPECT_FALSE(geometry.E);
  EXPECT_TRUE(geometry.F);
  EXPECT_FALSE(geometry.cam2_from_cam1);
  ASSERT_TRUE(EstimateTwoViewGeometryPose(camera1, first, camera2, second, &geometry));
  EXPECT_GT(geometry.cam2_from_cam1->t_cw.normalized().dot(Eigen::Vector3d(-1, -0.1, 0).normalized()), 0.999);
  EXPECT_GT(geometry.median_triangulation_angle, 1);
}

TEST(TwoViewGeometryVerifierTest, RecoversHomographyPoseForUnequalCamerasAndProjectiveScales) {
  auto camera1 = MakeCamera(), camera2 = MakeCamera();
  camera2.params = {1000, 900, 450, 390};
  const auto rotation = Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const auto target_pose = CameraAt({0.8, 0.1, 0}, rotation);
  const Eigen::Matrix3d normalized_h = rotation + target_pose.t_cw * Eigen::Vector3d(0, 0, 1).transpose() / 4.5;
  FeatureSet first, second;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) for (int x = 0; x < 6; ++x) {
    const Eigen::Vector3d xyz(-1.2 + 0.4*x, -0.7 + 0.3*y, 4.5);
    const auto a = *ProjectWorldPoint(camera1, Pose3{}, xyz);
    const auto b = *ProjectWorldPoint(camera2, target_pose, xyz);
    const auto idx = static_cast<Point2DIdx>(first.keypoints.size());
    first.keypoints.push_back({float(a.x()), float(a.y()), 1, 0});
    second.keypoints.push_back({float(b.x()), float(b.y()), 1, 0});
    matches.push_back({idx, idx});
  }
  for (double scale : {1.0, -3.0, 1e-14}) {
    TwoViewGeometry geometry;
    geometry.type = TwoViewGeometry::Type::kPlanarOrPanoramic;
    geometry.H = scale * camera2.CalibrationMatrix() * normalized_h * camera1.CalibrationMatrix().inverse();
    geometry.inlier_matches = matches;
    ASSERT_TRUE(EstimateTwoViewGeometryPose(camera1, first, camera2, second, &geometry));
    EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPlanarHomography);
    EXPECT_GT(geometry.cam2_from_cam1->t_cw.norm(), 0);
    EXPECT_GT(geometry.median_triangulation_angle, 0);
    for (const auto& match : matches) {
      const auto a = Eigen::Vector2d(first.keypoints[match.idx1].x, first.keypoints[match.idx1].y);
      const auto b = Eigen::Vector2d(second.keypoints[match.idx2].x, second.keypoints[match.idx2].y);
      const auto xyz = TriangulatePoint(camera1, Pose3{}, a, camera2, *geometry.cam2_from_cam1, b);
      ASSERT_TRUE(xyz);
      EXPECT_GT(xyz->z(), 0);
      EXPECT_LT(*ReprojectionError(camera2, *geometry.cam2_from_cam1, *xyz, b), 0.01);
    }
  }
  TwoViewGeometry invalid;
  invalid.type = TwoViewGeometry::Type::kPlanarOrPanoramic;
  invalid.H = Eigen::Matrix3d::Zero(); invalid.inlier_matches = matches;
  EXPECT_FALSE(EstimateTwoViewGeometryPose(camera1, first, camera2, second, &invalid));
  EXPECT_FALSE(invalid.cam2_from_cam1);
}

TEST(TwoViewGeometryVerifierTest, DistinguishesDeferredPlanarPanoramicAndPureRotationPose) {
  const auto camera = MakeCamera();
  FeatureSet first, second;
  std::vector<FeatureMatch> matches;
  const auto r = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitY()).toRotationMatrix();
  for (int y = 0; y < 5; ++y) for (int x = 0; x < 6; ++x) {
    AppendProjectedPoint(camera, Pose3{}, CameraAt({0, 0, 0}, r), {-1 + 0.4*x, -0.5 + 0.3*y, 5 + 0.1*x*y}, &first, &second, &matches);
  }
  TwoViewGeometryOptions options;
  options.random_seed = 0;
  options.force_H_use = true;
  auto geometry = EstimateTwoViewGeometry(camera, first, camera, second, matches, options);
  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPlanarOrPanoramic);
  EXPECT_FALSE(geometry.cam2_from_cam1);
  ASSERT_TRUE(EstimateTwoViewGeometryPose(camera, first, camera, second, &geometry));
  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPanoramic);
  EXPECT_EQ(geometry.cam2_from_cam1->t_cw.norm(), 0);
  EXPECT_NEAR(std::abs(geometry.cam2_from_cam1->q_cw.dot(Eigen::Quaterniond(r))), 1, 1e-10);
}

TEST(TwoViewGeometryVerifierTest, DetectsWatermarksAndFiltersStationaryMatches) {
  const auto camera = MakeCamera();
  FeatureSet first, second;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) for (int x = 0; x < 6; ++x) {
    first.keypoints.push_back({float(15 + 9*x), float(15 + 9*y), 1, 0});
    second.keypoints.push_back({float(32 + 9*x), float(24 + 9*y), 1, 0});
    const auto idx = static_cast<Point2DIdx>(matches.size()); matches.push_back({idx, idx});
  }
  TwoViewGeometryOptions options;
  options.random_seed = 0; options.force_H_use = true;
  EXPECT_EQ(EstimateTwoViewGeometry(camera, first, camera, second, matches, options).type, TwoViewGeometry::Type::kWatermark);
  options.detect_watermark = false;
  EXPECT_EQ(EstimateTwoViewGeometry(camera, first, camera, second, matches, options).type, TwoViewGeometry::Type::kPlanarOrPanoramic);
  options.filter_stationary_matches = true;
  options.stationary_matches_max_error = 20;
  EXPECT_EQ(EstimateTwoViewGeometry(camera, first, camera, second, matches, options).type, TwoViewGeometry::Type::kDegenerate);
  options.stationary_matches_max_error = 4;
  EXPECT_EQ(EstimateTwoViewGeometry(camera, first, camera, second, matches, options).inlier_matches.size(), matches.size());
}

TEST(TwoViewGeometryVerifierTest, CombinesMultipleModelsAndHonorsWatermarkPolicy) {
  const auto camera = MakeCamera();
  FeatureSet first, second;
  std::vector<FeatureMatch> matches;
  for (int model = 0; model < 2; ++model) for (int y = 0; y < 4; ++y) for (int x = 0; x < 5; ++x) {
    const float px = 15 + 9*x, py = 15 + 9*y;
    first.keypoints.push_back({px, py, 1, 0});
    second.keypoints.push_back({px + (model ? 9 : 27), py + (model ? 31 : 5), 1, 0});
    const auto idx = static_cast<Point2DIdx>(matches.size()); matches.push_back({idx, idx});
  }
  TwoViewGeometryOptions options;
  options.random_seed = 0; options.force_H_use = true; options.max_error_px = 0.1;
  options.multiple_models = true; options.detect_watermark = false;
  auto geometry = EstimateTwoViewGeometry(camera, first, camera, second, matches, options);
  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kMultiple);
  EXPECT_EQ(geometry.inlier_matches.size(), 40U);
  EXPECT_FALSE(geometry.E); EXPECT_FALSE(geometry.F); EXPECT_FALSE(geometry.H); EXPECT_FALSE(geometry.cam2_from_cam1);
  options.detect_watermark = true;
  EXPECT_EQ(EstimateTwoViewGeometry(camera, first, camera, second, matches, options).type, TwoViewGeometry::Type::kDegenerate);
  options.multiple_ignore_watermark = false;
  EXPECT_EQ(EstimateTwoViewGeometry(camera, first, camera, second, matches, options).inlier_matches.size(), 40U);
}

TEST(TwoViewGeometryVerifierTest, HomographyAxisTranslationHandlesExactlyZeroMinors) {
  const auto camera = MakeCamera();
  FeatureSet first, second;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kPlanarOrPanoramic;
  Eigen::Matrix3d h = Eigen::Matrix3d::Identity(); h(0, 2) = -140;
  geometry.H = h;
  for (int y = 0; y < 4; ++y) for (int x = 0; x < 5; ++x) {
    first.keypoints.push_back({float(300 + 50*x), float(250 + 50*y), 1, 0});
    second.keypoints.push_back({float(160 + 50*x), float(250 + 50*y), 1, 0});
    const auto idx = static_cast<Point2DIdx>(geometry.inlier_matches.size());
    geometry.inlier_matches.push_back({idx, idx});
  }
  ASSERT_TRUE(EstimateTwoViewGeometryPose(camera, first, camera, second, &geometry));
  EXPECT_EQ(geometry.type, TwoViewGeometry::Type::kPlanarHomography);
  for (const auto& match : geometry.inlier_matches) {
    const Eigen::Vector2d a(first.keypoints[match.idx1].x, first.keypoints[match.idx1].y);
    const Eigen::Vector2d b(second.keypoints[match.idx2].x, second.keypoints[match.idx2].y);
    const auto xyz = TriangulatePoint(camera, Pose3{}, a, camera, *geometry.cam2_from_cam1, b);
    ASSERT_TRUE(xyz);
    EXPECT_GT(xyz->z(), 0);
    EXPECT_LT(*ReprojectionError(camera, *geometry.cam2_from_cam1, *xyz, b), 1e-6);
  }
}

}  // namespace
}  // namespace colite
