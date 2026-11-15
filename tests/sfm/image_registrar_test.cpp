#include "colite/sfm/image_registrar.h"

#include <string>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/sfm/incremental_initializer.h"
#include "poselib/p3p.h"

namespace colite {
namespace {

Camera MakeCamera(CameraId id) {
  Camera camera;
  camera.id = id;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  camera.has_prior_focal_length = true;
  return camera;
}

Image MakeImage(ImageId image_id, CameraId camera_id) {
  Image image;
  image.id = image_id;
  image.camera_id = camera_id;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  return image;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

std::vector<Eigen::Vector3d> MakeScenePoints() {
  std::vector<Eigen::Vector3d> points;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      points.emplace_back(-1.0 + 0.4 * x, -0.6 + 0.3 * y,
                          4.0 + 0.07 * x + 0.11 * y + 0.03 * x * y);
    }
  }
  return points;
}

FeatureSet ProjectFeatures(const Camera& camera,
                           const Pose3& cam_from_world,
                           const std::vector<Eigen::Vector3d>& points) {
  FeatureSet features;
  for (const Eigen::Vector3d& point : points) {
    const auto pixel = ProjectWorldPoint(camera, cam_from_world, point);
    EXPECT_TRUE(pixel.has_value());
    features.keypoints.push_back(
        {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()), 1.0F,
         0.0F});
  }
  return features;
}

std::vector<FeatureMatch> IdentityMatches(std::size_t count) {
  std::vector<FeatureMatch> matches;
  for (std::size_t i = 0; i < count; ++i) {
    const Point2DIdx idx = static_cast<Point2DIdx>(i);
    matches.push_back({idx, idx});
  }
  return matches;
}

TwoViewGeometry EstimateInitialGeometry(const Camera& camera,
                                        const FeatureSet& features1,
                                        const FeatureSet& features2,
                                        const std::vector<FeatureMatch>& matches) {
  TwoViewGeometryOptions options;
  options.compute_relative_pose = true;
  options.random_seed = 0;
  options.max_error_px = 1.0;
  options.min_num_inliers = 20;
  return EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                                 options);
}

TwoViewGeometry MakeVerifiedGeometryWithInliers(
    const std::vector<FeatureMatch>& matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = matches;
  geometry.median_triangulation_angle = 1.0;
  return geometry;
}

TEST(ImageRegistrarTest, RegistersImageFromExistingPointsUsingPnPRansac) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const std::vector<FeatureMatch> matches12 = IdentityMatches(points.size());
  const std::vector<FeatureMatch> matches13 = IdentityMatches(points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, matches12);
  SetMatches(&graph, 1, 3, matches13);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateInitialGeometry(camera, features1, features2,
                                              matches12));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometryWithInliers(matches13));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));

  IncrementalInitializerOptions init_options;
  init_options.min_num_inliers = 20;
  init_options.min_num_triangulated_points = 20;
  init_options.min_triangulation_angle_deg = 0.1;
  init_options.max_reprojection_error_px = 1.0;
  const IncrementalInitializerSummary init_summary =
      InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                       init_options, &reconstruction);
  ASSERT_TRUE(init_summary.success);

  ImageRegistrarOptions register_options;
  register_options.min_num_correspondences = 20;
  register_options.min_num_inliers = 20;
  register_options.max_reprojection_error_px = 2.0;

  // Enough absolute inliers can still be a poor explanation of a large set of
  // correspondences. Enforce the COLMAP ratio before publishing the pose.
  MatchGraph outlier_graph = graph;
  FeatureSet noisy_features = features3;
  noisy_features.keypoints.back().x += 10000.0F;
  noisy_features.keypoints.back().y += 10000.0F;
  SetFeatures(&outlier_graph, 3, noisy_features);
  Reconstruction ratio_reconstruction = reconstruction;
  ImageRegistrarOptions ratio_options = register_options;
  ratio_options.min_num_inliers = 12;
  ratio_options.min_inlier_ratio = 1.0;
  const auto rejected = RegisterImageWithExistingPoints(
      outlier_graph, 3, ratio_options, &ratio_reconstruction);
  EXPECT_GE(rejected.num_inliers, 12U);
  EXPECT_LT(rejected.num_inliers, rejected.num_correspondences);
  EXPECT_FALSE(rejected.success);
  EXPECT_FALSE(ratio_reconstruction.IsImageRegistered(3));
  EXPECT_FALSE(ratio_reconstruction.GetImage(3).HasPose());

  const ImageRegistrationSummary summary =
      RegisterImageWithExistingPoints(graph, 3, register_options,
                                      &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.image_id, 3U);
  EXPECT_GE(summary.num_correspondences, 20U);
  EXPECT_GE(summary.num_ransac_inliers, 20U);
  EXPECT_GE(summary.num_inliers, 20U);
  EXPECT_GE(summary.num_inliers, summary.num_ransac_inliers);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_TRUE(reconstruction.GetImage(3).HasPose());
  EXPECT_GE(reconstruction.GetImage(3).NumPoints3D(), 20U);
}

TEST(ImageRegistrarTest, FailsWhenThereAreTooFewCorrespondences) {
  MatchGraph graph;
  Reconstruction reconstruction;

  ImageRegistrationSummary summary =
      RegisterImageWithExistingPoints(graph, 7, {}, &reconstruction);

  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.image_id, 7U);
  EXPECT_EQ(summary.num_correspondences, 0U);
}

TEST(P3PSolverTest, BuiltinReturnsGroundTruthAmongFiniteThreePointRoots) {
  const std::vector<Eigen::Vector3d> points = {{-.7, -.4, 3}, {.8, -.3, 4}, {-.1, .6, 5}};
  Pose3 truth;
  truth.q_cw = Eigen::AngleAxisd(.23, Eigen::Vector3d(1, 2, 3).normalized());
  truth.t_cw = {-.2, .1, .3};
  std::vector<Eigen::Vector3d> rays;
  for (const auto& point : points) rays.push_back(truth.TransformPoint(point).normalized());
  const auto original_rays = rays;
  const auto original_points = points;
  std::vector<third_party::poselib::CameraPose> poses;
  ASSERT_GT(third_party::poselib::p3p(rays, points, &poses), 0);
  EXPECT_LE(poses.size(), 4U);
  bool recovered = false;
  for (const auto& candidate : poses) {
    const auto matrix = candidate.Rt();
    ASSERT_TRUE(matrix.allFinite());
    const Eigen::Quaterniond q(matrix.leftCols<3>());
    if (q.angularDistance(truth.q_cw) < 1e-7 && (candidate.t - truth.t_cw).norm() < 1e-7)
      recovered = true;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const Eigen::Vector3d predicted = matrix.leftCols<3>() * points[i] + candidate.t;
      EXPECT_GT(predicted.dot(rays[i]), 0);
      EXPECT_LT(predicted.normalized().cross(rays[i]).norm(), 1e-7);
    }
  }
  EXPECT_TRUE(recovered);
  EXPECT_EQ(rays, original_rays);
  EXPECT_EQ(points, original_points);
}

TEST(P3PSolverTest, BuiltinRejectsInvalidAndCollinearSamplesWithoutStaleOutput) {
  std::vector<Eigen::Vector3d> points = {{0, 0, 3}, {1, 0, 4}, {2, 0, 5}};
  std::vector<Eigen::Vector3d> rays;
  for (const auto& point : points) rays.push_back(point.normalized());
  std::vector<third_party::poselib::CameraPose> poses(1);
  EXPECT_EQ(third_party::poselib::p3p(rays, points, &poses), 0);
  EXPECT_TRUE(poses.empty());
  points[2] = {-.1, .6, 5};
  points[0].x() = std::numeric_limits<double>::quiet_NaN();
  poses.resize(1);
  EXPECT_EQ(third_party::poselib::p3p(rays, points, &poses), 0);
  EXPECT_TRUE(poses.empty());
  EXPECT_EQ(third_party::poselib::p3p({}, {}, &poses), 0);
  EXPECT_EQ(third_party::poselib::p3p(rays, points, nullptr), 0);
}

TEST(ImageRegistrarTest, RegistersWithOnlyFourDistortedCorrespondences) {
  auto camera = MakeCamera(1);
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700, 500, 400, .08};
  const std::vector<Eigen::Vector3d> points = {
      {-.7, -.4, 3}, {.8, -.3, 4}, {-.1, .6, 5}, {.4, .5, 3.5}};
  Pose3 truth;
  truth.q_cw = Eigen::AngleAxisd(.07, Eigen::Vector3d::UnitY());
  truth.t_cw = {-.2, .1, .03};
  Reconstruction model;
  model.AddCamera(camera);
  MatchGraph graph;
  for (ImageId id = 1; id <= 2; ++id) {
    auto image = MakeImage(id, 1);
    const Pose3 pose = id == 1 ? Pose3{} : truth;
    auto features = ProjectFeatures(camera, pose, points);
    SetFeatures(&graph, id, features);
    for (Point2DIdx idx = 0; idx < points.size(); ++idx)
      image.points2d.push_back({Eigen::Vector2d(features.keypoints[idx].x, features.keypoints[idx].y),
                                idx, kInvalidPoint3DId});
    if (id == 1) image.cam_from_world = pose;
    model.AddImage(image);
    if (id == 1) model.RegisterImage(id);
  }
  for (Point2DIdx idx = 0; idx < points.size(); ++idx) {
    Track track;
    track.AddElement(1, idx);
    model.AddPoint3D(points[idx], track);
  }
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometryWithInliers(IdentityMatches(points.size())));
  ImageRegistrarOptions options;
  options.min_num_correspondences = 4;
  options.min_num_inliers = 4;
  options.max_reprojection_error_px = .5;
  options.random_seed = 0;
  const auto result = RegisterImageWithExistingPoints(graph, 2, options, &model);
  ASSERT_TRUE(result.success) << result.message;
  EXPECT_EQ(result.num_inliers, 4U);
  const auto& pose = *model.GetImage(2).cam_from_world;
  EXPECT_LT(pose.q_cw.angularDistance(truth.q_cw), 1e-5);
  EXPECT_LT((pose.CameraCenter() - truth.CameraCenter()).norm(), 1e-4);
}

TEST(AbsolutePoseEpnpTest, RecoversDistortedPoseAcrossWorldScaleAndTranslation) {
  auto camera = MakeCamera(1);
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700, 500, 400, .08};
  const std::vector<Eigen::Vector3d> base = {
      {-.7,-.4,3}, {.8,-.3,4}, {-.1,.6,5}, {.4,.5,3.5},
      {-.5,.3,6}, {.9,.1,5.5}, {-.3,-.6,4.5}, {.1,-.2,3.2}};
  for (const double scale : {1e-3, 1.0, 1e3}) {
    const Eigen::Vector3d offset(1e6, -2e6, 3e6);
    auto truth = CameraAt(offset + scale*Eigen::Vector3d(.2, .1, -.05));
    truth.q_cw = Eigen::AngleAxisd(.07, Eigen::Vector3d::UnitY());
    truth.t_cw = -(truth.q_cw * (offset + scale*Eigen::Vector3d(.2, .1, -.05)));
    std::vector<Eigen::Vector3d> points;
    std::vector<Eigen::Vector2d> pixels;
    for (const auto& point : base) {
      points.push_back(offset + scale*point);
      pixels.push_back(*ProjectWorldPoint(camera, truth, points.back()));
    }
    const auto original_points = points;
    const auto original_pixels = pixels;
    const auto result = EstimateAbsolutePoseEpnp(camera, pixels, points);
    ASSERT_TRUE(result) << scale;
    EXPECT_LT(result->q_cw.angularDistance(truth.q_cw), 1e-5);
    EXPECT_LT((result->CameraCenter()-truth.CameraCenter()).norm()/scale, 1e-4);
    for (std::size_t i = 0; i < points.size(); ++i)
      EXPECT_LT(*ReprojectionError(camera, *result, points[i], pixels[i]), .01);
    EXPECT_EQ(points, original_points);
    EXPECT_EQ(pixels, original_pixels);
  }
}

TEST(AbsolutePoseEpnpTest, RejectsDegenerateDataAndInvalidCameraRays) {
  auto camera = MakeCamera(1);
  std::vector<Eigen::Vector3d> points(8, Eigen::Vector3d(0,0,4));
  std::vector<Eigen::Vector2d> pixels(8, Eigen::Vector2d(500,400));
  EXPECT_FALSE(EstimateAbsolutePoseEpnp(camera, pixels, points));
  EXPECT_FALSE(EstimateAbsolutePoseEpnp(camera, {}, points));
  points[0].x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(EstimateAbsolutePoseEpnp(camera, pixels, points));
  points[0] = {0,0,4};
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700, 500, 400, -1};
  pixels[0] = {0,0};
  EXPECT_FALSE(EstimateAbsolutePoseEpnp(camera, pixels, points));
}

TEST(ImageRegistrarTest, SkipsTargetPixelsOutsideCameraDomain) {
  auto camera = MakeCamera(1);
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700, 500, 400, -1};
  std::vector<Eigen::Vector3d> points;
  for (int y = 0; y < 5; ++y) for (int x = 0; x < 6; ++x)
    points.emplace_back(-.2+.08*x, -.2+.08*y, 4+.02*x*x+.03*y);
  Reconstruction model; model.AddCamera(camera);
  MatchGraph graph;
  for (ImageId id = 1; id <= 3; ++id) {
    auto image = MakeImage(id, 1);
    const auto pose = CameraAt({.2*(id-1), .025*(id-1), 0});
    auto features = ProjectFeatures(camera, pose, points);
    if (id == 3) { features.keypoints[0].x = 0; features.keypoints[0].y = 0; }
    SetFeatures(&graph, id, features);
    for (Point2DIdx idx = 0; idx < points.size(); ++idx)
      image.points2d.push_back({Eigen::Vector2d(features.keypoints[idx].x, features.keypoints[idx].y), idx, kInvalidPoint3DId});
    if (id <= 2) { image.registered = true; image.cam_from_world = pose; }
    model.AddImage(image);
  }
  for (Point2DIdx idx = 0; idx < points.size(); ++idx) {
    Track track; track.AddElement(1, idx); track.AddElement(2, idx);
    model.AddPoint3D(points[idx], track);
  }
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = IdentityMatches(points.size());
  SetVerifiedGeometry(&graph, 1, 3, geometry);
  SetVerifiedGeometry(&graph, 2, 3, geometry);
  ImageRegistrarOptions options;
  options.min_num_correspondences = 20; options.min_num_inliers = 20;
  options.max_reprojection_error_px = .5;
  auto failed_model = model;
  auto invalid = options;
  invalid.refinement.max_num_iterations = -1;
  const auto failed = RegisterImageWithExistingPoints(graph, 3, invalid, &failed_model);
  EXPECT_FALSE(failed.success);
  EXPECT_FALSE(failed_model.IsImageRegistered(3));
  EXPECT_FALSE(failed_model.GetImage(3).cam_from_world);
  EXPECT_EQ(failed_model.GetImage(3).NumPoints3D(), 0U);
  for (const auto& [id, point] : failed_model.Points3D()) EXPECT_EQ(point.track.Length(), 2U);
  const auto summary = RegisterImageWithExistingPoints(graph, 3, options, &model);
  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_TRUE(summary.refinement.success);
  EXPECT_GE(summary.num_ransac_iterations, options.min_num_iterations);
  EXPECT_EQ(summary.refinement.num_observations, points.size()-1);
  EXPECT_EQ(summary.num_correspondences, points.size()-1);
  EXPECT_FALSE(model.GetImage(3).points2d[0].HasPoint3D());
  EXPECT_EQ(model.GetImage(3).NumPoints3D(), points.size()-1);
}

TEST(AbsolutePoseRefinementTest, RefinesPoseWhileKeepingCameraAndPointsFixed) {
  const auto camera = MakeCamera(1);
  const auto points = MakeScenePoints();
  const auto original_points = points;
  auto truth = CameraAt({.3, .1, .05});
  truth.q_cw = Eigen::AngleAxisd(.03, Eigen::Vector3d::UnitY());
  std::vector<Eigen::Vector2d> pixels;
  for (const auto& point : points) pixels.push_back(*ProjectWorldPoint(camera, truth, point));
  std::vector<char> inliers(points.size(), true);
  pixels.back() += Eigen::Vector2d(500, -300);
  inliers.back() = false;
  auto pose = truth;
  pose.q_cw = Eigen::Quaterniond(Eigen::AngleAxisd(.04, Eigen::Vector3d::UnitY())) * truth.q_cw;
  pose.t_cw += Eigen::Vector3d(.05, -.04, .03);
  const auto summary = RefineAbsolutePose({}, camera, pixels, points, inliers, &pose);
  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_EQ(summary.num_observations, points.size()-1);
  EXPECT_LT(summary.final_cost, summary.initial_cost*.001);
  EXPECT_LT(pose.q_cw.angularDistance(truth.q_cw), 1e-3);
  EXPECT_LT((pose.CameraCenter()-truth.CameraCenter()).norm(), 1e-2);
  EXPECT_EQ(points, original_points);
  EXPECT_EQ(camera.params, MakeCamera(1).params);
}

TEST(AbsolutePoseRefinementTest, FailedObjectiveLeavesInputPoseUnchanged) {
  const auto camera = MakeCamera(1);
  const std::vector<Eigen::Vector2d> pixels(4, Eigen::Vector2d(500,400));
  const std::vector<Eigen::Vector3d> behind(4, Eigen::Vector3d(0,0,-4));
  Pose3 pose = CameraAt({0,0,0});
  const auto original = pose;
  const auto summary = RefineAbsolutePose({}, camera, pixels, behind, std::vector<char>(4,true), &pose);
  EXPECT_FALSE(summary.success);
  EXPECT_EQ(pose.q_cw.coeffs(), original.q_cw.coeffs());
  EXPECT_EQ(pose.t_cw, original.t_cw);
  EXPECT_FALSE(RefineAbsolutePose({}, camera, pixels, behind, std::vector<char>(4,false), &pose).success);
  EXPECT_FALSE(RefineAbsolutePose({}, camera, pixels, behind, {}, &pose).success);
}

}  // namespace
}  // namespace colite
