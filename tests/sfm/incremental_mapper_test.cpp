#include "colite/sfm/incremental_mapper.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/sfm/incremental_initializer.h"

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
    for (int x = 0; x < 8; ++x) {
      points.emplace_back(-1.4 + 0.4 * x, -0.6 + 0.3 * y,
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

FeatureSet PerturbFeatures(FeatureSet features, double amplitude_px) {
  for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
    const double sign = (i % 2 == 0) ? 1.0 : -1.0;
    features.keypoints[i].x += static_cast<float>(sign * amplitude_px);
    features.keypoints[i].y += static_cast<float>(-sign * amplitude_px * 0.5);
  }
  return features;
}

std::vector<FeatureMatch> IdentityMatches(std::size_t begin, std::size_t end) {
  std::vector<FeatureMatch> matches;
  for (std::size_t i = begin; i < end; ++i) {
    const Point2DIdx idx = static_cast<Point2DIdx>(i);
    matches.push_back({idx, idx});
  }
  return matches;
}

TwoViewGeometry EstimateGeometry(const Camera& camera,
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

TwoViewGeometry MakeVerifiedGeometry(const std::vector<FeatureMatch>& matches) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = matches;
  geometry.median_triangulation_angle = 2.0;
  return geometry;
}

struct InitialRetryFixture {
  MatchGraph graph;
  Reconstruction model;
  IncrementalMappingOptions options;
};

InitialRetryFixture MakeInitialRetryFixture() {
  InitialRetryFixture fixture;
  const auto camera = MakeCamera(1);
  fixture.model.AddCamera(camera);
  const auto points = MakeScenePoints();
  for (ImageId id : {1,2,3}) {
    const auto pose = CameraAt({id==1?0.0:(id==2?1.0:.7),0,0});
    auto features = ProjectFeatures(camera,pose,points);
    if (id==2) for (std::size_t idx=0;idx<features.keypoints.size();++idx)
      features.keypoints[idx].y += idx%2==0?3.0F:-3.0F;
    SetFeatures(&fixture.graph,id,features);
    fixture.model.AddImage(MakeImage(id,1));
  }
  for (ImageId neighbor : {2,3}) {
    auto geometry = MakeVerifiedGeometry(IdentityMatches(0,neighbor==2?points.size():35));
    geometry.cam2_from_cam1 = CameraAt({neighbor==2?1.0:.7,0,0});
    SetVerifiedGeometry(&fixture.graph,1,neighbor,geometry);
  }
  auto& options=fixture.options;
  options.initialization.min_num_inliers=20;
  options.initialization.min_num_triangulated_points=1;
  options.initialization.min_triangulation_angle_deg=.1;
  options.initialization.max_reprojection_error_px=10;
  options.mapper.registration.min_num_inliers=20;
  options.global_observation_filter.max_reprojection_error_px=.001;
  options.global_observation_filter.min_triangulation_angle_deg=.1;
  options.max_num_registrations=0;
  options.max_num_global_refinements=0;
  options.run_global_track_completion=false;
  options.run_pose_guided_track_completion=false;
  options.run_final_global_bundle_adjustment=false;
  return fixture;
}

TEST(IncrementalMapperTest, RejectsFilteredInitialModelTransactionallyAndTriesNextPair) {
  auto fixture=MakeInitialRetryFixture();
  fixture.options.max_num_initialization_trials=1;
  const auto failed=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model);
  ASSERT_FALSE(failed.success);
  ASSERT_EQ(failed.initialization_refinements.size(),1U);
  const auto& rejected=failed.initialization_refinements[0];
  EXPECT_TRUE(rejected.bundle_adjustment_attempted);
  EXPECT_TRUE(rejected.bundle_adjustment.success);
  EXPECT_EQ(rejected.bundle_adjustment.solver_max_num_iterations,100);
  EXPECT_DOUBLE_EQ(rejected.bundle_adjustment.solver_gradient_tolerance,1e-5);
  EXPECT_LT(rejected.num_points_after,20U);
  EXPECT_TRUE(fixture.model.Points3D().empty());
  EXPECT_TRUE(fixture.model.RegisteredImageIds().empty());
  for (ImageId id : {1,2,3}) {
    EXPECT_FALSE(fixture.model.GetImage(id).HasPose());
    EXPECT_TRUE(fixture.model.GetImage(id).points2d.empty());
  }
  EXPECT_EQ(fixture.model.GetCamera(1).params,MakeCamera(1).params);
  fixture.options.max_num_initialization_trials=2;
  const auto recovered=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model);
  ASSERT_TRUE(recovered.success) << recovered.message;
  ASSERT_EQ(recovered.initialization_refinements.size(),2U);
  EXPECT_EQ(recovered.initialization.initial_pair,ImagePair::Ordered(1,3));
  EXPECT_TRUE(recovered.initialization_refinements.back().success);
  EXPECT_TRUE(recovered.initialization_refinements.back().normalized);
  EXPECT_FALSE(fixture.model.IsImageRegistered(2));
  EXPECT_TRUE(fixture.model.GetImage(2).points2d.empty());
}

TEST(IncrementalMapperTest, GeometryChecksDoNotConsumeInitialReconstructionTrials) {
  auto fixture=MakeInitialRetryFixture();
  fixture.graph.features.erase(2);
  fixture.options.max_num_initialization_trials=1;
  const auto result=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model);
  ASSERT_TRUE(result.success) << result.message;
  EXPECT_EQ(result.initialization.num_initialization_attempts,2U);
  ASSERT_EQ(result.initialization_refinements.size(),1U);
  EXPECT_EQ(result.initialization.initial_pair,ImagePair::Ordered(1,3));
}

TEST(IncrementalMapperTest, CreationAngleControlsInitialTriangulation) {
  auto fixture=MakeInitialRetryFixture();
  fixture.options.mapper.triangulation.create_max_angle_error_deg=.001;
  const auto result=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model);
  ASSERT_TRUE(result.success) << result.message;
  ASSERT_EQ(result.initialization_refinements.size(),2U);
  EXPECT_EQ(result.initialization_refinements[0].num_points_before,0U);
  EXPECT_FALSE(result.initialization_refinements[0].bundle_adjustment_attempted);
  EXPECT_EQ(result.initialization.initial_pair,ImagePair::Ordered(1,3));
}

TEST(IncrementalMapperTest, InitialTrialCountsBothImagesAndAllowsExhaustedSecondSeed) {
  auto fixture=MakeInitialRetryFixture();
  fixture.options.initialization.max_num_reg_trials=1;
  InitialPairTrialState state;
  const auto result=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model,&state);
  ASSERT_TRUE(result.success) << result.message;
  ASSERT_EQ(result.initialization_refinements.size(),2U);
  EXPECT_EQ(result.initialization.initial_pair,(ImagePair{3,1}));
  EXPECT_EQ(state.num_reg_trials.at(1),2U);
  EXPECT_EQ(state.num_reg_trials.at(2),1U);
  EXPECT_EQ(state.num_reg_trials.at(3),1U);
}

TEST(IncrementalMapperTest, SpecifiedSeedControlsOrientationAndNeverFallsBackToAnotherPair) {
  for (bool second_field : {false,true}) {
    auto fixture=MakeInitialRetryFixture();
    if (second_field) fixture.options.initialization.image_id2=3;
    else fixture.options.initialization.image_id1=3;
    const auto result=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model);
    ASSERT_TRUE(result.success) << result.message;
    EXPECT_EQ(result.initialization.initial_pair,(ImagePair{3,1}));
  }
  auto fixture=MakeInitialRetryFixture();
  fixture.options.initialization.image_id1=1;fixture.options.initialization.image_id2=2;
  const auto failed=RunIncrementalMapping(fixture.graph,fixture.options,&fixture.model);
  EXPECT_FALSE(failed.success);
  ASSERT_EQ(failed.initialization_refinements.size(),1U);
  EXPECT_EQ(failed.initialization.initial_pair,(ImagePair{1,2}));
  EXPECT_TRUE(fixture.model.Points3D().empty());
  EXPECT_TRUE(fixture.model.RegisteredImageIds().empty());
}

TEST(IncrementalMapperTest,
     RegistersImageTriangulatesAndRunsLocalBundleAdjustment) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(matches_all));

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
  ASSERT_TRUE(InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                               init_options, &reconstruction)
                  .success);

  // 模拟增量建图中常见的累积误差：已有位姿和点云并非完全干净。
  reconstruction.GetImage(2).cam_from_world->t_cw +=
      Eigen::Vector3d(0.04, -0.02, 0.02);
  std::vector<Point3DId> point3d_ids;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d;
    point3d_ids.push_back(point3d_id);
  }
  for (const Point3DId point3d_id : point3d_ids) {
    reconstruction.GetPoint3D(point3d_id).xyz +=
        Eigen::Vector3d(0.02, -0.015, 0.03);
  }

  const std::size_t num_points_before = reconstruction.Points3D().size();

  IncrementalMapperOptions mapper_options;
  mapper_options.registration.min_num_correspondences = 20;
  mapper_options.registration.min_num_inliers = 20;
  mapper_options.registration.max_reprojection_error_px = 3.0;
  mapper_options.triangulation.min_triangulation_angle_deg = 0.1;
  mapper_options.triangulation.create_max_angle_error_deg = 2.0;
  mapper_options.local_bundle_adjustment.max_num_iterations = 50;
  mapper_options.local_bundle_adjustment.loss_scale = 1.0;
  mapper_options.local_bundle_adjustment.constant_image_ids.insert(1);

  // Each control starts from the same perturbed model. Zero change threshold
  // forces all passes even when membership stabilizes; a high threshold exits.
  for (const auto& control : std::vector<std::pair<std::size_t, double>>{
           {1, 0.0}, {2, 0.0}, {3, 1000.0}}) {
    auto candidate = reconstruction;
    auto options = mapper_options;
    options.ba_local_max_refinements = control.first;
    options.ba_local_max_refinement_change = control.second;
    const auto refined = RegisterNextImage(graph, 3, options, &candidate);
    ASSERT_TRUE(refined.success) << refined.message;
    EXPECT_EQ(refined.num_local_refinements, control.second > 0 ? 1U : control.first);
    EXPECT_TRUE(std::isfinite(refined.last_local_refinement_change));
  }
  for (int invalid = 0; invalid < 3; ++invalid) {
    auto candidate = reconstruction;
    auto options = mapper_options;
    if (invalid == 0) options.ba_local_max_refinements = 0;
    if (invalid == 1) options.ba_local_max_refinement_change = -1.0;
    if (invalid == 2) options.ba_local_max_refinement_change = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(RegisterNextImage(graph, 3, options, &candidate).success);
    EXPECT_FALSE(candidate.IsImageRegistered(3));
    EXPECT_EQ(candidate.Points3D().size(), reconstruction.Points3D().size());
    IncrementalMappingOptions mapping;
    mapping.mapper = options;
    EXPECT_FALSE(RunIncrementalMapping(graph, mapping, &candidate).success);
    EXPECT_FALSE(candidate.IsImageRegistered(3));
  }
  const IncrementalMapperSummary summary =
      RegisterNextImage(graph, 3, mapper_options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_TRUE(summary.registration.success);
  EXPECT_GT(summary.triangulation.num_created_points, 0U);
  EXPECT_TRUE(summary.local_bundle_adjustment.success);
  EXPECT_GT(summary.local_bundle_adjustment.num_residuals, 0U);
  EXPECT_TRUE(summary.observation_filter.success);
  EXPECT_LT(summary.local_bundle_adjustment.final_mean_reprojection_error,
            summary.local_bundle_adjustment.initial_mean_reprojection_error);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_GT(reconstruction.Points3D().size(), num_points_before);
}

TEST(IncrementalMapperTest, FiltersSavedSeedEvenWithBundleAdjustmentDisabled) {
  Reconstruction reconstruction;
  const Camera camera = MakeCamera(1);
  reconstruction.AddCamera(camera);
  const auto points = MakeScenePoints();
  for (ImageId id = 1; id <= 2; ++id) {
    Image image = MakeImage(id, camera.id);
    image.cam_from_world = CameraAt(Eigen::Vector3d(id - 1, 0, 0));
    image.registered = true;
    for (std::size_t idx = 0; idx < points.size(); ++idx) {
      Point2D observation;
      observation.xy = *ProjectWorldPoint(camera, *image.cam_from_world, points[idx]);
      if (id == 2 && idx == 0) observation.xy.x() += 10.0;
      image.points2d.push_back(observation);
    }
    reconstruction.AddImage(std::move(image));
  }
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    Point3D point;
    point.id = idx + 1;
    point.xyz = points[idx];
    point.track.elements = {{1, static_cast<Point2DIdx>(idx)},
                            {2, static_cast<Point2DIdx>(idx)}};
    reconstruction.AddPoint3D(std::move(point));
  }
  IncrementalMappingOptions options;
  options.run_global_bundle_adjustment = false;
  options.run_global_track_completion = false;
  options.run_pose_guided_track_completion = false;
  const auto summary = RunIncrementalMapping({}, options, &reconstruction);
  ASSERT_TRUE(summary.success);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(1));
  EXPECT_EQ(reconstruction.Points3D().size(), points.size() - 1);
  EXPECT_GT(summary.global_observation_filter.num_removed_observations, 0U);
}

TEST(IncrementalMapperTest, GlobalRefinementHonorsChangeThresholdAndPassLimit) {
  Reconstruction seed;
  const auto camera = MakeCamera(1);
  seed.AddCamera(camera);
  const auto points = MakeScenePoints();
  for (ImageId id = 1; id <= 2; ++id) {
    auto image = MakeImage(id, camera.id);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(id - 1, 0, 0));
    for (const auto& xyz : points) {
      Point2D point;
      point.xy = *ProjectWorldPoint(camera, *image.cam_from_world, xyz);
      image.points2d.push_back(point);
    }
    seed.AddImage(image);
  }
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    Track track;
    for (ImageId id = 1; id <= 2; ++id) track.AddElement(id, idx);
    seed.AddPoint3D(points[idx] + Eigen::Vector3d(.01, -.02, .03), track);
  }
  IncrementalMappingOptions options;
  options.run_global_track_completion = false;
  options.run_pose_guided_track_completion = false;
  options.run_global_observation_filter = false;
  auto model = seed;
  auto summary = RunIncrementalMapping({}, options, &model);
  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_EQ(summary.num_global_refinements, 1U);
  EXPECT_DOUBLE_EQ(summary.last_global_refinement_change, 0);
  options.max_num_global_refinements = 2;
  options.global_refinement_change = 0;
  model = seed;
  summary = RunIncrementalMapping({}, options, &model);
  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_EQ(summary.num_global_refinements, 2U);
  EXPECT_TRUE(summary.final_global_bundle_adjustment.success);
  options.max_num_global_refinements = 0;
  model = seed;
  summary = RunIncrementalMapping({}, options, &model);
  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_global_refinements, 0U);
  EXPECT_EQ(model.Points3D().size(), seed.Points3D().size());
}

TEST(IncrementalMapperTest, RefinesStalledModelBeforeRetryingRegistration) {
  const Camera camera = MakeCamera(1);
  const auto points = MakeScenePoints();
  const std::vector<Pose3> poses = {
      CameraAt(Eigen::Vector3d::Zero()), CameraAt(Eigen::Vector3d(1, 0, 0)),
      CameraAt(Eigen::Vector3d(.4, .1, 0))};
  Reconstruction seed;
  seed.AddCamera(camera);
  MatchGraph graph;
  for (ImageId id = 1; id <= 3; ++id) {
    const auto features = ProjectFeatures(camera, poses[id - 1], points);
    SetFeatures(&graph, id, features);
    auto image = MakeImage(id, camera.id);
    if (id <= 2) {
      image.registered = true;
      image.cam_from_world = poses[id - 1];
      for (const auto& feature : features.keypoints) {
        Point2D point;
        point.xy = Eigen::Vector2d(feature.x, feature.y);
        image.points2d.push_back(point);
      }
    }
    seed.AddImage(std::move(image));
  }
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    Track track;
    track.AddElement(1, idx);
    track.AddElement(2, idx);
    const Eigen::Vector3d offset(.2 * (static_cast<int>(idx % 3) - 1),
                                 .18 * (static_cast<int>(idx % 5) - 2),
                                 .1 * static_cast<int>(idx % 7));
    seed.AddPoint3D(points[idx] + offset, track);
  }
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(IdentityMatches(0, points.size())));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(IdentityMatches(0, points.size())));
  IncrementalMappingOptions options;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 1;
  options.registration_retry_reprojection_error_multiplier = 1;
  options.registration_retry_max_reprojection_error_px = 1;
  options.mapper.run_local_bundle_adjustment = false;
  options.mapper.run_local_retriangulation_filter = false;
  options.mapper.run_observation_filter = false;
  options.run_global_track_completion = false;
  options.run_pose_guided_track_completion = false;
  options.run_global_observation_filter = false;
  options.global_bundle_adjustment.refine_camera_params = CameraParameterGroup::kNone;
  auto disabled = seed;
  options.run_global_bundle_adjustment = false;
  const auto no_recovery = RunIncrementalMapping(graph, options, &disabled);
  ASSERT_TRUE(no_recovery.success);
  EXPECT_FALSE(disabled.IsImageRegistered(3));
  EXPECT_EQ(no_recovery.num_registration_recovery_attempts, 0U);

  auto recovered = seed;
  options.run_global_bundle_adjustment = true;
  const auto summary = RunIncrementalMapping(graph, options, &recovered);
  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_TRUE(recovered.IsImageRegistered(3));
  EXPECT_EQ(summary.num_registration_recovery_attempts, 1U);
  EXPECT_EQ(summary.num_registration_recovery_successes, 1U);
  ASSERT_EQ(summary.registrations.size(), 1U);
  EXPECT_EQ(summary.registrations.front().registration_attempts_before_success, 1U);
  EXPECT_DOUBLE_EQ(summary.registrations.front().registration_max_reprojection_error_px, 1);
  ASSERT_FALSE(summary.failed_registrations.empty());
  EXPECT_LT(summary.failed_registrations.front().registration_num_inliers, 20U);
  for (std::size_t idx = 0; idx < summary.global_refinement_diagnostics.size(); ++idx) {
    EXPECT_EQ(summary.global_refinement_diagnostics[idx].iteration, idx + 1);
  }
}

TEST(IncrementalMapperTest, RunsCompleteIncrementalMappingLoop) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));
  const Pose3 cam4_from_world = CameraAt(Eigen::Vector3d(0.75, -0.2, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const FeatureSet features4 = ProjectFeatures(camera, cam4_from_world, points);
  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetFeatures(&graph, 4, features4);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetMatches(&graph, 1, 4, matches_all);
  SetMatches(&graph, 2, 4, matches_all);
  SetMatches(&graph, 3, 4, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3,
                      EstimateGeometry(camera, features1, features3,
                                       matches_all));
  SetVerifiedGeometry(&graph, 2, 3,
                      EstimateGeometry(camera, features2, features3,
                                       matches_all));
  SetVerifiedGeometry(&graph, 1, 4,
                      EstimateGeometry(camera, features1, features4,
                                       matches_all));
  SetVerifiedGeometry(&graph, 2, 4,
                      EstimateGeometry(camera, features2, features4,
                                       matches_all));
  SetVerifiedGeometry(&graph, 3, 4,
                      EstimateGeometry(camera, features3, features4,
                                       matches_all));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));
  reconstruction.AddImage(MakeImage(4, 1));

  IncrementalMappingOptions options;
  options.initialization.min_num_inliers = 20;
  options.initialization.min_num_triangulated_points = 20;
  options.initialization.min_triangulation_angle_deg = 0.1;
  options.initialization.max_reprojection_error_px = 1.0;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 3.0;
  options.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.mapper.triangulation.create_max_angle_error_deg = 2.0;
  options.mapper.local_bundle_adjustment.max_num_iterations = 50;
  options.mapper.local_bundle_adjustment.loss_scale = 1.0;
  options.max_num_registrations = 4;
  options.max_num_global_refinements = 2;
  options.global_refinement_change = 0;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_TRUE(summary.initialization.success);
  EXPECT_TRUE(summary.global_bundle_adjustment.success);
  EXPECT_GT(summary.global_bundle_adjustment.num_residuals, 0U);
  EXPECT_TRUE(summary.global_observation_filter.success);
  EXPECT_TRUE(summary.final_global_bundle_adjustment.success);
  // Two growth triggers (3 and 4 frames), each with the requested two passes.
  EXPECT_EQ(summary.num_growth_global_refinements, 2U);
  EXPECT_EQ(summary.num_global_refinements, 4U);
  EXPECT_GT(summary.final_global_bundle_adjustment.num_residuals, 0U);
  EXPECT_EQ(summary.registrations.size(), 2U);
  EXPECT_EQ(summary.registrations[0].registration_order, 1U);
  EXPECT_EQ(summary.registrations[1].registration_order, 2U);
  EXPECT_GE(summary.registrations[0].candidate_num_visible_points,
            options.mapper.registration.min_num_correspondences);
  EXPECT_GE(summary.registrations[0].candidate_num_registered_neighbors, 2U);
  EXPECT_GE(
      summary.registrations[0].candidate_num_verified_inliers_to_registered_images,
      options.mapper.registration.min_num_inliers);
  EXPECT_GT(summary.registrations[0].candidate_num_visible_observations, 0U);
  EXPECT_GT(summary.registrations[0].candidate_mean_track_length, 0.0);
  EXPECT_GT(summary.registrations[0].candidate_score, 0.0);
  EXPECT_EQ(summary.num_registered_images, 4U);
  EXPECT_EQ(reconstruction.RegisteredImageIds().size(), 4U);
  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  EXPECT_TRUE(reconstruction.IsImageRegistered(2));
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_TRUE(reconstruction.IsImageRegistered(4));
  EXPECT_GT(summary.num_points3d, options.initialization.min_num_triangulated_points);
  for (const IncrementalMapperSummary& registration : summary.registrations) {
    EXPECT_TRUE(registration.success);
    EXPECT_TRUE(registration.local_bundle_adjustment.success);
    EXPECT_TRUE(registration.observation_filter.success);
  }
}

TEST(IncrementalMapperTest, DemotesFailedCandidatesAndStopsAfterTrialLimit) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam4_from_world = CameraAt(Eigen::Vector3d(0.5, 0.1, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features4 = ProjectFeatures(camera, cam4_from_world, points);
  FeatureSet bad_features3;
  for (std::size_t i = 0; i < points.size(); ++i) {
    bad_features3.keypoints.push_back({12.0F, 12.0F, 1.0F, 0.0F});
  }

  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, bad_features3);
  SetFeatures(&graph, 4, features4);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetMatches(&graph, 1, 4, matches_all);
  SetMatches(&graph, 2, 4, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 1, 4, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 4, MakeVerifiedGeometry(matches_all));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));
  reconstruction.AddImage(MakeImage(4, 1));

  IncrementalInitializerOptions init_options;
  init_options.min_num_inliers = 20;
  init_options.min_num_triangulated_points = 20;
  init_options.min_triangulation_angle_deg = 0.1;
  init_options.max_reprojection_error_px = 1.0;
  ASSERT_TRUE(InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                               init_options, &reconstruction)
                  .success);

  IncrementalMappingOptions options;
  options.max_num_registrations = 4;
  options.max_num_registration_trials_per_image = 1;
  options.run_global_track_completion = false;
  options.run_global_bundle_adjustment = false;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 0.25;
  options.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.mapper.triangulation.create_max_angle_error_deg = 2.0;
  options.mapper.run_local_bundle_adjustment = false;
  options.mapper.run_local_retriangulation_filter = false;
  options.mapper.run_observation_filter = false;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  ASSERT_EQ(summary.failed_registrations.size(), 1U);
  EXPECT_EQ(summary.failed_registrations[0].image_id, 3U);
  EXPECT_EQ(summary.failed_registrations[0].attempt, 1U);
  EXPECT_GT(summary.failed_registrations[0].candidate_score, 0.0);
  ASSERT_EQ(summary.registrations.size(), 1U);
  EXPECT_EQ(summary.registrations[0].image_id, 4U);
  EXPECT_EQ(summary.registrations[0].registration_attempts_before_success, 0U);
  EXPECT_TRUE(reconstruction.IsImageRegistered(4));
  EXPECT_FALSE(reconstruction.IsImageRegistered(3));
}

TEST(IncrementalMapperTest, RelaxesRegistrationThresholdsAcrossRetries) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.35, 0.15, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 =
      PerturbFeatures(ProjectFeatures(camera, cam3_from_world, points), 0.8);
  const std::vector<FeatureMatch> matches_existing = IdentityMatches(0, 30);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetMatches(&graph, 1, 2, matches_existing);
  SetMatches(&graph, 1, 3, matches_all);
  SetMatches(&graph, 2, 3, matches_all);
  SetVerifiedGeometry(&graph, 1, 2,
                      EstimateGeometry(camera, features1, features2,
                                       matches_existing));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(matches_all));

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
  ASSERT_TRUE(InitializeReconstructionFromPair(graph, ImagePair::Ordered(1, 2),
                                               init_options, &reconstruction)
                  .success);

  IncrementalMappingOptions options;
  options.max_num_registrations = 2;
  options.max_num_registration_trials_per_image = 2;
  options.registration_retry_reprojection_error_multiplier = 80.0;
  options.registration_retry_max_reprojection_error_px = 4.0;
  options.run_global_track_completion = false;
  options.run_global_bundle_adjustment = false;
  options.mapper.registration.min_num_correspondences = 20;
  options.mapper.registration.min_num_inliers = 20;
  options.mapper.registration.max_reprojection_error_px = 0.05;
  options.mapper.triangulation.min_triangulation_angle_deg = 0.1;
  options.mapper.run_local_bundle_adjustment = false;
  options.mapper.run_local_retriangulation_filter = false;
  options.mapper.run_observation_filter = false;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  ASSERT_EQ(summary.failed_registrations.size(), 1U);
  EXPECT_EQ(summary.failed_registrations[0].image_id, 3U);
  EXPECT_EQ(summary.failed_registrations[0].attempt, 1U);
  EXPECT_DOUBLE_EQ(summary.failed_registrations[0].registration_max_reprojection_error_px,
                   0.05);
  ASSERT_EQ(summary.registrations.size(), 1U);
  EXPECT_EQ(summary.registrations[0].image_id, 3U);
  EXPECT_EQ(summary.registrations[0].registration_attempts_before_success, 1U);
  EXPECT_DOUBLE_EQ(summary.registrations[0].registration_max_reprojection_error_px,
                   4.0);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
}

TEST(IncrementalMapperTest, SelectsCandidateWithStrongerRegisteredConnectivity) {
  const Camera camera = MakeCamera(1);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (const ImageId image_id : {1U, 2U, 3U, 4U}) {
    Image image = MakeImage(image_id, 1);
    if (image_id == 1U || image_id == 4U) {
      image.registered = true;
      image.cam_from_world = CameraAt(Eigen::Vector3d::Zero());
      for (int i = 0; i < 10; ++i) {
        image.points2d.push_back(
            {{100.0 + i, 120.0 + i}, 0, kInvalidPoint3DId});
      }
    } else {
      for (int i = 0; i < 10; ++i) {
        image.points2d.push_back(
            {{200.0 + i, 220.0 + i}, 0, kInvalidPoint3DId});
      }
    }
    reconstruction.AddImage(image);
  }

  for (int i = 0; i < 10; ++i) {
    Track track1;
    track1.AddElement(1, static_cast<Point2DIdx>(i));
    reconstruction.AddPoint3D(Eigen::Vector3d(i, 0.0, 5.0), track1);

    Track track4;
    track4.AddElement(4, static_cast<Point2DIdx>(i));
    reconstruction.AddPoint3D(Eigen::Vector3d(i, 1.0, 5.0), track4);
  }

  MatchGraph graph;
  FeatureSet dummy_features;
  for (int i = 0; i < 10; ++i) {
    dummy_features.keypoints.push_back(
        {static_cast<float>(i), static_cast<float>(i), 1.0F, 0.0F});
  }
  SetFeatures(&graph, 2, dummy_features);
  SetFeatures(&graph, 3, dummy_features);

  const std::vector<FeatureMatch> all_matches = IdentityMatches(0, 10);
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(all_matches));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(IdentityMatches(0, 5)));
  SetVerifiedGeometry(&graph, 4, 3, MakeVerifiedGeometry(IdentityMatches(5, 10)));

  IncrementalMapperOptions options;
  options.registration.min_num_correspondences = 8;

  const std::optional<ImageId> next_image =
      SelectNextImage(graph, options, reconstruction);

  ASSERT_TRUE(next_image.has_value());
  EXPECT_EQ(*next_image, 3U);
}

TEST(IncrementalMapperTest, RetriangulatesRegisteredNeighborhoodPairs) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.5, 0.1, 0.0));

  const FeatureSet features1 = ProjectFeatures(camera, cam1_from_world, points);
  const FeatureSet features2 = ProjectFeatures(camera, cam2_from_world, points);
  const FeatureSet features3 = ProjectFeatures(camera, cam3_from_world, points);
  const std::vector<FeatureMatch> matches_all = IdentityMatches(0, 10);
  const std::vector<FeatureMatch> seed_matches = IdentityMatches(0, 8);

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetFeatures(&graph, 3, features3);
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(matches_all));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(seed_matches));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry(seed_matches));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  Image image3 = MakeImage(3, 1);
  image3.registered = true;
  image3.cam_from_world = cam3_from_world;
  for (std::size_t i = 0; i < points.size(); ++i) {
    image1.points2d.push_back({{features1.keypoints[i].x, features1.keypoints[i].y},
                               static_cast<Point2DIdx>(i), kInvalidPoint3DId});
    image2.points2d.push_back({{features2.keypoints[i].x, features2.keypoints[i].y},
                               static_cast<Point2DIdx>(i), kInvalidPoint3DId});
    image3.points2d.push_back({{features3.keypoints[i].x, features3.keypoints[i].y},
                               static_cast<Point2DIdx>(i), kInvalidPoint3DId});
  }
  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);
  reconstruction.AddImage(image3);

  IncrementalTriangulatorOptions options;
  options.ignore_two_view_tracks = false;
  options.min_triangulation_angle_deg = 0.1;
  options.create_max_angle_error_deg = 1.0;

  Reconstruction bounded = reconstruction;
  const std::set<ImageId> root_window{3};
  RetriangulateLocalNeighborhood(graph, 3, options, nullptr, &bounded,
                                nullptr, &root_window);
  EXPECT_EQ(bounded.Points3D().size(), 8U);
  RetriangulateAllRegisteredImages(graph, options, &bounded);
  EXPECT_EQ(bounded.Points3D().size(), 10U);

  const IncrementalTriangulatorSummary summary =
      RetriangulateLocalNeighborhood(graph, 3, options, nullptr,
                                     &reconstruction);

  EXPECT_GE(summary.num_created_points, 10U);
  EXPECT_EQ(reconstruction.Points3D().size(), 10U);
}

TEST(IncrementalMapperTest, RetriangulationSkipsLowAnglePairLayer) {
  const Camera camera = MakeCamera(1);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  image1.points2d.push_back({*pixel1, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);
  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  image2.points2d.push_back({*pixel2, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  MatchGraph graph;
  TwoViewGeometry geometry = MakeVerifiedGeometry({{0, 0}});
  geometry.median_triangulation_angle = 0.2;
  SetVerifiedGeometry(&graph, 1, 2, geometry);

  IncrementalTriangulatorOptions options;
  options.min_triangulation_angle_deg = 0.0;
  options.min_pair_median_triangulation_angle_deg = 1.0;
  options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      RetriangulateLocalNeighborhood(graph, 1, options, nullptr,
                                     &reconstruction);

  EXPECT_EQ(summary.num_created_points, 0U);
  EXPECT_TRUE(reconstruction.Points3D().empty());
}

TEST(IncrementalMapperTest, RetriangulationFiltersNewLowAnglePoints) {
  const Camera camera = MakeCamera(1);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(0.01, 0.0, 0.0));
  const Eigen::Vector3d point_world(0.0, 0.0, 10.0);
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  image1.points2d.push_back({*pixel1, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);
  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  image2.points2d.push_back({*pixel2, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions triangulation_options;
  triangulation_options.ignore_two_view_tracks = false;
  triangulation_options.min_triangulation_angle_deg = 0.0;
  triangulation_options.create_max_angle_error_deg = 1.0;

  ObservationFilterOptions filter_options;
  filter_options.max_reprojection_error_px = 1.0;
  filter_options.min_triangulation_angle_deg = 1.0;
  filter_options.min_track_length = 2;

  const IncrementalTriangulatorSummary summary =
      RetriangulateLocalNeighborhood(graph, 1, triangulation_options,
                                     &filter_options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, 1U);
  EXPECT_EQ(summary.num_filtered_points, 1U);
  EXPECT_TRUE(reconstruction.Points3D().empty());
}

TEST(IncrementalMapperTest, RetriangulationPreservesExistingSparseImportedIds) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d xyz(0.0, 0.0, 10.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 2; ++image_id) {
    Image image = MakeImage(image_id, camera.id);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(0.01 * (image_id - 1), 0, 0));
    image.points2d.push_back({*ProjectWorldPoint(camera, *image.cam_from_world,
                                               xyz), 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  Point3D imported;
  imported.id = 1000000;
  imported.xyz = xyz;
  reconstruction.AddPoint3D(imported);
  imported.id = 9000000;
  reconstruction.AddPoint3D(imported);
  ASSERT_TRUE(reconstruction.DeletePoint3D(imported.id));
  const Point3DId boundary = reconstruction.NextPoint3DId();
  EXPECT_EQ(boundary, 9000001U);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  IncrementalTriangulatorOptions options;
  options.ignore_two_view_tracks = false;
  options.min_triangulation_angle_deg = 0.0;
  ObservationFilterOptions filter;
  filter.min_triangulation_angle_deg = 1.0;
  const auto summary = RetriangulateLocalNeighborhood(
      graph, 1, options, &filter, &reconstruction);
  EXPECT_EQ(summary.num_created_points, 1U);
  EXPECT_EQ(summary.num_filtered_points, 1U);
  EXPECT_EQ(reconstruction.NextPoint3DId(), boundary + 1);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_TRUE(reconstruction.ExistsPoint3D(1000000));
}

TEST(IncrementalMapperTest, CompletesTracksAcrossAllRegisteredImagesAtEnd) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  Image image1 = MakeImage(1, 1);
  image1.registered = true;
  image1.cam_from_world = cam1_from_world;
  image1.points2d.push_back({*pixel1, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image1);

  Image image2 = MakeImage(2, 1);
  image2.registered = true;
  image2.cam_from_world = cam2_from_world;
  image2.points2d.push_back({*pixel2, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image2);

  Track track1;
  track1.AddElement(1, 0);
  reconstruction.AddPoint3D(point_world, track1);

  Track track2;
  track2.AddElement(2, 0);
  reconstruction.AddPoint3D(point_world, track2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalMappingOptions options;
  options.max_num_registrations = 0;
  options.run_global_bundle_adjustment = false;

  const IncrementalMappingSummary summary =
      RunIncrementalMapping(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  const Point3DId merged_id = reconstruction.GetImage(1).points2d[0].point3d_id;
  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id, merged_id);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(merged_id));
  EXPECT_EQ(reconstruction.GetPoint3D(merged_id).track.Length(), 2U);
}

TEST(IncrementalMapperTest,
     RetriangulatesAllRegisteredImagesBeforeGlobalBundleAdjustment) {
  const Camera camera = MakeCamera(1);
  const std::vector<Eigen::Vector3d> points = MakeScenePoints();
  const Pose3 pose1 = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 pose2 = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const FeatureSet features1 = ProjectFeatures(camera, pose1, points);
  const FeatureSet features2 = ProjectFeatures(camera, pose2, points);
  const std::vector<FeatureMatch> matches = IdentityMatches(0, points.size());

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetMatches(&graph, 1, 2, matches);
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(matches));

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeImage(1, 1);
  Image image2 = MakeImage(2, 1);
  image1.cam_from_world = pose1;
  image2.cam_from_world = pose2;
  image1.registered = true;
  image2.registered = true;
  for (std::size_t i = 0; i < points.size(); ++i) {
    image1.points2d.push_back(
        {Eigen::Vector2d(features1.keypoints[i].x, features1.keypoints[i].y),
         static_cast<Point2DIdx>(i), kInvalidPoint3DId});
    image2.points2d.push_back(
        {Eigen::Vector2d(features2.keypoints[i].x, features2.keypoints[i].y),
         static_cast<Point2DIdx>(i), kInvalidPoint3DId});
  }
  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);

  IncrementalTriangulatorOptions options;
  options.ignore_two_view_tracks = false;
  options.min_triangulation_angle_deg = 0.1;
  const IncrementalTriangulatorSummary summary =
      RetriangulateAllRegisteredImages(graph, options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, points.size());
  EXPECT_EQ(reconstruction.Points3D().size(), points.size());
}

TEST(IncrementalMapperTest, HonorsCompletionTransitivityAfterBundleAdjustment) {
  // The fifth/seventh-hop observation is outside the completion gate before BA and
  // inside it afterwards. Intermediate features conflict with the two seed
  // images, so traversal cannot shorten the path by attaching them first.
  for (const std::size_t transitivity : {4U, 5U, 6U, 7U}) {
    SCOPED_TRACE(transitivity);
    const std::size_t required_hops = transitivity <= 5 ? 5 : 7;
    const std::size_t intermediates_per_image = (required_hops - 1) / 2;
    const Camera camera = MakeCamera(1);
    const Eigen::Vector3d world(0.0, 0.0, 5.0);
    Reconstruction model;
    model.AddCamera(camera);
    MatchGraph graph;
    for (ImageId id = 1; id <= 3; ++id) {
      const auto pose = CameraAt(id == 3 ? Eigen::Vector3d(0.0, 0.0, 4.5)
                                       : Eigen::Vector3d(id - 1.0, 0.0, 0.0));
      Image image = MakeImage(id, 1);
      image.registered = true;
      image.cam_from_world = pose;
      const auto pixel = ProjectWorldPoint(camera, pose, world);
      ASSERT_TRUE(pixel.has_value());
      image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
      FeatureSet features;
      features.keypoints.push_back({static_cast<float>(pixel->x()),
                                   static_cast<float>(pixel->y()), 1.0F, 0.0F});
      if (id < 3) {
        for (std::size_t i = 0; i < intermediates_per_image; ++i) {
          image.points2d.push_back({Eigen::Vector2d(10, 10),
              static_cast<Point2DIdx>(i + 1), kInvalidPoint3DId});
          features.keypoints.push_back({10.0F, 10.0F, 1.0F, 0.0F});
        }
      }
      model.AddImage(image);
      SetFeatures(&graph, id, features);
    }
    Track track;
    track.AddElement(1, 0);
    track.AddElement(2, 0);
    const Point3DId point = model.AddPoint3D(world + Eigen::Vector3d(0.01, 0, 0), track);
    std::vector<FeatureMatch> chain{{0, 1}};
    for (Point2DIdx i = 1; i <= intermediates_per_image; ++i) {
      chain.push_back({i, i});
      if (i < intermediates_per_image) chain.push_back({i, i + 1});
    }
    SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(chain));
    SetVerifiedGeometry(&graph, 1, 3,
        MakeVerifiedGeometry({{static_cast<Point2DIdx>(intermediates_per_image), 0}}));
    IncrementalMappingOptions options;
    options.max_num_registrations = 0;
    options.run_pose_guided_track_completion = false;
    options.run_global_observation_filter = false;
    options.global_bundle_adjustment.refine_poses = false;
    options.global_bundle_adjustment.refine_camera_params = CameraParameterGroup::kNone;
    options.mapper.triangulation.complete_max_transitivity = transitivity;
    options.mapper.triangulation.complete_max_reprojection_error_px = 4.0;
    const auto summary = RunIncrementalMapping(graph, options, &model);
    ASSERT_TRUE(summary.success) << summary.message;
    ASSERT_TRUE(model.ExistsPoint3D(point));
    EXPECT_EQ(model.GetImage(3).points2d[0].point3d_id,
              transitivity >= required_hops ? point : kInvalidPoint3DId);
    EXPECT_EQ(model.GetPoint3D(point).track.Length(), transitivity >= required_hops ? 3U : 2U);
  }
}

TEST(IncrementalMapperTest, SchedulesGlobalRefinementFromFrameAndPointGrowth) {
  struct Case {
    double frames_ratio, points_ratio;
    std::size_t frames_freq, points_freq, seed_points, expected;
  };
  const std::vector<Case> cases = {
      {1.5, 100, 1000, 1000, 40, 1},  // equality at 3 frames, then reset
      {2.0, 100, 1000, 1000, 40, 1},  // equality at 4 frames
      {100, 100, 1, 1000, 40, 2},
      {100, 100, 2, 1000, 40, 1},
      {100, 100, 1000, 1000, 40, 0},
      {100, 4.0 / 3.0, 1000, 1000, 30, 1},  // triangulate 10 new points
      {100, 1.34, 1000, 1000, 30, 0},
      {100, 100, 1000, 10, 30, 1},
      {100, 100, 1000, 11, 30, 0},
  };
  for (std::size_t case_idx = 0; case_idx < cases.size(); ++case_idx) {
    SCOPED_TRACE(case_idx);
    const auto& test = cases[case_idx];
    const Camera camera = MakeCamera(1);
    const auto points = MakeScenePoints();
    Reconstruction model;
    model.AddCamera(camera);
    MatchGraph graph;
    for (ImageId id = 1; id <= 4; ++id) {
      const auto pose = CameraAt(Eigen::Vector3d((id - 1) * 0.5, 0, 0));
      auto image = MakeImage(id, 1);
      image.registered = id <= 2;
      if (image.registered) image.cam_from_world = pose;
      const auto features = ProjectFeatures(camera, pose, points);
      SetFeatures(&graph, id, features);
      for (Point2DIdx i = 0; i < points.size(); ++i) {
        const auto& key = features.keypoints[i];
        image.points2d.push_back({Eigen::Vector2d(key.x, key.y), i, kInvalidPoint3DId});
      }
      model.AddImage(image);
      for (ImageId earlier = 1; earlier < id; ++earlier) {
        SetVerifiedGeometry(&graph, earlier, id,
                            MakeVerifiedGeometry(IdentityMatches(0, points.size())));
      }
    }
    for (Point2DIdx i = 0; i < test.seed_points; ++i) {
      Track track;
      track.AddElement(1, i);
      track.AddElement(2, i);
      model.AddPoint3D(points[i], track);
    }
    IncrementalMappingOptions options;
    options.max_num_registrations = 2;
    options.run_global_track_completion = false;
    options.run_pose_guided_track_completion = false;
    options.run_global_observation_filter = false;
    options.mapper.run_local_bundle_adjustment = false;
    options.mapper.run_local_retriangulation_filter = false;
    options.mapper.run_observation_filter = false;
    options.mapper.registration.random_seed = 0;
    options.mapper.triangulation.ignore_two_view_tracks = true;
    options.global_bundle_adjustment.refine_poses = false;
    options.ba_global_frames_ratio = test.frames_ratio;
    options.ba_global_points_ratio = test.points_ratio;
    options.ba_global_frames_freq = test.frames_freq;
    options.ba_global_points_freq = test.points_freq;
    const auto summary = RunIncrementalMapping(graph, options, &model);
    ASSERT_TRUE(summary.success) << summary.message;
    EXPECT_EQ(model.RegisteredImageIds().size(), 4U);
    EXPECT_EQ(model.Points3D().size(), points.size());
    EXPECT_EQ(summary.num_growth_global_refinements, test.expected);
  }
}

TEST(IncrementalMapperTest, RejectsInvalidGlobalGrowthConfigurationBeforeMutation) {
  for (int kind = 0; kind < 4; ++kind) {
    IncrementalMappingOptions options;
    if (kind == 0) options.ba_global_frames_ratio = 1.0;
    if (kind == 1) options.ba_global_points_ratio = 0.0;
    if (kind == 2) options.ba_global_frames_freq = 0;
    if (kind == 3) options.ba_global_points_freq = 0;
    Reconstruction model;
    const auto summary = RunIncrementalMapping({}, options, &model);
    EXPECT_FALSE(summary.success);
    EXPECT_NE(summary.message.find("Global BA growth"), std::string::npos);
    EXPECT_TRUE(model.Images().empty());
    EXPECT_TRUE(model.Points3D().empty());
  }
}

}  // namespace
}  // namespace colite
