#include "colite/sfm/incremental_initializer.h"

#include <vector>
#include <algorithm>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"

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

void AddProjectedMatch(const Camera& camera,
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

MatchGraph MakeVerifiedGraph(const Camera& camera) {
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));

  FeatureSet features1;
  FeatureSet features2;
  std::vector<FeatureMatch> matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      AddProjectedMatch(camera, cam1_from_world, cam2_from_world,
                        Eigen::Vector3d(-1.0 + 0.4 * x,
                                        -0.6 + 0.3 * y,
                                        4.0 + 0.07 * x + 0.11 * y +
                                            0.03 * x * y),
                        &features1, &features2, &matches);
    }
  }

  TwoViewGeometryOptions verify_options;
  verify_options.compute_relative_pose = true;
  verify_options.random_seed = 0;
  verify_options.max_error_px = 1.0;
  verify_options.min_num_inliers = 20;
  const TwoViewGeometry geometry =
      EstimateTwoViewGeometry(camera, features1, camera, features2, matches,
                              verify_options);

  MatchGraph graph;
  SetFeatures(&graph, 1, features1);
  SetFeatures(&graph, 2, features2);
  SetMatches(&graph, 1, 2, matches);
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  return graph;
}

TEST(IncrementalInitializerTest, ChoosesBestInitialPairByInliersAndAngle) {
  MatchGraph graph;

  TwoViewGeometry weak_geometry;
  weak_geometry.cam2_from_cam1 = CameraAt({1, 0, 0});
  weak_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  weak_geometry.inlier_matches = std::vector<FeatureMatch>(24, {0, 0});
  weak_geometry.median_triangulation_angle = 1.0;
  SetVerifiedGeometry(&graph, 1, 2, weak_geometry);

  TwoViewGeometry strong_geometry = weak_geometry;
  strong_geometry.inlier_matches = std::vector<FeatureMatch>(30, {0, 0});
  strong_geometry.median_triangulation_angle = 2.0;
  SetVerifiedGeometry(&graph, 3, 4, strong_geometry);

  const auto pair = SelectInitialImagePair(graph);

  ASSERT_TRUE(pair.has_value());
  EXPECT_EQ(pair->image_id1, 3U);
  EXPECT_EQ(pair->image_id2, 4U);
}

TEST(IncrementalInitializerTest, RejectsForwardMotionAtThresholdIndependentlyOfScale) {
  MatchGraph graph;
  TwoViewGeometry lateral;
  lateral.type = TwoViewGeometry::Type::kCalibratedEssential;
  lateral.cam2_from_cam1 = CameraAt({1, 0, 0});
  lateral.inlier_matches = std::vector<FeatureMatch>(24, {0, 0});
  lateral.median_triangulation_angle = 20;
  SetVerifiedGeometry(&graph, 3, 4, lateral);
  auto forward = lateral;
  forward.inlier_matches.resize(30);
  forward.cam2_from_cam1 = CameraAt({.6, 0, .8});
  SetVerifiedGeometry(&graph, 1, 2, forward);
  IncrementalInitializerOptions options;
  options.max_forward_motion = .8;
  auto pair = SelectInitialImagePair(graph, options);
  ASSERT_TRUE(pair.has_value());
  EXPECT_EQ(pair->image_id1, 3U);
  for (const double scale : {0.5, 10.0, -2.0}) {
    forward.cam2_from_cam1 = CameraAt(scale * Eigen::Vector3d(.6, 0, .8));
    SetVerifiedGeometry(&graph, 1, 2, forward);
    EXPECT_EQ(SelectInitialImagePair(graph, options)->image_id1, 3U);
  }
  options.max_forward_motion = .81;
  EXPECT_EQ(SelectInitialImagePair(graph, options)->image_id1, 1U);
  options.max_forward_motion = 0;
  EXPECT_FALSE(SelectInitialImagePair(graph, options).has_value());
  options.max_forward_motion = 1;
  forward.cam2_from_cam1 = CameraAt({0, 0, 1});
  SetVerifiedGeometry(&graph, 1, 2, forward);
  EXPECT_EQ(SelectInitialImagePair(graph, options)->image_id1, 3U);
}

TEST(IncrementalInitializerTest, SeedsPreferGlobalSupportAndFocalPriors) {
  MatchGraph graph;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.cam2_from_cam1 = CameraAt({1, 0, 0});
  geometry.median_triangulation_angle = 20;
  geometry.inlier_matches.resize(40);
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  geometry.inlier_matches.resize(30);
  SetVerifiedGeometry(&graph, 3, 4, geometry);
  geometry.median_triangulation_angle = 0;  // support, but not a seed
  geometry.inlier_matches.resize(100);
  SetVerifiedGeometry(&graph, 3, 5, geometry);
  EXPECT_EQ(SelectInitialImagePair(graph), ImagePair::Ordered(3, 4));
  IncrementalInitializerOptions options;
  options.allowed_image_ids = {1, 2, 3, 4};
  EXPECT_EQ(SelectInitialImagePair(graph, options), ImagePair::Ordered(1, 2));

  Reconstruction model;
  auto prior = MakeCamera(1);
  model.AddCamera(prior);
  auto unknown = MakeCamera(2); unknown.has_prior_focal_length = false;
  model.AddCamera(unknown);
  for (ImageId id = 1; id <= 5; ++id) model.AddImage(MakeImage(id, id <= 2 ? 1 : 2));
  EXPECT_EQ(SelectInitialImagePair(graph, {}, &model), ImagePair::Ordered(1, 2));
  EXPECT_EQ(SelectInitialImagePair(graph), ImagePair::Ordered(3, 4));
}

TEST(IncrementalInitializerTest, InitializesReconstructionFromVerifiedPair) {
  const Camera camera = MakeCamera(1);
  MatchGraph graph = MakeVerifiedGraph(camera);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));

  IncrementalInitializerOptions options;
  options.min_num_inliers = 20;
  options.min_num_triangulated_points = 20;
  options.min_triangulation_angle_deg = 0.1;
  options.max_reprojection_error_px = 1.0;

  const IncrementalInitializerSummary summary =
      InitializeReconstructionFromBestPair(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.initial_pair.image_id1, 1U);
  EXPECT_EQ(summary.initial_pair.image_id2, 2U);
  EXPECT_GE(summary.num_triangulated_points, 20U);
  EXPECT_TRUE(reconstruction.IsImageRegistered(1));
  EXPECT_TRUE(reconstruction.IsImageRegistered(2));
  EXPECT_EQ(reconstruction.RegisteredImageIds().size(), 2U);
  EXPECT_EQ(reconstruction.Points3D().size(), summary.num_triangulated_points);
  EXPECT_TRUE(reconstruction.GetImage(1).HasPose());
  EXPECT_TRUE(reconstruction.GetImage(2).HasPose());
  EXPECT_EQ(reconstruction.GetImage(1).points2d.size(),
            graph.features.at(1).keypoints.size());
  EXPECT_GT(reconstruction.GetImage(1).NumPoints3D(), 0U);
  EXPECT_GT(reconstruction.GetImage(2).NumPoints3D(), 0U);
}

TEST(IncrementalInitializerTest, TriesNextInitialPairWhenBestCandidateFails) {
  const Camera camera = MakeCamera(1);
  MatchGraph graph;

  TwoViewGeometry bad_best_geometry;
  bad_best_geometry.cam2_from_cam1 = CameraAt({1, 0, 0});
  bad_best_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  bad_best_geometry.inlier_matches = std::vector<FeatureMatch>(40, {0, 0});
  bad_best_geometry.median_triangulation_angle = 3.0;
  SetVerifiedGeometry(&graph, 1, 2, bad_best_geometry);

  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam4_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  FeatureSet features3;
  FeatureSet features4;
  std::vector<FeatureMatch> fallback_matches;
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 6; ++x) {
      AddProjectedMatch(camera, cam3_from_world, cam4_from_world,
                        Eigen::Vector3d(-1.0 + 0.4 * x,
                                        -0.6 + 0.3 * y,
                                        4.0 + 0.07 * x + 0.11 * y +
                                            0.03 * x * y),
                        &features3, &features4, &fallback_matches);
    }
  }
  TwoViewGeometryOptions verify_options;
  verify_options.compute_relative_pose = true;
  verify_options.random_seed = 0;
  verify_options.max_error_px = 1.0;
  verify_options.min_num_inliers = 20;
  const TwoViewGeometry fallback_geometry = EstimateTwoViewGeometry(
      camera, features3, camera, features4, fallback_matches, verify_options);
  SetFeatures(&graph, 3, features3);
  SetFeatures(&graph, 4, features4);
  SetMatches(&graph, 3, 4, fallback_matches);
  SetVerifiedGeometry(&graph, 3, 4, fallback_geometry);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 1));
  reconstruction.AddImage(MakeImage(2, 1));
  reconstruction.AddImage(MakeImage(3, 1));
  reconstruction.AddImage(MakeImage(4, 1));

  IncrementalInitializerOptions options;
  options.min_num_inliers = 20;
  options.min_num_triangulated_points = 20;
  options.min_triangulation_angle_deg = 0.1;
  options.max_reprojection_error_px = 1.0;
  options.max_num_pair_trials = 3;

  const IncrementalInitializerSummary summary =
      InitializeReconstructionFromBestPair(graph, options, &reconstruction);

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.initial_pair.image_id1, 3U);
  EXPECT_EQ(summary.initial_pair.image_id2, 4U);
  EXPECT_EQ(summary.num_initialization_attempts, 2U);
  ASSERT_EQ(summary.attempted_initial_pairs.size(), 2U);
  EXPECT_EQ(summary.attempted_initial_pairs[0], ImagePair::Ordered(1, 2));
  EXPECT_EQ(summary.attempted_initial_pairs[1], ImagePair::Ordered(3, 4));
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_TRUE(reconstruction.IsImageRegistered(4));
}

TEST(IncrementalInitializerTest, FailsWhenNoUsableInitialPairExists) {
  MatchGraph graph;
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kDegenerate;
  SetVerifiedGeometry(&graph, 1, 2, geometry);

  Reconstruction reconstruction;
  IncrementalInitializerSummary summary =
      InitializeReconstructionFromBestPair(graph, {}, &reconstruction);

  EXPECT_FALSE(summary.success);
  EXPECT_EQ(summary.num_triangulated_points, 0U);
}

TEST(IncrementalInitializerTest, InitializesUncalibratedAndPlanarPosesButRejectsPureRotation) {
  const auto camera = MakeCamera(1);
  const auto original = MakeVerifiedGraph(camera);
  for (auto type : {TwoViewGeometry::Type::kUncalibratedFundamental, TwoViewGeometry::Type::kPlanarHomography}) {
    auto graph = original;
    graph.verified_geometries.begin()->second.type = type;
    Reconstruction model;
    model.AddCamera(camera);
    model.AddImage(MakeImage(1, 1)); model.AddImage(MakeImage(2, 1));
    IncrementalInitializerOptions options;
    options.min_num_inliers = 20; options.min_num_triangulated_points = 20;
    options.min_triangulation_angle_deg = 0.1;
    const auto result = InitializeReconstructionFromBestPair(graph, options, &model);
    EXPECT_TRUE(result.success) << result.message;
    EXPECT_GE(model.Points3D().size(), 20U);
  }
  auto graph = original;
  graph.verified_geometries.begin()->second.cam2_from_cam1->t_cw.setZero();
  EXPECT_FALSE(SelectInitialImagePair(graph));
}

TEST(IncrementalInitializerTest, ReestimatesMultipleWithoutPoseOrFocalPrior) {
  auto camera = MakeCamera(1);
  auto graph = MakeVerifiedGraph(camera);
  auto& cached = graph.verified_geometries.begin()->second;
  cached.type = TwoViewGeometry::Type::kMultiple;
  cached.cam2_from_cam1.reset(); cached.E.reset(); cached.F.reset(); cached.H.reset();
  cached.median_triangulation_angle = -1;
  camera.has_prior_focal_length = false;
  Reconstruction model;
  model.AddCamera(camera); model.AddImage(MakeImage(1, 1)); model.AddImage(MakeImage(2, 1));
  IncrementalInitializerOptions options;
  options.reestimate_geometry = true; options.min_num_inliers = 20;
  options.min_num_triangulated_points = 20; options.min_triangulation_angle_deg = .1;
  options.max_reprojection_error_px = 1;
  const auto summary = InitializeReconstructionFromBestPair(graph, options, &model);
  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_TRUE(summary.geometry_reestimated);
  EXPECT_GE(summary.num_geometry_inliers, 20U);
  EXPECT_EQ(model.RegisteredImageIds().size(), 2U);
  EXPECT_FALSE(model.GetCamera(1).has_prior_focal_length);
  EXPECT_FALSE(cached.cam2_from_cam1); EXPECT_FALSE(cached.E);
}

TEST(IncrementalInitializerTest, ReestimationRejectsBadMatchesDespiteCachedPose) {
  const auto camera = MakeCamera(1);
  auto graph = MakeVerifiedGraph(camera);
  auto& cached = graph.verified_geometries.begin()->second;
  cached.inlier_matches.assign(30, {0, 0});
  cached.median_triangulation_angle = 30;
  Reconstruction model;
  model.AddCamera(camera); model.AddImage(MakeImage(1, 1)); model.AddImage(MakeImage(2, 1));
  IncrementalInitializerOptions options;
  options.reestimate_geometry = true; options.min_num_inliers = 20;
  options.min_triangulation_angle_deg = .1;
  const auto summary = InitializeReconstructionFromBestPair(graph, options, &model);
  EXPECT_FALSE(summary.success);
  EXPECT_TRUE(summary.geometry_reestimated);
  EXPECT_TRUE(model.RegisteredImageIds().empty());
  EXPECT_TRUE(model.Points3D().empty());
}

TEST(IncrementalInitializerTest, ImageTrialLimitOnlyConstrainsAutomaticFirstSeed) {
  auto camera=MakeCamera(1);
  auto graph=MakeVerifiedGraph(camera);
  auto geometry=graph.verified_geometries.begin()->second;
  geometry.inlier_matches.resize(25); SetVerifiedGeometry(&graph,1,3,geometry);
  geometry.inlier_matches.resize(20); SetVerifiedGeometry(&graph,2,3,geometry);
  Reconstruction model;model.AddCamera(camera);
  for (ImageId id : {1,2,3}) model.AddImage(MakeImage(id,1));
  IncrementalInitializerOptions options;
  InitialPairTrialState state;state.num_reg_trials[1]=2;
  auto candidates=SelectInitialImagePairCandidates(graph,options,&model,&state);
  ASSERT_FALSE(candidates.empty());
  EXPECT_EQ(candidates.front(),(ImagePair{2,1}));  // exhausted image 1 remains a valid second seed
  options.image_id1=1;
  candidates=SelectInitialImagePairCandidates(graph,options,&model,&state);
  ASSERT_FALSE(candidates.empty());EXPECT_EQ(candidates.front(),(ImagePair{1,2}));
  options.image_id1=kInvalidImageId;options.image_id2=3;
  candidates=SelectInitialImagePairCandidates(graph,options,&model,&state);
  ASSERT_FALSE(candidates.empty());EXPECT_EQ(candidates.front(),(ImagePair{3,1}));
  options.image_id1=3;options.image_id2=2;
  candidates=SelectInitialImagePairCandidates(graph,options,&model,&state);
  ASSERT_EQ(candidates.size(),1U);EXPECT_EQ(candidates.front(),(ImagePair{3,2}));
}

TEST(IncrementalInitializerTest, ReversedInitialPairSwapsIndicesAndRelativePose) {
  const auto camera=MakeCamera(1);
  auto graph=MakeVerifiedGraph(camera);
  auto features2=graph.features.at(2);
  std::rotate(features2.keypoints.begin(),features2.keypoints.begin()+2,features2.keypoints.end());
  SetFeatures(&graph,2,features2);
  auto& geometry=graph.verified_geometries.begin()->second;
  const auto count=geometry.inlier_matches.size();
  for (auto& match:geometry.inlier_matches) match.idx2=(match.idx1+count-2)%count;
  const auto original_pose=*geometry.cam2_from_cam1;
  const auto original_first=geometry.inlier_matches.front();
  Reconstruction model;model.AddCamera(camera);model.AddImage(MakeImage(1,1));model.AddImage(MakeImage(2,1));
  IncrementalInitializerOptions options;
  options.min_num_inliers=20;options.min_num_triangulated_points=20;options.min_triangulation_angle_deg=.1;
  const auto result=InitializeReconstructionFromPair(graph,{2,1},options,&model);
  ASSERT_TRUE(result.success) << result.message;
  EXPECT_LT(model.GetImage(2).cam_from_world->t_cw.norm(),1e-12);
  EXPECT_LT((model.GetImage(1).cam_from_world->t_cw-original_pose.Inverse().t_cw).norm(),1e-12);
  for (const auto& [id,point]:model.Points3D()) for(const auto& link:point.track.elements) {
    const auto& image=model.GetImage(link.image_id);
    EXPECT_LT(*ReprojectionError(camera,*image.cam_from_world,point.xyz,image.points2d[link.point2d_idx].xy),.001);
  }
  EXPECT_EQ(geometry.inlier_matches.front().idx1,original_first.idx1);
  EXPECT_EQ(geometry.inlier_matches.front().idx2,original_first.idx2);
  EXPECT_EQ(geometry.cam2_from_cam1->t_cw,original_pose.t_cw);
}

}  // namespace
}  // namespace colite
