#include "colite/sfm/structureless_registrar.h"

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite {
namespace {

Pose3 CameraAt(const double center_x) {
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(-center_x, 0.0, 0.0);
  return pose;
}

Image MakeImage(const ImageId id, const double center_x, const bool registered) {
  Image image;
  image.id = id;
  image.camera_id = 1;
  image.registered = registered;
  if (registered) {
    Pose3 pose;
    pose.t_cw = Eigen::Vector3d(-center_x, 0.0, 0.0);
    image.cam_from_world = pose;
  }
  return image;
}

TwoViewGeometry RelativeGeometry(const double translation_x) {
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  if (!geometry.cam2_from_cam1) geometry.cam2_from_cam1.emplace();
  geometry.cam2_from_cam1->t_cw =
      Eigen::Vector3d(translation_x, 0.0, 0.0);
  geometry.inlier_matches.resize(20);
  return geometry;
}

TEST(StructurelessRegistrarTest, RejectsCollinearNeighborRays) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 0.0, true));
  reconstruction.AddImage(MakeImage(2, 2.0, true));
  reconstruction.AddImage(MakeImage(3, 0.0, false));
  reconstruction.RegisterImage(1);
  reconstruction.RegisterImage(2);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 3, RelativeGeometry(-1.0));
  SetVerifiedGeometry(&graph, 2, 3, RelativeGeometry(1.0));

  const StructurelessRegistrationSummary summary =
      EstimateStructurelessPose(graph, 3, reconstruction);

  EXPECT_FALSE(summary.success);
}

TEST(StructurelessRegistrarTest, RejectsSingleNeighborHypothesis) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  reconstruction.AddCamera(camera);
  reconstruction.AddImage(MakeImage(1, 0.0, true));
  reconstruction.AddImage(MakeImage(3, 0.0, false));
  reconstruction.RegisterImage(1);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 3, RelativeGeometry(-1.0));

  const StructurelessRegistrationSummary summary =
      EstimateStructurelessPose(graph, 3, reconstruction);
  EXPECT_FALSE(summary.success);
}

TEST(StructurelessRegistrarTest, IncrementalMapperUsesExplicitFallback) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 800;
  camera.height = 600;
  camera.params = {700.0, 700.0, 400.0, 300.0};
  reconstruction.AddCamera(camera);
  const std::vector<double> centers{0.0, 2.0, 1.0};
  MatchGraph graph;
  std::vector<Eigen::Vector3d> points;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 5; ++col) {
      points.emplace_back(-0.7 + 0.35 * col, -0.4 + 0.28 * row,
                          4.0 + 0.05 * col + 0.03 * row * col);
    }
  }
  for (std::size_t image_idx = 0; image_idx < centers.size(); ++image_idx) {
    Image image;
    image.id = static_cast<ImageId>(image_idx + 1);
    image.camera_id = 1;
    FeatureSet features;
    Pose3 pose = CameraAt(centers[image_idx]);
    if (image_idx == 2) pose.t_cw.y() = -0.6;
    for (const Eigen::Vector3d& point : points) {
      const auto pixel = ProjectWorldPoint(camera, pose, point);
      ASSERT_TRUE(pixel.has_value());
      features.keypoints.push_back(
          {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
           1.0F, 0.0F, 1.0F});
      if (image_idx < 2) {
        image.points2d.push_back(
            {*pixel, static_cast<Point2DIdx>(features.keypoints.size() - 1),
             kInvalidPoint3DId});
      }
    }
    SetFeatures(&graph, image.id, features);
    if (image_idx < 2) {
      image.cam_from_world = pose;
    }
    reconstruction.AddImage(image);
    if (image_idx < 2) {
      reconstruction.RegisterImage(image.id);
    }
  }
  TwoViewGeometry geometry13 = RelativeGeometry(-1.0);
  TwoViewGeometry geometry23 = RelativeGeometry(1.0);
  if (!geometry13.cam2_from_cam1) geometry13.cam2_from_cam1.emplace();
  geometry13.cam2_from_cam1->t_cw = Eigen::Vector3d(-1, -0.6, 0).normalized();
  if (!geometry23.cam2_from_cam1) geometry23.cam2_from_cam1.emplace();
  geometry23.cam2_from_cam1->t_cw = Eigen::Vector3d(1, -0.6, 0).normalized();
  geometry13.inlier_matches.clear();
  geometry23.inlier_matches.clear();
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    const FeatureMatch match{static_cast<Point2DIdx>(idx),
                             static_cast<Point2DIdx>(idx)};
    geometry13.inlier_matches.push_back(match);
    geometry23.inlier_matches.push_back(match);
  }
  SetVerifiedGeometry(&graph, 1, 3, geometry13);
  SetVerifiedGeometry(&graph, 2, 3, geometry23);

  IncrementalMapperOptions options;
  options.use_structureless_registration = true;
  options.structureless_registration.min_num_inliers_per_edge = 8;
  options.triangulation.min_triangulation_angle_deg = 0.1;
  options.run_local_bundle_adjustment = false;
  options.run_local_retriangulation_filter = false;
  options.run_observation_filter = false;
  const Reconstruction schema = reconstruction;
  MatchGraph invalid_graph = graph;
  for (auto& keypoint : invalid_graph.features.at(3).keypoints) keypoint.y += 150.0F;
  const auto invalid = RegisterNextImage(invalid_graph, 3, options, &reconstruction);
  EXPECT_FALSE(invalid.success);
  EXPECT_TRUE(invalid.structureless_registration.success);
  EXPECT_FALSE(reconstruction.IsImageRegistered(3));
  EXPECT_FALSE(reconstruction.GetImage(3).cam_from_world.has_value());
  EXPECT_TRUE(reconstruction.Points3D().empty());
  reconstruction = schema;
  const IncrementalMapperSummary summary =
      RegisterNextImage(graph, 3, options, &reconstruction);

  ASSERT_TRUE(summary.success) << summary.message;
  EXPECT_TRUE(summary.used_structureless_registration);
  EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  EXPECT_GT(reconstruction.Points3D().size(), 0U);
  EXPECT_EQ(summary.registration.num_inliers, points.size());
}

TEST(StructurelessRegistrarTest, GeneralizedResectionUsesMatchesWithoutCalibratedPairPoses) {
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700, 500, 400, .04};
  for (const double scale : {.001, 1.0, 1000.0}) {
    SCOPED_TRACE(scale);
    Reconstruction reconstruction;
    reconstruction.AddCamera(camera);
    MatchGraph graph;
    std::vector<Eigen::Vector3d> points;
    for (int row = 0; row < 5; ++row) {
      for (int col = 0; col < 8; ++col) {
        points.emplace_back(scale * (-1.2 + .35 * col), scale * (-.6 + .3 * row),
                            scale * (4 + .08 * col + .03 * row * col));
      }
    }
    const Eigen::Vector3d origin(3 * scale, -2 * scale, scale);
    const std::vector<Eigen::Vector3d> centers = {
        origin, origin + Eigen::Vector3d(2 * scale, 0, 0),
        origin + Eigen::Vector3d(.8 * scale, .6 * scale, 0)};
    Pose3 truth;
    for (ImageId id = 1; id <= 3; ++id) {
      Pose3 pose;
      pose.q_cw = Eigen::Quaterniond(Eigen::AngleAxisd(.04 * id, Eigen::Vector3d::UnitY()));
      pose.t_cw = -(pose.q_cw * centers[id - 1]);
      if (id == 3) truth = pose;
      auto image = MakeImage(id, 0, false);
      FeatureSet features;
      for (std::size_t idx = 0; idx < points.size(); ++idx) {
        const auto pixel = ProjectWorldPoint(camera, pose, points[idx] + origin);
        ASSERT_TRUE(pixel.has_value());
        features.keypoints.push_back({static_cast<float>(pixel->x()), static_cast<float>(pixel->y()), 1, 0});
      }
      SetFeatures(&graph, id, features);
      if (id <= 2) image.cam_from_world = pose;
      reconstruction.AddImage(image);
      if (id <= 2) reconstruction.RegisterImage(id);
    }
    for (ImageId neighbor : {1U, 2U}) {
      TwoViewGeometry geometry;
      geometry.type = neighbor == 1 ? TwoViewGeometry::Type::kUncalibratedFundamental
                                    : TwoViewGeometry::Type::kPlanarHomography;
      for (Point2DIdx idx = 0; idx < points.size(); ++idx) geometry.inlier_matches.push_back({idx, idx});
      SetVerifiedGeometry(&graph, neighbor, 3, geometry);
    }
    IncrementalMapperOptions options;
    options.use_structureless_registration = true;
    options.registration.min_num_inliers = 30;
    options.registration.min_num_correspondences = 30;
    options.run_local_bundle_adjustment = false;
    options.run_observation_filter = false;
    options.run_local_retriangulation_filter = false;
    const auto estimated = EstimateStructurelessPoseFromMatches(
        graph, 3, reconstruction, options.registration);
    if (estimated.message == "Generalized structure-less resection requires PoseLib") GTEST_SKIP();
    ASSERT_TRUE(estimated.success) << estimated.message;
    EXPECT_GE(estimated.num_inlier_correspondences, 60U);
    EXPECT_GE(estimated.num_unique_inlier_points, 30U);
    EXPECT_LT((estimated.pose.CameraCenter() - truth.CameraCenter()).norm() / scale, 1e-3);
    // A generalized resection constrains one query against several cameras;
    // it does not require fifteen inliers in every constituent pair.
    // Keep one/eight exact constraints at the second center and the dominant edge.
    for (const std::size_t weak_count : {1U, 8U}) {
      auto asymmetric = graph;
      auto& weak_edge = asymmetric.verified_geometries.at(ImagePair::Ordered(2, 3)).inlier_matches;
      weak_edge.resize(weak_count);
      auto weak_options = options;
      weak_options.registration.min_num_inliers = 20;
      weak_options.registration.min_num_correspondences = 20;
      const auto weak_estimated = EstimateStructurelessPoseFromMatches(
          asymmetric, 3, reconstruction, weak_options.registration);
      ASSERT_TRUE(weak_estimated.success) << weak_estimated.message;
      EXPECT_GE(weak_estimated.num_inlier_correspondences, 40U);
      EXPECT_EQ(weak_estimated.num_consistent_hypotheses, 2U);
      EXPECT_LT((weak_estimated.pose.CameraCenter() - truth.CameraCenter()).norm() / scale, 1e-3);
      auto asymmetric_model = reconstruction;
      const auto weak_registered = RegisterNextImage(asymmetric, 3, weak_options, &asymmetric_model);
      ASSERT_TRUE(weak_registered.success) << weak_registered.message;
      EXPECT_TRUE(weak_registered.structureless_registration.used_generalized_pose);
    }
    auto duplicates = graph;
    for (auto& [pair, geometry] : duplicates.verified_geometries) {
      geometry.inlier_matches.assign(120, FeatureMatch{0, 0});
    }
    EXPECT_FALSE(EstimateStructurelessPoseFromMatches(
        duplicates, 3, reconstruction, options.registration).success);
    auto panoramic = reconstruction;
    panoramic.GetImage(2).cam_from_world = panoramic.GetImage(1).cam_from_world;
    EXPECT_FALSE(EstimateStructurelessPoseFromMatches(
        graph, 3, panoramic, options.registration).success);
    auto outliers = graph;
    auto& bad_matches = outliers.verified_geometries.at(ImagePair::Ordered(2, 3)).inlier_matches;
    for (auto& match : bad_matches) match.idx2 = (match.idx2 + 17) % points.size();
    EXPECT_FALSE(EstimateStructurelessPoseFromMatches(
        outliers, 3, reconstruction, options.registration).success);
    EXPECT_FALSE(reconstruction.IsImageRegistered(3));
    const auto registered = RegisterNextImage(graph, 3, options, &reconstruction);
    ASSERT_TRUE(registered.success) << registered.message;
    EXPECT_TRUE(registered.structureless_registration.used_generalized_pose);
    EXPECT_TRUE(reconstruction.IsImageRegistered(3));
  }
}

TEST(StructurelessRegistrarTest, IntersectsRaysIndependentlyOfWorldScale) {
  for (const double scale : {0.001, 1.0, 10.0, 1000.0}) {
    SCOPED_TRACE(scale);
    Reconstruction reconstruction;
    Camera camera;
    camera.id = 1;
    camera.model_id = CameraModelId::kPinhole;
    camera.width = 640;
    camera.height = 480;
    camera.params = {500, 500, 320, 240};
    reconstruction.AddCamera(camera);
    reconstruction.AddImage(MakeImage(1, 0.0, true));
    reconstruction.AddImage(MakeImage(4, 2.0 * scale, true));
    reconstruction.AddImage(MakeImage(3, 0.0, false));
    reconstruction.RegisterImage(1);
    reconstruction.RegisterImage(4);
    MatchGraph graph;
    auto edge13 = RelativeGeometry(-1);
    auto edge23 = RelativeGeometry(1);
    if (!edge13.cam2_from_cam1) edge13.cam2_from_cam1.emplace();
    edge13.cam2_from_cam1->t_cw = Eigen::Vector3d(-1, -0.6, 0).normalized();
    if (!edge23.cam2_from_cam1) edge23.cam2_from_cam1.emplace();
    edge23.cam2_from_cam1->t_cw = Eigen::Vector3d(1, -0.6, 0).normalized();
    SetVerifiedGeometry(&graph, 1, 3, edge13);
    // The target is the first image of the ordered second pair, exercising
    // inverse relative-pose composition without violating the graph contract.
    edge23.cam2_from_cam1 = edge23.cam2_from_cam1->Inverse();
    SetVerifiedGeometry(&graph, 3, 4, edge23);
    const auto summary = EstimateStructurelessPose(graph, 3, reconstruction);
    ASSERT_TRUE(summary.success) << summary.message;
    EXPECT_EQ(summary.num_consistent_hypotheses, 2U);
    EXPECT_LT((summary.pose.CameraCenter() / scale - Eigen::Vector3d(1, 0.6, 0)).norm(), 1e-9);
  }
}

}  // namespace
}  // namespace colite
