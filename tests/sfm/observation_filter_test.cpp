#include "colite/sfm/observation_filter.h"

#include <string>
#include <limits>

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
  return camera;
}

Pose3 CameraAt(const Eigen::Vector3d& center_world) {
  Pose3 pose;
  pose.q_cw = Eigen::Quaterniond::Identity();
  pose.t_cw = -center_world;
  return pose;
}

Image MakeRegisteredImage(ImageId image_id, const Pose3& cam_from_world) {
  Image image;
  image.id = image_id;
  image.camera_id = 1;
  image.name = "image_" + std::to_string(image_id) + ".jpg";
  image.cam_from_world = cam_from_world;
  image.registered = true;
  return image;
}

void AddProjectedObservation(const Camera& camera,
                             const Pose3& pose,
                             const Eigen::Vector3d& point,
                             Image* image) {
  const auto pixel = ProjectWorldPoint(camera, pose, point);
  ASSERT_TRUE(pixel.has_value());
  image->points2d.push_back(
      {*pixel, static_cast<Point2DIdx>(image->points2d.size()),
       kInvalidPoint3DId});
}

TEST(ObservationFilterTest,
     NegativeDepthCleanupPreservesPositiveAndUnregisteredObservations) {
  Reconstruction model;
  model.AddCamera(MakeCamera());
  for (ImageId id = 1; id <= 4; ++id) {
    auto image = MakeRegisteredImage(id, CameraAt(
        Eigen::Vector3d(0, 0, id == 3 ? 6 : 0)));
    image.registered = id != 4;
    for (Point2DIdx idx = 0; idx < 7; ++idx)
      image.points2d.push_back({Eigen::Vector2d(-100, -200), idx, kInvalidPoint3DId});
    model.AddImage(image);
  }
  const auto add = [&](const double z, const Point2DIdx idx,
                       const std::vector<ImageId>& images) {
    Track track;
    for (const auto id : images) track.AddElement(id, idx);
    return model.AddPoint3D(Eigen::Vector3d(0, 0, z), track);
  };
  const double epsilon = std::numeric_limits<double>::epsilon();
  const auto partial = add(4, 0, {1, 2, 3});
  const auto short_track = add(4, 1, {1, 3});
  const auto positive = add(8, 2, {1, 2, 3});
  const auto zero = add(0, 3, {1, 2});
  const auto boundary = add(epsilon, 4, {1, 2});
  const auto unregistered = add(-1, 5, {4});
  const auto sub_epsilon = add(epsilon / 2, 6, {1, 2});
  const auto summary = FilterObservationsWithNegativeDepth(&model);
  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_filtered_observations, 4U);
  EXPECT_EQ(summary.num_removed_points, 3U);
  EXPECT_EQ(summary.num_remaining_observations, 7U);
  EXPECT_EQ(model.Points3D().size(), 4U);
  EXPECT_EQ(model.GetPoint3D(partial).track.Length(), 2U);
  EXPECT_EQ(model.GetPoint3D(positive).track.Length(), 3U);
  EXPECT_DOUBLE_EQ(model.GetPoint3D(boundary).xyz.z(), epsilon);
  EXPECT_DOUBLE_EQ(model.GetPoint3D(unregistered).xyz.z(), -1);
  for (const auto id : {short_track, zero, sub_epsilon})
    EXPECT_FALSE(model.ExistsPoint3D(id));
  EXPECT_FALSE(model.GetImage(1).points2d[1].HasPoint3D());
  EXPECT_FALSE(model.GetImage(3).points2d[0].HasPoint3D());
  for (const auto& [id, point] : model.Points3D()) {
    for (const auto& link : point.track.elements)
      EXPECT_EQ(model.GetImage(link.image_id).points2d[link.point2d_idx].point3d_id, id);
  }
  const auto repeat = FilterObservationsWithNegativeDepth(&model);
  EXPECT_TRUE(repeat.success);
  EXPECT_EQ(repeat.num_filtered_observations, 0U);
  EXPECT_EQ(repeat.num_removed_points, 0U);
  EXPECT_EQ(repeat.num_remaining_observations, 7U);
  EXPECT_FALSE(FilterObservationsWithNegativeDepth(nullptr).success);
}

TEST(ObservationFilterTest, PixelNoiseDoesNotInventParallaxForCoincidentCameras) {
  const auto camera = MakeCamera();
  Reconstruction model; model.AddCamera(camera);
  const Eigen::Vector3d point(0,0,4);
  for (ImageId id : {1,2}) {
    auto image = MakeRegisteredImage(id, CameraAt({0,0,0}));
    AddProjectedObservation(camera, *image.cam_from_world, point, &image);
    image.points2d[0].xy.x() += id == 1 ? 20 : -20;
    model.AddImage(image);
  }
  Track track; track.AddElement(1,0); track.AddElement(2,0);
  const auto id = model.AddPoint3D(point, track);
  ObservationFilterOptions options;
  options.max_reprojection_error_px = 25;
  options.min_triangulation_angle_deg = 1.5;
  const auto summary = FilterObservationsAndPoints(options, &model);
  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_reprojection_observations, 0U);
  EXPECT_EQ(summary.num_low_angle_points, 1U);
  EXPECT_FALSE(model.ExistsPoint3D(id));
}

TEST(ObservationFilterTest, GeometricParallaxDoesNotRequireInverseCameraRay) {
  auto camera = MakeCamera();
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700,500,400,-1};
  Reconstruction model; model.AddCamera(camera);
  const Eigen::Vector3d point(1.2,0,1);
  for (ImageId id : {1,2}) {
    auto image = MakeRegisteredImage(id, CameraAt({.1*(id-1),0,0}));
    AddProjectedObservation(camera, *image.cam_from_world, point, &image);
    model.AddImage(image);
  }
  ASSERT_FALSE(TryPixelToBearing(camera, model.GetImage(1).points2d[0].xy));
  Track track; track.AddElement(1,0); track.AddElement(2,0);
  const auto id = model.AddPoint3D(point, track);
  ObservationFilterOptions options;
  options.max_reprojection_error_px = 1e-6;
  options.min_triangulation_angle_deg = 1;
  const auto summary = FilterObservationsAndPoints(options, &model);
  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_removed_points, 0U);
  ASSERT_TRUE(model.ExistsPoint3D(id));
  EXPECT_EQ(model.GetPoint3D(id).track.Length(), 2U);
}

TEST(ObservationFilterTest,
     RemovesHighErrorObservationsAndDeletesInvalidPoints) {
  const Camera camera = MakeCamera();
  const Pose3 pose1 = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 pose2 = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 pose3 = CameraAt(Eigen::Vector3d(0.4, 0.2, 0.0));
  const Eigen::Vector3d point_a(-0.2, 0.1, 4.2);
  const Eigen::Vector3d point_b(0.35, -0.15, 4.6);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  Image image1 = MakeRegisteredImage(1, pose1);
  Image image2 = MakeRegisteredImage(2, pose2);
  Image image3 = MakeRegisteredImage(3, pose3);
  AddProjectedObservation(camera, pose1, point_a, &image1);
  AddProjectedObservation(camera, pose2, point_a, &image2);
  AddProjectedObservation(camera, pose3, point_a, &image3);
  AddProjectedObservation(camera, pose1, point_b, &image1);
  AddProjectedObservation(camera, pose2, point_b, &image2);

  // 第三个观测明显偏离，用来模拟局部 BA 后仍然不可信的外点观测。
  image3.points2d[0].xy += Eigen::Vector2d(80.0, -40.0);
  // 第二个点只剩两条观测，其中一条是外点，过滤后 track 长度不足，应删除整点。
  image2.points2d[1].xy += Eigen::Vector2d(-90.0, 60.0);

  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);
  reconstruction.AddImage(image3);

  Track track_a;
  track_a.AddElement(1, 0);
  track_a.AddElement(2, 0);
  track_a.AddElement(3, 0);
  const Point3DId point_a_id = reconstruction.AddPoint3D(point_a, track_a);

  Track track_b;
  track_b.AddElement(1, 1);
  track_b.AddElement(2, 1);
  const Point3DId point_b_id = reconstruction.AddPoint3D(point_b, track_b);

  ObservationFilterOptions options;
  options.max_reprojection_error_px = 2.0;
  options.min_track_length = 2;
  options.min_triangulation_angle_deg = 0.1;

  const ObservationFilterSummary summary =
      FilterObservationsAndPoints(options, &reconstruction);

  EXPECT_EQ(summary.num_removed_observations, 2U);
  EXPECT_EQ(summary.num_removed_points, 1U);
  EXPECT_TRUE(reconstruction.ExistsPoint3D(point_a_id));
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point_b_id));
  EXPECT_FALSE(reconstruction.GetImage(3).points2d[0].HasPoint3D());
  EXPECT_FALSE(reconstruction.GetImage(1).points2d[1].HasPoint3D());
  EXPECT_FALSE(reconstruction.GetImage(2).points2d[1].HasPoint3D());
  EXPECT_EQ(reconstruction.GetPoint3D(point_a_id).track.Length(), 2U);
}

TEST(ObservationFilterTest, FiltersOnlyPointsObservedBySelectedImages) {
  const Camera camera = MakeCamera();
  const Pose3 pose1 = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 pose2 = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 pose3 = CameraAt(Eigen::Vector3d(2.0, 0.0, 0.0));
  const Pose3 pose4 = CameraAt(Eigen::Vector3d(3.0, 0.0, 0.0));
  const Eigen::Vector3d point_a(0.0, 0.1, 5.0);
  const Eigen::Vector3d point_b(1.0, -0.1, 5.5);

  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  Image image1 = MakeRegisteredImage(1, pose1);
  Image image2 = MakeRegisteredImage(2, pose2);
  Image image3 = MakeRegisteredImage(3, pose3);
  Image image4 = MakeRegisteredImage(4, pose4);
  AddProjectedObservation(camera, pose1, point_a, &image1);
  AddProjectedObservation(camera, pose2, point_a, &image2);
  AddProjectedObservation(camera, pose3, point_b, &image3);
  AddProjectedObservation(camera, pose4, point_b, &image4);
  image2.points2d[0].xy += Eigen::Vector2d(80.0, 0.0);
  image4.points2d[0].xy += Eigen::Vector2d(-80.0, 0.0);
  reconstruction.AddImage(image1);
  reconstruction.AddImage(image2);
  reconstruction.AddImage(image3);
  reconstruction.AddImage(image4);

  Track track_a;
  track_a.AddElement(1, 0);
  track_a.AddElement(2, 0);
  const Point3DId point_a_id = reconstruction.AddPoint3D(point_a, track_a);
  Track track_b;
  track_b.AddElement(3, 0);
  track_b.AddElement(4, 0);
  const Point3DId point_b_id = reconstruction.AddPoint3D(point_b, track_b);

  ObservationFilterOptions options;
  options.max_reprojection_error_px = 2.0;
  options.min_track_length = 2;
  options.min_triangulation_angle_deg = 0.1;
  const ObservationFilterSummary summary =
      FilterObservationsAndPointsInImages(options, {1, 2}, &reconstruction);

  EXPECT_EQ(summary.num_removed_observations, 1U);
  EXPECT_EQ(summary.num_removed_points, 1U);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point_a_id));
  EXPECT_TRUE(reconstruction.ExistsPoint3D(point_b_id));
  EXPECT_TRUE(reconstruction.GetImage(4).points2d[0].HasPoint3D());

  // A changed point outside the local images must also be filtered; deleted
  // IDs in the change collection are harmless and are not counted twice.
  const std::set<Point3DId> changed = {point_a_id, point_b_id};
  const auto changed_summary = FilterObservationsAndPointsInImages(
      options, {1, 2}, &reconstruction, &changed);
  EXPECT_EQ(changed_summary.num_removed_observations, 1U);
  EXPECT_EQ(changed_summary.num_removed_points, 1U);
  EXPECT_FALSE(reconstruction.ExistsPoint3D(point_b_id));
}

}  // namespace
}  // namespace colite
