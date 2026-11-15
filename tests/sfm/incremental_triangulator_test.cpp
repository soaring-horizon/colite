#include "colite/sfm/incremental_triangulator.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "colite/core/geometry.h"
#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/sfm/image_registrar.h"
#include "colite/sfm/incremental_initializer.h"

namespace colite {
namespace {

TEST(IncrementalTriangulatorCacheTest, PreservesSparseIdsAndNeighborOrderInCsr) {
  MatchGraph graph;
  graph.features[7].keypoints.resize(4);
  graph.features[42].keypoints.resize(3);
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  geometry.inlier_matches = {{2, 1}, {0, 2}, {2, 1}, {kInvalidPoint2DIdx, 0}};
  SetVerifiedGeometry(&graph, 7, 42, geometry);
  // Absent feature data and an out-of-range row must keep the same graph
  // nodes as before. Pose/feature bounds are checked by triangulation callers.
  geometry.inlier_matches = {{2, 5}, {6, 1}};
  SetVerifiedGeometry(&graph, 7, 1001, geometry);
  geometry.type = TwoViewGeometry::Type::kDegenerate;
  geometry.inlier_matches = {{0, 0}};
  SetVerifiedGeometry(&graph, 42, 1001, geometry);
  const IncrementalTriangulatorCache cache(graph);
  const auto& adjacency = cache.correspondence_adjacency;
  using Key = IncrementalTriangulatorCache::ObservationKey;
  const auto collect = [&](const Key& key) {
    const auto neighbors = adjacency.Find(key);
    EXPECT_TRUE(neighbors.has_value());
    return neighbors ? std::vector<Key>(neighbors->begin(), neighbors->end())
                     : std::vector<Key>{};
  };
  EXPECT_EQ(collect({7, 2}), (std::vector<Key>{{42, 1}, {42, 1}, {1001, 5}}));
  EXPECT_EQ(collect({42, 1}), (std::vector<Key>{{7, 2}, {7, 2}}));
  EXPECT_EQ(collect({7, 0}), (std::vector<Key>{{42, 2}}));
  EXPECT_EQ(collect({1001, 1}), (std::vector<Key>{{7, 6}}));
  EXPECT_TRUE(collect({7, 1}).empty());
  EXPECT_TRUE(collect({42, 0}).empty());
  EXPECT_FALSE(adjacency.Find({8, 0}).has_value());
  EXPECT_FALSE(adjacency.Find({7, 7}).has_value());
  EXPECT_EQ(adjacency.observations.size(), 16U);
  EXPECT_EQ(adjacency.edge_offsets.size(), 17U);
  EXPECT_EQ(adjacency.edges.size(), 10U);
  for (const auto index : adjacency.edges) EXPECT_LT(index, adjacency.observations.size());
}

TEST(IncrementalTriangulatorCacheTest, EmptyGraphAndCopiesKeepValidViews) {
  const IncrementalTriangulatorCache empty(MatchGraph{});
  EXPECT_EQ(empty.correspondence_adjacency.edge_offsets.size(), 1U);
  EXPECT_FALSE(empty.correspondence_adjacency.Find({1, 0}).has_value());
  MatchGraph graph;
  graph.features[1].keypoints.resize(2);
  const IncrementalTriangulatorCache original(graph);
  const auto copied = original;
  const auto neighbors = copied.correspondence_adjacency.Find({1, 1});
  ASSERT_TRUE(neighbors.has_value());
  EXPECT_EQ(neighbors->begin(), neighbors->end());
  EXPECT_TRUE(neighbors->empty());
}

class RecordingDescriptorSource final : public DescriptorSource {
 public:
  explicit RecordingDescriptorSource(DescriptorMatrix values)
      : values_(std::move(values)) {}
  Eigen::Index Rows() const override { return values_.rows(); }
  Eigen::Index Cols() const override { return values_.cols(); }
  DescriptorMatrix ReadRows(const std::vector<Point2DIdx>& indices) const override {
    requested.insert(requested.end(), indices.begin(), indices.end());
    DescriptorMatrix result(indices.size(), values_.cols());
    for (std::size_t i = 0; i < indices.size(); ++i) {
      result.row(i) = values_.row(indices[i]);
    }
    return result;
  }
  mutable std::vector<Point2DIdx> requested;

 private:
  DescriptorMatrix values_;
};

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

TEST(IncrementalTriangulatorTest,
     CreatesPointFromTransitiveMultiViewCorrespondences) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.1, -0.2, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(static_cast<double>(image_id - 1), 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry({{0, 0}}));
  IncrementalTriangulatorOptions options;
  options.max_transitivity = 2;
  options.min_triangulation_angle_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, 1U);
  EXPECT_EQ(summary.num_created_observations, 3U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_EQ(reconstruction.Points3D().begin()->second.track.Length(), 3U);
}

TEST(IncrementalTriangulatorTest,
     PoseGuidedCompletionRequiresMultiDescriptorAgreement) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(0.4 * static_cast<double>(image_id - 1), 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back(
        {*pixel + (image_id == 3 ? Eigen::Vector2d(1.0, 0.0)
                                 : Eigen::Vector2d::Zero()),
         0, kInvalidPoint3DId});
    if (image_id == 3) {
      image.points2d.push_back({*pixel, 1, kInvalidPoint3DId});
      image.points2d.push_back({*pixel + Eigen::Vector2d(100.0, 0.0),
                                2, kInvalidPoint3DId});
    }
    reconstruction.AddImage(image);

    FeatureSet features;
    features.keypoints.push_back(
        {static_cast<float>(image.points2d[0].xy.x()),
         static_cast<float>(image.points2d[0].xy.y()), 1.0F, 0.0F});
    if (image_id == 3) {
      features.keypoints.push_back(
          {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
           1.0F, 0.0F});
      features.keypoints.push_back(
          {static_cast<float>(pixel->x() + 100.0),
           static_cast<float>(pixel->y()), 1.0F, 0.0F});
    }
    features.descriptors = DescriptorMatrix::Zero(
        static_cast<Eigen::Index>(features.keypoints.size()), 128);
    features.descriptors(0, 0) = 1.0F;
    if (image_id == 3) {
      features.descriptors(1, 1) = 1.0F;
    }
    SetFeatures(&graph, image_id, features);
  }

  Track track;
  track.AddElement(1, 0);
  track.AddElement(2, 0);
  const Point3DId point3d_id =
      reconstruction.AddPoint3D(point_world, track);
  Reconstruction deferred_reconstruction = reconstruction;
  IncrementalTriangulatorOptions options;
  const IncrementalTriangulatorSummary summary =
      CompleteTracksByPoseAndDescriptors(graph, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id).track.Length(), 3U);
  EXPECT_EQ(reconstruction.GetImage(3).points2d[0].point3d_id, point3d_id);
  EXPECT_FALSE(reconstruction.GetImage(3).points2d[1].HasPoint3D());

  std::map<ImageId, std::shared_ptr<RecordingDescriptorSource>> readers;
  for (auto& [image_id, features] : graph.features) {
    readers[image_id] = std::make_shared<RecordingDescriptorSource>(
        std::move(features.descriptors));
    features.descriptors.resize(0, 0);
    features.descriptor_source = readers[image_id];
  }
  const auto deferred_summary = CompleteTracksByPoseAndDescriptors(
      graph, options, &deferred_reconstruction);
  EXPECT_EQ(deferred_summary.num_completed_observations,
            summary.num_completed_observations);
  EXPECT_EQ(deferred_reconstruction.GetPoint3D(point3d_id).track.Length(), 3U);
  EXPECT_EQ(deferred_reconstruction.GetImage(3).points2d[0].point3d_id, point3d_id);
  EXPECT_FALSE(deferred_reconstruction.GetImage(3).points2d[1].HasPoint3D());
  EXPECT_FALSE(deferred_reconstruction.GetImage(3).points2d[2].HasPoint3D());
  EXPECT_EQ(readers.at(1)->requested, (std::vector<Point2DIdx>{0}));
  EXPECT_EQ(readers.at(2)->requested, (std::vector<Point2DIdx>{0}));
  EXPECT_EQ(readers.at(3)->requested, (std::vector<Point2DIdx>{0, 1}));
}

TEST(IncrementalTriangulatorTest,
     PoseGuidedCompletionDeduplicatesLongTrackReadsAcrossTargets) {
  Reconstruction reconstruction;
  reconstruction.AddCamera(MakeCamera(1));
  const Eigen::Vector3d xyz(0.0, 0.0, 5.0);
  constexpr ImageId kTargetImage = 260;
  constexpr int kTargetFeatures = 600;
  MatchGraph graph;
  Track track;
  std::map<ImageId, std::shared_ptr<RecordingDescriptorSource>> readers;
  for (ImageId id = 1; id <= kTargetImage; ++id) {
    Image image = MakeImage(id, 1);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(0.002 * id, 0, 0));
    const auto xy = *ProjectWorldPoint(reconstruction.GetCamera(1),
                                       *image.cam_from_world, xyz);
    const int count = id == kTargetImage ? kTargetFeatures : 1;
    FeatureSet features;
    for (int row = 0; row < count; ++row) {
      image.points2d.push_back({xy, static_cast<Point2DIdx>(row), kInvalidPoint3DId});
      features.keypoints.push_back({static_cast<float>(xy.x()),
                                   static_cast<float>(xy.y()), 1.0F, 0.0F});
    }
    DescriptorMatrix values = DescriptorMatrix::Zero(count, 128);
    values.col(0).setOnes();
    readers[id] = std::make_shared<RecordingDescriptorSource>(std::move(values));
    features.descriptor_source = readers[id];
    SetFeatures(&graph, id, std::move(features));
    reconstruction.AddImage(std::move(image));
    if (id != kTargetImage) track.AddElement(id, 0);
  }
  const auto point_id = reconstruction.AddPoint3D(xyz, track);
  const auto summary = CompleteTracksByPoseAndDescriptors(graph, {}, &reconstruction);
  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(reconstruction.GetPoint3D(point_id).track.Length(), kTargetImage);
  EXPECT_EQ(reconstruction.GetImage(kTargetImage).points2d[0].point3d_id, point_id);
  for (ImageId id = 1; id < kTargetImage; ++id) {
    EXPECT_EQ(readers.at(id)->requested, (std::vector<Point2DIdx>{0}));
  }
  EXPECT_EQ(readers.at(kTargetImage)->requested.size(), kTargetFeatures);
}

TEST(IncrementalTriangulatorTest,
     MultiViewCreateRejectsAngularOutlier) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const Eigen::Vector3d outlier_world(1.5, 1.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(0.6 * static_cast<double>(image_id - 1), 0.0, 0.0));
    const Eigen::Vector3d& observed_world =
        image_id == 4 ? outlier_world : point_world;
    const auto pixel = ProjectWorldPoint(camera, pose, observed_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }

  MatchGraph graph;
  for (ImageId image_id = 2; image_id <= 4; ++image_id) {
    SetVerifiedGeometry(&graph, 1, image_id,
                        MakeVerifiedGeometry({{0, 0}}));
  }
  IncrementalTriangulatorOptions options;
  options.min_triangulation_angle_deg = 1.0;
  options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, 1U);
  EXPECT_EQ(summary.num_created_observations, 3U);
  EXPECT_FALSE(reconstruction.GetImage(4).points2d[0].HasPoint3D());
}

TEST(IncrementalTriangulatorTest,
     MultiViewCreateKeepsBestConflictingObservationPerImage) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);

  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(0.7 * static_cast<double>(image_id - 1), 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    if (image_id == 2) {
      image.points2d.push_back(
          {*pixel + Eigen::Vector2d(3.0, 0.0), 1, kInvalidPoint3DId});
    }
    reconstruction.AddImage(image);
  }

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2,
                      MakeVerifiedGeometry({{0, 0}, {0, 1}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 4, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.min_triangulation_angle_deg = 0.1;
  options.create_max_angle_error_deg = 2.0;
  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, 1U);
  EXPECT_EQ(summary.num_created_observations, 4U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_EQ(reconstruction.Points3D().begin()->second.track.Length(), 4U);
  EXPECT_TRUE(reconstruction.GetImage(2).points2d[0].HasPoint3D());
  EXPECT_FALSE(reconstruction.GetImage(2).points2d[1].HasPoint3D());
}

TEST(IncrementalTriangulatorTest,
     ContinueChoosesCandidateWithSmallestAngularError) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d correct_world(0.0, 0.0, 5.0);
  const Eigen::Vector3d wrong_world(1.5, 1.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(static_cast<double>(image_id - 1), 0.0, 0.0));
    const Eigen::Vector3d& observed_world =
        image_id == 3 ? wrong_world : correct_world;
    const auto pixel = ProjectWorldPoint(camera, pose, observed_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  Track correct_track;
  correct_track.AddElement(2, 0);
  const Point3DId correct_id =
      reconstruction.AddPoint3D(correct_world, correct_track);
  Track wrong_track;
  wrong_track.AddElement(3, 0);
  reconstruction.AddPoint3D(wrong_world, wrong_track);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));
  IncrementalTriangulatorOptions options;
  options.continue_max_angle_error_deg = 2.0;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id, correct_id);
  EXPECT_EQ(reconstruction.GetPoint3D(correct_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest,
     TriangulatesNewPointsAfterImageRegistration) {
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
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_existing));
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

  ImageRegistrarOptions register_options;
  register_options.min_num_correspondences = 20;
  register_options.min_num_inliers = 20;
  register_options.max_reprojection_error_px = 2.0;
  ASSERT_TRUE(RegisterImageWithExistingPoints(graph, 3, register_options,
                                             &reconstruction)
                  .success);

  const std::size_t num_points_before = reconstruction.Points3D().size();

  IncrementalTriangulatorOptions tri_options;
  tri_options.ignore_two_view_tracks = false;
  tri_options.min_triangulation_angle_deg = 0.1;
  tri_options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 3, tri_options, &reconstruction);

  EXPECT_EQ(summary.image_id, 3U);
  EXPECT_GT(summary.num_created_points, 0U);
  EXPECT_EQ(summary.num_created_observations, summary.num_created_points * 2);
  EXPECT_GT(reconstruction.Points3D().size(), num_points_before);
  EXPECT_GT(reconstruction.GetImage(3).NumPoints3D(),
            register_options.min_num_inliers);
}

TEST(IncrementalTriangulatorTest,
     CompletesTracksForExistingPointsAfterCreatingNewPoints) {
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
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_existing));
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

  ImageRegistrarOptions register_options;
  register_options.min_num_correspondences = 20;
  register_options.min_num_inliers = 20;
  register_options.max_reprojection_error_px = 2.0;
  ASSERT_TRUE(RegisterImageWithExistingPoints(graph, 3, register_options,
                                             &reconstruction)
                  .success);

  IncrementalTriangulatorOptions tri_options;
  tri_options.ignore_two_view_tracks = false;
  tri_options.min_triangulation_angle_deg = 0.1;
  tri_options.create_max_angle_error_deg = 1.0;
  ASSERT_GT(TriangulateRegisteredImage(graph, 3, tri_options, &reconstruction)
                .num_created_points,
            0U);
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry(matches_all));

  const std::size_t image1_points_before =
      reconstruction.GetImage(1).NumPoints3D();

  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 1, tri_options, &reconstruction);

  EXPECT_EQ(summary.image_id, 1U);
  EXPECT_GT(summary.num_completed_observations, 0U);
  EXPECT_GT(reconstruction.GetImage(1).NumPoints3D(), image1_points_before);

  std::size_t num_tracks_with_three_observations = 0;
  for (const auto& [point3d_id, point3d] : reconstruction.Points3D()) {
    (void)point3d_id;
    if (point3d.track.Length() >= 3) {
      ++num_tracks_with_three_observations;
    }
  }
  EXPECT_GT(num_tracks_with_three_observations, 0U);
}

TEST(IncrementalTriangulatorTest,
     ContinuesExistingPointBeforeCreatingDuplicateTrack) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(0.5, 0.1, 0.0));
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  const auto pixel3 = ProjectWorldPoint(camera, cam3_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());
  ASSERT_TRUE(pixel3.has_value());

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

  Image image3 = MakeImage(3, 1);
  image3.registered = true;
  image3.cam_from_world = cam3_from_world;
  image3.points2d.push_back({*pixel3, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image3);

  Track track2;
  track2.AddElement(2, 0);
  const Point3DId existing_point3d_id =
      reconstruction.AddPoint3D(point_world, track2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.ignore_two_view_tracks = false;
  options.min_triangulation_angle_deg = 0.1;
  options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 3, options, &reconstruction);

  EXPECT_EQ(summary.num_created_points, 0U);
  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(summary.num_merged_points, 0U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_FALSE(reconstruction.GetImage(1).points2d[0].HasPoint3D());
  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id,
            existing_point3d_id);
  EXPECT_EQ(reconstruction.GetImage(3).points2d[0].point3d_id,
            existing_point3d_id);
  EXPECT_EQ(reconstruction.GetPoint3D(existing_point3d_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest,
     GlobalCompletionPropagatesToNeighborEndpoint) {
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

  Track track;
  track.AddElement(2, 0);
  const Point3DId point3d_id =
      reconstruction.AddPoint3D(point_world, track);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.complete_neighbor_observations = true;
  options.create_max_angle_error_deg = 1.0;
  auto disabled = reconstruction;
  options.complete_max_transitivity = 0;
  EXPECT_EQ(CompleteTracksForRegisteredImage(graph, 2, options, &disabled)
                .num_completed_observations, 0U);
  EXPECT_EQ(CompleteAndMergeAllTracks(graph, options, &disabled)
                .num_completed_observations, 0U);
  EXPECT_FALSE(disabled.GetImage(1).points2d[0].HasPoint3D());
  options.complete_max_transitivity = 1;
  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 2, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id, point3d_id);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest, IgnoresPureTwoViewTrackByDefault) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 2; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(static_cast<double>(image_id - 1), 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.min_triangulation_angle_deg = 0.1;
  EXPECT_EQ(TriangulateRegisteredImage(graph, 1, options, &reconstruction)
                .num_created_points,
            0U);
  EXPECT_TRUE(reconstruction.Points3D().empty());

  options.ignore_two_view_tracks = false;
  EXPECT_EQ(TriangulateRegisteredImage(graph, 1, options, &reconstruction)
                .num_created_points,
            1U);
}

TEST(IncrementalTriangulatorTest, MergesMatchedExistingPointsIntoLongerTrack) {
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
  const Point3DId point3d_id1 =
      reconstruction.AddPoint3D(point_world, track1);

  Track track2;
  track2.AddElement(2, 0);
  const Point3DId point3d_id2 =
      reconstruction.AddPoint3D(point_world, track2);
  ASSERT_NE(point3d_id1, point3d_id2);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.create_max_angle_error_deg = 1.0;

  options.complete_max_transitivity = 0;  // Merging is independent of completion.
  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.image_id, 1U);
  EXPECT_EQ(summary.num_merged_points, 1U);
  EXPECT_EQ(summary.num_merged_observations, 2U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  const Point3DId merged_id = reconstruction.GetImage(1).points2d[0].point3d_id;
  EXPECT_EQ(reconstruction.GetImage(2).points2d[0].point3d_id, merged_id);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(merged_id));
  EXPECT_EQ(reconstruction.GetPoint3D(merged_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest,
     MergeUsesJointMultiViewTriangulationWhenWeightedPositionFails) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(0.5 * static_cast<double>(image_id - 1), 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }

  Track first_track;
  first_track.AddElement(1, 0);
  first_track.AddElement(2, 0);
  reconstruction.AddPoint3D(point_world + Eigen::Vector3d(0.15, 0.0, 0.0),
                            first_track);
  Track second_track;
  second_track.AddElement(3, 0);
  second_track.AddElement(4, 0);
  reconstruction.AddPoint3D(point_world + Eigen::Vector3d(0.15, 0.0, 0.0),
                            second_track);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry({{0, 0}}));
  IncrementalTriangulatorOptions options;
  options.merge_max_reprojection_error_px = 1.0;
  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 2, options, &reconstruction);

  EXPECT_EQ(summary.num_merged_points, 1U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_EQ(reconstruction.Points3D().begin()->second.track.Length(), 4U);
  EXPECT_NEAR(reconstruction.Points3D().begin()->second.xyz.x(), 0.0, 1e-6);
}

TEST(IncrementalTriangulatorTest,
     MergePrunesSingleConflictingObservation) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(0.5 * static_cast<double>(image_id - 1), 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    if (image_id == 4) {
      image.points2d[0].xy += Eigen::Vector2d(80.0, 40.0);
    }
    reconstruction.AddImage(image);
  }

  Track first_track;
  first_track.AddElement(1, 0);
  first_track.AddElement(2, 0);
  reconstruction.AddPoint3D(point_world, first_track);
  Track second_track;
  second_track.AddElement(3, 0);
  second_track.AddElement(4, 0);
  reconstruction.AddPoint3D(point_world, second_track);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry({{0, 0}}));
  IncrementalTriangulatorOptions options;
  options.merge_max_reprojection_error_px = 1.0;
  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 2, options, &reconstruction);

  EXPECT_EQ(summary.num_merged_points, 1U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_EQ(reconstruction.Points3D().begin()->second.track.Length(), 3U);
  EXPECT_FALSE(reconstruction.GetImage(4).points2d[0].HasPoint3D());
}

TEST(IncrementalTriangulatorTest,
     GlobalCompletionTransitivelyCompletesAndMergesTracks) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(static_cast<double>(image_id - 1) * 0.4, 0.0, 0.0));
    const auto pixel = ProjectWorldPoint(camera, pose, point_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }

  Track first_track;
  first_track.AddElement(1, 0);
  const Point3DId first_id =
      reconstruction.AddPoint3D(point_world, first_track);
  Track second_track;
  second_track.AddElement(4, 0);
  reconstruction.AddPoint3D(point_world, second_track);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 3, 4, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.complete_max_transitivity = 5;
  options.complete_max_reprojection_error_px = 1.0;
  options.merge_max_reprojection_error_px = 1.0;
  const IncrementalTriangulatorSummary summary =
      CompleteAndMergeAllTracks(graph, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 2U);
  EXPECT_EQ(summary.num_merged_points, 1U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(first_id));
  EXPECT_EQ(reconstruction.GetPoint3D(first_id).track.Length(), 4U);
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    EXPECT_EQ(reconstruction.GetImage(image_id).points2d[0].point3d_id,
              first_id);
  }
}

TEST(IncrementalTriangulatorTest,
     GlobalCompletionTraversesAssignedIntermediateObservation) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d first_world(0.0, 0.0, 5.0);
  const Eigen::Vector3d intermediate_world(0.2, 0.0, 5.0);
  Reconstruction reconstruction;
  reconstruction.AddCamera(camera);
  for (ImageId image_id = 1; image_id <= 3; ++image_id) {
    const Pose3 pose = CameraAt(
        Eigen::Vector3d(static_cast<double>(image_id - 1) * 0.4, 0.0, 0.0));
    const Eigen::Vector3d& observed_world =
        image_id == 2 ? intermediate_world : first_world;
    const auto pixel = ProjectWorldPoint(camera, pose, observed_world);
    ASSERT_TRUE(pixel.has_value());
    Image image = MakeImage(image_id, 1);
    image.registered = true;
    image.cam_from_world = pose;
    image.points2d.push_back({*pixel, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }

  Track first_track;
  first_track.AddElement(1, 0);
  const Point3DId first_id =
      reconstruction.AddPoint3D(first_world, first_track);
  Track intermediate_track;
  intermediate_track.AddElement(2, 0);
  const Point3DId intermediate_id =
      reconstruction.AddPoint3D(intermediate_world, intermediate_track);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 2, 3, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.complete_max_transitivity = 2;
  options.complete_max_reprojection_error_px = 1.0;
  options.merge_max_reprojection_error_px = 1.0;
  const IncrementalTriangulatorSummary summary =
      CompleteAndMergeAllTracks(graph, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(summary.num_merged_points, 0U);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(first_id));
  ASSERT_TRUE(reconstruction.ExistsPoint3D(intermediate_id));
  EXPECT_EQ(reconstruction.GetPoint3D(first_id).track.Length(), 2U);
  EXPECT_EQ(reconstruction.GetImage(3).points2d[0].point3d_id, first_id);
}

TEST(IncrementalTriangulatorTest,
     CompletesObservationAfterEarlierNeighborWithoutPointFails) {
  const Camera camera = MakeCamera(1);
  const Eigen::Vector3d point_world(0.0, 0.0, 5.0);
  const Pose3 cam1_from_world = CameraAt(Eigen::Vector3d(0.0, 0.0, 0.0));
  const Pose3 cam2_from_world = CameraAt(Eigen::Vector3d(1.0, 0.0, 0.0));
  const Pose3 cam3_from_world = CameraAt(Eigen::Vector3d(2.0, 0.0, 0.0));
  const auto pixel1 = ProjectWorldPoint(camera, cam1_from_world, point_world);
  const auto pixel2 = ProjectWorldPoint(camera, cam2_from_world, point_world);
  const auto pixel3 = ProjectWorldPoint(camera, cam3_from_world, point_world);
  ASSERT_TRUE(pixel1.has_value());
  ASSERT_TRUE(pixel2.has_value());
  ASSERT_TRUE(pixel3.has_value());

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

  Image image3 = MakeImage(3, 1);
  image3.registered = true;
  image3.cam_from_world = cam3_from_world;
  image3.points2d.push_back({*pixel3, 0, kInvalidPoint3DId});
  reconstruction.AddImage(image3);

  Track track3;
  track3.AddElement(3, 0);
  const Point3DId point3d_id = reconstruction.AddPoint3D(point_world, track3);

  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry(
      {{0, 0}, {kInvalidPoint2DIdx, 0}, {0, kInvalidPoint2DIdx},
       {999999, 0}, {0, 999999}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));

  IncrementalTriangulatorOptions options;
  options.create_max_angle_error_deg = 1.0;

  const IncrementalTriangulatorSummary summary =
      CompleteTracksForRegisteredImage(graph, 1, options, &reconstruction);

  EXPECT_EQ(summary.num_completed_observations, 1U);
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id, point3d_id);
  EXPECT_EQ(reconstruction.GetPoint3D(point3d_id).track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest,
     ExistingSharedPointDoesNotSuppressLaterMergeCandidate) {
  Reconstruction reconstruction;
  const Camera camera = MakeCamera(1);
  reconstruction.AddCamera(camera);
  const Eigen::Vector3d xyz(0.1, -0.2, 5.0);
  for (ImageId id = 1; id <= 3; ++id) {
    Image image = MakeImage(id, camera.id);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(0.5 * (id - 1), 0, 0));
    image.points2d.push_back({*ProjectWorldPoint(camera, *image.cam_from_world,
                                               xyz), 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  Track first;
  first.AddElement(1, 0);
  first.AddElement(2, 0);
  const auto first_id = reconstruction.AddPoint3D(xyz, first);
  Track second;
  second.AddElement(3, 0);
  reconstruction.AddPoint3D(xyz, second);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));
  const auto summary = CompleteTracksForRegisteredImage(graph, 1, {},
                                                       &reconstruction);
  EXPECT_EQ(summary.num_merged_points, 1U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_EQ(reconstruction.GetPoint3D(first_id).track.Length(), 3U);
  EXPECT_EQ(reconstruction.GetImage(3).points2d[0].point3d_id, first_id);
}

TEST(IncrementalTriangulatorTest,
     RejectsTwoConflictingImagesButCanPruneOneOutlier) {
  Reconstruction reconstruction;
  const Camera camera = MakeCamera(1);
  reconstruction.AddCamera(camera);
  const Eigen::Vector3d xyz(0.1, -0.2, 5.0);
  for (ImageId id = 1; id <= 4; ++id) {
    Image image = MakeImage(id, camera.id);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(0.5 * (id - 1), 0, 0));
    const auto xy = *ProjectWorldPoint(camera, *image.cam_from_world, xyz);
    image.points2d.push_back({xy, 0, kInvalidPoint3DId});
    if (id <= 2) {
      image.points2d.push_back({xy + Eigen::Vector2d(40, 0),
                               1, kInvalidPoint3DId});
    }
    reconstruction.AddImage(image);
  }
  Track first;
  first.AddElement(1, 0);
  first.AddElement(2, 0);
  first.AddElement(3, 0);
  const auto first_id = reconstruction.AddPoint3D(xyz, first);
  Track second;
  second.AddElement(1, 1);
  second.AddElement(2, 1);
  second.AddElement(4, 0);
  const auto second_id = reconstruction.AddPoint3D(xyz, second);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 3, 4, MakeVerifiedGeometry({{0, 0}}));
  EXPECT_EQ(CompleteTracksForRegisteredImage(graph, 3, {}, &reconstruction)
                .num_merged_points, 0U);
  EXPECT_EQ(reconstruction.Points3D().size(), 2U);
  ASSERT_TRUE(reconstruction.DeleteObservation(second_id, 2, 1));
  const auto summary = CompleteTracksForRegisteredImage(graph, 3, {},
                                                       &reconstruction);
  EXPECT_EQ(summary.num_merged_points, 1U);
  ASSERT_TRUE(reconstruction.ExistsPoint3D(first_id));
  EXPECT_EQ(reconstruction.GetPoint3D(first_id).track.Length(), 4U);
  EXPECT_FALSE(reconstruction.GetImage(1).points2d[1].HasPoint3D());
}

TEST(IncrementalTriangulatorTest,
     SharedImagesWithinMergeToleranceAreNotRejected) {
  Reconstruction reconstruction;
  const Camera camera = MakeCamera(1);
  reconstruction.AddCamera(camera);
  const Eigen::Vector3d xyz(0.1, -0.2, 5.0);
  for (ImageId id = 1; id <= 2; ++id) {
    Image image = MakeImage(id, camera.id);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(id - 1, 0, 0));
    const auto xy = *ProjectWorldPoint(camera, *image.cam_from_world, xyz);
    image.points2d.push_back({xy, 0, kInvalidPoint3DId});
    image.points2d.push_back({xy + Eigen::Vector2d(3, 0), 1, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  Track first;
  first.AddElement(1, 0);
  first.AddElement(2, 0);
  reconstruction.AddPoint3D(xyz, first);
  Track second;
  second.AddElement(1, 1);
  second.AddElement(2, 1);
  reconstruction.AddPoint3D(xyz, second);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 1}}));
  const auto summary = CompleteTracksForRegisteredImage(graph, 1, {},
                                                       &reconstruction);
  EXPECT_EQ(summary.num_merged_points, 1U);
  ASSERT_EQ(reconstruction.Points3D().size(), 1U);
  EXPECT_EQ(reconstruction.Points3D().begin()->second.track.Length(), 2U);
}

TEST(IncrementalTriangulatorTest, SkipsUnregisteredImage) {
  MatchGraph graph;
  Reconstruction reconstruction;

  const IncrementalTriangulatorSummary summary =
      TriangulateRegisteredImage(graph, 9, {}, &reconstruction);

  EXPECT_EQ(summary.image_id, 9U);
  EXPECT_EQ(summary.num_created_points, 0U);
  EXPECT_EQ(summary.num_created_observations, 0U);
}


TEST(IncrementalTriangulatorTest,
     AssignedReferenceStillCreatesPointFromUnassignedDirectNeighbors) {
  Reconstruction reconstruction;
  const Camera camera = MakeCamera(1);
  reconstruction.AddCamera(camera);
  const Eigen::Vector3d existing(0.2, 0.1, 5.0), missing(0.4, -0.2, 6.0);
  for (ImageId id = 1; id <= 4; ++id) {
    Image image = MakeImage(id, 1);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(id - 1, 0, 0));
    const auto xy = ProjectWorldPoint(camera, *image.cam_from_world,
                                     id == 1 || id == 4 ? existing : missing);
    ASSERT_TRUE(xy.has_value());
    image.points2d.push_back({*xy, 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  Track track;
  track.AddElement(1, 0);
  track.AddElement(4, 0);
  const auto original_id = reconstruction.AddPoint3D(existing, track);
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}, {0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));
  const auto summary = TriangulateRegisteredImage(graph, 1, {}, &reconstruction);
  EXPECT_EQ(summary.num_created_points, 1);
  EXPECT_EQ(summary.num_created_observations, 2);
  EXPECT_EQ(reconstruction.GetImage(1).points2d[0].point3d_id, original_id);
  const auto new_id = reconstruction.GetImage(2).points2d[0].point3d_id;
  EXPECT_NE(new_id, original_id);
  EXPECT_EQ(new_id, reconstruction.GetImage(3).points2d[0].point3d_id);
  EXPECT_LT((reconstruction.GetPoint3D(new_id).xyz - missing).norm(), 1e-8);
  EXPECT_EQ(TriangulateRegisteredImage(graph, 1, {}, &reconstruction).num_created_points, 0);
}

TEST(IncrementalTriangulatorTest,
     RefreshesTriangulationDataAfterPoseAndCalibrationChanges) {
  Reconstruction reconstruction;
  Camera camera = MakeCamera(1);
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {650.0, 500.0, 400.0, 0.08};
  reconstruction.AddCamera(camera);
  const Eigen::Vector3d expected(0.3, -0.2, 5.0);
  for (ImageId id = 1; id <= 3; ++id) {
    Image image = MakeImage(id, camera.id);
    image.registered = true;
    image.cam_from_world = CameraAt(Eigen::Vector3d(id - 1, 0.0, 0.0));
    image.points2d.push_back({*ProjectWorldPoint(camera, *image.cam_from_world,
                                               expected), 0, kInvalidPoint3DId});
    reconstruction.AddImage(image);
  }
  MatchGraph graph;
  SetVerifiedGeometry(&graph, 1, 2, MakeVerifiedGeometry({{0, 0}}));
  SetVerifiedGeometry(&graph, 1, 3, MakeVerifiedGeometry({{0, 0}}));
  IncrementalTriangulatorCache cache(graph);
  for (int pass = 0; pass < 2; ++pass) {
    ASSERT_EQ(TriangulateRegisteredImage(graph, 1, {}, &reconstruction, &cache)
                  .num_created_points, 1U);
    const Point3DId id = reconstruction.GetImage(1).points2d[0].point3d_id;
    EXPECT_LT((reconstruction.GetPoint3D(id).xyz - expected).norm(), 1e-8);
    EXPECT_EQ(reconstruction.GetPoint3D(id).track.Length(), 3U);
    reconstruction.DeletePoint3D(id);
    auto& updated_camera = reconstruction.GetCamera(camera.id);
    updated_camera.params[0] = 850.0;
    updated_camera.params[3] = -0.05;
    for (ImageId image_id = 1; image_id <= 3; ++image_id) {
      auto& image = reconstruction.GetImage(image_id);
      image.cam_from_world->q_cw =
          Eigen::AngleAxisd(0.02 * image_id, Eigen::Vector3d::UnitY());
      image.points2d[0].xy = *ProjectWorldPoint(
          updated_camera, *image.cam_from_world, expected);
    }
  }
}

TEST(IncrementalTriangulatorTest, PartialCameraDomainKeepsValidTrackSupport) {
  Reconstruction model;
  auto camera = MakeCamera(1);
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.params = {700, 500, 400, -1};
  model.AddCamera(camera);
  const Eigen::Vector3d expected(.1, .1, 5);
  MatchGraph graph;
  for (ImageId id = 1; id <= 4; ++id) {
    auto image = MakeImage(id, 1);
    image.registered = true; image.cam_from_world = CameraAt({.2*(id-1), 0, 0});
    const Eigen::Vector2d pixel = id == 4 ? Eigen::Vector2d(0, 0) :
        *ProjectWorldPoint(camera, *image.cam_from_world, expected);
    image.points2d.push_back({pixel, 0, kInvalidPoint3DId});
    model.AddImage(image);
    if (id > 1) SetVerifiedGeometry(&graph, 1, id, MakeVerifiedGeometry({{0, 0}}));
  }
  IncrementalTriangulatorCache cache(graph);
  const auto summary = TriangulateRegisteredImage(graph, 1, {}, &model, &cache);
  ASSERT_EQ(summary.num_created_points, 1U);
  const auto id = model.GetImage(1).points2d[0].point3d_id;
  EXPECT_EQ(model.GetPoint3D(id).track.Length(), 3U);
  EXPECT_LT((model.GetPoint3D(id).xyz-expected).norm(), 1e-8);
  EXPECT_FALSE(model.GetImage(4).points2d[0].HasPoint3D());
  EXPECT_EQ(TriangulateRegisteredImage(graph, 4, {}, &model, &cache).num_created_points, 0U);
}

}  // namespace
}  // namespace colite
