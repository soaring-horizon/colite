#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "colite/controllers/mapper.h"
#include "colite/controllers/recon_sfm_controller.h"
#include "colite/controllers/sequential_matcher.h"
#include "colite/core/camera.h"
#include "colite/core/geometry.h"
#include "colite/core/match_graph.h"
#include "colite/core/parallel.h"
#include "colite/exe/command.h"
#include "colite/exe/sfm_options.h"
#include "colite/exe/compare_colmap.h"
#include "colite/exe/feature_extractor.h"
#include "colite/exe/mapper.h"
#include "colite/exe/sequential_matcher.h"
#include "colite/exe/sfm_options.h"
#include "colite/io/camera_prior.h"
#include "colite/io/sfm_workspace.h"

namespace colite {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream file(path);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

template <typename T>
void WriteBinaryTestValue(std::ofstream* file, const T& value) {
  file->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

FeatureSet MakeTinyFeatures() {
  FeatureSet features;
  features.keypoints = {{10.0F, 20.0F, 1.5F, 0.1F, 2.0F},
                        {30.0F, 40.0F, 2.5F, 0.2F, 3.0F}};
  features.descriptors.resize(2, 128);
  features.descriptors.setZero();
  features.descriptors(0, 0) = 1.0F;
  features.descriptors(1, 127) = 2.0F;
  return features;
}

FeatureSet MakeTinyAlikedFeatures() {
  FeatureSet features = MakeTinyFeatures();
  features.extractor_type = FeatureExtractorType::kAlikedN16Rot;
  features.descriptor_metric = DescriptorMetric::kCosine;
  features.descriptor_normalization = DescriptorNormalization::kL2;
  features.descriptors.resize(2, 4);
  features.descriptors.setZero();
  features.descriptors(0, 0) = 1.0F;
  features.descriptors(1, 1) = 1.0F;
  return features;
}

Pose3 MappingCameraAt(const double x) {
  Pose3 pose;
  pose.t_cw = Eigen::Vector3d(-x, 0.0, 0.0);
  return pose;
}

std::vector<Eigen::Vector3d> MappingScenePoints(const double x_offset) {
  std::vector<Eigen::Vector3d> points;
  for (int row = 0; row < 4; ++row) {
    for (int column = 0; column < 6; ++column) {
      points.emplace_back(x_offset - 1.0 + 0.4 * column,
                          -0.5 + 0.3 * row,
                          4.0 + 0.07 * column + 0.09 * row);
    }
  }
  return points;
}

FeatureSet ProjectMappingFeatures(
    const Camera& camera,
    const Pose3& pose,
    const std::vector<Eigen::Vector3d>& points) {
  FeatureSet features;
  for (const Eigen::Vector3d& point : points) {
    const std::optional<Eigen::Vector2d> pixel =
        ProjectWorldPoint(camera, pose, point);
    EXPECT_TRUE(pixel.has_value());
    features.keypoints.push_back(
        {static_cast<float>(pixel->x()), static_cast<float>(pixel->y()),
         1.0F, 0.0F});
  }
  return features;
}

std::vector<FeatureMatch> IdentityFeatureMatches(const std::size_t count) {
  std::vector<FeatureMatch> matches;
  for (std::size_t idx = 0; idx < count; ++idx) {
    matches.push_back({static_cast<Point2DIdx>(idx),
                       static_cast<Point2DIdx>(idx)});
  }
  return matches;
}

void WriteTextFile(const std::filesystem::path& path,
                   const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  file << contents;
}

void WriteTinyColmapModel(const std::filesystem::path& path,
                          const bool candidate) {
  WriteTextFile(path / "cameras.txt",
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
                "1 PINHOLE 640 480 500 500 320 240\n");
  if (!candidate) {
    WriteTextFile(path / "images.txt",
                  "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                  "# POINTS2D[] as X Y POINT3D_ID\n"
                  "1 1 0 0 0 0 0 0 1 image1.jpg\n"
                  "10 20 1 30 40 2\n"
                  "2 1 0 0 0 1 0 0 1 image2.jpg\n"
                  "15 25 1\n");
    WriteTextFile(path / "points3D.txt",
                  "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                  "1 0 0 5 255 255 255 0.5 1 0 2 0\n"
                  "2 1 0 5 255 255 255 0.7 1 1\n");
    return;
  }

  WriteTextFile(path / "images.txt",
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                "# POINTS2D[] as X Y POINT3D_ID\n"
                "1 1 0 0 0 0.1 0 0 1 image1.jpg\n"
                "10 20 1 30 40 3\n"
                "2 1 0 0 0 1 0 0 1 image2.jpg\n"
                "15 25 1 35 45 3\n"
                "3 1 0 0 0 2 0 0 1 image3.jpg\n"
                "\n");
  WriteTextFile(path / "points3D.txt",
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
                "1 0.1 0 5 255 255 255 0.6 1 0 2 0\n"
                "3 2 0 5 255 255 255 0.9 1 1 2 1\n");
}

TEST(SfmWorkspaceTest, RoundTripsAllCameraModelsThroughColmapTextAndBinary) {
  const std::vector<CameraModelId> models = {
      CameraModelId::kSimplePinhole,
      CameraModelId::kPinhole,
      CameraModelId::kSimpleRadial,
      CameraModelId::kRadial,
      CameraModelId::kOpenCV,
      CameraModelId::kOpenCVFisheye,
      CameraModelId::kFullOpenCV,
      CameraModelId::kFOV,
      CameraModelId::kSimpleRadialFisheye,
      CameraModelId::kRadialFisheye,
      CameraModelId::kThinPrismFisheye,
      CameraModelId::kRadTanThinPrismFisheye};
  std::vector<Camera> cameras;
  Reconstruction reconstruction;
  CameraId camera_id = 1;
  for (const CameraModelId model : models) {
    Camera camera;
    camera.id = camera_id++;
    camera.model_id = model;
    camera.width = 1000;
    camera.height = 800;
    camera.params = InitializeCameraParams(model, 700.0, 1000, 800);
    for (const std::size_t idx :
         GetCameraModelSpec(model).distortion_idxs) {
      camera.params[idx] = model == CameraModelId::kFOV ? 0.2 : 0.001;
    }
    cameras.push_back(camera);
    reconstruction.AddCamera(camera);
  }

  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      "colite_all_camera_model_io_test";
  const std::filesystem::path workspace = root / "workspace";
  const std::filesystem::path colmap = root / "colmap";
  std::filesystem::remove_all(root);
  WriteWorkspaceCameras(workspace, cameras);
  const std::vector<Camera> workspace_cameras =
      ReadWorkspaceCameras(workspace);
  ASSERT_EQ(workspace_cameras.size(), cameras.size());
  for (std::size_t idx = 0; idx < cameras.size(); ++idx) {
    EXPECT_EQ(workspace_cameras[idx].model_id, cameras[idx].model_id);
    EXPECT_EQ(workspace_cameras[idx].params, cameras[idx].params);
  }

  WriteColmapReconstruction(colmap, reconstruction,
                            ColmapModelFormat::kTextAndBinary);
  const Reconstruction loaded = ReadColmapTextReconstruction(colmap);
  const Reconstruction loaded_binary =
      ReadColmapBinaryReconstruction(colmap);
  EXPECT_EQ(DetectColmapModelFormat(colmap), ColmapModelFormat::kBinary);
  ASSERT_EQ(loaded.Cameras().size(), cameras.size());
  ASSERT_EQ(loaded_binary.Cameras().size(), cameras.size());
  for (const Camera& camera : cameras) {
    ASSERT_TRUE(loaded.ExistsCamera(camera.id));
    EXPECT_EQ(loaded.GetCamera(camera.id).model_id, camera.model_id);
    EXPECT_EQ(loaded.GetCamera(camera.id).params, camera.params);
    ASSERT_TRUE(loaded_binary.ExistsCamera(camera.id));
    EXPECT_EQ(loaded_binary.GetCamera(camera.id).model_id, camera.model_id);
    EXPECT_EQ(loaded_binary.GetCamera(camera.id).params, camera.params);
  }
}

TEST(SfmWorkspaceTest, RoundTripsBinaryPosesColorsAndTracks) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 3;
  camera.model_id = CameraModelId::kSimpleRadial;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 320.0, 240.0, 0.01};
  reconstruction.AddCamera(camera);

  for (ImageId image_id : {7U, 9U}) {
    Image image;
    image.id = image_id;
    image.camera_id = camera.id;
    image.name = "folder/image " + std::to_string(image_id) + ".jpg";
    image.registered = true;
    image.cam_from_world = MappingCameraAt(static_cast<double>(image_id));
    image.points2d.push_back(
        {Eigen::Vector2d(10.5 + image_id, 20.25), 0, 42});
    reconstruction.AddImage(std::move(image));
  }
  Point3D point;
  point.id = 42;
  point.xyz = Eigen::Vector3d(1.25, -2.5, 7.75);
  point.color << 17, 101, 249;
  point.reprojection_error = 0.125;
  point.track.AddElement(7, 0);
  point.track.AddElement(9, 0);
  reconstruction.AddPoint3D(std::move(point));

  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_binary_model_test";
  std::filesystem::remove_all(root);
  WriteColmapReconstruction(root, reconstruction,
                            ColmapModelFormat::kTextAndBinary);
  const Reconstruction loaded = ReadColmapReconstruction(root);
  const Reconstruction loaded_text =
      ReadColmapReconstruction(root, ColmapModelFormat::kText);

  ASSERT_EQ(loaded.RegisteredImageIds(), (std::set<ImageId>{7U, 9U}));
  EXPECT_EQ(loaded.GetImage(7).name, "folder/image 7.jpg");
  EXPECT_EQ(loaded_text.GetImage(7).name, "folder/image 7.jpg");
  ASSERT_TRUE(loaded.GetImage(9).cam_from_world.has_value());
  EXPECT_DOUBLE_EQ(loaded.GetImage(9).cam_from_world->t_cw.x(), -9.0);
  ASSERT_TRUE(loaded.ExistsPoint3D(42));
  const Point3D& loaded_point = loaded.GetPoint3D(42);
  EXPECT_EQ(loaded_point.color.x(), 17);
  EXPECT_EQ(loaded_point.color.y(), 101);
  EXPECT_EQ(loaded_point.color.z(), 249);
  EXPECT_EQ(loaded_point.track.elements.size(), 2U);
  EXPECT_EQ(loaded.GetImage(7).points2d[0].point3d_id, 42U);
  EXPECT_DOUBLE_EQ(loaded_point.reprojection_error, 0.125);
  EXPECT_DOUBLE_EQ(loaded_text.GetPoint3D(42).reprojection_error, 0.125);
  std::filesystem::remove_all(root);
}

TEST(SfmWorkspaceTest, ConvertsLargePointIdsAndCrLfWithoutLosingObservations) {
  const auto root = std::filesystem::temp_directory_path() / "colite_model_interop_test";
  std::filesystem::remove_all(root);
  WriteTextFile(root / "input/cameras.txt", "3 PINHOLE 640 480 500 510 320 240\r\n");
  WriteTextFile(root / "input/images.txt",
      "7 0.5 0.5 0.5 0.5 1.25 -2.5 3.75 3 folder/image 7.jpg\r\n"
      "10.5 20.25 9223372036854775815 30.5 40.25 -1\r\n"
      "9 1 0 0 0 -1 0 0 3 empty.jpg\r\n\r\n");
  WriteTextFile(root / "input/points3D.txt",
      "9223372036854775815 1.25 -2.5 7.75 17 101 249 0.125 7 0\r\n");
  std::ostringstream out, err;
  ASSERT_EQ(exe::RunColite({"colite", "model_converter", "--input_path",
                            (root / "input").string(), "--output_path",
                            (root / "output").string(), "--output_type", "both"},
                           out, err), 0) << err.str();
  constexpr Point3DId id = 9223372036854775815ULL;
  for (const auto format : {ColmapModelFormat::kText, ColmapModelFormat::kBinary}) {
    const Reconstruction model = ReadColmapReconstruction(root / "output", format);
    ASSERT_TRUE(model.ExistsPoint3D(id));
    const Image& image = model.GetImage(7);
    EXPECT_EQ(image.name, "folder/image 7.jpg");
    ASSERT_EQ(image.points2d.size(), 2U);
    EXPECT_EQ(image.points2d[0].point3d_id, id);
    EXPECT_FALSE(image.points2d[1].HasPoint3D());
    EXPECT_TRUE(model.GetImage(9).points2d.empty());
    EXPECT_DOUBLE_EQ(image.cam_from_world->q_cw.z(), 0.5);
    EXPECT_DOUBLE_EQ(image.cam_from_world->t_cw.y(), -2.5);
    EXPECT_DOUBLE_EQ(model.GetPoint3D(id).reprojection_error, 0.125);
    EXPECT_EQ(model.GetPoint3D(id).track.elements.size(), 1U);
  }
  std::filesystem::remove_all(root);
}

TEST(SfmWorkspaceTest, TrivialFramesControlPosesAndMvsIndicesInBothEncodings) {
  const auto root = std::filesystem::temp_directory_path() / "colite_trivial_frame_interop_test";
  std::filesystem::remove_all(root);
  WriteTinyColmapModel(root, false);
  const auto legacy = ReadColmapReconstruction(root);
  WriteColmapReconstruction(root, legacy, ColmapModelFormat::kTextAndBinary);
  WriteTextFile(root / "rigs.txt", "1 1 CAMERA 1\n");
  WriteTextFile(root / "frames.txt", "20 1 1 0 0 0 4 0 0 1 CAMERA 1 2\n"
                                         "10 1 1 0 0 0 -3 0 0 1 CAMERA 1 1\n");
  {
    std::ofstream rigs(root / "rigs.bin", std::ios::binary);
    WriteBinaryTestValue(&rigs, std::uint64_t{1});
    WriteBinaryTestValue(&rigs, std::uint32_t{1});
    WriteBinaryTestValue(&rigs, std::uint32_t{1});
    WriteBinaryTestValue(&rigs, std::int32_t{0});
    WriteBinaryTestValue(&rigs, std::uint32_t{1});
    std::ofstream frames(root / "frames.bin", std::ios::binary);
    WriteBinaryTestValue(&frames, std::uint64_t{2});
    for (const std::uint32_t image : {2U, 1U}) {
      WriteBinaryTestValue(&frames, image * 10);
      WriteBinaryTestValue(&frames, std::uint32_t{1});
      for (const double value : {1.0, 0.0, 0.0, 0.0, image == 2 ? 4.0 : -3.0, 0.0, 0.0}) {
        WriteBinaryTestValue(&frames, value);
      }
      WriteBinaryTestValue(&frames, std::uint32_t{1});
      WriteBinaryTestValue(&frames, std::int32_t{0});
      WriteBinaryTestValue(&frames, std::uint32_t{1});
      WriteBinaryTestValue(&frames, std::uint64_t{image});
    }
  }
  for (const auto format : {ColmapModelFormat::kText, ColmapModelFormat::kBinary}) {
    std::vector<ImageId> order{99};
    const auto model = ReadColmapReconstruction(root, format, &order);
    EXPECT_EQ(order, (std::vector<ImageId>{2, 1}));
    EXPECT_DOUBLE_EQ(model.GetImage(2).cam_from_world->t_cw.x(), 4.0);
    EXPECT_DOUBLE_EQ(model.GetImage(1).cam_from_world->t_cw.x(), -3.0);
    // A three-file export is a deliberate independent-camera representation.
    const auto output = root / (format == ColmapModelFormat::kText ? "text" : "binary");
    WriteColmapReconstruction(output, model, format);
    const auto restored = ReadColmapReconstruction(output, format);
    EXPECT_DOUBLE_EQ(restored.GetImage(2).cam_from_world->t_cw.x(), 4.0);
  }
  for (const auto& invalid : {"20 1 1 0 0 0 4 0 0 2 CAMERA 1 2 CAMERA 1 1\n",
                             "20 1 1 0 0 0 4 0 0 1 CAMERA 1 2\n"}) {
    WriteTextFile(root / "frames.txt", invalid);
    EXPECT_THROW(ReadColmapReconstruction(root, ColmapModelFormat::kText), std::runtime_error);
  }
  WriteTextFile(root / "rigs.txt", "1 2 CAMERA 1 CAMERA 2 0\n");
  EXPECT_THROW(ReadColmapReconstruction(root, ColmapModelFormat::kText), std::runtime_error);
  const auto frame_file = root / "frames.bin";
  std::filesystem::resize_file(frame_file, std::filesystem::file_size(frame_file) - 1);
  EXPECT_THROW(ReadColmapReconstruction(root, ColmapModelFormat::kBinary), std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST(SfmWorkspaceTest, ReplacingModelFormatDoesNotLeaveStaleAutoInput) {
  const auto root = std::filesystem::temp_directory_path() / "colite_model_format_switch_test";
  std::filesystem::remove_all(root);
  WriteTinyColmapModel(root, false);
  Reconstruction model = ReadColmapReconstruction(root);
  WriteColmapReconstruction(root, model, ColmapModelFormat::kTextAndBinary);
  WriteTextFile(root / "rigs.txt", "stale metadata");
  WriteTextFile(root / "frames.bin", "stale metadata");
  model.GetPoint3D(1).xyz.x() = 12.5;
  WriteColmapReconstruction(root, model, ColmapModelFormat::kText);
  EXPECT_EQ(DetectColmapModelFormat(root), ColmapModelFormat::kText);
  EXPECT_FALSE(std::filesystem::exists(root / "images.bin"));
  EXPECT_FALSE(std::filesystem::exists(root / "rigs.txt"));
  EXPECT_FALSE(std::filesystem::exists(root / "frames.bin"));
  EXPECT_DOUBLE_EQ(ReadColmapReconstruction(root).GetPoint3D(1).xyz.x(), 12.5);
  WriteColmapReconstruction(root, model, ColmapModelFormat::kTextAndBinary);
  model.GetPoint3D(1).xyz.x() = -7.25;
  WriteColmapReconstruction(root, model, ColmapModelFormat::kBinary);
  EXPECT_FALSE(std::filesystem::exists(root / "images.txt"));
  EXPECT_DOUBLE_EQ(ReadColmapReconstruction(root).GetPoint3D(1).xyz.x(), -7.25);
  std::filesystem::remove_all(root);
}

TEST(SfmWorkspaceTest, RejectsTruncatedTextObservationsAndTracks) {
  const auto root = std::filesystem::temp_directory_path() / "colite_model_truncated_test";
  std::filesystem::remove_all(root);
  WriteTinyColmapModel(root, false);
  WriteTextFile(root / "images.txt", "1 1 0 0 0 0 0 0 1 image.jpg\n10 20\n");
  EXPECT_THROW(ReadColmapTextReconstruction(root), std::runtime_error);
  WriteTinyColmapModel(root, false);
  WriteTextFile(root / "points3D.txt", "1 0 0 5 255 255 255 0.5 1\n");
  EXPECT_THROW(ReadColmapTextReconstruction(root), std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST(SfmWorkspaceTest, RoundTripsManifestFeaturesAndMatches) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_sfm_workspace_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  camera.has_prior_focal_length = true;

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.file_path = "/tmp/image1.jpg";

  Image image2;
  image2.id = 2;
  image2.camera_id = 1;
  image2.name = "image2.jpg";
  image2.file_path = "/tmp/image2.jpg";
  image2.prior_position = Eigen::Vector3d(1.0, 2.0, 3.0);

  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyFeatures());

  MatchGraph graph;
  SetMatches(&graph, 1, 2, {{0, 1}, {1, 0}});
  TwoViewGeometry geometry;
  geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  if (!geometry.cam2_from_cam1) geometry.cam2_from_cam1.emplace();
  geometry.cam2_from_cam1->t_cw = Eigen::Vector3d(1.0, 0.0, 0.0);
  geometry.median_triangulation_angle = 3.5;
  geometry.inlier_matches = {{0, 1}, {1, 0}};
  SetVerifiedGeometry(&graph, 1, 2, geometry);
  WriteWorkspaceMatches(workspace, graph);
  WriteWorkspaceVerifiedGeometries(workspace, graph);

  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  const std::vector<Image> images = ReadWorkspaceImages(workspace);
  const FeatureSet loaded_features = ReadWorkspaceFeatureSet(workspace, 1);
  MatchGraph loaded_graph;
  ReadWorkspaceMatches(workspace, &loaded_graph);
  ReadWorkspaceVerifiedGeometries(workspace, &loaded_graph);

  ASSERT_EQ(cameras.size(), 1U);
  EXPECT_EQ(cameras[0].width, 640U);
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
  ASSERT_EQ(images.size(), 2U);
  EXPECT_EQ(images[1].name, "image2.jpg");
  ASSERT_TRUE(images[1].prior_position.has_value());
  EXPECT_NEAR(images[1].prior_position->x(), 1.0, 1e-12);
  EXPECT_NEAR(images[1].prior_position->y(), 2.0, 1e-12);
  EXPECT_NEAR(images[1].prior_position->z(), 3.0, 1e-12);
  ASSERT_EQ(loaded_features.keypoints.size(), 2U);
  EXPECT_EQ(loaded_features.extractor_type, FeatureExtractorType::kSift);
  EXPECT_EQ(loaded_features.descriptor_scalar_type,
            DescriptorScalarType::kFloat32);
  EXPECT_EQ(loaded_features.descriptor_metric,
            DescriptorMetric::kSiftAngular);
  EXPECT_EQ(loaded_features.descriptor_normalization,
            DescriptorNormalization::kRootL1);
  EXPECT_FLOAT_EQ(loaded_features.keypoints[1].response, 3.0F);
  ASSERT_EQ(loaded_features.descriptors.rows(), 2);
  EXPECT_FLOAT_EQ(loaded_features.descriptors(1, 127), 2.0F);

  const std::vector<FeatureMatch>* matches =
      FindMatches(loaded_graph, 1, 2);
  ASSERT_NE(matches, nullptr);
  EXPECT_EQ(matches->size(), 2U);
  const TwoViewGeometry* loaded_geometry =
      FindVerifiedGeometry(loaded_graph, 1, 2);
  ASSERT_NE(loaded_geometry, nullptr);
  EXPECT_EQ(loaded_geometry->type, TwoViewGeometry::Type::kCalibratedEssential);
  EXPECT_NEAR(loaded_geometry->median_triangulation_angle, 3.5, 1e-12);
  EXPECT_EQ(loaded_geometry->inlier_matches.size(), 2U);

  EXPECT_TRUE(std::filesystem::exists(workspace / "features" / "1.bin"));
  EXPECT_TRUE(std::filesystem::exists(workspace / "matches" /
                                      "raw_matches.tsv"));
  EXPECT_TRUE(std::filesystem::exists(workspace / "matches" /
                                      "verified_matches.tsv"));
}

TEST(SfmWorkspaceTest, ReadsLegacyVersionOneFeatureFilesAsSift) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_feature_v1_test";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(workspace / "features");
  const FeatureSet source = MakeTinyFeatures();
  std::ofstream file(workspace / "features" / "7.bin", std::ios::binary);
  const std::uint32_t magic = 0x434C4654U;
  const std::uint32_t version = 1;
  const std::uint64_t count = source.keypoints.size();
  const std::uint64_t rows = source.descriptors.rows();
  const std::uint64_t cols = source.descriptors.cols();
  WriteBinaryTestValue(&file, magic);
  WriteBinaryTestValue(&file, version);
  WriteBinaryTestValue(&file, count);
  WriteBinaryTestValue(&file, rows);
  WriteBinaryTestValue(&file, cols);
  for (const FeatureKeypoint& keypoint : source.keypoints) {
    WriteBinaryTestValue(&file, keypoint.x);
    WriteBinaryTestValue(&file, keypoint.y);
    WriteBinaryTestValue(&file, keypoint.scale);
    WriteBinaryTestValue(&file, keypoint.orientation);
    WriteBinaryTestValue(&file, keypoint.response);
  }
  file.write(reinterpret_cast<const char*>(source.descriptors.data()),
             source.descriptors.size() * sizeof(float));
  file.close();

  const FeatureSet loaded = ReadWorkspaceFeatureSet(workspace, 7);
  EXPECT_EQ(loaded.extractor_type, FeatureExtractorType::kSift);
  EXPECT_EQ(loaded.descriptor_metric, DescriptorMetric::kSiftAngular);
  ASSERT_EQ(loaded.descriptors.rows(), source.descriptors.rows());
  EXPECT_FLOAT_EQ(loaded.descriptors(1, 127), 2.0F);
  const FeatureSet deferred = ReadWorkspaceFeatureSet(
      workspace, 7, WorkspaceFeatureLoading::kDeferredDescriptors);
  EXPECT_EQ(deferred.descriptors.size(), 0);
  ASSERT_EQ(deferred.DescriptorRows(), source.descriptors.rows());
  EXPECT_TRUE(deferred.ReadDescriptorRows({1, 0, 1}).row(0).isApprox(
      source.descriptors.row(1), 0.0F));
}

TEST(SfmWorkspaceTest, RejectsCorruptedFeaturePayload) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_feature_v2_corrupt_test";
  std::filesystem::remove_all(workspace);
  WriteWorkspaceFeatureSet(workspace, 9, MakeTinyFeatures());
  const std::filesystem::path path = workspace / "features" / "9.bin";
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  file.seekg(-1, std::ios::end);
  char value = 0;
  file.read(&value, 1);
  value ^= 0x01;
  file.seekp(-1, std::ios::end);
  file.write(&value, 1);
  file.close();

  EXPECT_THROW(ReadWorkspaceFeatureSet(workspace, 9), std::runtime_error);
  EXPECT_THROW(ReadWorkspaceFeatureSet(
      workspace, 9, WorkspaceFeatureLoading::kDeferredDescriptors), std::runtime_error);
}

TEST(SfmWorkspaceTest, PreservesOriginalAffineShapeAndDeferredDescriptorOffset) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_feature_affine_v3_test";
  std::filesystem::remove_all(workspace);
  auto source = MakeTinyFeatures();
  source.source_keypoint_columns = 6;
  source.source_keypoint_shapes.resize(2, 4);
  source.source_keypoint_shapes << 0.5F, -3.F, 2.F, 0.25F,
                                   1.5F, -0.75F, 4.F, 0.125F;
  WriteWorkspaceFeatureSet(workspace, 7, source);
  const auto loaded = ReadWorkspaceFeatureSet(workspace, 7);
  EXPECT_EQ(loaded.source_keypoint_columns, 6);
  EXPECT_TRUE(loaded.source_keypoint_shapes.isApprox(source.source_keypoint_shapes, 0));
  EXPECT_TRUE(loaded.descriptors.isApprox(source.descriptors, 0));
  const auto deferred = ReadWorkspaceFeatureSet(workspace, 7, WorkspaceFeatureLoading::kDeferredDescriptors);
  EXPECT_TRUE(deferred.source_keypoint_shapes.isApprox(source.source_keypoint_shapes, 0));
  const auto rows = deferred.ReadDescriptorRows({1, 0, 1});
  EXPECT_TRUE(rows.row(0).isApprox(source.descriptors.row(1), 0));
  EXPECT_TRUE(rows.row(1).isApprox(source.descriptors.row(0), 0));
  // Corrupt only a finite affine coefficient: full and streamed readers must
  // validate it along with the descriptors rather than silently dropping it.
  const auto path = workspace / "features/7.bin";
  std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
  file.seekp(72 + 2 * 5 * sizeof(float));
  const float changed = 9.F;
  file.write(reinterpret_cast<const char*>(&changed), sizeof(changed));
  file.close();
  EXPECT_THROW(ReadWorkspaceFeatureSet(workspace, 7), std::runtime_error);
  EXPECT_THROW(ReadWorkspaceFeatureSet(workspace, 7, WorkspaceFeatureLoading::kDeferredDescriptors), std::runtime_error);
  source.source_keypoint_shapes.resize(1, 4);
  EXPECT_THROW(WriteWorkspaceFeatureSet(workspace, 8, source), std::invalid_argument);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, RawMatchIOKeepsEmptyPairsAndReordersReversedIndices) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_raw_match_exchange_test";
  std::filesystem::remove_all(workspace);
  WriteTextFile(workspace / "matches/raw_matches.tsv", "23 7 4 2\n7 31 -1 -1\n");
  MatchGraph loaded;
  ReadWorkspaceMatches(workspace, &loaded);
  ASSERT_EQ(loaded.matches.size(), 2U);
  ASSERT_EQ(loaded.matches.at(ImagePair::Ordered(7, 23)).size(), 1U);
  EXPECT_EQ(loaded.matches.at(ImagePair::Ordered(7, 23))[0].idx1, 2U);
  EXPECT_EQ(loaded.matches.at(ImagePair::Ordered(7, 23))[0].idx2, 4U);
  EXPECT_TRUE(loaded.matches.at(ImagePair::Ordered(7, 31)).empty());
  WriteWorkspaceMatches(workspace, loaded);
  MatchGraph returned;
  ReadWorkspaceMatches(workspace, &returned);
  EXPECT_EQ(returned.matches.size(), 2U);
  EXPECT_TRUE(returned.matches.at(ImagePair::Ordered(7, 31)).empty());
  WriteTextFile(workspace / "matches/raw_matches.tsv", "7 23 -1 1\n");
  EXPECT_THROW(ReadWorkspaceMatches(workspace, &returned), std::runtime_error);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, DeferredDescriptorsPreserveRequestedRowsAndMetadata) {
  const auto workspace = std::filesystem::temp_directory_path() /
                         "colite_deferred_feature_rows_test";
  std::filesystem::remove_all(workspace);
  FeatureSet source;
  source.extractor_type = FeatureExtractorType::kAlikedN32;
  source.descriptor_metric = DescriptorMetric::kCosine;
  source.keypoints.resize(400);
  source.descriptors.resize(400, 128);
  for (int i = 0; i < 400; ++i) {
    source.keypoints[i].x = static_cast<float>(i);
    for (int j = 0; j < 128; ++j) source.descriptors(i, j) = i * 128.0F + j;
  }
  WriteWorkspaceFeatureSet(workspace, 1, source);
  const FeatureSet deferred = ReadWorkspaceFeatureSet(
      workspace, 1, WorkspaceFeatureLoading::kDeferredDescriptors);
  EXPECT_EQ(deferred.extractor_type, source.extractor_type);
  EXPECT_EQ(deferred.descriptor_metric, source.descriptor_metric);
  EXPECT_EQ(deferred.keypoints.size(), 400U);
  EXPECT_FLOAT_EQ(deferred.keypoints[399].x, 399.0F);
  EXPECT_EQ(deferred.descriptors.size(), 0);
  EXPECT_EQ(deferred.DescriptorRows(), 400);
  EXPECT_EQ(deferred.DescriptorCols(), 128);
  const std::vector<Point2DIdx> indices{399, 2, 127, 128, 2, 0, 257};
  const DescriptorMatrix rows = deferred.ReadDescriptorRows(indices);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    EXPECT_TRUE(rows.row(i).isApprox(source.descriptors.row(indices[i]), 0.0F));
  }
  EXPECT_EQ(deferred.ReadDescriptorRows({}).rows(), 0);
  EXPECT_THROW(deferred.ReadDescriptorRows({400}), std::out_of_range);
  EXPECT_THROW(WriteWorkspaceFeatureSet(workspace, 2, deferred), std::invalid_argument);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, DeferredDescriptorsRejectTruncationAndLaterFileChanges) {
  const auto workspace = std::filesystem::temp_directory_path() /
                         "colite_deferred_feature_changed_test";
  std::filesystem::remove_all(workspace);
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  const FeatureSet deferred = ReadWorkspaceFeatureSet(
      workspace, 1, WorkspaceFeatureLoading::kDeferredDescriptors);
  const auto path = workspace / "features/1.bin";
  const auto original_size = std::filesystem::file_size(path);
  std::filesystem::resize_file(path, original_size - 1);
  EXPECT_THROW(deferred.ReadDescriptorRows({0}), std::runtime_error);
  EXPECT_THROW(ReadWorkspaceFeatureSet(
      workspace, 1, WorkspaceFeatureLoading::kDeferredDescriptors), std::runtime_error);
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  const FeatureSet replacement = ReadWorkspaceFeatureSet(
      workspace, 1, WorkspaceFeatureLoading::kDeferredDescriptors);
  std::filesystem::last_write_time(path, std::filesystem::last_write_time(path) +
      std::chrono::seconds(2));
  EXPECT_THROW(replacement.ReadDescriptorRows({0}), std::runtime_error);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, RoundTripsLearnedDescriptorMetadata) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_feature_metadata_test";
  std::filesystem::remove_all(workspace);
  FeatureSet source = MakeTinyFeatures();
  source.extractor_type = FeatureExtractorType::kAlikedN32;
  source.descriptor_metric = DescriptorMetric::kCosine;
  WriteWorkspaceFeatureSet(workspace, 11, source);

  const FeatureSet loaded = ReadWorkspaceFeatureSet(workspace, 11);
  EXPECT_EQ(loaded.extractor_type, FeatureExtractorType::kAlikedN32);
  EXPECT_EQ(loaded.descriptor_scalar_type, DescriptorScalarType::kFloat32);
  EXPECT_EQ(loaded.descriptor_metric, DescriptorMetric::kCosine);
  EXPECT_EQ(loaded.descriptor_normalization,
            DescriptorNormalization::kRootL1);
  EXPECT_TRUE(loaded.descriptors.isApprox(source.descriptors));
}

TEST(SfmWorkspaceTest, RejectsPartialDescriptorRows) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_feature_rows_test";
  std::filesystem::remove_all(workspace);
  FeatureSet features = MakeTinyFeatures();
  features.descriptors.conservativeResize(1, Eigen::NoChange);
  EXPECT_THROW(WriteWorkspaceFeatureSet(workspace, 12, features),
               std::invalid_argument);
}

TEST(SfmWorkspaceTest, RoundTripsStageCacheMetadata) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_sfm_cache_test";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(workspace / "features");
  {
    std::ofstream file(workspace / "features" / "1.bin");
    file << "feature bytes";
  }

  const std::string hash = StableConfigHash(
      {{"stage", "features"}, {"backend", "vlsift"}, {"max_num_features", "64"}});
  WriteWorkspaceCacheMetadata(workspace, "features", hash,
                              {{"images", "1"},
                               {"feature_extraction_seconds", "0.25"}},
                              {"features/1.bin"});

  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace, "features");

  EXPECT_EQ(metadata.stage, "features");
  EXPECT_EQ(metadata.config_hash, hash);
  EXPECT_EQ(metadata.values.at("images"), "1");
  EXPECT_TRUE(WorkspaceCacheIsValid(workspace, "features", hash));
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", hash + "x"));

  std::filesystem::remove(workspace / "features" / "1.bin");
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", hash));
}

TEST(SfmWorkspaceTest, CacheRejectsTruncatedArtifactsAndPartialMetadata) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_cache_integrity";
  std::filesystem::remove_all(workspace);
  WriteTextFile(workspace / "features/1.bin", "feature bytes");
  const auto publish = [&] {
    WriteWorkspaceCacheMetadata(workspace, "features", "fixed-options",
        {{"images", "1"}}, {"features/1.bin"});
  };
  publish();
  ASSERT_TRUE(WorkspaceCacheIsValid(workspace, "features", "fixed-options"));
  const auto modified = std::filesystem::last_write_time(workspace / "features/1.bin");
  std::filesystem::resize_file(workspace / "features/1.bin", 3);
  std::filesystem::last_write_time(workspace / "features/1.bin", modified);
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", "fixed-options"));
  publish();
  ASSERT_TRUE(WorkspaceCacheIsValid(workspace, "features", "fixed-options"));
  const auto path = WorkspaceCacheMetadataPath(workspace, "features");
  const auto valid = ReadTextFile(path);
  const auto second_line = valid.find('\n', valid.find('\n') + 1);
  WriteTextFile(path, valid.substr(0, second_line + 1));
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", "fixed-options"));
  EXPECT_TRUE(ReadWorkspaceCacheMetadata(workspace, "features").values.empty());
  WriteTextFile(path, valid);
  auto edited = valid;
  const auto count = edited.find("value\timages\t1");
  ASSERT_NE(count, std::string::npos);
  edited[count + std::string("value\timages\t").size()] = '2';
  WriteTextFile(path, edited);
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", "fixed-options"));
  WriteTextFile(path, "stage\tfeatures\nconfig_hash\tfixed-options\n");
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "features", "fixed-options"));
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, CacheTracksRecursiveInputsAndDoesNotReuseFailure) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_cache_inputs";
  std::filesystem::remove_all(workspace);
  WriteTextFile(workspace / "features/1.bin", "first feature");
  WriteTextFile(workspace / "matches/verified_matches.tsv", "verified output");
  WriteWorkspaceCacheMetadata(workspace, "geometry", "fixed-options", {},
      {"features", "matches/verified_matches.tsv"});
  ASSERT_TRUE(WorkspaceCacheIsValid(workspace, "geometry", "fixed-options"));
  WriteTextFile(workspace / "features/2.bin", "new input");
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "geometry", "fixed-options"));
  WriteWorkspaceCacheMetadata(workspace, "sparse", "fixed-options",
      {{"success", "0"}}, {"matches/verified_matches.tsv"});
  EXPECT_FALSE(WorkspaceCacheIsValid(workspace, "sparse", "fixed-options"));
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, ReadsRegisteredImageWithEmptyPointsLine) {
  const std::filesystem::path model_path =
      std::filesystem::temp_directory_path() /
      "colite_empty_image_points_model_test";
  std::filesystem::remove_all(model_path);
  WriteTinyColmapModel(model_path, true);

  const Reconstruction model = ReadColmapTextReconstruction(model_path);

  ASSERT_EQ(model.RegisteredImageIds().size(), 3U);
  ASSERT_TRUE(model.ExistsImage(3));
  EXPECT_TRUE(model.GetImage(3).points2d.empty());
  EXPECT_EQ(model.Points3D().size(), 2U);
}

TEST(SfmWorkspaceTest, RemapsCheckpointImagesByName) {
  const std::filesystem::path model_path =
      std::filesystem::temp_directory_path() /
      "colite_checkpoint_name_remap_test";
  std::filesystem::remove_all(model_path);
  WriteTinyColmapModel(model_path, true);

  Reconstruction schema;
  Camera camera;
  camera.id = 7;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {450.0, 450.0, 320.0, 240.0};
  schema.AddCamera(camera);
  for (ImageId image_id : {11U, 12U, 13U, 14U}) {
    Image image;
    image.id = image_id;
    image.camera_id = 7;
    image.name = image_id == 11U   ? "new_image.jpg"
                 : image_id == 12U ? "image3.jpg"
                 : image_id == 13U ? "image1.jpg"
                                   : "image2.jpg";
    image.file_path = "/current/" + image.name;
    schema.AddImage(std::move(image));
  }

  const Reconstruction resumed = ApplyColmapTextModel(schema, model_path);

  EXPECT_EQ(resumed.RegisteredImageIds(),
            (std::set<ImageId>{12U, 13U, 14U}));
  EXPECT_FALSE(resumed.IsImageRegistered(11));
  EXPECT_EQ(resumed.GetImage(13).file_path, "/current/image1.jpg");
  EXPECT_DOUBLE_EQ(resumed.GetCamera(7).params[0], 500.0);
  ASSERT_TRUE(resumed.ExistsPoint3D(1));
  const Track& track = resumed.GetPoint3D(1).track;
  ASSERT_EQ(track.elements.size(), 2U);
  EXPECT_EQ(track.elements[0].image_id, 13U);
  EXPECT_EQ(track.elements[1].image_id, 14U);
}

TEST(SfmWorkspaceTest, InitialImageControlsValidateBeforeDataIOAcrossEntryPoints) {
  const auto base=std::filesystem::temp_directory_path()/"colite_initial_control_validation_test";
  std::filesystem::remove_all(base);
  struct InvalidCase {std::vector<std::string> flags; std::string error;};
  const std::vector<InvalidCase> cases={
      {{"--Mapper.init_image_id1","-2"},"must be -1"},
      {{"--Mapper.init_max_reg_trials","0"},"must be positive"},
      {{"--Mapper.init_image_id1","2","--Mapper.init_image_id2","2"},"different images"}};
  for (const auto program : {"mapper","recon_sfm","recon_auto"}) for(const auto& invalid:cases) {
    std::vector<std::string> args={"colite",program};
    if (std::string(program)=="mapper") args.insert(args.end(),{"--Colite.workspace_path",(base/"missing_workspace").string()});
    else args.insert(args.end(),{"--image_path",(base/"missing_images").string(),"--output_path",(base/"output").string()});
    if(std::string(program)=="recon_auto") args.push_back("--dry_run");
    for(const auto& flag:invalid.flags)
      args.push_back(std::string(program)=="recon_auto" && flag.rfind("--Mapper.",0)==0 ? "--sfm."+flag.substr(2) : flag);
    std::ostringstream out,err;
    EXPECT_NE(exe::RunColite(args,out,err),0) << program;
    EXPECT_NE(err.str().find(invalid.error),std::string::npos) << program << ": " << err.str();
    EXPECT_FALSE(std::filesystem::exists(base));
  }
  auto parsed=controllers::ColmapSfmMappingDefaults();
  controllers::ApplySfmMapperOptions({{"--Mapper.init_image_id1","0"},{"--Mapper.init_image_id2","2"},
      {"--Mapper.init_max_reg_trials","3"}},&parsed);
  EXPECT_EQ(parsed.initialization.image_id1,0U);
  EXPECT_EQ(parsed.initialization.image_id2,2U);
  EXPECT_EQ(parsed.initialization.max_num_reg_trials,3U);
  EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.init_image_id1","2147483648",&parsed),std::out_of_range);
  std::ostringstream out,err;
  EXPECT_EQ(exe::RunColite({"colite","recon_auto","--image_path",(base/"missing_images").string(),
      "--output_path",(base/"output").string(),"--dry_run","--sfm.Mapper.init_image_id1","3",
      "--sfm.Mapper.init_image_id2","-1","--sfm.Mapper.init_max_reg_trials","2"},out,err),0) << err.str();
  EXPECT_FALSE(std::filesystem::exists(base));
}

TEST(SfmWorkspaceTest, StandaloneGlobalMapperReusesFeaturesAndKeysCacheByMethod) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_standalone_global_mapper_test";
  std::filesystem::remove_all(workspace);
  Camera camera; camera.id = 1; camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000; camera.height = 800; camera.params = {700, 700, 500, 400};
  WriteWorkspaceCameras(workspace, {camera});
  const auto points = MappingScenePoints(0);
  std::vector<Image> images;
  MatchGraph graph;
  for (ImageId id = 1; id <= 3; ++id) {
    Image image; image.id = id; image.camera_id = 1; image.name = "missing" + std::to_string(id) + ".jpg";
    images.push_back(image);
    auto features = ProjectMappingFeatures(camera, MappingCameraAt(double(id - 1)), points);
    SetFeatures(&graph, id, features); WriteWorkspaceFeatureSet(workspace, id, features);
  }
  WriteWorkspaceImages(workspace, images);
  for (ImageId a = 1; a <= 3; ++a) for (ImageId b = a + 1; b <= 3; ++b) {
    auto matches = IdentityFeatureMatches(points.size());
    SetMatches(&graph, a, b, matches);
    TwoViewGeometry geometry; geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
    geometry.cam2_from_cam1 = MappingCameraAt(double(b - a));
    geometry.median_triangulation_angle = 5; geometry.inlier_matches = matches;
    SetVerifiedGeometry(&graph, a, b, geometry);
  }
  WriteWorkspaceMatches(workspace, graph); WriteWorkspaceVerifiedGeometries(workspace, graph);
  const auto feature_bytes = ReadTextFile(workspace / "features" / "1.bin");
  controllers::MapperControllerOptions options; options.workspace_path = workspace;
  options.global_mapping = true; options.mapper_options["--Mapper.num_threads"] = "2";
  auto first = controllers::MapperController(options).Run();
  ASSERT_TRUE(first.success); EXPECT_EQ(first.num_registered_images, 3U); EXPECT_FALSE(first.cache_hit);
  EXPECT_TRUE(controllers::MapperController(options).Run().cache_hit);
  options.global_options.view_graph.min_num_inliers = 16;
  EXPECT_FALSE(controllers::MapperController(options).Run().cache_hit);
  options.global_mapping = false;
  options.mapper_options["--Mapper.init_min_num_inliers"] = "20";
  options.mapper_options["--Mapper.abs_pose_min_num_inliers"] = "20";
  options.mapper_options["--Mapper.init_min_tri_angle"] = "0.5";
  const auto incremental = controllers::MapperController(options).Run();
  EXPECT_TRUE(incremental.success); EXPECT_FALSE(incremental.cache_hit);
  EXPECT_EQ(ReadTextFile(workspace / "features" / "1.bin"), feature_bytes);
  std::ostringstream out, err;
  EXPECT_NE(exe::RunColite({"colite", "mapper", "--Colite.workspace_path", workspace.string(),
                          "--GlobalMapper.min_num_inliers", "20"}, out, err), 0);
  EXPECT_NE(err.str().find("require --Colite.mapper global"), std::string::npos);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, MapperWritesAndResumesMultipleSparseModels) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_mapper_multimodel_resume_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000;
  camera.height = 800;
  camera.params = {700.0, 700.0, 500.0, 400.0};
  WriteWorkspaceCameras(workspace, {camera});

  std::vector<Image> images;
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    Image image;
    image.id = image_id;
    image.camera_id = 1;
    image.name = "image_" + std::to_string(image_id) + ".jpg";
    images.push_back(std::move(image));
  }
  WriteWorkspaceImages(workspace, images);

  const std::vector<Eigen::Vector3d> points1 = MappingScenePoints(0.0);
  const std::vector<Eigen::Vector3d> points2 = MappingScenePoints(0.4);
  MatchGraph graph;
  for (ImageId image_id = 1; image_id <= 4; ++image_id) {
    const bool second_view = image_id == 2 || image_id == 4;
    const std::vector<Eigen::Vector3d>& points =
        image_id <= 2 ? points1 : points2;
    const FeatureSet features = ProjectMappingFeatures(
        camera, MappingCameraAt(second_view ? 1.0 : 0.0), points);
    SetFeatures(&graph, image_id, features);
    WriteWorkspaceFeatureSet(workspace, image_id, features);
  }
  for (const ImagePair pair :
       {ImagePair::Ordered(1, 2), ImagePair::Ordered(3, 4)}) {
    const std::vector<FeatureMatch> matches =
        IdentityFeatureMatches(points1.size());
    SetMatches(&graph, pair.image_id1, pair.image_id2, matches);
    TwoViewGeometry geometry;
    geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
    geometry.cam2_from_cam1 = MappingCameraAt(1.0);
    geometry.median_triangulation_angle = 2.0;
    geometry.inlier_matches = matches;
    SetVerifiedGeometry(&graph, pair.image_id1, pair.image_id2, geometry);
  }
  WriteWorkspaceMatches(workspace, graph);
  WriteWorkspaceVerifiedGeometries(workspace, graph);

  controllers::MapperControllerOptions options;
  options.workspace_path = workspace;
  options.merge_models = false;
  // Default ten is capped at half the four connected input images.
  options.min_num_images_per_model = 10;
  options.mapper_options["--Mapper.init_min_num_inliers"] = "20";
  options.mapper_options["--Mapper.abs_pose_min_num_inliers"] = "20";
  options.mapper_options["--Mapper.tri_ignore_two_view_tracks"] = "0";
  options.mapper_options["--Mapper.init_min_tri_angle"] = "0.5";
  options.mapper_options["--Mapper.num_threads"] = "4";
  struct RestoreThreads { int previous; ~RestoreThreads() { parallel::setNumThreads(previous); } }
      restore_threads{parallel::getNumThreads()};
  parallel::setNumThreads(3);
  const controllers::MapperControllerSummary first =
      controllers::MapperController(options).Run();
  EXPECT_EQ(parallel::getNumThreads(), 3);

  ASSERT_TRUE(first.success);
  EXPECT_EQ(first.num_models, 2U);
  EXPECT_EQ(first.num_unregistered_images, 0U);
  EXPECT_TRUE(std::filesystem::exists(workspace / "sparse" / "models.txt"));
  EXPECT_TRUE(std::filesystem::exists(
      workspace / "sparse" / "models" / "0" / "points3D.ply"));
  EXPECT_TRUE(std::filesystem::exists(
      workspace / "sparse" / "models" / "1" / "points3D.ply"));

  // Keep the input immutable: continuing directly from a cache output would
  // change its fingerprint when that same cache is rewritten.
  options.input_model_path = workspace / "resume_input";
  std::filesystem::copy(workspace / "sparse/models/0", options.input_model_path,
                        std::filesystem::copy_options::recursive);
  const controllers::MapperControllerSummary resumed =
      controllers::MapperController(options).Run();
  EXPECT_TRUE(resumed.success);
  EXPECT_FALSE(resumed.cache_hit);
  EXPECT_EQ(resumed.num_models, 2U);
  EXPECT_EQ(resumed.num_unregistered_images, 0U);
  options.output_path = workspace / "continued_export";
  for (int replay = 0; replay < 2; ++replay) {
    if (replay == 1) {
      std::filesystem::resize_file(options.output_path / "points3D.bin", 1);
    }
    const auto exported_resume = controllers::MapperController(options).Run();
    ASSERT_TRUE(exported_resume.success);
    EXPECT_TRUE(exported_resume.cache_hit);
    EXPECT_EQ(exported_resume.num_models, 2U);
    EXPECT_FALSE(std::filesystem::exists(options.output_path / "0"));
    EXPECT_FALSE(std::filesystem::exists(options.output_path / "1"));
    for (const std::string file : {"cameras.txt", "images.txt", "points3D.txt",
                                   "cameras.bin", "images.bin", "points3D.bin"}) {
      ASSERT_TRUE(std::filesystem::exists(options.output_path / file));
      EXPECT_EQ(ReadTextFile(options.output_path / file),
                ReadTextFile(workspace / "sparse/models/0" / file));
    }
    const auto sorted_ply_lines = [](const std::filesystem::path& path) {
      std::istringstream stream(ReadTextFile(path));
      std::vector<std::string> lines;
      for (std::string line; std::getline(stream, line);) lines.push_back(line);
      std::sort(lines.begin(), lines.end());
      return lines;
    };
    EXPECT_EQ(sorted_ply_lines(options.output_path / "points3D.ply"),
              sorted_ply_lines(workspace / "sparse/models/0/points3D.ply"));
    for (const auto format : {ColmapModelFormat::kText, ColmapModelFormat::kBinary}) {
      EXPECT_EQ(ReadColmapReconstruction(options.output_path, format)
                    .RegisteredImageIds().size(), 2U);
    }
    EXPECT_TRUE(std::filesystem::exists(workspace / "sparse/models/1/images.bin"));
  }
  // Content changes must invalidate the seed even at identical size and mtime.
  const auto seed_points = options.input_model_path / "points3D.bin";
  const auto seed_mtime = std::filesystem::last_write_time(seed_points);
  const auto seed_fingerprint = ColmapModelFingerprint(options.input_model_path);
  EXPECT_EQ(seed_fingerprint, ColmapModelFingerprint(options.input_model_path,
                                                     ColmapModelFormat::kBinary));
  {
    std::fstream file(seed_points, std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(16);  // uint64 count, uint64 first point ID, then XYZ.
    double x = 0.0;
    file.read(reinterpret_cast<char*>(&x), sizeof(x));
    x += 0.001;
    file.seekp(16);
    file.write(reinterpret_cast<const char*>(&x), sizeof(x));
    ASSERT_TRUE(file.good());
  }
  std::filesystem::last_write_time(seed_points, seed_mtime);
  EXPECT_NE(seed_fingerprint, ColmapModelFingerprint(options.input_model_path));
  ASSERT_FALSE(controllers::MapperController(options).Run().cache_hit);
  ASSERT_TRUE(controllers::MapperController(options).Run().cache_hit);
  auto fresh_options = options;
  fresh_options.input_model_path.clear();
  fresh_options.output_path = workspace / "fresh_after_resume";
  const auto fresh_after_resume = controllers::MapperController(fresh_options).Run();
  ASSERT_TRUE(fresh_after_resume.success);
  EXPECT_FALSE(fresh_after_resume.cache_hit);
  EXPECT_TRUE(std::filesystem::exists(fresh_options.output_path / "0/images.bin"));
  std::ostringstream cli_out, cli_err;
  const std::vector<std::string> canonical_args = {"colite", "mapper",
      "--Colite.workspace_path", workspace.string(), "--Mapper.init_min_num_inliers", "20",
      "--Mapper.abs_pose_min_num_inliers", "20", "--Mapper.tri_ignore_two_view_tracks", "0",
      "--Mapper.tri_min_angle", "1", "--Mapper.ba_refine_focal_length", "0",
      "--Mapper.ba_refine_extra_params", "0", "--Mapper.max_num_models", "2",
      "--Colite.merge_models", "0", "--Mapper.min_model_size", "2", "--Mapper.init_min_tri_angle", "0.5"};
  ASSERT_EQ(exe::RunColite(canonical_args, cli_out, cli_err), 0) << cli_err.str();
  EXPECT_NE(cli_out.str().find("Sparse models: 2"), std::string::npos);
  cli_out.str("");
  ASSERT_EQ(exe::RunColite(canonical_args, cli_out, cli_err), 0) << cli_err.str();
  EXPECT_NE(cli_out.str().find("Sparse cache hit: 1"), std::string::npos);
  auto exported_args = canonical_args;
  const auto exported = workspace / "exported";
  exported_args.push_back("--output_path=" + exported.string());
  for (int replay = 0; replay < 2; ++replay) {
    if (replay == 1) std::filesystem::remove(exported / "1" / "points3D.bin");
    cli_out.str("");
    ASSERT_EQ(exe::RunColite(exported_args, cli_out, cli_err), 0) << cli_err.str();
    EXPECT_NE(cli_out.str().find("Sparse cache hit: 1"), std::string::npos);
    for (const std::string idx : {"0", "1"}) {
      for (const std::string file : {"cameras.txt", "images.txt", "points3D.txt",
                                     "cameras.bin", "images.bin", "points3D.bin"}) {
        ASSERT_TRUE(std::filesystem::exists(exported / idx / file));
        EXPECT_EQ(ReadTextFile(exported / idx / file),
                  ReadTextFile(workspace / "sparse/models" / idx / file));
      }
    }
  }
  const auto damaged_model = workspace / "sparse/models/1/images.bin";
  std::filesystem::resize_file(damaged_model, std::filesystem::file_size(damaged_model) - 1);
  cli_out.str("");
  ASSERT_EQ(exe::RunColite(canonical_args, cli_out, cli_err), 0) << cli_err.str();
  EXPECT_EQ(cli_out.str().find("Sparse cache hit: 1"), std::string::npos);
  EXPECT_EQ(ReadColmapReconstruction(workspace / "sparse/models/1").RegisteredImageIds().size(), 2U);
  for (const std::string minimum : {"0", "1"}) {
    auto small_args = canonical_args;
    const auto flag = std::find(small_args.begin(), small_args.end(),
                                "--Mapper.min_model_size");
    ASSERT_NE(flag, small_args.end());
    *(flag + 1) = minimum;
    cli_out.str("");
    cli_err.str("");
    ASSERT_EQ(exe::RunColite(small_args, cli_out, cli_err), 0) << cli_err.str();
    EXPECT_NE(cli_out.str().find("Sparse models: 2"), std::string::npos);
  }
  auto strict_args = canonical_args;
  strict_args[5] = "1000";  // Fewer than 250 inliers even after relaxation: initialization must fail.
  cli_out.str("");
  EXPECT_EQ(exe::RunColite(strict_args, cli_out, cli_err), 1);
  EXPECT_EQ(cli_out.str().find("Sparse cache hit: 1"), std::string::npos);

  // Standalone triangulation must use the same controls as the mapper.
  Reconstruction seed = ReadColmapReconstruction(options.input_model_path);
  std::vector<Point3DId> removed_points;
  for (const auto& [point_id, point] : seed.Points3D()) {
    (void)point;
    removed_points.push_back(point_id);
  }
  for (const auto point_id : removed_points) seed.DeletePoint3D(point_id);
  WriteColmapReconstruction(workspace / "seed", seed,
                            ColmapModelFormat::kTextAndBinary);
  const auto triangulate = [&](const std::string& ignore_two_view,
                               const std::string& output) {
    cli_out.str("");
    cli_err.str("");
    return exe::RunColite({"colite", "point_triangulator", "--Colite.workspace_path",
        workspace.string(), "--input_path", (workspace / "seed").string(),
        "--output_path", (workspace / output).string(),
        "--Mapper.tri_min_angle", "1", "--Mapper.tri_ignore_two_view_tracks",
        ignore_two_view}, cli_out, cli_err);
  };
  ASSERT_EQ(triangulate("0", "triangulated"), 0) << cli_err.str();
  EXPECT_GT(ReadColmapReconstruction(workspace / "triangulated").Points3D().size(), 0U);
  ASSERT_EQ(triangulate("1", "ignored_two_view"), 0) << cli_err.str();
  EXPECT_TRUE(ReadColmapReconstruction(workspace / "ignored_two_view").Points3D().empty());

  // Register a third view from existing points, then raise the actual PnP
  // inlier threshold beyond the available correspondences and require failure.
  const FeatureSet third_features = ProjectMappingFeatures(
      camera, MappingCameraAt(2.0), points1);
  SetFeatures(&graph, 3, third_features);
  WriteWorkspaceFeatureSet(workspace, 3, third_features);
  SetMatches(&graph, 2, 3, IdentityFeatureMatches(points1.size()));
  TwoViewGeometry third_geometry;
  third_geometry.type = TwoViewGeometry::Type::kCalibratedEssential;
  third_geometry.cam2_from_cam1 = MappingCameraAt(1.0);
  third_geometry.inlier_matches = IdentityFeatureMatches(points1.size());
  SetVerifiedGeometry(&graph, 2, 3, third_geometry);
  WriteWorkspaceMatches(workspace, graph);
  WriteWorkspaceVerifiedGeometries(workspace, graph);
  const auto register_image = [&](const std::string& min_inliers,
                                 const std::string& output) {
    cli_out.str("");
    cli_err.str("");
    return exe::RunColite({"colite", "image_registrator", "--Colite.workspace_path",
        workspace.string(), "--input_path", (workspace / "triangulated").string(),
        "--output_path", (workspace / output).string(), "--image_id", "3",
        "--Mapper.abs_pose_min_num_inliers", min_inliers,
        "--Mapper.abs_pose_max_error", "4", "--Mapper.abs_pose_min_inlier_ratio",
        "0.25"}, cli_out, cli_err);
  };
  ASSERT_EQ(register_image("12", "registered"), 0) << cli_err.str();
  EXPECT_NE(cli_out.str().find("registered_images 1\n"), std::string::npos);
  ASSERT_EQ(register_image("100", "rejected"), 0) << cli_err.str();
  EXPECT_NE(cli_out.str().find("registered_images 0\n"), std::string::npos);
  cli_out.str(""); cli_err.str("");
  ASSERT_EQ(exe::RunColite({"colite", "image_registrator", "--Colite.workspace_path", workspace.string(),
      "--input_path", (workspace / "triangulated").string(), "--output_path", (workspace / "default_registration").string(),
      "--image_id", "3", "--Mapper.random_seed", "0"}, cli_out, cli_err), 0) << cli_err.str();
  EXPECT_NE(cli_out.str().find("registered_images 0\n"), std::string::npos);  // 24 < default 30
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, CudaMatcherPipelinePreservesIndependentRawAndGeometryCaches) {
  if (!IsCudaDescriptorMatcherAvailable()) GTEST_SKIP() << "CUDA matcher unavailable";
  const auto workspace = std::filesystem::temp_directory_path() / "colite_pipeline_cache_test";
  std::filesystem::remove_all(workspace);
  Camera camera;
  camera.id = 1; camera.model_id = CameraModelId::kPinhole;
  camera.width = 640; camera.height = 480;
  camera.params = {500, 500, 320, 240};
  Image first;
  first.id = 1; first.camera_id = 1; first.name = "first.jpg";
  Image second = first;
  second.id = 2; second.name = "second.jpg";
  auto features = MakeTinyFeatures();
  features.descriptors.setZero();
  features.descriptors.row(0).head(64).setConstant(0.125F);
  features.descriptors.row(1).tail(64).setConstant(0.125F);
  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {first, second});
  WriteWorkspaceFeatureSet(workspace, 1, features);
  WriteWorkspaceFeatureSet(workspace, 2, features);
  controllers::SequentialMatcherControllerOptions options;
  options.workspace_path = workspace;
  options.execution.use_gpu = true;
  options.execution.num_threads = 2;
  options.matching_strategy = MatchingStrategy::kExhaustive;
  options.guided_matching = true;
  const auto initial = controllers::SequentialMatcherController(options).Run();
  EXPECT_TRUE(initial.frontend_pipeline_used);
  EXPECT_GT(initial.frontend_pipeline_seconds, 0.0);
  const auto raw = ReadTextFile(workspace / "matches/raw_matches.tsv");
  const auto geometry = ReadTextFile(workspace / "matches/verified_matches.tsv");
  const auto cached = controllers::SequentialMatcherController(options).Run();
  EXPECT_TRUE(cached.raw_match_cache_hit);
  EXPECT_TRUE(cached.geometry_cache_hit);
  EXPECT_FALSE(cached.frontend_pipeline_used);
  std::filesystem::remove(workspace / "cache/geometry.txt");
  const auto raw_only = controllers::SequentialMatcherController(options).Run();
  EXPECT_TRUE(raw_only.raw_match_cache_hit);
  EXPECT_FALSE(raw_only.geometry_cache_hit);
  EXPECT_FALSE(raw_only.frontend_pipeline_used);
  EXPECT_EQ(ReadTextFile(workspace / "matches/raw_matches.tsv"), raw);
  EXPECT_EQ(ReadTextFile(workspace / "matches/verified_matches.tsv"), geometry);
  std::filesystem::remove(workspace / "matches/raw_matches.tsv");
  const auto geometry_only = controllers::SequentialMatcherController(options).Run();
  EXPECT_FALSE(geometry_only.raw_match_cache_hit);
  EXPECT_TRUE(geometry_only.geometry_cache_hit);
  EXPECT_FALSE(geometry_only.frontend_pipeline_used);
  EXPECT_EQ(ReadTextFile(workspace / "matches/raw_matches.tsv"), raw);
  EXPECT_EQ(ReadTextFile(workspace / "matches/verified_matches.tsv"), geometry);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, SequentialMatcherReusesRawAndGeometryCaches) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_match_cache_test";
  std::filesystem::remove_all(workspace);

  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};

  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  image1.file_path = "/tmp/image1.jpg";

  Image image2 = image1;
  image2.id = 2;
  image2.name = "image2.jpg";
  image2.file_path = "/tmp/image2.jpg";

  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyFeatures());

  controllers::SequentialMatcherControllerOptions options;
  options.workspace_path = workspace;
  options.matching_strategy = MatchingStrategy::kExhaustive;
  options.matcher_backend = DescriptorMatcherBackend::kBruteForce;
  options.max_ratio = 1.0;

  const controllers::SequentialMatcherControllerSummary first =
      controllers::SequentialMatcherController(options).Run();
  const controllers::SequentialMatcherControllerSummary second =
      controllers::SequentialMatcherController(options).Run();

  EXPECT_TRUE(first.success);
  EXPECT_FALSE(first.raw_match_cache_hit);
  EXPECT_FALSE(first.geometry_cache_hit);
  EXPECT_TRUE(second.success);
  EXPECT_TRUE(second.raw_match_cache_hit);
  EXPECT_TRUE(second.geometry_cache_hit);
  EXPECT_EQ(second.num_pairs, first.num_pairs);
  EXPECT_EQ(second.num_verified_inliers, first.num_verified_inliers);
  options.two_view_geometry.max_error_px += 1e-8;
  const auto geometry_changed = controllers::SequentialMatcherController(options).Run();
  EXPECT_TRUE(geometry_changed.raw_match_cache_hit);
  EXPECT_FALSE(geometry_changed.geometry_cache_hit);
  options.max_distance += 1e-8;
  const auto matching_changed = controllers::SequentialMatcherController(options).Run();
  EXPECT_FALSE(matching_changed.raw_match_cache_hit);
  EXPECT_FALSE(matching_changed.geometry_cache_hit);
  // External feature files have no extractor cache metadata. Their actual
  // payload generation must still invalidate both downstream checkpoints.
  auto reordered = MakeTinyFeatures();
  reordered.descriptors.row(0).swap(reordered.descriptors.row(1));
  WriteWorkspaceFeatureSet(workspace, 1, reordered);
  const auto changed_input = controllers::SequentialMatcherController(options).Run();
  EXPECT_FALSE(changed_input.raw_match_cache_hit);
  EXPECT_FALSE(changed_input.geometry_cache_hit);
  const auto repaired = controllers::SequentialMatcherController(options).Run();
  EXPECT_TRUE(repaired.raw_match_cache_hit);
  EXPECT_TRUE(repaired.geometry_cache_hit);
  camera.params[0] += 5;
  WriteWorkspaceCameras(workspace, {camera});
  const auto changed_camera = controllers::SequentialMatcherController(options).Run();
  EXPECT_FALSE(changed_camera.raw_match_cache_hit);
  EXPECT_FALSE(changed_camera.geometry_cache_hit);
  std::ostringstream cli_out, cli_err;
  EXPECT_EQ(exe::RunColite({"colite", "exhaustive_matcher", "--Colite.workspace_path", workspace.string(),
      "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--FeatureMatching.use_gpu", "0",
      "--SiftMatching.max_ratio", "1", "--SiftMatching.max_distance", "0.7",
      "--SiftMatching.cross_check", "0", "--TwoViewGeometry.max_error", "2.5"},
      cli_out, cli_err), 0) << cli_err.str();
  std::filesystem::remove_all(workspace);
}

TEST(SfmCommandExeTest, RejectsDuplicateOptionsAndInvalidValuesBeforeRunningStages) {
  const std::vector<std::vector<std::string>> invalid_args = {
      {"colite", "feature_extractor", "--ImageReader.camera_model", "PINHOLE", "--ImageReader.camera_model", "PINHOLE"},
      {"colite", "feature_extractor", "--ImageReader.single_camera", "1", "--ImageReader.single_camera_per_folder", "1"},
      {"colite", "feature_extractor", "--SiftExtraction.max_num_features", "8192junk"},
      {"colite", "mapper", "--Mapper.max_num_models", "-1"},
      {"colite", "mapper", "--Mapper.abs_pose_max_error", "nan"},
      {"colite", "mapper", "--Mapper.abs_pose_min_inlier_ratio", "1.1"},
      {"colite", "mapper", "--Mapper.tri_ignore_two_view_tracks", "2"},
      {"colite", "point_triangulator", "--Mapper.tri_min_angle", "181"},
      {"colite", "image_registrator", "--Mapper.abs_pose_min_inlier_ratio", "-0.1"},
      {"colite", "image_registrator", "--Mapper.abs_pose_max_error", "4", "--Mapper.abs_pose_max_error", "4"},
      {"colite", "exhaustive_matcher", "--SiftMatching.cross_check", "2"},
      {"colite", "exhaustive_matcher", "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--matcher", "flann"},
      {"colite", "exhaustive_matcher", "--TwoViewGeometry.confidence", "nan"},
      {"colite", "exhaustive_matcher", "--TwoViewGeometry.max_num_trials", "-1"}};
  for (const auto& args : invalid_args) {
    std::ostringstream out, err;
    EXPECT_EQ(exe::RunColite(args, out, err), 1);
    EXPECT_NE(err.str().find("Error:"), std::string::npos);
    EXPECT_EQ(err.str().find("required"), std::string::npos) << err.str();
  }
  std::ostringstream out, err;
  EXPECT_EQ(exe::RunColite({"colite", "mapper", "-h"}, out, err), 0);
  EXPECT_NE(out.str().find("--Mapper.tri_min_angle"), std::string::npos);
}

TEST(SfmCommandExeTest, RejectsRemovedNamesAndInactiveFamiliesBeforeReadingData) {
  const std::vector<std::vector<std::string>> cases = {
      {"colite", "feature_extractor", "--max_num_features", "1024"},
      {"colite", "feature_extractor", "--SiftExtraction.max_num_features", "0"},
      {"colite", "feature_extractor", "--camera_mode", "single"},
      {"colite", "feature_extractor", "--focal_length", "500"},
      {"colite", "feature_extractor", "--FeatureExtraction.type", "ALIKED_N32", "--SiftExtraction.max_num_features", "1024"},
      {"colite", "feature_extractor", "--FeatureExtraction.type", "ALIKED_N32", "--AlikedExtraction.n16rot_model_path", "wrong.onnx"},
      {"colite", "exhaustive_matcher", "--matcher", "bf"},
      {"colite", "exhaustive_matcher", "--Colite.matcher_backend", "flann"},
      {"colite", "exhaustive_matcher", "--Colite.matcher_backend", "cuda"},
      {"colite", "exhaustive_matcher", "--FeatureMatching.num_threads", "0"},
      {"colite", "feature_extractor", "--FeatureExtraction.gpu_index", "0,1"},
      {"colite", "exhaustive_matcher", "--SequentialMatching.overlap", "10"},
      {"colite", "exhaustive_matcher", "--FeatureMatching.type", "ALIKED_BRUTEFORCE", "--SiftMatching.max_ratio", "0.8"},
      {"colite", "exhaustive_matcher", "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--SiftMatching.lightglue_min_score", "0.1"},
      {"colite", "mapper", "--input_model_path", "old-model"},
      {"colite", "bundle_adjuster", "--max_num_iterations", "10"},
      {"colite", "bundle_adjuster", "--BundleAdjustment.max_num_iterations", "10"},
      {"colite", "image_registrator", "--max_reprojection_error", "4"}};
  for (const auto& args : cases) {
    std::ostringstream out, err;
    EXPECT_EQ(exe::RunColite(args, out, err), 1);
    EXPECT_EQ(err.str().find("required"), std::string::npos) << err.str();
  }
}

TEST(SfmCommandExeTest, StandaloneAndMapperBundleAdjustmentUseSameDefaults) {
  const auto root = std::filesystem::temp_directory_path() / "colite_ba_entry_parity";
  std::filesystem::remove_all(root);
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500, 500, 320, 240};
  Reconstruction model;
  model.AddCamera(camera);
  const auto points = MappingScenePoints(0);
  for (ImageId id = 1; id <= 3; ++id) {
    Image image;
    image.id = id;
    image.camera_id = 1;
    image.name = std::to_string(id) + ".jpg";
    image.registered = true;
    image.cam_from_world = MappingCameraAt(.5 * (id - 1));
    for (std::size_t idx = 0; idx < points.size(); ++idx) {
      auto xy = *ProjectWorldPoint(camera, *image.cam_from_world, points[idx]);
      if (id == 3 && idx == 0) xy.x() += 20;
      image.points2d.push_back({xy, static_cast<Point2DIdx>(idx), kInvalidPoint3DId});
    }
    model.AddImage(image);
  }
  for (std::size_t idx = 0; idx < points.size(); ++idx) {
    Track track;
    for (ImageId id = 1; id <= 3; ++id) track.AddElement(id, idx);
    model.AddPoint3D(points[idx] + Eigen::Vector3d(.03, -.02, .01), track);
  }
  WriteColmapReconstruction(root / "input", model, ColmapModelFormat::kTextAndBinary);
  auto expected = ReadColmapReconstruction(root / "input");
  auto options = controllers::ColmapSfmMappingDefaults().global_bundle_adjustment;
  options.max_num_iterations = 15;
  ASSERT_DOUBLE_EQ(options.loss_scale, GlobalBundleAdjustmentOptions{}.loss_scale);
  ASSERT_TRUE(AdjustGlobalBundle(options, &expected).success);
  std::ostringstream out, err;
  ASSERT_EQ(exe::RunColite({"colite", "bundle_adjuster", "--input_path", (root / "input").string(),
      "--output_path", (root / "output").string(), "--BundleAdjustmentCeres.max_num_iterations", "15",
      "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_extra_params", "0"}, out, err), 0) << err.str();
  const auto actual = ReadColmapReconstruction(root / "output");
  ASSERT_EQ(actual.Points3D().size(), expected.Points3D().size());
  for (const auto& [id, point] : expected.Points3D()) {
    EXPECT_LT((actual.GetPoint3D(id).xyz - point.xyz).norm(), 1e-9);
  }
  for (const auto id : expected.RegisteredImageIds()) {
    EXPECT_LT((actual.GetImage(id).cam_from_world->t_cw -
               expected.GetImage(id).cam_from_world->t_cw).norm(), 1e-9);
  }
  std::filesystem::remove_all(root);
}

TEST(SfmCommandExeTest, BundleAdjustmentControlsConstrainTheActualSolve) {
  const auto root = std::filesystem::temp_directory_path() / "colite_ba_controls_test";
  std::filesystem::remove_all(root);
  Camera camera;
  camera.id = 1; camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000; camera.height = 800;
  camera.params = {700, 700, 500, 400};
  const auto points = MappingScenePoints(0);
  Reconstruction model;
  model.AddCamera(camera);
  for (ImageId id = 1; id <= 3; ++id) {
    Image image;
    image.id = id; image.camera_id = camera.id; image.name = std::to_string(id) + ".jpg";
    image.registered = true; image.cam_from_world = MappingCameraAt(0.5 * (id - 1));
    for (std::size_t i = 0; i < points.size(); ++i) {
      image.points2d.push_back({*ProjectWorldPoint(camera, *image.cam_from_world, points[i]),
          static_cast<Point2DIdx>(i), kInvalidPoint3DId});
    }
    if (id == 3) image.cam_from_world->t_cw += Eigen::Vector3d(0.03, -0.02, 0.01);
    model.AddImage(image);
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    Track track;
    for (ImageId id = 1; id <= 3; ++id) track.AddElement(id, i);
    model.AddPoint3D(points[i], track);
  }
  WriteColmapReconstruction(root / "input", model, ColmapModelFormat::kTextAndBinary);
  std::ostringstream out, err;
  ASSERT_EQ(exe::RunColite({"colite", "bundle_adjuster", "--input_path", (root / "input").string(),
      "--output_path", (root / "output").string(), "--BundleAdjustment.refine_points3D", "0",
      "--BundleAdjustment.constant_rig_from_world_rotation", "1",
      "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_extra_params", "0",
      "--BundleAdjustment.min_track_length", "3", "--BundleAdjustmentCeres.max_num_iterations", "30",
      "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver", "0",
      "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver", "1"}, out, err), 0) << err.str();
  const auto actual = ReadColmapReconstruction(root / "output");
  EXPECT_NE(out.str().find("requested_linear_solver_type ITERATIVE_SCHUR"), std::string::npos);
  EXPECT_NE(out.str().find("num_variable_points 0"), std::string::npos);
  for (const auto& [id, point] : model.Points3D()) EXPECT_EQ(actual.GetPoint3D(id).xyz, point.xyz);
  for (const auto id : model.RegisteredImageIds()) {
    EXPECT_LT(actual.GetImage(id).cam_from_world->q_cw.angularDistance(model.GetImage(id).cam_from_world->q_cw), 1e-12);
  }
  const auto mean_error = [](const Reconstruction& reconstruction) {
    double sum = 0; std::size_t count = 0;
    for (const auto& [id, point] : reconstruction.Points3D()) {
      for (const auto& observation : point.track.elements) {
        const auto& image = reconstruction.GetImage(observation.image_id);
        sum += ReprojectionError(reconstruction.GetCamera(image.camera_id),
            *image.cam_from_world, point.xyz, image.points2d[observation.point2d_idx].xy).value();
        ++count;
      }
    }
    return sum / count;
  };
  EXPECT_LT(mean_error(actual), mean_error(model) * 0.01);
  out.str(""); err.str("");
  EXPECT_EQ(exe::RunColite({"colite", "bundle_adjuster", "--input_path", "missing-model",
      "--output_path", (root / "invalid").string(),
      "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver", "2",
      "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver", "1"}, out, err), 1);
  EXPECT_NE(err.str().find("dense < sparse"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(root / "invalid"));
  std::filesystem::remove_all(root);
}

TEST(SfmCommandExeTest, BundleAdjusterCleansNegativeDepthAndSavesEmptyModels) {
  const auto root = std::filesystem::temp_directory_path() / "colite_ba_negative_depth_test";
  std::filesystem::remove_all(root);
  Camera camera;
  camera.id = 1; camera.model_id = CameraModelId::kPinhole;
  camera.width = 1000; camera.height = 800;
  camera.params = {700, 700, 500, 400};
  const auto points = MappingScenePoints(0);
  for (const bool all_negative : {false, true}) {
    Reconstruction model;
    model.AddCamera(camera);
    for (ImageId id = 1; id <= 3; ++id) {
      Image image;
      image.id = id; image.camera_id = 1; image.name = std::to_string(id) + ".jpg";
      image.registered = true; image.cam_from_world = MappingCameraAt(.5 * (id - 1));
      for (std::size_t i = 0; i < points.size(); ++i)
        image.points2d.push_back({*ProjectWorldPoint(camera, *image.cam_from_world, points[i]),
            static_cast<Point2DIdx>(i), kInvalidPoint3DId});
      model.AddImage(image);
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
      Track track;
      for (ImageId id = 1; id <= 3; ++id) track.AddElement(id, i);
      auto xyz = points[i];
      if (all_negative || i == 0) xyz.z() = -xyz.z();
      model.AddPoint3D(xyz, track);
    }
    const auto dataset = root / (all_negative ? "all" : "partial");
    WriteColmapReconstruction(dataset / "input", model, ColmapModelFormat::kTextAndBinary);
    for (const auto format : {"BIN", "TXT"}) {
      const auto output = dataset / format;
      std::ostringstream out, err;
      ASSERT_EQ(exe::RunColite({"colite", "bundle_adjuster", "--input_path", (dataset / "input").string(),
          "--output_path", output.string(), "--Colite.output_format", format,
          "--BundleAdjustment.refine_rig_from_world", "0", "--BundleAdjustment.refine_points3D", "0",
          "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_extra_params", "0"}, out, err), 0) << err.str();
      const auto actual = ReadColmapReconstruction(output);
      EXPECT_EQ(actual.RegisteredImageIds().size(), 3U);
      EXPECT_EQ(actual.Points3D().size(), all_negative ? 0U : points.size() - 1);
      for (const auto& [id, image] : actual.Images()) {
        EXPECT_FALSE(image.points2d[0].HasPoint3D());
        EXPECT_EQ(image.NumPoints3D(), all_negative ? 0U : points.size() - 1);
      }
      for (const auto& [id, point] : actual.Points3D()) {
        EXPECT_EQ(point.xyz, model.GetPoint3D(id).xyz);
        EXPECT_EQ(point.track.Length(), 3U);
      }
    }
    EXPECT_EQ(ReadColmapReconstruction(dataset / "input").Points3D().size(), points.size());
  }
  std::filesystem::remove_all(root);
}

TEST(SfmCommandExeTest, OpenCvSiftUsesStandardControlsAcrossEntryPoints) {
  const auto root = std::filesystem::temp_directory_path() / "colite_cv_sift_controls";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root / "images");
  cv::Mat image(256, 256, CV_8UC1);
  cv::RNG random(91);
  random.fill(image, cv::RNG::UNIFORM, 0, 255);
  ASSERT_TRUE(cv::imwrite((root / "images/one.png").string(), image));
  const std::vector<std::string> controls = {
      "--Colite.feature_extractor", "cvsift", "--FeatureExtraction.use_gpu", "0",
      "--SiftExtraction.first_octave", "0", "--SiftExtraction.num_octaves", "1",
      "--SiftExtraction.octave_resolution", "3", "--SiftExtraction.peak_threshold", "0.001",
      "--SiftExtraction.edge_threshold", "10", "--SiftExtraction.max_num_orientations", "1"};
  std::vector<std::string> extract = {"colite", "feature_extractor", "--image_path",
      (root / "images").string(), "--Colite.workspace_path", (root / "workspace").string()};
  extract.insert(extract.end(), controls.begin(), controls.end());
  std::ostringstream out, err;
  ASSERT_EQ(exe::RunColite(extract, out, err), 0) << err.str();
  const auto features = ReadWorkspaceFeatureSet(root / "workspace", 1);
  EXPECT_FALSE(features.keypoints.empty());
  std::map<std::tuple<float, float, float>, int> groups;
  for (const auto& point : features.keypoints) ++groups[{point.x, point.y, point.scale}];
  for (const auto& group : groups) EXPECT_EQ(group.second, 1);
  std::vector<std::string> automatic = {"colite", "recon_auto", "--image_path",
      (root / "images").string(), "--output_path", (root / "planned").string(), "--dry_run"};
  for (std::size_t i = 0; i < controls.size(); i += 2) {
    automatic.push_back("--sfm." + controls[i].substr(2));
    automatic.push_back(controls[i + 1]);
  }
  out.str(""); err.str("");
  EXPECT_EQ(exe::RunColite(automatic, out, err), 0) << err.str();
  EXPECT_FALSE(std::filesystem::exists(root / "planned"));
  std::filesystem::remove_all(root);
}

TEST(SfmCommandExeTest, StandardFrontendAndMapperControlsHaveColmapSemantics) {
  controllers::SfmFeatureExecutionOptions feature;
  ASSERT_TRUE(exe::ApplySfmFeatureExecutionOption("--FeatureExtraction.use_gpu", "0", &feature));
  ASSERT_TRUE(exe::ApplySfmFeatureExecutionOption("--FeatureExtraction.max_image_size", "-1", &feature));
  EXPECT_EQ(controllers::EffectiveSfmImageSize(feature, FeatureExtractorType::kSift), 3200);
  EXPECT_EQ(controllers::EffectiveSfmImageSize(feature, FeatureExtractorType::kAlikedN32), 1600);
  ASSERT_TRUE(exe::ApplySfmFeatureExecutionOption("--SiftExtraction.octave_resolution", "2", &feature));
  // Changing octave resolution must not implicitly change the independent
  // peak_threshold option's default.
  EXPECT_DOUBLE_EQ(feature.sift.peak_threshold, 0.02 / 3.0);
  ASSERT_TRUE(exe::ApplySfmFeatureExecutionOption("--descriptor_normalization", "l2", &feature));
  EXPECT_FALSE(feature.sift.use_root_sift);
  EXPECT_EQ(controllers::ResolveSfmSiftOptions(feature, SiftBackend::kAuto, 8192).backend, SiftBackend::kVlSift);
  EXPECT_THROW(controllers::ResolveSfmSiftOptions(feature, SiftBackend::kCudaVlSift, 8192), std::invalid_argument);
  const auto mapping = controllers::ColmapSfmMappingDefaults();
  const auto standalone_ba = controllers::ColmapStandaloneBundleAdjustmentDefaults();
  EXPECT_EQ(standalone_ba.max_num_iterations, 100);
  EXPECT_DOUBLE_EQ(standalone_ba.solver.function_tolerance, 0);
  EXPECT_DOUBLE_EQ(standalone_ba.solver.gradient_tolerance, 1e-4);
  EXPECT_DOUBLE_EQ(standalone_ba.solver.parameter_tolerance, 0);
  EXPECT_EQ(standalone_ba.solver.max_linear_solver_iterations, 200);
  EXPECT_EQ(mapping.initialization.min_num_inliers, 100U);
  EXPECT_EQ(mapping.max_num_initialization_trials, 200U);
  EXPECT_DOUBLE_EQ(mapping.initialization.min_triangulation_angle_deg, 16);
  EXPECT_DOUBLE_EQ(mapping.initialization.max_forward_motion, .95);
  EXPECT_TRUE(mapping.initialization.reestimate_geometry);
  EXPECT_EQ(mapping.initialization.random_seed, -1);
  EXPECT_EQ(mapping.initialization.image_id1,kInvalidImageId);
  EXPECT_EQ(mapping.initialization.image_id2,kInvalidImageId);
  EXPECT_EQ(mapping.initialization.max_num_reg_trials,2U);
  EXPECT_EQ(mapping.mapper.registration.random_seed, -1);
  auto initialization = mapping;
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.random_seed", "7", &initialization));
  EXPECT_EQ(initialization.initialization.random_seed, 7);
  EXPECT_EQ(initialization.mapper.registration.random_seed, 7);
  EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.random_seed", "-2", &initialization), std::invalid_argument);
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.init_max_forward_motion", "0.7", &initialization));
  EXPECT_DOUBLE_EQ(initialization.initialization.max_forward_motion, .7);
  for (const auto value : {"-0.1", "1.1", "nan"})
    EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.init_max_forward_motion", value, &initialization), std::invalid_argument);
  EXPECT_DOUBLE_EQ(mapping.mapper.registration.max_reprojection_error_px, 12);
  EXPECT_EQ(mapping.mapper.registration.min_num_inliers, 30U);
  EXPECT_TRUE(mapping.relax_initialization_constraints);
  EXPECT_TRUE(mapping.mapper.use_structureless_registration);
  EXPECT_DOUBLE_EQ(mapping.registration_retry_reprojection_error_multiplier, 1);
  EXPECT_DOUBLE_EQ(mapping.mapper.local_bundle_adjustment.loss_scale, 1);
  EXPECT_DOUBLE_EQ(mapping.global_bundle_adjustment.loss_scale, 0);
  auto refinement = mapping;
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_local_function_tolerance", "0.01", &refinement));
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_global_function_tolerance", "0.02", &refinement));
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_min_num_residuals_for_cpu_multi_threading", "0", &refinement));
  EXPECT_DOUBLE_EQ(refinement.mapper.local_bundle_adjustment.solver.function_tolerance, 0.01);
  EXPECT_DOUBLE_EQ(refinement.global_bundle_adjustment.solver.function_tolerance, 0.02);
  EXPECT_EQ(refinement.mapper.local_bundle_adjustment.solver.min_num_residuals_for_cpu_multi_threading, 0);
  EXPECT_EQ(refinement.global_bundle_adjustment.solver.min_num_residuals_for_cpu_multi_threading, 0);
  EXPECT_EQ(mapping.mapper.ba_local_max_refinements, 2U);
  EXPECT_DOUBLE_EQ(mapping.mapper.ba_local_max_refinement_change, 0.001);
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_local_max_refinements", "3", &refinement));
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_local_max_refinement_change", "0", &refinement));
  EXPECT_EQ(refinement.mapper.ba_local_max_refinements, 3U);
  EXPECT_DOUBLE_EQ(refinement.mapper.ba_local_max_refinement_change, 0);
  for (const auto* value : {"0", "-1", "1.5"})
    EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.ba_local_max_refinements", value, &refinement), std::invalid_argument);
  for (const auto* value : {"-1", "nan", "inf"})
    EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.ba_local_max_refinement_change", value, &refinement), std::invalid_argument);
  EXPECT_EQ(refinement.max_num_global_refinements, 5U);
  EXPECT_DOUBLE_EQ(refinement.global_refinement_change, 0.0005);
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.num_threads", "4", &refinement));
  EXPECT_EQ(refinement.num_threads, 4);
  EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.num_threads", "-2", &refinement), std::invalid_argument);
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.num_threads", "0", &refinement));
  EXPECT_EQ(refinement.num_threads, 0);
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_global_max_refinements", "0", &refinement));
  EXPECT_EQ(refinement.max_num_global_refinements, 0U);
  ASSERT_TRUE(controllers::ApplySfmMapperOption("--Mapper.ba_global_max_refinement_change", "0", &refinement));
  EXPECT_DOUBLE_EQ(refinement.global_refinement_change, 0);
  EXPECT_THROW(controllers::ApplySfmMapperOption("--Mapper.ba_global_max_refinement_change", "-1", &refinement), std::invalid_argument);
  EXPECT_DOUBLE_EQ(mapping.ba_global_frames_ratio, 1.1);
  EXPECT_DOUBLE_EQ(mapping.ba_global_points_ratio, 1.1);
  EXPECT_EQ(mapping.ba_global_frames_freq, 500U);
  EXPECT_EQ(mapping.ba_global_points_freq, 250000U);
  for (const auto* name : {"--Mapper.ba_global_frames_ratio", "--Mapper.ba_global_points_ratio"}) {
    ASSERT_TRUE(controllers::ApplySfmMapperOption(name, "1.5", &refinement));
    for (const auto* invalid : {"1", "0", "-1", "nan", "inf"}) {
      EXPECT_THROW(controllers::ApplySfmMapperOption(name, invalid, &refinement), std::invalid_argument);
    }
  }
  EXPECT_DOUBLE_EQ(refinement.ba_global_frames_ratio, 1.5);
  EXPECT_DOUBLE_EQ(refinement.ba_global_points_ratio, 1.5);
  for (const auto* name : {"--Mapper.ba_global_frames_freq", "--Mapper.ba_global_points_freq"}) {
    ASSERT_TRUE(controllers::ApplySfmMapperOption(name, "2", &refinement));
    for (const auto* invalid : {"0", "-1", "1.5"}) {
      EXPECT_THROW(controllers::ApplySfmMapperOption(name, invalid, &refinement), std::invalid_argument);
    }
    EXPECT_THROW(controllers::ApplySfmMapperOption(name, "2147483648", &refinement), std::out_of_range);
  }
  EXPECT_EQ(refinement.ba_global_frames_freq, 2U);
  EXPECT_EQ(refinement.ba_global_points_freq, 2U);
  EXPECT_FALSE(controllers::ApplySfmMapperOption("--Mapper.ba_global_images_ratio", "1.1", &refinement));
  EXPECT_EQ(mapping.mapper.triangulation.complete_max_transitivity, 5U);
  for (const auto* name : {"--Mapper.tri_max_transitivity", "--Mapper.tri_complete_max_transitivity"}) {
    ASSERT_TRUE(controllers::ApplySfmMapperOption(name, "0", &refinement));
    EXPECT_THROW(controllers::ApplySfmMapperOption(name, "-1", &refinement), std::invalid_argument);
    EXPECT_THROW(controllers::ApplySfmMapperOption(name, "2147483648", &refinement), std::out_of_range);
  }
  EXPECT_EQ(refinement.mapper.triangulation.max_transitivity, 0U);
  EXPECT_EQ(refinement.mapper.triangulation.complete_max_transitivity, 0U);

}

TEST(SfmCommandExeTest, ReconAutoValidatesSfmOptionsEvenForDryRun) {
  std::ostringstream out, err;
  EXPECT_EQ(exe::RunColite({"colite", "recon_auto", "--image_path", "images",
      "--output_path", "out", "--sfm.camera_mode", "single", "--dry_run"}, out, err), 1);
  EXPECT_NE(err.str().find("Unknown argument"), std::string::npos);
  EXPECT_EQ(out.str().find("PLAN recon_sfm"), std::string::npos);
}

TEST(SfmWorkspaceTest, ExhaustiveMatcherSupportsPersistedAlikedDescriptors) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() / "colite_aliked_match_test";
  std::filesystem::remove_all(workspace);
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = 640;
  camera.height = 480;
  camera.params = {500.0, 500.0, 320.0, 240.0};
  Image image1;
  image1.id = 1;
  image1.camera_id = 1;
  image1.name = "image1.jpg";
  Image image2 = image1;
  image2.id = 2;
  image2.name = "image2.jpg";
  WriteWorkspaceCameras(workspace, {camera});
  WriteWorkspaceImages(workspace, {image1, image2});
  WriteWorkspaceFeatureSet(workspace, 1, MakeTinyAlikedFeatures());
  WriteWorkspaceFeatureSet(workspace, 2, MakeTinyAlikedFeatures());

  controllers::SequentialMatcherControllerOptions options;
  options.workspace_path = workspace;
  options.matching_strategy = MatchingStrategy::kExhaustive;
  options.matcher_type = FeatureMatcherType::kAlikedBruteForce;
  options.min_cosine_similarity = 0.9;
  options.max_ratio = 1.0;
  const auto summary = controllers::SequentialMatcherController(options).Run();

  EXPECT_TRUE(summary.success);
  EXPECT_EQ(summary.num_pairs, 1U);
  MatchGraph graph;
  ReadWorkspaceMatches(workspace, &graph);
  const auto* matches = FindMatches(graph, 1, 2);
  ASSERT_NE(matches, nullptr);
  EXPECT_EQ(matches->size(), 2U);
  std::filesystem::remove_all(workspace);
}

TEST(SfmCommandExeTest, PrintsHelpForStageCommands) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunFeatureExtractor({"feature_extractor", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("--Colite.workspace_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunSequentialMatcher({"sequential_matcher", "--help"}, out,
                                      err),
            0);
  EXPECT_NE(out.str().find("--SequentialMatching.overlap"), std::string::npos);
  EXPECT_NE(out.str().find("--FeatureMatching.type"), std::string::npos);
  EXPECT_NE(out.str().find("--FeatureMatching.guided_matching"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.guided_match_ratio"), std::string::npos);
  EXPECT_NE(out.str().find("--FeatureMatching.use_gpu"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.learned_cuda_memory_limit_mb"),
            std::string::npos);
  EXPECT_NE(out.str().find("--Colite.learned_use_io_binding"), std::string::npos);

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunMapper({"mapper", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("--Colite.workspace_path"), std::string::npos);
  EXPECT_NE(out.str().find("--input_path"), std::string::npos);
  EXPECT_NE(out.str().find("--Mapper.multiple_models"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.merge_models"), std::string::npos);
}

TEST(ColiteCommandExeTest, PrintsTopLevelHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("Usage:"), std::string::npos);
  EXPECT_NE(out.str().find("feature_extractor"), std::string::npos);
  EXPECT_NE(out.str().find("exhaustive_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("vocab_tree_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("vocab_tree_builder"), std::string::npos);
  EXPECT_NE(out.str().find("compare_colmap"), std::string::npos);
  EXPECT_NE(out.str().find("recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("recon_auto"), std::string::npos);
  EXPECT_NE(out.str().find("image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("patch_match_stereo"), std::string::npos);
  EXPECT_NE(out.str().find("stereo_fusion"), std::string::npos);
  EXPECT_NE(out.str().find("poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("delaunay_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("mesh_texturer"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesExhaustiveSpatialAndVocabTreeMatcherHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "exhaustive_matcher", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("colite exhaustive_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.workspace_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "spatial_matcher", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("colite spatial_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--SpatialMatching.max_distance"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "vocab_tree_matcher", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("colite vocab_tree_matcher"), std::string::npos);
  EXPECT_NE(out.str().find("--VocabTreeMatching.num_images"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.vocab_tree_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());

  out.str("");
  out.clear();
  EXPECT_EQ(exe::RunColite({"colite", "vocab_tree_builder", "--help"}, out,
                           err),
            0);
  EXPECT_NE(out.str().find("--output_path"), std::string::npos);
  EXPECT_NE(out.str().find("--Colite.branching"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesSubcommandHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "recon_sfm", "--help"}, out, err), 0);
  EXPECT_NE(out.str().find("colite recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("--image_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, DispatchesStandaloneMvsToolHelp) {
  for (const std::string& command : {"poisson_mesher", "delaunay_mesher",
                                     "mesh_simplifier", "mesh_texturer"}) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite({"colite", command, "--help"}, out, err), 0)
        << command << ": " << err.str();
    EXPECT_NE(out.str().find("--input_path"), std::string::npos) << command;
    EXPECT_NE(out.str().find("--output_path"), std::string::npos) << command;
    EXPECT_TRUE(err.str().empty()) << command;
  }
}

TEST(ColiteCommandExeTest, DispatchesStagedPipelineHelp) {
  for (const std::string& command : {"recon_auto", "image_undistorter",
                                     "patch_match_stereo", "stereo_fusion"}) {
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite({"colite", command, "--help"}, out, err), 0)
        << command << ": " << err.str();
    EXPECT_NE(out.str().find("Usage:"), std::string::npos) << command;
    EXPECT_TRUE(err.str().empty()) << command;
  }
}

TEST(ColiteCommandExeTest, ReconAutoDryRunPrintsAllStageBoundaries) {
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite({"colite", "recon_auto", "--image_path", "images",
                            "--output_path", "result", "--dry_run"},
                           out, err),
            0)
      << err.str();
  EXPECT_NE(out.str().find("PLAN recon_sfm"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN image_undistorter"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN patch_match_stereo"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN stereo_fusion"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN poisson_mesher"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN mesh_simplifier"), std::string::npos);
  EXPECT_NE(out.str().find("PLAN mesh_texturer"), std::string::npos);
}

TEST(ColiteCommandExeTest, ReconAutoAcceptsSfmCameraMode) {
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(
      exe::RunColite({"colite", "recon_auto", "--image_path", "images",
                      "--output_path", "result", "--stop_after", "sfm",
                      "--sfm.ImageReader.single_camera", "1", "--dry_run"},
                     out, err),
      0)
      << err.str();
  EXPECT_NE(out.str().find("PLAN recon_sfm"), std::string::npos);
}

TEST(ColiteCommandExeTest, SurfaceCommandsSupportDryRunStageContract) {
  struct Case {
    std::string command;
    std::vector<std::string> arguments;
    std::string stage;
  };
  const std::vector<Case> cases = {
      {"poisson_mesher",
       {"--input_path", "fused.ply", "--output_path", "mesh.ply"},
       "poisson_mesher"},
      {"delaunay_mesher",
       {"--input_path", "dense", "--output_path", "mesh.ply"},
       "delaunay_mesher"},
      {"mesh_simplifier",
       {"--input_path", "mesh.ply", "--output_path", "simple.ply"},
       "mesh_simplifier"},
      {"mesh_texturer",
       {"--input_path", "mesh.ply", "--output_path", "textured",
        "--workspace_path", "workspace",
        "--Colite.texture_visibility_resolution", "512",
        "--Colite.texture_occlusion_relative_tolerance", "0.02"},
       "mesh_texturer"}};
  for (const Case& test_case : cases) {
    std::vector<std::string> command{"colite", test_case.command};
    command.insert(command.end(), test_case.arguments.begin(),
                   test_case.arguments.end());
    command.push_back("--dry_run");
    std::ostringstream out;
    std::ostringstream err;
    EXPECT_EQ(exe::RunColite(command, out, err), 0)
        << test_case.command << ": " << err.str();
    EXPECT_NE(out.str().find("PLAN " + test_case.stage), std::string::npos)
        << test_case.command;
  }
}

TEST(ColiteCommandExeTest, MeshSimplifierWritesAndResumesStageManifest) {
  const auto root = std::filesystem::temp_directory_path() /
                    "colite_mesh_simplifier_stage_test";
  std::filesystem::remove_all(root);
  const auto input = root / "input.ply";
  const auto output = root / "output.ply";
  WriteTextFile(input,
                "ply\nformat ascii 1.0\n"
                "element vertex 4\nproperty float x\nproperty float y\n"
                "property float z\nelement face 2\n"
                "property list uchar int vertex_indices\nend_header\n"
                "0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
                "3 0 1 2\n3 0 2 3\n");
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite(
                {"colite", "mesh_simplifier", "--input_path", input.string(),
                 "--output_path", output.string(),
                 "--MeshSimplification.target_face_ratio", "0.75"},
                out, err),
            0)
      << err.str();
  EXPECT_TRUE(std::filesystem::exists(output));
  EXPECT_TRUE(std::filesystem::exists(
      root / ".colite" / "stages" / "mesh_simplifier.json"));

  out.str("");
  out.clear();
  err.str("");
  err.clear();
  EXPECT_EQ(exe::RunColite(
                {"colite", "mesh_simplifier", "--input_path", input.string(),
                 "--output_path", output.string(),
                 "--MeshSimplification.target_face_ratio", "0.75", "--resume"},
                out, err),
            0)
      << err.str();
  EXPECT_NE(out.str().find("SKIP mesh_simplifier"), std::string::npos);
  std::filesystem::remove_all(root);
}

TEST(ColiteCommandExeTest, ReconAutoCanSelectDelaunayOrStopAtFusion) {
  std::ostringstream delaunay_out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunColite(
                {"colite", "recon_auto", "--image_path", "images",
                 "--output_path", "result", "--mesher", "delaunay",
                 "--dry_run"},
                delaunay_out, err),
            0)
      << err.str();
  EXPECT_NE(delaunay_out.str().find("PLAN delaunay_mesher"),
            std::string::npos);

  std::ostringstream fusion_out;
  err.str("");
  err.clear();
  EXPECT_EQ(exe::RunColite(
                {"colite", "recon_auto", "--image_path", "images",
                 "--output_path", "result", "--stop_after", "fusion",
                 "--dry_run"},
                fusion_out, err),
            0)
      << err.str();
  EXPECT_EQ(fusion_out.str().find("PLAN poisson_mesher"), std::string::npos);
}

TEST(ColiteCompareExeTest, WritesFixedColmapDifferenceReports) {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() / "colite_compare_colmap_test";
  const std::filesystem::path reference = root / "reference";
  const std::filesystem::path candidate = root / "candidate";
  const std::filesystem::path output = root / "report";
  std::filesystem::remove_all(root);
  WriteTinyColmapModel(reference, false);
  WriteTinyColmapModel(candidate, true);
  WriteColmapReconstruction(candidate, ReadColmapTextReconstruction(candidate),
                            ColmapModelFormat::kBinary);

  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(exe::RunCompareColmap({"compare_colmap",
                                   "--reference_path",
                                   reference.string(),
                                   "--candidate_path",
                                   candidate.string(),
                                   "--output_path",
                                   output.string()},
                                  out,
                                  err),
            0)
      << err.str();

  EXPECT_NE(out.str().find("Common images: 2"), std::string::npos);
  EXPECT_NE(out.str().find("Common points by id: 1"), std::string::npos);
  const std::string image_diff = ReadTextFile(output / "image_diff.tsv");
  const std::string pair_diff = ReadTextFile(output / "pair_diff.tsv");
  const std::string point_diff = ReadTextFile(output / "point_diff.tsv");
  const std::string pose_diff = ReadTextFile(output / "pose_diff.tsv");
  const std::string summary = ReadTextFile(output / "summary.txt");

  EXPECT_NE(image_diff.find("image_name\tstatus\tref_image_id"),
            std::string::npos);
  EXPECT_NE(image_diff.find("image3.jpg\tcandidate_only"), std::string::npos);
  EXPECT_NE(pair_diff.find("image1.jpg\timage2.jpg\t1\t2\t1"),
            std::string::npos);
  EXPECT_NE(point_diff.find("1\tcommon\t0.1"), std::string::npos);
  EXPECT_NE(point_diff.find("2\treference_only"), std::string::npos);
  EXPECT_NE(point_diff.find("3\tcandidate_only"), std::string::npos);
  EXPECT_NE(pose_diff.find("image1.jpg\t1\t1\t0\t0.1\t0.1"),
            std::string::npos);
  EXPECT_NE(summary.find("common_pairs 1"), std::string::npos);
  EXPECT_NE(summary.find("candidate_only_images 1"), std::string::npos);
  // Explicit format selection also allows comparing TXT and BIN in one folder.
  WriteColmapReconstruction(reference, ReadColmapTextReconstruction(reference),
                            ColmapModelFormat::kTextAndBinary);
  std::ostringstream same_out, same_err;
  EXPECT_EQ(exe::RunCompareColmap({"compare_colmap", "--reference_path",
              reference.string(), "--candidate_path", reference.string(),
              "--reference_format", "txt", "--candidate_format", "bin",
              "--output_path", (root / "same_report").string()},
              same_out, same_err), 0) << same_err.str();
  EXPECT_NE(same_out.str().find("Common points by id: 2"), std::string::npos);
  std::filesystem::remove_all(root);
}

TEST(ColiteCommandExeTest, DispatchesCompareColmapHelp) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "compare_colmap", "--help"}, out, err),
            0);
  EXPECT_NE(out.str().find("colite compare_colmap"), std::string::npos);
  EXPECT_NE(out.str().find("--reference_path"), std::string::npos);
  EXPECT_TRUE(err.str().empty());
}

TEST(ColiteCommandExeTest, RejectsUnknownSubcommand) {
  std::ostringstream out;
  std::ostringstream err;

  EXPECT_EQ(exe::RunColite({"colite", "does_not_exist"}, out, err), 1);
  EXPECT_TRUE(out.str().empty());
  EXPECT_NE(err.str().find("Unknown command: does_not_exist"),
            std::string::npos);
  EXPECT_NE(err.str().find("Available commands:"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorUsesExifFocalPriorByDefault) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_exif_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  const std::filesystem::path source_image =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(source_image)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }
  std::filesystem::copy_file(
      source_image,
      image_dir / "P1180166.JPG",
      std::filesystem::copy_options::overwrite_existing);

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--Colite.workspace_path",
       workspace.string(),
       "--SiftExtraction.max_num_features",
       "32",
       "--FeatureExtraction.num_threads",
       "1"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  ASSERT_EQ(cameras.size(), 1U);
  ASSERT_EQ(cameras[0].params.size(), 4U);
  EXPECT_EQ(cameras[0].model_id, CameraModelId::kSimpleRadial);
  EXPECT_NEAR(cameras[0].params[0], 2484.86249133349, 1e-6);
  EXPECT_TRUE(cameras[0].has_prior_focal_length);
}

TEST(SfmCommandExeTest,
     FeatureExtractorCreatesIndividualCamerasForMixedImageDimensions) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_individual_camera_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image1.png").string(),
      cv::Mat(80, 100, CV_8UC3, cv::Scalar(20, 40, 60))));
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image2.png").string(),
      cv::Mat(60, 120, CV_8UC3, cv::Scalar(60, 40, 20))));

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--Colite.workspace_path",
       workspace.string(),
       "--SiftExtraction.max_num_features",
       "32",
       "--FeatureExtraction.num_threads",
       "1",
       "--Colite.feature_extractor",
       "vlsift"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  const std::vector<Image> images = ReadWorkspaceImages(workspace);
  ASSERT_EQ(cameras.size(), 2U);
  ASSERT_EQ(images.size(), 2U);
  EXPECT_EQ(cameras[0].width, 100U);
  EXPECT_EQ(cameras[0].height, 80U);
  EXPECT_EQ(cameras[1].width, 120U);
  EXPECT_EQ(cameras[1].height, 60U);
  EXPECT_EQ(images[0].camera_id, cameras[0].id);
  EXPECT_EQ(images[1].camera_id, cameras[1].id);
  EXPECT_NE(out.str().find("Cameras: 2"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorSharedModeRejectsMixedDimensions) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_shared_camera_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image1.png").string(),
      cv::Mat(80, 100, CV_8UC3, cv::Scalar(20, 40, 60))));
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image2.png").string(),
      cv::Mat(60, 120, CV_8UC3, cv::Scalar(60, 40, 20))));

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--Colite.workspace_path",
       workspace.string(),
       "--ImageReader.single_camera",
       "1"},
      out,
      err);

  EXPECT_NE(exit_code, 0);
  EXPECT_NE(err.str().find("identical image dimensions"), std::string::npos);
}

TEST(SfmCommandExeTest, FeatureExtractorAcceptsCameraModelAndParameters) {
  const std::filesystem::path workspace =
      std::filesystem::temp_directory_path() /
      "colite_feature_extractor_camera_model_test";
  const std::filesystem::path image_dir = workspace / "images";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(image_dir);
  ASSERT_TRUE(cv::imwrite(
      (image_dir / "image.png").string(),
      cv::Mat(80, 100, CV_8UC3, cv::Scalar(20, 40, 60))));

  std::ostringstream out;
  std::ostringstream err;
  const int exit_code = exe::RunFeatureExtractor(
      {"feature_extractor",
       "--image_path",
       image_dir.string(),
       "--Colite.workspace_path",
       workspace.string(),
       "--FeatureExtraction.type",
       "SIFT",
       "--ImageReader.camera_model",
       "OPENCV",
       "--ImageReader.camera_params",
       "90,91,50,40,0.01,-0.002,0.0001,-0.0002",
       "--ImageReader.single_camera_per_image",
       "1",
       "--SiftExtraction.max_num_features",
       "16",
       "--FeatureExtraction.num_threads",
       "1"},
      out,
      err);

  EXPECT_EQ(exit_code, 0) << err.str();
  const std::vector<Camera> cameras = ReadWorkspaceCameras(workspace);
  ASSERT_EQ(cameras.size(), 1U);
  EXPECT_EQ(cameras[0].model_id, CameraModelId::kOpenCV);
  EXPECT_EQ(cameras[0].params,
            (std::vector<double>{90, 91, 50, 40, 0.01, -0.002,
                                 0.0001, -0.0002}));
}

TEST(CameraPriorTest, EstimatesFocalLengthFromJpegExif) {
  const std::filesystem::path image_path =
      std::filesystem::path(COLITE_SOURCE_DIR) / "data" / "building-18" /
      "P1180166.JPG";
  if (!std::filesystem::exists(image_path)) {
    GTEST_SKIP() << "Local building-18 dataset is not available.";
  }

  EXPECT_NEAR(EstimateInitialFocalLengthFromImage(image_path, 0.0),
              2484.86249133349,
              1e-6);
  const ExifCameraMetadata metadata = ReadExifCameraMetadata(image_path);
  EXPECT_TRUE(metadata.make.has_value());
  EXPECT_TRUE(metadata.model.has_value());
  EXPECT_TRUE(metadata.HasAutoCameraIdentity());
}

}  // namespace
TEST(SfmWorkspaceTest, PreservesOptionalVerifiedGeometryAndAllColmapConfigurations) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_optional_geometry_test";
  std::filesystem::remove_all(workspace);
  MatchGraph graph;
  for (int config = 0; config <= 9; ++config) {
    TwoViewGeometry geometry;
    geometry.type = static_cast<TwoViewGeometry::Type>(config);
    if (config % 2) geometry.F = Eigen::Matrix3d::Zero();
    if (config % 3) {
      Eigen::Matrix3d matrix;
      matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;
      geometry.E = matrix;
    }
    if (config % 4) geometry.H = Eigen::Matrix3d::Identity();
    if (config % 2) geometry.cam2_from_cam1 = Pose3{}; // Present pure rotation / zero translation.
    if (config > 1) geometry.inlier_matches = {{2, 7}, {9, 4}};
    SetVerifiedGeometry(&graph, 1, 2 + config, geometry);
  }
  WriteWorkspaceVerifiedGeometries(workspace, graph);
  MatchGraph loaded;
  ReadWorkspaceVerifiedGeometries(workspace, &loaded);
  ASSERT_EQ(loaded.verified_geometries.size(), 10U);
  for (const auto& [pair, original] : graph.verified_geometries) {
    const auto& result = loaded.verified_geometries.at(pair);
    EXPECT_EQ(result.type, original.type);
    EXPECT_EQ(result.cam2_from_cam1.has_value(), original.cam2_from_cam1.has_value());
    for (const auto& matrices : {std::make_pair(&original.F, &result.F),
                                std::make_pair(&original.E, &result.E),
                                std::make_pair(&original.H, &result.H)}) {
      ASSERT_EQ(matrices.first->has_value(), matrices.second->has_value());
      if (*matrices.first) EXPECT_TRUE((**matrices.first).isApprox(**matrices.second, 0.0));
    }
    EXPECT_EQ(result.inlier_matches.size(), original.inlier_matches.size());
    const int config = static_cast<int>(original.type);
    EXPECT_EQ(result.HasUsableCorrespondences(), config >= 2 && config != 7 && config != 9);
  }
  // The pair count is authoritative; truncated correspondence files must fail.
  const auto path = workspace / "matches/verified_matches.tsv";
  std::ifstream input(path);
  std::string text((std::istreambuf_iterator<char>(input)), {});
  input.close();
  text.erase(text.find_last_of('\n', text.size() - 2) + 1);
  std::ofstream(path) << text;
  loaded.verified_geometries.begin()->second.type = TwoViewGeometry::Type::kMultiple;
  EXPECT_THROW(ReadWorkspaceVerifiedGeometries(workspace, &loaded), std::runtime_error);
  EXPECT_EQ(loaded.verified_geometries.begin()->second.type, TwoViewGeometry::Type::kMultiple);
  std::filesystem::remove_all(workspace);
}

TEST(SfmWorkspaceTest, ReadsLegacyGeometryWithoutInventingMissingPose) {
  const auto workspace = std::filesystem::temp_directory_path() / "colite_legacy_geometry_test";
  std::filesystem::remove_all(workspace);
  std::filesystem::create_directories(workspace / "matches");
  std::ofstream(workspace / "matches/verified_matches.tsv")
      << "1\t2\t6\t-1\t1\t0\t0\t0\t0\t0\t0\t-1\t-1\n"
         "3\t2\t2\t3\t1\t0\t0\t0\t1\t0\t0\t4\t7\n";
  MatchGraph graph;
  ReadWorkspaceVerifiedGeometries(workspace, &graph);
  const auto& multiple = graph.verified_geometries.at(ImagePair::Ordered(1, 2));
  EXPECT_EQ(multiple.type, TwoViewGeometry::Type::kMultiple);
  EXPECT_FALSE(multiple.cam2_from_cam1);
  EXPECT_FALSE(multiple.F);
  const auto& reversed = graph.verified_geometries.at(ImagePair::Ordered(2, 3));
  EXPECT_DOUBLE_EQ(reversed.cam2_from_cam1->t_cw.x(), -1.0);
  EXPECT_EQ(reversed.inlier_matches[0].idx1, 7U);
  std::filesystem::remove_all(workspace);
}

TEST(SfmCommandExeTest, AlignsAllExposedColmap4TwoViewOptionsAndCacheIdentity) {
  const std::vector<std::pair<std::string, std::string>> choices = {
    {"min_num_inliers", "0"}, {"multiple_models", "1"}, {"compute_relative_pose", "1"},
    {"detect_watermark", "0"}, {"multiple_ignore_watermark", "0"},
    {"watermark_detection_max_error", "2"}, {"filter_stationary_matches", "1"},
    {"stationary_matches_max_error", "0"}, {"max_error", "2"}, {"confidence", "1"},
    {"max_num_trials", "100"}, {"min_inlier_ratio", "0.5"}, {"random_seed", "0"}};
  const auto baseline = controllers::SfmGeometryKey(TwoViewGeometryOptions{});
  std::ostringstream help;
  exe::PrintSfmOptionsHelp(help, exe::SfmOptionScope::kMatching);
  for (const auto& [name, value] : choices) {
    TwoViewGeometryOptions options;
    ASSERT_TRUE(controllers::ApplySfmGeometryOption("--TwoViewGeometry." + name, value, &options));
    EXPECT_NE(controllers::SfmGeometryKey(options), baseline) << name;
    EXPECT_NE(help.str().find("--TwoViewGeometry." + name), std::string::npos) << name;
  }
  TwoViewGeometryOptions boundary;
  EXPECT_TRUE(controllers::ApplySfmGeometryOption("--TwoViewGeometry.confidence", "0", &boundary));
  for (const auto& [name, value] : std::vector<std::pair<std::string, std::string>>{
      {"min_num_inliers", "-1"}, {"compute_relative_pose", "2"}, {"random_seed", "-2"},
      {"min_inlier_ratio", "1.1"}, {"watermark_detection_max_error", "0"}, {"stationary_matches_max_error", "-1"}}) {
    std::ostringstream out, err;
    EXPECT_EQ(exe::RunColite({"colite", "exhaustive_matcher", "--TwoViewGeometry." + name, value}, out, err), 1);
    EXPECT_EQ(err.str().find("required"), std::string::npos) << err.str();
  }
}

}  // namespace colite
