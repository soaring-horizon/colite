#include "colite/sfm/similarity_transform.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include "colite/core/geometry.h"

namespace colite {
namespace {

TEST(SimilarityTransformTest, NormalizationPreservesProjectionAndUsesCameraExtent) {
  Reconstruction model;
  Camera camera; camera.id=1; camera.model_id=CameraModelId::kSimplePinhole;
  camera.width=100; camera.height=100; camera.params={80,50,50};
  model.AddCamera(camera);
  const Eigen::Vector3d point(11,21,35);
  Track track;
  for (ImageId id : {1,2}) {
    Image image; image.id=id; image.camera_id=1; image.registered=true;
    Pose3 pose; pose.t_cw=-Eigen::Vector3d(10+2*(id-1),20,30);
    image.cam_from_world=pose;
    image.points2d.push_back({*ProjectWorldPoint(camera,pose,point),0,kInvalidPoint3DId});
    model.AddImage(image); track.AddElement(id,0);
  }
  const auto point_id=model.AddPoint3D(point,track);
  auto fixed_scale=model;
  const auto normalized=NormalizeReconstruction(&model);
  ASSERT_TRUE(normalized.valid);
  EXPECT_DOUBLE_EQ(normalized.scale,5);
  EXPECT_LT((model.GetImage(1).cam_from_world->CameraCenter()-Eigen::Vector3d(-5,0,0)).norm(),1e-12);
  EXPECT_LT((model.GetImage(2).cam_from_world->CameraCenter()-Eigen::Vector3d(5,0,0)).norm(),1e-12);
  for (ImageId id : {1,2}) {
    const auto& image=model.GetImage(id);
    EXPECT_LT(*ReprojectionError(camera,*image.cam_from_world,model.GetPoint3D(point_id).xyz,image.points2d[0].xy),1e-12);
  }
  const auto centered=NormalizeReconstruction(&fixed_scale,true);
  EXPECT_DOUBLE_EQ(centered.scale,1);
  EXPECT_LT((fixed_scale.GetImage(1).cam_from_world->CameraCenter()-Eigen::Vector3d(-1,0,0)).norm(),1e-12);
  EXPECT_THROW(NormalizeReconstruction(&fixed_scale,false,0),std::invalid_argument);
}

TEST(SimilarityTransformTest, NormalizationTrimsCoordinateOutliersAndHandlesOneCamera) {
  Reconstruction model;
  Camera camera; camera.id=1; camera.model_id=CameraModelId::kSimplePinhole;
  camera.width=100; camera.height=100; camera.params={80,50,50}; model.AddCamera(camera);
  for (ImageId id=1;id<=31;++id) {
    Image image; image.id=id; image.camera_id=1; image.registered=true;
    image.cam_from_world=Pose3{};
    image.cam_from_world->t_cw.x()=-(id==31?1e6:static_cast<double>(id-1));
    model.AddImage(image);
    if (id==1) {
      const auto no_op=NormalizeReconstruction(&model);
      EXPECT_DOUBLE_EQ(no_op.scale,1);
      EXPECT_DOUBLE_EQ(no_op.translation.norm(),0);
    }
  }
  const auto normalized=NormalizeReconstruction(&model);
  EXPECT_NEAR(normalized.scale,10.0/24,1e-12);
  EXPECT_NEAR(normalized.translation.x(),-15*10.0/24,1e-12);
}

TEST(SimilarityTransformTest, RecoversKnownTransform) {
  const std::vector<Eigen::Vector3d> source = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const double scale = 2.5;
  const Eigen::Vector3d translation(4.0, -3.0, 7.0);
  std::vector<Eigen::Vector3d> target;
  for (const Eigen::Vector3d& point : source) {
    target.push_back(scale * rotation * point + translation);
  }

  const SimilarityTransform3 transform =
      EstimateSimilarityTransform(source, target);

  ASSERT_TRUE(transform.valid);
  EXPECT_NEAR(transform.scale, scale, 1e-10);
  EXPECT_LT((transform.rotation - rotation).norm(), 1e-10);
  EXPECT_LT((transform.translation - translation).norm(), 1e-10);
}

TEST(SimilarityTransformTest, AlignsCameraCentersToPositionPriors) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kSimplePinhole;
  camera.width = 100;
  camera.height = 100;
  camera.params = {80.0, 50.0, 50.0};
  reconstruction.AddCamera(camera);
  const std::vector<Eigen::Vector3d> centers = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0}, {0.5, 0.5, 1.0}};
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()).toRotationMatrix();
  for (std::size_t idx = 0; idx < centers.size(); ++idx) {
    Image image;
    image.id = idx + 1;
    image.camera_id = 1;
    image.registered = true;
    image.cam_from_world = Pose3{};
    image.cam_from_world->t_cw = -centers[idx];
    image.prior_position = 3.0 * rotation * centers[idx] +
                           Eigen::Vector3d(20.0, -5.0, 2.0);
    reconstruction.AddImage(image);
  }

  const PositionPriorAlignmentSummary summary =
      AlignReconstructionToPositionPriors(0.01, &reconstruction);

  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_inliers, centers.size());
  EXPECT_NEAR(summary.scale, 3.0, 1e-10);
  EXPECT_LT(summary.mean_error, 1e-9);
}

TEST(SimilarityTransformTest, RejectsPositionPriorOutlier) {
  Reconstruction reconstruction;
  Camera camera;
  camera.id = 1;
  camera.model_id = CameraModelId::kSimplePinhole;
  camera.width = 100;
  camera.height = 100;
  camera.params = {80.0, 50.0, 50.0};
  reconstruction.AddCamera(camera);
  const std::vector<Eigen::Vector3d> centers = {
      {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 3.0, 0.0},
      {1.0, 1.0, 2.0}, {2.0, 2.0, 1.0}};
  for (std::size_t idx = 0; idx < centers.size(); ++idx) {
    Image image;
    image.id = idx + 1;
    image.camera_id = 1;
    image.registered = true;
    image.cam_from_world = Pose3{};
    image.cam_from_world->t_cw = -centers[idx];
    image.prior_position = 2.0 * centers[idx] + Eigen::Vector3d(4.0, 5.0, 6.0);
    if (idx == centers.size() - 1) *image.prior_position += Eigen::Vector3d(100.0, 0.0, 0.0);
    reconstruction.AddImage(std::move(image));
  }

  const PositionPriorAlignmentSummary summary =
      AlignReconstructionToPositionPriors(0.1, &reconstruction);

  ASSERT_TRUE(summary.success);
  EXPECT_EQ(summary.num_inliers, 4U);
  EXPECT_NEAR(summary.scale, 2.0, 1e-8);
  EXPECT_LT(summary.mean_error, 1e-8);
}

}  // namespace
}  // namespace colite
