#include "colite/mvs/meshing.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>

#include "colite/core/parallel.h"

namespace colite::mvs {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<FusedPoint> MakeSpherePoints(std::size_t count) {
  std::vector<FusedPoint> points;
  points.reserve(count);
  const double golden_angle = kPi * (3.0 - std::sqrt(5.0));
  for (std::size_t i = 0; i < count; ++i) {
    const double y =
        1.0 - 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(count);
    const double radius = std::sqrt(std::max(0.0, 1.0 - y * y));
    const double angle = golden_angle * static_cast<double>(i);
    FusedPoint point;
    point.xyz =
        Eigen::Vector3d(radius * std::cos(angle), y, radius * std::sin(angle));
    point.normal = point.xyz.cast<float>().normalized();
    point.color = Eigen::Matrix<std::uint8_t, 3, 1>(128, 180, 220);
    points.push_back(point);
  }
  return points;
}

void WriteOrientedPointCloudPly(const std::filesystem::path& path,
                                const std::vector<FusedPoint>& points) {
  std::ofstream file(path);
  ASSERT_TRUE(file.is_open());
  file << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n"
       << "property double x\nproperty double y\nproperty double z\n"
       << "property float nx\nproperty float ny\nproperty float nz\n"
       << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
       << "end_header\n";
  for (const FusedPoint& point : points) {
    file << point.xyz.x() << ' ' << point.xyz.y() << ' ' << point.xyz.z() << ' '
         << point.normal.x() << ' ' << point.normal.y() << ' '
         << point.normal.z() << ' ' << static_cast<int>(point.color.x()) << ' '
         << static_cast<int>(point.color.y()) << ' '
         << static_cast<int>(point.color.z()) << '\n';
  }
}

DenseTriangleMesh MakeGridMesh(int side) {
  DenseTriangleMesh mesh;
  for (int y = 0; y < side; ++y) {
    for (int x = 0; x < side; ++x) {
      FusedPoint point;
      point.xyz = Eigen::Vector3d(x, y, 0.05 * std::sin(0.4 * x) *
                                           std::sin(0.4 * y));
      point.normal = Eigen::Vector3f::UnitZ();
      point.color = Eigen::Matrix<std::uint8_t, 3, 1>(80, 120, 160);
      mesh.vertices.push_back(point);
    }
  }
  for (int y = 0; y + 1 < side; ++y) {
    for (int x = 0; x + 1 < side; ++x) {
      const int a = y * side + x;
      const int b = a + 1;
      const int c = a + side;
      const int d = c + 1;
      mesh.triangles.emplace_back(a, b, d);
      mesh.triangles.emplace_back(a, d, c);
    }
  }
  return mesh;
}

TEST(PoissonMeshingTest, SingleThreadCallsAreRepeatable) {
  if (!IsPoissonMeshingAvailable()) GTEST_SKIP();
  const auto root = std::filesystem::temp_directory_path() / "colite_poisson_single_thread";
  std::filesystem::create_directories(root);
  WriteOrientedPointCloudPly(root / "input.ply", MakeSpherePoints(256));
  PoissonMeshingOptions options;
  options.num_threads = 1;
  options.depth = 4;
  options.trim = 0;
  ASSERT_TRUE(PoissonMeshing(options, root / "input.ply", root / "a.ply"));
  ASSERT_TRUE(PoissonMeshing(options, root / "input.ply", root / "b.ply"));
  const auto contents = [](const auto& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  };
  EXPECT_EQ(contents(root / "a.ply"), contents(root / "b.ply"));
  options.color = false;
  options.depth = 5;
  ASSERT_TRUE(PoissonMeshing(options, root / "input.ply", root / "no_color.ply"));
  EXPECT_EQ(contents(root / "no_color.ply").find("property uchar red"), std::string::npos);
  options.color = true;
  options.depth = 4;
  ASSERT_TRUE(PoissonMeshing(options, root / "input.ply", root / "restored.ply"));
  EXPECT_EQ(contents(root / "a.ply"), contents(root / "restored.ply"));
  std::filesystem::remove_all(root);
}

TEST(PoissonMeshingOptionsTest, ValidatesParameters) {
  PoissonMeshingOptions options;
  EXPECT_TRUE(options.Check());
  options.depth = 0;
  EXPECT_FALSE(options.Check());
  options.depth = 8;
  options.trim = -1.0;
  EXPECT_FALSE(options.Check());
  options.trim = 0.0;
  options.point_weight = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(options.Check());
}

TEST(PoissonMeshingTest, RejectsMissingInput) {
  const std::filesystem::path missing = std::filesystem::temp_directory_path() /
                                        "colite_missing_poisson_input.ply";
  EXPECT_THROW(PoissonMeshing({}, missing, missing.parent_path() / "out.ply"),
               std::invalid_argument);
}

TEST(PoissonMeshingTest, ReconstructsSyntheticOrientedSphere) {
  if (!IsPoissonMeshingAvailable()) {
    GTEST_SKIP() << "Poisson meshing is disabled in this build.";
  }
  const std::filesystem::path root = std::filesystem::temp_directory_path() /
                                     "colite_poisson_meshing_execution_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path input = root / "sphere.ply";
  const std::filesystem::path output = root / "mesh.ply";
  WriteOrientedPointCloudPly(input, MakeSpherePoints(256));

  PoissonMeshingOptions options;
  options.depth = 5;
  options.color = false;
  options.trim = 0.0;
  options.num_threads = 1;
  EXPECT_TRUE(PoissonMeshing(options, input, output));
  ASSERT_TRUE(std::filesystem::exists(output));
  EXPECT_GT(std::filesystem::file_size(output), 100U);
  const DenseTriangleMesh mesh = ReadDenseTriangleMeshPly(output);
  ASSERT_FALSE(mesh.triangles.empty());
  MeshSimplificationOptions simplification_options;
  simplification_options.target_face_ratio = 0.8;
  const DenseTriangleMesh simplified =
      SimplifyMesh(mesh, simplification_options);
  EXPECT_LT(simplified.triangles.size(), mesh.triangles.size());
  std::filesystem::remove_all(root);
}

TEST(DelaunayMeshingOptionsTest, ValidatesParameters) {
  DelaunayMeshingOptions options;
  EXPECT_TRUE(options.Check());
  options.max_depth_dist = 1.01;
  EXPECT_FALSE(options.Check());
  options.max_depth_dist = 0.05;
  options.visibility_sigma = 0.0;
  EXPECT_FALSE(options.Check());
  options.visibility_sigma = 3.0;
  options.max_side_length_percentile = 101.0;
  EXPECT_FALSE(options.Check());
}

TEST(DelaunayMeshingTest, RejectsMismatchedVisibility) {
  std::vector<FusedPoint> points(1);
  points.front().xyz = Eigen::Vector3d::Zero();
  if (IsDelaunayMeshingAvailable()) {
    EXPECT_THROW(DelaunayMeshing({}, points, {}, {}), std::invalid_argument);
  } else {
    EXPECT_THROW(DelaunayMeshing({}, points, {}, {}), std::runtime_error);
  }
}

TEST(DelaunayMeshingTest, ReconstructsSyntheticVisibleSphere) {
  if (!IsDelaunayMeshingAvailable()) {
    GTEST_SKIP() << "CGAL Delaunay meshing is disabled in this build.";
  }
  const std::vector<FusedPoint> points = MakeSpherePoints(48);
  const std::array<Eigen::Vector3d, 6> centers = {
      Eigen::Vector3d(3.0, 0.0, 0.0), Eigen::Vector3d(-3.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 3.0, 0.0), Eigen::Vector3d(0.0, -3.0, 0.0),
      Eigen::Vector3d(0.0, 0.0, 3.0), Eigen::Vector3d(0.0, 0.0, -3.0)};
  std::vector<PatchMatchView> views;
  for (std::size_t i = 0; i < centers.size(); ++i) {
    PatchMatchView view;
    view.image_id = static_cast<ImageId>(i + 1);
    view.cam_from_world.t_cw = -centers[i];
    views.push_back(view);
  }
  std::vector<std::vector<std::uint32_t>> visibility(points.size());
  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    for (std::size_t i = 0; i < views.size(); ++i) {
      if (points[point_index].xyz.dot(centers[i] - points[point_index].xyz) > 0) {
        visibility[point_index].push_back(static_cast<std::uint32_t>(i));
      }
    }
  }

  DelaunayMeshingOptions options;
  options.max_proj_dist = 0.0;
  options.max_side_length_factor = 1000.0;
  options.quality_regularization = 0.0;
  options.distance_sigma_factor = 2.0;
  const DenseTriangleMesh mesh =
      DelaunayMeshing(options, points, visibility, views);
  EXPECT_FALSE(mesh.vertices.empty());
  EXPECT_FALSE(mesh.triangles.empty());
  const auto repeated = DelaunayMeshing(options, points, visibility, views);
  ASSERT_EQ(mesh.vertices.size(), repeated.vertices.size());
  ASSERT_EQ(mesh.triangles.size(), repeated.triangles.size());
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    EXPECT_TRUE(mesh.vertices[i].xyz.isApprox(repeated.vertices[i].xyz, 0.0));
  }
  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    EXPECT_TRUE(mesh.triangles[i].isApprox(repeated.triangles[i], 0));
  }

  Reconstruction sparse;
  Camera camera;
  camera.id = 17;
  camera.model_id = CameraModelId::kPinhole;
  camera.width = camera.height = 640;
  camera.params = {400, 400, 320, 320};
  sparse.AddCamera(camera);
  for (std::size_t i = 0; i < views.size(); ++i) {
    Image image;
    image.id = static_cast<ImageId>(11 + 13 * i);
    image.camera_id = camera.id;
    image.name = std::to_string(image.id) + ".png";
    image.cam_from_world = views[i].cam_from_world;
    image.registered = true;
    image.points2d.resize(points.size());
    sparse.AddImage(std::move(image));
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    Point3D point;
    point.id = static_cast<Point3DId>(101 + 23 * i);
    point.xyz = points[i].xyz;
    point.color = points[i].color;
    for (const auto view_index : visibility[i]) {
      point.track.AddElement(static_cast<ImageId>(11 + 13 * view_index), static_cast<Point2DIdx>(i));
    }
    sparse.AddPoint3D(std::move(point));
  }
  const auto from_sparse = SparseDelaunayMeshing(options, sparse);
  ASSERT_EQ(mesh.vertices.size(), from_sparse.vertices.size());
  ASSERT_EQ(mesh.triangles.size(), from_sparse.triangles.size());
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    EXPECT_TRUE(mesh.vertices[i].xyz.isApprox(from_sparse.vertices[i].xyz, 0));
  }
  for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
    EXPECT_TRUE(mesh.triangles[i].isApprox(from_sparse.triangles[i], 0));
  }
  auto invalid = sparse;
  const auto first_observation = sparse.GetPoint3D(101).track.elements.front();
  invalid.GetPoint3D(101).track.elements.push_back(first_observation);
  EXPECT_THROW(SparseDelaunayMeshing(options, invalid), std::invalid_argument);
  invalid = sparse;
  invalid.GetImage(first_observation.image_id).points2d[first_observation.point2d_idx].point3d_id = kInvalidPoint3DId;
  EXPECT_THROW(SparseDelaunayMeshing(options, invalid), std::invalid_argument);
  invalid = sparse;
  invalid.DeregisterImage(first_observation.image_id);
  EXPECT_THROW(SparseDelaunayMeshing(options, invalid), std::invalid_argument);
}

TEST(MeshSimplificationTest, KeepsSingleTriangleForSubunitTargetRatio) {
  DenseTriangleMesh mesh;
  for (const auto& point : {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0)}) {
    FusedPoint vertex;
    vertex.xyz = point;
    mesh.vertices.push_back(vertex);
  }
  mesh.triangles.emplace_back(0, 1, 2);
  EXPECT_EQ(SimplifyMesh(mesh).triangles.size(), 1U);
}

TEST(MeshSimplificationTest, KeepsTetrahedronTopologyUnderAggressiveTarget) {
  DenseTriangleMesh mesh;
  for (const auto& position : {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0),
                              Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 1)}) {
    FusedPoint vertex;
    vertex.xyz = position;
    mesh.vertices.push_back(vertex);
  }
  mesh.triangles = {Eigen::Vector3i(0, 2, 1), Eigen::Vector3i(0, 1, 3),
                    Eigen::Vector3i(1, 2, 3), Eigen::Vector3i(2, 0, 3)};
  const auto reduced = SimplifyMesh(mesh);
  EXPECT_EQ(reduced.vertices.size(), 4U);
  EXPECT_EQ(reduced.triangles.size(), 4U);
}

TEST(MeshSimplificationOptionsTest, ValidatesParameters) {
  MeshSimplificationOptions options;
  EXPECT_TRUE(options.Check());
  options.target_face_ratio = 0.0;
  EXPECT_FALSE(options.Check());
  options.target_face_ratio = 0.5;
  options.max_error = -1.0;
  EXPECT_FALSE(options.Check());
  options.max_error = 0.0;
  options.boundary_weight = -1.0;
  EXPECT_FALSE(options.Check());
}

TEST(MeshSimplificationTest, ReducesGridWithValidTriangles) {
  const DenseTriangleMesh mesh = MakeGridMesh(12);
  MeshSimplificationOptions options;
  options.target_face_ratio = 0.35;
  const DenseTriangleMesh simplified = SimplifyMesh(mesh, options);
  EXPECT_LT(simplified.triangles.size(), mesh.triangles.size());
  EXPECT_LE(simplified.triangles.size(),
            static_cast<std::size_t>(mesh.triangles.size() * 0.40));
  EXPECT_FALSE(simplified.vertices.empty());
  for (const Eigen::Vector3i& triangle : simplified.triangles) {
    EXPECT_GE(triangle.minCoeff(), 0);
    EXPECT_LT(triangle.maxCoeff(),
              static_cast<int>(simplified.vertices.size()));
    EXPECT_EQ(std::set<int>({triangle.x(), triangle.y(), triangle.z()}).size(),
              3U);
    const Eigen::Vector3d cross =
        (simplified.vertices[triangle.y()].xyz -
         simplified.vertices[triangle.x()].xyz)
            .cross(simplified.vertices[triangle.z()].xyz -
                   simplified.vertices[triangle.x()].xyz);
    EXPECT_GT(cross.squaredNorm(), 0.0);
  }
}

TEST(MeshSimplificationTest, WeightedQuadricsKeepOpenBoundaryClose) {
  const DenseTriangleMesh mesh = MakeGridMesh(14);
  MeshSimplificationOptions options;
  options.target_face_ratio = 0.2;
  const DenseTriangleMesh simplified = SimplifyMesh(mesh, options);

  const auto boundary = [](const DenseTriangleMesh& input) {
    std::map<std::array<int, 2>, int> counts;
    for (const auto& face : input.triangles) {
      for (int j = 0; j < 3; ++j) {
        const int a = face[j], b = face[(j + 1) % 3];
        ++counts[{std::min(a, b), std::max(a, b)}];
      }
    }
    std::vector<std::array<int, 2>> result;
    for (const auto& [edge, count] : counts) if (count == 1) result.push_back(edge);
    return result;
  };
  const auto original_boundary = boundary(mesh);
  const auto drift = [&](const DenseTriangleMesh& reduced) {
    double maximum = 0;
    for (const auto& edge : boundary(reduced)) {
      for (const int id : edge) {
        const auto& point = reduced.vertices[id].xyz;
        double closest = std::numeric_limits<double>::infinity();
        for (const auto& original : original_boundary) {
          const auto& a = mesh.vertices[original[0]].xyz;
          const Eigen::Vector3d direction = mesh.vertices[original[1]].xyz - a;
          const double t = std::clamp((point - a).dot(direction) / direction.squaredNorm(), 0.0, 1.0);
          closest = std::min(closest, (point - a - t * direction).norm());
        }
        maximum = std::max(maximum, closest);
      }
    }
    return maximum;
  };
  options.boundary_weight = 0;
  const auto unweighted = SimplifyMesh(mesh, options);
  EXPECT_LT(drift(simplified), 0.01);
  EXPECT_LT(drift(simplified), drift(unweighted) * 0.1);
}

TEST(MeshSimplificationTest, ColorModesAndThreadBudgetsPreserveGeometry) {
  auto mesh = MakeGridMesh(14);
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    mesh.vertices[i].color = i % 2 ? Eigen::Matrix<std::uint8_t, 3, 1>(255, 0, 0)
                                  : Eigen::Matrix<std::uint8_t, 3, 1>(0, 0, 255);
  }
  const int previous = parallel::getNumThreads();
  struct Restore { int threads; ~Restore() { parallel::setNumThreads(threads); } } restore{previous};
  parallel::setNumThreads(3);
  MeshSimplificationOptions options;
  options.target_face_ratio = .2;
  options.num_threads = 1;
  const auto interpolated = SimplifyMesh(mesh, options);
  EXPECT_EQ(parallel::getNumThreads(), 3);
  options.num_threads = 4;
  const auto threaded = SimplifyMesh(mesh, options);
  EXPECT_EQ(parallel::getNumThreads(), 3);
  options.interpolate_colors = false;
  const auto selected = SimplifyMesh(mesh, options);
  ASSERT_EQ(interpolated.vertices.size(), threaded.vertices.size());
  ASSERT_EQ(interpolated.vertices.size(), selected.vertices.size());
  ASSERT_EQ(interpolated.triangles.size(), threaded.triangles.size());
  bool blended = false;
  for (std::size_t i = 0; i < interpolated.vertices.size(); ++i) {
    EXPECT_TRUE(interpolated.vertices[i].xyz.isApprox(threaded.vertices[i].xyz, 0));
    EXPECT_TRUE(interpolated.vertices[i].xyz.isApprox(selected.vertices[i].xyz, 0));
    EXPECT_EQ(interpolated.vertices[i].color, threaded.vertices[i].color);
    const auto red = selected.vertices[i].color[0];
    EXPECT_TRUE(red == 0 || red == 255);
    blended |= interpolated.vertices[i].color[0] > 0 && interpolated.vertices[i].color[0] < 255;
  }
  EXPECT_TRUE(blended);
  for (std::size_t i = 0; i < interpolated.triangles.size(); ++i) {
    EXPECT_TRUE(interpolated.triangles[i].isApprox(threaded.triangles[i], 0));
  }
}

TEST(MeshSimplificationTest, RatioOnePreservesAttributesAndInvalidIndicesFail) {
  auto mesh = MakeGridMesh(3);
  mesh.vertices.push_back(mesh.vertices.front());
  mesh.vertices.back().xyz = Eigen::Vector3d(100, 200, 300);
  MeshSimplificationOptions options;
  options.target_face_ratio = 1;
  const auto unchanged = SimplifyMesh(mesh, options);
  ASSERT_EQ(unchanged.vertices.size(), mesh.vertices.size());
  EXPECT_TRUE(unchanged.vertices.back().xyz.isApprox(mesh.vertices.back().xyz, 0));
  EXPECT_TRUE(unchanged.vertices.front().normal.isApprox(mesh.vertices.front().normal, 0));
  mesh.triangles.front()[0] = -1;
  EXPECT_THROW(SimplifyMesh(mesh, options), std::invalid_argument);
}

TEST(MeshSimplificationTest, HonorsRatioAndErrorLimit) {
  const DenseTriangleMesh mesh = MakeGridMesh(10);
  MeshSimplificationOptions options;
  options.target_face_ratio = 0.9;
  options.target_face_ratio = 40.0 / mesh.triangles.size();
  const DenseTriangleMesh targeted = SimplifyMesh(mesh, options);
  EXPECT_LE(targeted.triangles.size(), 42U);

  options.max_error = 1e-30;
  const DenseTriangleMesh limited = SimplifyMesh(mesh, options);
  EXPECT_GE(limited.triangles.size(), targeted.triangles.size());
}

TEST(MeshSimplificationTest, ReadsTexturedPlyWithFractionalFaceCoordinates) {
  const auto path = std::filesystem::temp_directory_path() / "colite_fractional_uv.ply";
  {
    std::ofstream stream(path);
    stream << "ply\nformat ascii 1.0\ncomment TextureFile texture.png\n"
        "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nproperty list uchar float texcoord\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n3 0 1 2 6 0.1 0.2 0.3 0.4 0.5 0.6\n";
  }
  EXPECT_EQ(ReadDenseTriangleMeshPly(path).triangles.size(), 1U);
  std::filesystem::remove(path);
}

TEST(MeshSimplificationTest, ReadsWrittenAsciiPly) {
  const std::filesystem::path root = std::filesystem::temp_directory_path() /
                                     "colite_mesh_simplifier_ply_test";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  const std::filesystem::path path = root / "grid.ply";
  const DenseTriangleMesh expected = MakeGridMesh(4);
  WriteDenseTriangleMeshPly(path, expected);
  const DenseTriangleMesh actual = ReadDenseTriangleMeshPly(path);
  EXPECT_EQ(actual.vertices.size(), expected.vertices.size());
  EXPECT_EQ(actual.triangles.size(), expected.triangles.size());
  EXPECT_TRUE(
      actual.vertices.front().xyz.isApprox(expected.vertices.front().xyz));
  std::filesystem::remove_all(root);
}

TEST(MeshSimplificationTest, ReadsBinaryLittleEndianPly) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     "colite_binary_mesh_test.ply";
  std::ofstream file(path, std::ios::binary);
  file << "ply\nformat binary_little_endian 1.0\n"
       << "element vertex 3\nproperty float x\nproperty float y\n"
       << "property float z\nelement face 1\n"
       << "property list uchar int vertex_indices\nend_header\n";
  for (const std::array<float, 3>& point :
       {std::array<float, 3>{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}) {
    file.write(reinterpret_cast<const char*>(point.data()), sizeof(point));
  }
  const std::uint8_t count = 3;
  const std::array<std::int32_t, 3> indices = {0, 1, 2};
  file.write(reinterpret_cast<const char*>(&count), sizeof(count));
  file.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
  file.close();
  const DenseTriangleMesh mesh = ReadDenseTriangleMeshPly(path);
  ASSERT_EQ(mesh.vertices.size(), 3U);
  ASSERT_EQ(mesh.triangles.size(), 1U);
  EXPECT_EQ(mesh.triangles.front(), Eigen::Vector3i(0, 1, 2));
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace colite::mvs
