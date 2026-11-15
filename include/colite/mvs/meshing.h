#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "colite/core/reconstruction.h"
#include "colite/mvs/dense_cloud.h"

namespace colite::mvs {

struct PoissonMeshingOptions {
  double point_weight = 1.0;
  int depth = 13;
  // Matches COLMAP 4.0 / current PoissonRecon: enable vertex colors.
  bool color = true;
  double trim = 10.0;
  int num_threads = -1;

  bool Check() const;
};

bool IsPoissonMeshingAvailable();

// Runs the bundled PoissonRecon implementation on an oriented
// point cloud stored as PLY. The input must contain x/y/z and nx/ny/nz.
bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::filesystem::path& input_path,
                    const std::filesystem::path& output_path);

struct DelaunayMeshingOptions {
  double max_proj_dist = 20.0;
  double max_depth_dist = 0.05;
  double visibility_sigma = 3.0;
  double distance_sigma_factor = 1.0;
  double quality_regularization = 1.0;
  double max_side_length_factor = 25.0;
  double max_side_length_percentile = 95.0;
  int num_threads = -1;

  bool Check() const;
};

bool IsDelaunayMeshingAvailable();

// Visibility-aware Delaunay meshing following Labatut et al. The visibility
// entries contain indices into views, as produced by stereo fusion.
DenseTriangleMesh DelaunayMeshing(
    const DelaunayMeshingOptions& options,
    const std::vector<FusedPoint>& points,
    const std::vector<std::vector<std::uint32_t>>& points_visibility,
    const std::vector<PatchMatchView>& views);

// Uses registered poses and sparse point tracks directly, without dense maps
// or a separately stored visibility file. IDs need not be contiguous.
DenseTriangleMesh SparseDelaunayMeshing(
    const DelaunayMeshingOptions& options,
    const Reconstruction& reconstruction);

struct MeshSimplificationOptions {
  // Fraction of faces retained, in (0, 1]. Matches COLMAP 4.0.
  double target_face_ratio = 0.1;
  // Zero disables the accumulated quadric-error limit.
  double max_error = 0.0;
  double boundary_weight = 1000.0;
  bool interpolate_colors = true;
  // CPU budget for initialization; -1 selects available cores.
  int num_threads = -1;

  bool Check() const;
};

// Simplifies a triangular manifold mesh using quadric-error edge collapses.
// Invalid indices are rejected; degenerate faces are discarded when reducing.
// Weighted boundary quadrics, link and normal-flip checks protect the surface.
DenseTriangleMesh SimplifyMesh(
    const DenseTriangleMesh& mesh,
    const MeshSimplificationOptions& options = {});

// Reads ASCII or binary little/big-endian triangular PLY meshes.
DenseTriangleMesh ReadDenseTriangleMeshPly(
    const std::filesystem::path& path,
    std::vector<Eigen::Vector2f>* face_uvs = nullptr,
    std::string* texture_file = nullptr);

}  // namespace colite::mvs
