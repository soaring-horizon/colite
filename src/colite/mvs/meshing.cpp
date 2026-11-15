// Portions adapted from COLMAP.
// Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the names of ETH Zurich and UNC Chapel Hill nor the names of their
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colite/mvs/meshing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(COLITE_ENABLE_POISSON_MESHING)
#include "PoissonRecon.h"
#include "MultiThreading.h"
#include "SurfaceTrimmer.h"
#endif

#if defined(COLITE_ENABLE_DELAUNAY_MESHING)
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/intersections.h>

#include "min_st_graph_cut.h"
#endif

namespace colite::mvs {
namespace {

bool IsValidThreadCount(int count) { return count == -1 || count > 0; }

#if defined(COLITE_ENABLE_POISSON_MESHING)
std::vector<char*> MutableArguments(std::vector<std::string>* arguments) {
  std::vector<char*> result;
  result.reserve(arguments->size());
  for (std::string& argument : *arguments) {
    result.push_back(argument.data());
  }
  return result;
}

class ScopedOpenMpThreads {
 public:
  explicit ScopedOpenMpThreads(int requested) {
    previous_poisson_ = ::PoissonRecon::ThreadPool::NumThreads();
    ::PoissonRecon::ThreadPool::SetNumThreads(requested > 0
        ? static_cast<unsigned>(requested)
        : std::max(1U, std::thread::hardware_concurrency()));
#if defined(_OPENMP)
    previous_ = omp_get_max_threads();
    if (requested > 0) omp_set_num_threads(requested);
#else
    (void)requested;
#endif
  }
  ~ScopedOpenMpThreads() {
    ::PoissonRecon::ThreadPool::SetNumThreads(previous_poisson_);
#if defined(_OPENMP)
    omp_set_num_threads(previous_);
#endif
  }

 private:
  int previous_ = 1;
  unsigned previous_poisson_ = 1;
};
#endif

template <typename T>
T Percentile(std::vector<T> values, double percentile) {
  if (values.empty()) {
    return T{};
  }
  std::sort(values.begin(), values.end());
  const double rank = std::clamp(percentile, 0.0, 100.0) * 0.01 *
                      static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
  const double fraction = rank - static_cast<double>(lower);
  return static_cast<T>((1.0 - fraction) * values[lower] +
                        fraction * values[upper]);
}

}  // namespace

bool PoissonMeshingOptions::Check() const {
  return std::isfinite(point_weight) && point_weight >= 0.0 && depth > 0 &&
         std::isfinite(trim) && trim >= 0.0 && IsValidThreadCount(num_threads);
}

bool DelaunayMeshingOptions::Check() const {
  return std::isfinite(max_proj_dist) && max_proj_dist >= 0.0 &&
         std::isfinite(max_depth_dist) && max_depth_dist >= 0.0 &&
         max_depth_dist <= 1.0 && std::isfinite(visibility_sigma) &&
         visibility_sigma > 0.0 && std::isfinite(distance_sigma_factor) &&
         distance_sigma_factor > 0.0 && std::isfinite(quality_regularization) &&
         quality_regularization >= 0.0 &&
         std::isfinite(max_side_length_factor) &&
         max_side_length_factor >= 0.0 &&
         std::isfinite(max_side_length_percentile) &&
         max_side_length_percentile >= 0.0 &&
         max_side_length_percentile <= 100.0 && IsValidThreadCount(num_threads);
}

bool IsPoissonMeshingAvailable() {
#if defined(COLITE_ENABLE_POISSON_MESHING)
  return true;
#else
  return false;
#endif
}

bool PoissonMeshing(const PoissonMeshingOptions& options,
                    const std::filesystem::path& input_path,
                    const std::filesystem::path& output_path) {
  if (!options.Check()) {
    throw std::invalid_argument("Invalid Poisson meshing options.");
  }
  if (input_path.extension() != ".ply" ||
      !std::filesystem::exists(input_path)) {
    throw std::invalid_argument("Poisson input must be an existing PLY file.");
  }
  if (output_path.extension() != ".ply") {
    throw std::invalid_argument("Poisson output must be a PLY file.");
  }
#if !defined(COLITE_ENABLE_POISSON_MESHING)
  throw std::runtime_error("Poisson meshing was disabled at build time.");
#else
  // The bundled command parser and thread pool hold process-wide state.
  static std::mutex poisson_mutex;
  const std::lock_guard<std::mutex> lock(poisson_mutex);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  const ScopedOpenMpThreads thread_scope(options.num_threads);

  std::vector<std::string> arguments = {"colite_poisson_recon",
                                        "--in",
                                        input_path.string(),
                                        "--out",
                                        output_path.string(),
                                        "--pointWeight",
                                        std::to_string(options.point_weight),
                                        "--depth",
                                        std::to_string(options.depth),
                                        "--ascii"};
  // Set this on every embedded invocation: parser values survive calls.
  arguments.insert(arguments.end(),
                   {"--fullDepth", std::to_string(std::min(5, options.depth))});
  if (options.color) {
    arguments.push_back("--colors");
  }
  if (options.num_threads > 0) {
    arguments.insert(arguments.end(),
                     {"--parallel", "0"});
  }
  if (options.trim > 0.0) {
    arguments.push_back("--density");
  }
  std::vector<char*> mutable_arguments = MutableArguments(&arguments);
  if (RunPoissonRecon(static_cast<int>(mutable_arguments.size()),
                      mutable_arguments.data()) != EXIT_SUCCESS) {
    return false;
  }
  if (options.trim == 0.0) {
    return true;
  }

  arguments = {
      "colite_surface_trimmer", "--in",   output_path.string(),        "--out",
      output_path.string(),     "--trim", std::to_string(options.trim), "--ascii"};
  mutable_arguments = MutableArguments(&arguments);
  return RunSurfaceTrimmer(static_cast<int>(mutable_arguments.size()),
                           mutable_arguments.data()) == EXIT_SUCCESS;
#endif
}

#if !defined(COLITE_ENABLE_DELAUNAY_MESHING)

bool IsDelaunayMeshingAvailable() { return false; }

DenseTriangleMesh DelaunayMeshing(
    const DelaunayMeshingOptions&, const std::vector<FusedPoint>&,
    const std::vector<std::vector<std::uint32_t>>&,
    const std::vector<PatchMatchView>&) {
  throw std::runtime_error(
      "Delaunay meshing requires a build configured with CGAL.");
}

#else

namespace {

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using VertexBase =
    CGAL::Triangulation_vertex_base_with_info_3<std::size_t, Kernel>;
using CellBase = CGAL::Delaunay_triangulation_cell_base_3<Kernel>;
using TriangulationData =
    CGAL::Triangulation_data_structure_3<VertexBase, CellBase>;
using Delaunay = CGAL::Delaunay_triangulation_3<Kernel, TriangulationData,
                                                CGAL::Fast_location>;

template <typename Handle>
struct HandleHash {
  std::size_t operator()(const Handle& handle) const {
    return reinterpret_cast<std::size_t>(&*handle);
  }
};

Kernel::Point_3 ToCgal(const Eigen::Vector3d& point) {
  return {point.x(), point.y(), point.z()};
}

Eigen::Vector3d FromCgal(const Kernel::Point_3& point) {
  return {point.x(), point.y(), point.z()};
}

Eigen::Vector3d CameraCenter(const PatchMatchView& view) {
  return view.cam_from_world.Inverse().TransformPoint(Eigen::Vector3d::Zero());
}

std::optional<Eigen::Vector2d> ProjectWorld(const PatchMatchView& view,
                                            const Eigen::Vector3d& point) {
  const Eigen::Vector3d local = view.cam_from_world.TransformPoint(point);
  if (!local.allFinite() || local.z() <= 0.0) {
    return std::nullopt;
  }
  try {
    return ProjectPoint(view.camera, local);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

Delaunay BuildTriangulation(
    const DelaunayMeshingOptions& options,
    const std::vector<FusedPoint>& points,
    const std::vector<std::vector<std::uint32_t>>& visibility,
    const std::vector<PatchMatchView>& views) {
  Delaunay triangulation;
  std::vector<std::size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  // Match COLMAP 4.0's deterministic Fisher-Yates permutation. The insertion
  // order affects the selected tetrahedra for nearly co-spherical point sets,
  // so std::shuffle (whose mapping is implementation-defined) is not enough.
  std::mt19937 generator(0);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    std::uniform_int_distribution<std::size_t> distribution(
        i, indices.size() - 1);
    std::swap(indices[i], indices[distribution(generator)]);
  }

  const double max_squared_projection =
      options.max_proj_dist * options.max_proj_dist;
  const double minimum_depth_ratio = 1.0 - options.max_depth_dist;
  const double maximum_depth_ratio = 1.0 + options.max_depth_dist;

  for (const std::size_t index : indices) {
    // COLMAP's dense Delaunay input stores positions as float. Matching that
    // precision is important because CGAL tetrahedralization is sensitive to
    // tiny perturbations of nearly co-spherical points.
    const Eigen::Vector3d point_xyz =
        points[index].xyz.cast<float>().cast<double>();
    const Kernel::Point_3 position = ToCgal(point_xyz);
    bool insert =
        options.max_proj_dist == 0.0 || triangulation.number_of_vertices() < 4;
    if (!insert) {
      const Delaunay::Cell_handle cell = triangulation.locate(position);
      insert = triangulation.is_infinite(cell);
      for (const std::uint32_t view_index : visibility[index]) {
        if (insert) {
          break;
        }
        const PatchMatchView& view = views[view_index];
        const Eigen::Vector3d point_local =
            view.cam_from_world.TransformPoint(point_xyz);
        const std::optional<Eigen::Vector2d> projected =
            ProjectWorld(view, point_xyz);
        if (!projected || point_local.z() <= 0.0) {
          insert = true;
          break;
        }
        for (int corner = 0; corner < 4; ++corner) {
          const Eigen::Vector3d cell_position =
              FromCgal(cell->vertex(corner)->point());
          const Eigen::Vector3d cell_local =
              view.cam_from_world.TransformPoint(cell_position);
          const std::optional<Eigen::Vector2d> cell_projected =
              ProjectWorld(view, cell_position);
          if (!cell_projected || cell_local.z() <= 0.0) {
            insert = true;
            break;
          }
          const double depth_ratio = point_local.z() / cell_local.z();
          if (depth_ratio < minimum_depth_ratio ||
              depth_ratio > maximum_depth_ratio ||
              (*projected - *cell_projected).squaredNorm() >
                  max_squared_projection) {
            insert = true;
            break;
          }
        }
      }
    }
    if (insert) {
      const Delaunay::Vertex_handle vertex = triangulation.insert(position);
      vertex->info() = index;
    }
  }
  return triangulation;
}

class RayCaster {
 public:
  struct Intersection {
    Delaunay::Facet facet;
    double target_distance_squared = 0.0;
  };

  explicit RayCaster(const Delaunay& triangulation)
      : triangulation_(triangulation) {
    for (auto cell = triangulation_.all_cells_begin();
         cell != triangulation_.all_cells_end(); ++cell) {
      if (!triangulation_.is_infinite(cell)) {
        continue;
      }
      for (int side = 0; side < 4; ++side) {
        if (!triangulation_.is_infinite(cell, side)) {
          hull_facets_.emplace_back(cell, side);
        }
      }
    }
  }

  void Cast(const Kernel::Segment_3& segment,
            std::vector<Intersection>* intersections) const {
    intersections->clear();
    Delaunay::Cell_handle next_cell = triangulation_.locate(segment.start());
    bool found = true;
    std::size_t guard = 0;
    while (found && ++guard <= triangulation_.number_of_cells() + 1) {
      found = false;
      if (triangulation_.is_infinite(next_cell)) {
        for (const Delaunay::Facet& hull_facet : hull_facets_) {
          const Kernel::Triangle_3 triangle =
              triangulation_.triangle(hull_facet);
          if (CGAL::orientation(triangle[0], triangle[1], triangle[2],
                                segment.start()) ==
              Kernel::Orientation::NEGATIVE) {
            continue;
          }
          Kernel::Point_3 point;
          if (!CGAL::assign(point, CGAL::intersection(segment, triangle))) {
            continue;
          }
          const double target_distance =
              (point - segment.end()).squared_length();
          if (!intersections->empty() &&
              intersections->back().target_distance_squared < target_distance) {
            continue;
          }
          intersections->push_back({hull_facet, target_distance});
          next_cell = hull_facet.first->neighbor(hull_facet.second);
          found = true;
          break;
        }
      } else {
        for (int side = 0; side < 4; ++side) {
          const Kernel::Triangle_3 triangle =
              triangulation_.triangle(next_cell, side);
          if (CGAL::orientation(triangle[0], triangle[1], triangle[2],
                                segment.start()) ==
              Kernel::Orientation::NEGATIVE) {
            continue;
          }
          Kernel::Point_3 point;
          if (!CGAL::assign(point, CGAL::intersection(segment, triangle))) {
            continue;
          }
          const double target_distance =
              (point - segment.end()).squared_length();
          if (!intersections->empty() &&
              intersections->back().target_distance_squared < target_distance) {
            continue;
          }
          intersections->push_back({{next_cell, side}, target_distance});
          next_cell = next_cell->neighbor(side);
          found = true;
          break;
        }
      }
    }
  }

 private:
  const Delaunay& triangulation_;
  std::vector<Delaunay::Facet> hull_facets_;
};

class EdgeWeightComputer {
 public:
  EdgeWeightComputer(const Delaunay& triangulation, double visibility_sigma,
                     double distance_sigma_factor)
      : visibility_threshold_(5.0 * visibility_sigma),
        visibility_normalization_(-0.5 /
                                  (visibility_sigma * visibility_sigma)) {
    std::vector<double> squared_lengths;
    squared_lengths.reserve(triangulation.number_of_finite_edges());
    for (auto edge = triangulation.finite_edges_begin();
         edge != triangulation.finite_edges_end(); ++edge) {
      squared_lengths.push_back((edge->first->vertex(edge->second)->point() -
                                 edge->first->vertex(edge->third)->point())
                                    .squared_length());
    }
    distance_sigma_ =
        distance_sigma_factor *
        std::max(std::sqrt(Percentile(squared_lengths, 25.0)), 1e-7);
    distance_threshold_ = 5.0 * distance_sigma_;
    distance_normalization_ = -0.5 / (distance_sigma_ * distance_sigma_);
  }

  double DistanceSigma() const { return distance_sigma_; }

  double VisibilityProbability(double squared_visibility) const {
    if (squared_visibility >= visibility_threshold_) {
      return 1.0;
    }
    return std::max(
        0.0, 1.0 - std::exp(squared_visibility * visibility_normalization_));
  }

  double DistanceProbability(double squared_distance) const {
    if (squared_distance >= distance_threshold_) {
      return 1.0;
    }
    return std::max(0.0,
                    1.0 - std::exp(squared_distance * distance_normalization_));
  }

 private:
  double visibility_threshold_ = 0.0;
  double visibility_normalization_ = 0.0;
  double distance_sigma_ = 0.0;
  double distance_threshold_ = 0.0;
  double distance_normalization_ = 0.0;
};

double FacetCellCosine(const Delaunay& triangulation,
                       const Delaunay::Facet& facet) {
  if (triangulation.is_infinite(facet.first)) {
    return 1.0;
  }
  const Kernel::Triangle_3 triangle = triangulation.triangle(facet);
  const Kernel::Vector_3 normal =
      CGAL::cross_product(triangle[1] - triangle[0], triangle[2] - triangle[0]);
  const double normal_squared = normal.squared_length();
  if (normal_squared == 0.0) {
    return 0.5;
  }
  const Kernel::Vector_3 cotangent = facet.first->circumcenter() - triangle[0];
  const double cotangent_squared = cotangent.squared_length();
  if (cotangent_squared == 0.0) {
    return 0.5;
  }
  // Keep the orientation sign: the Labatut shape prior and COLMAP 4.0 use
  // the signed facet/cell angle. Taking the absolute value makes obtuse and
  // acute configurations indistinguishable and changes graph-cut capacities.
  return (normal * cotangent) /
         std::sqrt(normal_squared * cotangent_squared);
}

struct CellData {
  explicit CellData(std::size_t index = 0) : index(index) {}
  std::size_t index = 0;
  float source_weight = 0.0F;
  float sink_weight = 0.0F;
  std::array<float, 4> edge_weights = {0.0F, 0.0F, 0.0F, 0.0F};
};

}  // namespace

bool IsDelaunayMeshingAvailable() { return true; }

DenseTriangleMesh DelaunayMeshing(
    const DelaunayMeshingOptions& options,
    const std::vector<FusedPoint>& points,
    const std::vector<std::vector<std::uint32_t>>& points_visibility,
    const std::vector<PatchMatchView>& views) {
  if (!options.Check()) {
    throw std::invalid_argument("Invalid Delaunay meshing options.");
  }
  if (points.size() != points_visibility.size()) {
    throw std::invalid_argument(
        "Delaunay points and visibility must have identical sizes.");
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (!points[i].xyz.allFinite()) {
      throw std::invalid_argument(
          "Delaunay input contains a non-finite point.");
    }
    for (const std::uint32_t view_index : points_visibility[i]) {
      if (view_index >= views.size()) {
        throw std::invalid_argument(
            "Delaunay visibility contains an invalid view index.");
      }
    }
  }

  DenseTriangleMesh mesh;
  if (points.size() < 4 || views.empty()) {
    return mesh;
  }
  const Delaunay triangulation =
      BuildTriangulation(options, points, points_visibility, views);
  if (triangulation.dimension() != 3 ||
      triangulation.number_of_vertices() < 4) {
    return mesh;
  }

  using CellMap = std::unordered_map<Delaunay::Cell_handle, CellData,
                                     HandleHash<Delaunay::Cell_handle>>;
  CellMap cells;
  cells.reserve(triangulation.number_of_cells());
  for (auto cell = triangulation.all_cells_begin();
       cell != triangulation.all_cells_end(); ++cell) {
    cells.emplace(cell, CellData(cells.size()));
  }

  const RayCaster ray_caster(triangulation);
  const EdgeWeightComputer weights(triangulation, options.visibility_sigma,
                                   options.distance_sigma_factor);
  std::vector<std::vector<std::size_t>> visible_points(views.size());
  for (std::size_t point_index = 0; point_index < points.size();
       ++point_index) {
    for (const std::uint32_t view_index : points_visibility[point_index]) {
      visible_points[view_index].push_back(point_index);
    }
  }

  std::vector<RayCaster::Intersection> intersections;
  for (std::size_t view_index = 0; view_index < views.size(); ++view_index) {
    const Kernel::Point_3 camera_position =
        ToCgal(CameraCenter(views[view_index]).cast<float>().cast<double>());
    for (const std::size_t point_index : visible_points[view_index]) {
      const Kernel::Point_3 point_position = ToCgal(
          points[point_index].xyz.cast<float>().cast<double>());
      const double visible_count =
          static_cast<double>(points_visibility[point_index].size());
      const float alpha = static_cast<float>(
          weights.VisibilityProbability(visible_count * visible_count));
      const Kernel::Vector_3 direction = point_position - camera_position;
      const double squared_direction = direction.squared_length();
      if (squared_direction <= 0.0) {
        continue;
      }
      const Kernel::Vector_3 epsilon =
          (0.001 * weights.DistanceSigma() / std::sqrt(squared_direction)) *
          direction;

      ray_caster.Cast(
          Kernel::Segment_3(camera_position, point_position - epsilon),
          &intersections);
      if (!intersections.empty()) {
        cells.at(intersections.front().facet.first).source_weight += alpha;
      }
      for (const RayCaster::Intersection& intersection : intersections) {
        cells.at(intersection.facet.first)
            .edge_weights[intersection.facet.second] +=
            alpha * static_cast<float>(weights.DistanceProbability(
                        intersection.target_distance_squared));
      }

      const Delaunay::Cell_handle behind =
          triangulation.locate(point_position + epsilon);
      int selected_side = -1;
      double selected_distance = 0.0;
      for (int side = 0; side < 4; ++side) {
        // An infinite facet contains CGAL's undefined infinite vertex point.
        // Only the finite hull facet of an infinite cell has geometry.
        if (triangulation.is_infinite(Delaunay::Facet(behind, side))) continue;
        Kernel::Point_3 intersection_point;
        if (!CGAL::assign(intersection_point,
                          CGAL::intersection(
                              Kernel::Ray_3(camera_position, point_position),
                              triangulation.triangle(behind, side)))) {
          continue;
        }
        const double distance =
            (intersection_point - point_position).squared_length();
        if (distance > selected_distance) {
          selected_distance = distance;
          selected_side = side;
        }
      }
      if (selected_side >= 0) {
        cells.at(behind).edge_weights[selected_side] +=
            alpha *
            static_cast<float>(weights.DistanceProbability(selected_distance));
        cells.at(behind->neighbor(selected_side)).sink_weight += alpha;
      }
    }
  }

  MinSTGraphCut<float> graph(cells.size());
  for (const auto& entry : cells) {
    const Delaunay::Cell_handle cell = entry.first;
    const CellData& data = entry.second;
    graph.AddNode(data.index, data.source_weight, data.sink_weight);
    for (int side = 0; side < 4; ++side) {
      const Delaunay::Facet facet(cell, side);
      const Delaunay::Facet mirror = triangulation.mirror_facet(facet);
      const CellData& mirror_data = cells.at(mirror.first);
      if (data.index < mirror_data.index) {
        continue;
      }
      const float regularization = static_cast<float>(
          options.quality_regularization *
          (1.0 - std::min(FacetCellCosine(triangulation, facet),
                          FacetCellCosine(triangulation, mirror))));
      graph.AddEdge(data.index, mirror_data.index,
                    data.edge_weights[side] + regularization,
                    mirror_data.edge_weights[mirror.second] + regularization);
    }
  }
  graph.Compute();

  std::vector<Delaunay::Facet> surface_facets;
  std::vector<double> side_lengths;
  for (auto facet = triangulation.finite_facets_begin();
       facet != triangulation.finite_facets_end(); ++facet) {
    const CellData& first = cells.at(facet->first);
    const CellData& second = cells.at(facet->first->neighbor(facet->second));
    const bool first_is_outside = graph.IsConnectedToSource(first.index);
    const bool second_is_outside = graph.IsConnectedToSource(second.index);
    if (first_is_outside == second_is_outside) {
      continue;
    }
    const Kernel::Triangle_3 triangle = triangulation.triangle(*facet);
    side_lengths.push_back(
        std::sqrt(std::max({(triangle[0] - triangle[1]).squared_length(),
                            (triangle[0] - triangle[2]).squared_length(),
                            (triangle[1] - triangle[2]).squared_length()})));
    surface_facets.push_back(
        first_is_outside ? *facet : triangulation.mirror_facet(*facet));
  }
  if (surface_facets.empty()) {
    return mesh;
  }

  const double maximum_side_length =
      options.max_side_length_factor *
      Percentile(side_lengths, options.max_side_length_percentile);
  // CGAL cell traversal depends on allocation/insertion history. Preserve
  // winding, but emit the surface in stable input-point order so downstream
  // simplification receives identical meshes from standalone and pipelines.
  std::vector<std::array<std::size_t, 3>> faces;
  std::vector<std::size_t> vertex_ids;
  for (std::size_t facet_index = 0; facet_index < surface_facets.size();
       ++facet_index) {
    if (side_lengths[facet_index] > maximum_side_length) {
      continue;
    }
    const Delaunay::Facet& facet = surface_facets[facet_index];
    std::array<std::size_t, 3> triangle;
    for (int corner = 0; corner < 3; ++corner) {
      const Delaunay::Vertex_handle vertex = facet.first->vertex(
          triangulation.vertex_triple_index(facet.second, corner));
      triangle[corner] = vertex->info();
      vertex_ids.push_back(vertex->info());
    }
    std::rotate(triangle.begin(), std::min_element(triangle.begin(), triangle.end()), triangle.end());
    faces.push_back(triangle);
  }
  std::sort(vertex_ids.begin(), vertex_ids.end());
  vertex_ids.erase(std::unique(vertex_ids.begin(), vertex_ids.end()), vertex_ids.end());
  for (const auto id : vertex_ids) mesh.vertices.push_back(points.at(id));
  std::sort(faces.begin(), faces.end());
  for (const auto& face : faces) {
    Eigen::Vector3i triangle;
    for (int corner = 0; corner < 3; ++corner) {
      triangle[corner] = static_cast<int>(std::lower_bound(vertex_ids.begin(), vertex_ids.end(), face[corner]) - vertex_ids.begin());
    }
    mesh.triangles.push_back(triangle);
  }
  return mesh;
}

#endif

DenseTriangleMesh SparseDelaunayMeshing(
    const DelaunayMeshingOptions& options,
    const Reconstruction& reconstruction) {
  if (!IsDelaunayMeshingAvailable()) {
    throw std::runtime_error("Delaunay meshing requires CGAL support.");
  }
  if (!options.Check()) throw std::invalid_argument("Invalid Delaunay meshing options.");
  if (reconstruction.RegisteredImageIds().size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Too many registered images for Delaunay visibility.");
  }
  std::vector<PatchMatchView> views;
  std::unordered_map<ImageId, std::uint32_t> view_indices;
  for (const ImageId id : reconstruction.RegisteredImageIds()) {
    const auto& image = reconstruction.GetImage(id);
    if (!image.cam_from_world) throw std::invalid_argument("Registered image has no pose.");
    PatchMatchView view;
    view.image_id = id;
    view.camera = reconstruction.GetCamera(image.camera_id);
    view.cam_from_world = *image.cam_from_world;
    view_indices.emplace(id, static_cast<std::uint32_t>(views.size()));
    views.push_back(std::move(view));
  }
  // Stable IDs avoid unordered-map iteration changing subsampling across IO
  // formats or hash-table capacity. Coordinates still enter CGAL as float.
  std::vector<Point3DId> point_ids;
  point_ids.reserve(reconstruction.Points3D().size());
  for (const auto& [id, point] : reconstruction.Points3D()) point_ids.push_back(id);
  std::sort(point_ids.begin(), point_ids.end());
  std::vector<FusedPoint> points;
  std::vector<std::vector<std::uint32_t>> visibility;
  points.reserve(point_ids.size());
  visibility.reserve(point_ids.size());
  for (const auto id : point_ids) {
    const auto& source = reconstruction.GetPoint3D(id);
    FusedPoint point;
    point.xyz = source.xyz;
    point.color = source.color;
    std::vector<std::uint32_t> visible;
    visible.reserve(source.track.Length());
    for (const auto& observation : source.track.elements) {
      const auto view = view_indices.find(observation.image_id);
      if (view == view_indices.end()) {
        throw std::invalid_argument("Sparse point track references an unregistered image.");
      }
      const auto& image = reconstruction.GetImage(observation.image_id);
      if (observation.point2d_idx >= image.points2d.size() ||
          image.points2d[observation.point2d_idx].point3d_id != id) {
        throw std::invalid_argument("Sparse point track has an inconsistent 2D observation.");
      }
      visible.push_back(view->second);
    }
    std::sort(visible.begin(), visible.end());
    if (std::adjacent_find(visible.begin(), visible.end()) != visible.end()) {
      throw std::invalid_argument("Sparse point track contains duplicate image observations.");
    }
    points.push_back(std::move(point));
    visibility.push_back(std::move(visible));
  }
  return DelaunayMeshing(options, points, visibility, views);
}

}  // namespace colite::mvs
