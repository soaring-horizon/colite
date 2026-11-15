#include "colite/exe/recon_mvs.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "colite/core/version.h"
#include "colite/mvs/meshing.h"
#include "colite/mvs/patch_match.h"

namespace colite::exe {
namespace {
using Base = controllers::ReconMvsControllerOptions;
constexpr unsigned P = 1, U = 2, S = 4, F = 8, A = P | U | S | F;
unsigned Mask(MvsOptionScope scope) {
  switch (scope) {
    case MvsOptionScope::kPipeline: return P;
    case MvsOptionScope::kUndistorter: return U;
    case MvsOptionScope::kPatchMatch: return S;
    case MvsOptionScope::kFusion: return F;
  }
  throw std::invalid_argument("Invalid MVS option scope");
}
const char* Command(MvsOptionScope scope) {
  switch (scope) {
    case MvsOptionScope::kPipeline: return "recon_mvs";
    case MvsOptionScope::kUndistorter: return "image_undistorter";
    case MvsOptionScope::kPatchMatch: return "patch_match_stereo";
    case MvsOptionScope::kFusion: return "stereo_fusion";
  }
  return "recon_mvs";
}

template <typename T>
T Value(const std::string& text, const std::string& name) {
  if constexpr (std::is_same_v<T, bool>) {
    if (text == "1" || text == "true") return true;
    if (text == "0" || text == "false") return false;
  } else if constexpr (std::is_integral_v<T>) {
    T value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec == std::errc{} && result.ptr == text.data() + text.size()) return value;
  } else if constexpr (std::is_floating_point_v<T>) {
    std::size_t consumed = 0;
    try {
      const double value = std::stod(text, &consumed);
      if (consumed == text.size() && std::isfinite(value) &&
          std::abs(value) <= std::numeric_limits<T>::max()) return static_cast<T>(value);
    } catch (const std::exception&) {}
  } else {
    return T(text);
  }
  throw std::invalid_argument("Invalid value for " + name + ": " + text);
}
struct Option {
  std::string name;
  unsigned scopes;
  std::string description;
  std::string type;
  std::string default_value;
  std::function<void(ReconMvsCliOptions&, const std::string&)> set;
};
template <typename T, typename Owner>
Option Bind(const char* name, T Owner::* member, unsigned scopes, const char* description) {
  ReconMvsCliOptions defaults;
  std::ostringstream value;
  if constexpr (std::is_same_v<T, std::filesystem::path>) value << (defaults.*member).string();
  else value << defaults.*member;
  const char* type = std::is_same_v<T, bool> ? "0|1" :
      (std::is_integral_v<T> ? "integer" : (std::is_floating_point_v<T> ? "number" : "value"));
  return {name, scopes, description, type, value.str(),
          [member, name](ReconMvsCliOptions& options, const std::string& text) {
            const auto value = Value<T>(text, name);
            if constexpr (std::is_same_v<T, std::size_t>) {
              if (std::string(name).rfind("--Colite.", 0) != 0 &&
                  value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument(std::string(name) + " exceeds the COLMAP integer range");
              }
            }
            options.*member = value;
          }};
}
const std::vector<Option>& Options() {
  static const std::vector<Option> options{
      Bind("--input_path", &Base::model_path, A, "model path"),
      Bind("--image_path", &Base::image_path, A, "image path"),
      Bind("--output_path", &Base::output_path, P | U | S, "output path"),
      Bind("--output_path", &Base::fused_output_path, F, "fused output path"),
      Bind("--input_type", &Base::fusion_input_type, P | F, "fusion input type"),
      Bind("--config_path", &Base::patch_match_config_path, P | S, "PatchMatch configuration; default workspace/stereo/patch-match.cfg"),
      Bind("--num_patch_match_src_images", &Base::undistort_num_patch_match_src_images, P | U, "automatic source count written to patch-match.cfg"),
      Bind("--PatchMatchStereo.num_threads", &Base::patch_match_num_threads, P | S, "patch match num threads"),
      Bind("--StereoFusion.num_threads", &Base::fusion_num_threads, P | F, "fusion num threads"),
      Bind("--num_threads", &Base::undistort_num_threads, U, "undistort num threads"),
      Bind("--PatchMatchStereo.max_image_size", &Base::patch_match_max_image_size, P | S, "patch match max image size"),
      Bind("--StereoFusion.max_image_size", &Base::fusion_max_image_size, P | F, "fusion max image size"),
      Bind("--Colite.max_num_source_images", &Base::max_num_source_images, P | S, "max num source images"),
      Bind("--Colite.min_num_shared_points", &Base::min_num_shared_points, P | S | F, "min num shared points"),
      Bind("--Colite.source_selection_mode", &Base::source_selection_mode, P | S, "source selection mode"),
      Bind("--PatchMatchStereo.min_triangulation_angle", &Base::source_min_triangulation_angle_degrees, P | S, "source min triangulation angle degrees"),
      Bind("--Colite.source_max_triangulation_angle_degrees", &Base::source_max_triangulation_angle_degrees, P | S, "source max triangulation angle degrees"),
      Bind("--Colite.source_triangulation_angle_weight", &Base::source_triangulation_angle_weight, P | S, "source triangulation angle weight"),
      Bind("--Colite.source_shared_points_weight", &Base::source_shared_points_weight, P | S, "source shared points weight"),
      Bind("--Colite.source_scale_consistency_weight", &Base::source_scale_consistency_weight, P | S, "source scale consistency weight"),
      Bind("--Colite.source_coverage_weight", &Base::source_coverage_weight, P | S, "source coverage weight"),
      Bind("--Colite.source_coverage_grid_size", &Base::source_coverage_grid_size, P | S, "source coverage grid size"),
      Bind("--Colite.initialize_patch_match", &Base::initialize_patch_match, P, "initialize patch match"),
      Bind("--Colite.run_patch_match", &Base::run_patch_match, P, "run patch match"),
      Bind("--Colite.patch_match_backend", &Base::patch_match_backend, P | S, "patch match backend"),
      Bind("--PatchMatchStereo.gpu_index", &Base::patch_match_gpu_index, P | S, "patch match gpu index"),
      Bind("--PatchMatchStereo.num_iterations", &Base::patch_match_iterations, P | S, "patch match iterations"),
      Bind("--Colite.patch_match_pyramid_levels", &Base::patch_match_pyramid_levels, P | S, "patch match pyramid levels"),
      Bind("--PatchMatchStereo.window_radius", &Base::patch_match_window_radius, P | S, "patch match window radius"),
      Bind("--PatchMatchStereo.window_step", &Base::patch_match_window_step, P | S, "patch match window step"),
      Bind("--PatchMatchStereo.depth_min", &Base::patch_match_depth_min, P | S, "patch match depth min"),
      Bind("--PatchMatchStereo.depth_max", &Base::patch_match_depth_max, P | S, "patch match depth max"),
      Bind("--PatchMatchStereo.incident_angle_sigma", &Base::patch_match_incident_angle_sigma, P | S, "patch match incident angle sigma"),
      Bind("--PatchMatchStereo.geom_consistency", &Base::patch_match_geom_consistency, P | S, "patch match geom consistency"),
      Bind("--PatchMatchStereo.write_consistency_graph", &Base::patch_match_write_consistency_graph, P | S, "write COLMAP consistency graphs"),
      Bind("--PatchMatchStereo.allow_missing_files", &Base::patch_match_allow_missing_files, P | S, "skip unavailable geometric source maps"),
      Bind("--PatchMatchStereo.geom_consistency_max_cost", &Base::patch_match_geom_consistency_max_cost, P | S, "patch match geom consistency max cost"),
      Bind("--Colite.randomize_initial_normals", &Base::patch_match_randomize_initial_normals, P | S, "patch match randomize initial normals"),
      Bind("--Colite.patch_match_initial_normal_hemisphere_degrees", &Base::patch_match_initial_normal_hemisphere_degrees, P | S, "patch match initial normal hemisphere degrees"),
      Bind("--Colite.patch_match_normal_perturbation_degrees", &Base::patch_match_normal_perturbation_degrees, P | S, "patch match normal perturbation degrees"),
      Bind("--Colite.patch_match_bilateral_ncc", &Base::patch_match_bilateral_ncc, P | S, "patch match bilateral ncc"),
      Bind("--PatchMatchStereo.sigma_spatial", &Base::patch_match_sigma_spatial, P | S, "patch match sigma spatial"),
      Bind("--PatchMatchStereo.sigma_color", &Base::patch_match_sigma_color, P | S, "patch match sigma color"),
      Bind("--Colite.patch_match_dynamic_source_selection", &Base::patch_match_dynamic_source_selection, P | S, "patch match dynamic source selection"),
      Bind("--Colite.patch_match_directional_hmm", &Base::patch_match_directional_hmm, P | S, "patch match directional hmm"),
      Bind("--Colite.source_geometric_priors", &Base::patch_match_colmap_source_priors, P | S, "patch match colmap source priors"),
      Bind("--PatchMatchStereo.num_samples", &Base::patch_match_source_samples, P | S, "patch match source samples"),
      Bind("--PatchMatchStereo.ncc_sigma", &Base::patch_match_ncc_sigma, P | S, "patch match ncc sigma"),
      Bind("--Colite.patch_match_min_selection_probability", &Base::patch_match_min_selection_probability, P | S, "patch match min selection probability"),
      Bind("--Colite.patch_match_max_selection_probability", &Base::patch_match_max_selection_probability, P | S, "patch match max selection probability"),
      Bind("--Colite.patch_match_perturbation_samples", &Base::patch_match_perturbation_samples, P | S, "patch match perturbation samples"),
      Bind("--Colite.patch_match_min_num_valid_sources", &Base::patch_match_min_num_valid_sources, P | S, "patch match min num valid sources"),
      Bind("--Colite.patch_match_max_num_aggregated_costs", &Base::patch_match_max_num_aggregated_costs, P | S, "patch match max num aggregated costs"),
      Bind("--Colite.patch_match_propagation_schedule", &Base::patch_match_propagation_schedule, P | S, "patch match propagation schedule"),
      Bind("--Colite.patch_match_checkerboard_rounds", &Base::patch_match_checkerboard_rounds, P | S, "patch match checkerboard rounds"),
      Bind("--Colite.patch_match_diagnostics", &Base::patch_match_diagnostics, P | S, "patch match diagnostics"),
      Bind("--Colite.patch_match_geometric_refinement_iterations", &Base::patch_match_geometric_refinement_iterations, P | S, "patch match geometric refinement iterations"),
      Bind("--PatchMatchStereo.geom_consistency_regularizer", &Base::patch_match_geometric_consistency_weight, P | S, "patch match geometric consistency weight"),
      Bind("--Colite.robust_geometric_aggregation", &Base::patch_match_robust_geometric_aggregation, P | S, "patch match robust geometric aggregation"),
      Bind("--Colite.patch_match_geometric_max_relative_depth_error", &Base::patch_match_geometric_max_relative_depth_error, P | S, "patch match geometric max relative depth error"),
      Bind("--Colite.refine_normals_from_depth", &Base::refine_normals_from_depth, P | S, "refine normals from depth"),
      Bind("--Colite.normal_refinement_max_relative_depth_difference", &Base::normal_refinement_max_relative_depth_difference, P | S, "normal refinement max relative depth difference"),
      Bind("--Colite.filter_photometric_max_cost", &Base::filter_photometric_max_cost, P | S, "filter photometric max cost"),
      Bind("--PatchMatchStereo.filter", &Base::patch_match_filter, P | S, "patch match filter"),
      Bind("--PatchMatchStereo.filter_min_ncc", &Base::patch_match_filter_min_ncc, P | S, "patch match filter min ncc"),
      Bind("--PatchMatchStereo.filter_min_triangulation_angle", &Base::patch_match_filter_min_triangulation_angle_degrees, P | S, "patch match filter min triangulation angle degrees"),
      Bind("--PatchMatchStereo.filter_min_num_consistent", &Base::patch_match_filter_min_num_consistent, P | S, "patch match filter min num consistent"),
      Bind("--PatchMatchStereo.filter_geom_consistency_max_cost", &Base::patch_match_filter_geom_consistency_max_cost, P | S, "patch match filter geom consistency max cost"),
      Bind("--Colite.check_geometric_consistency", &Base::check_geometric_consistency, P | S, "check geometric consistency"),
      Bind("--Colite.filter_geometric_min_num_consistent_sources", &Base::filter_geometric_min_num_consistent_sources, P | S, "filter geometric min num consistent sources"),
      Bind("--Colite.fuse_stereo", &Base::fuse_stereo, P, "fuse stereo"),
      Bind("--StereoFusion.use_cache", &Base::fusion_use_cache, P | F, "fusion use cache"),
      Bind("--StereoFusion.cache_size", &Base::fusion_cache_size_gb, P | F, "fusion cache size gb"),
      Bind("--StereoFusion.check_num_images", &Base::fusion_max_num_source_images, P | F, "fusion max num source images"),
      Bind("--StereoFusion.min_num_pixels", &Base::fusion_min_num_observations, P | F, "fusion min num observations"),
      Bind("--StereoFusion.max_num_pixels", &Base::fusion_max_num_observations, P | F, "fusion max num observations"),
      Bind("--StereoFusion.max_traversal_depth", &Base::fusion_max_traversal_depth, P | F, "fusion max traversal depth"),
      Bind("--StereoFusion.max_reproj_error", &Base::fusion_max_reprojection_error_pixels, P | F, "fusion max reprojection error pixels"),
      Bind("--StereoFusion.max_depth_error", &Base::fusion_max_relative_depth_error, P | F, "fusion max relative depth error"),
      Bind("--StereoFusion.max_normal_error", &Base::fusion_max_normal_error_degrees, P | F, "fusion max normal error degrees"),
      Bind("--Colite.fusion_max_view_angle_degrees", &Base::fusion_max_view_angle_degrees, P | F, "fusion max view angle degrees"),
      Bind("--Colite.fusion_source_pixel_search_radius", &Base::fusion_source_pixel_search_radius, P | F, "fusion source pixel search radius"),
      Bind("--Colite.fusion_diagnostics", &Base::fusion_diagnostics, P | F, "fusion diagnostics"),
      Bind("--Colite.filter_dense_cloud", &Base::filter_dense_cloud, P, "filter dense cloud"),
      Bind("--Colite.dense_cloud_min_track_observations", &Base::dense_cloud_min_track_observations, P, "dense cloud min track observations"),
      Bind("--Colite.dense_cloud_voxel_size", &Base::dense_cloud_voxel_size, P, "dense cloud voxel size"),
      Bind("--Colite.dense_cloud_spatial_bounds_quantile", &Base::dense_cloud_spatial_bounds_quantile, P, "dense cloud spatial bounds quantile"),
      Bind("--Colite.mesh_dense_cloud", &Base::mesh_dense_cloud, P, "mesh dense cloud"),
      Bind("--Colite.mesh_max_edge_length", &Base::mesh_max_edge_length, P, "mesh max edge length"),
      Bind("--Colite.mesh_max_normal_angle_degrees", &Base::mesh_max_normal_angle_degrees, P, "mesh max normal angle degrees"),
      Bind("--Colite.poisson_mesher", &Base::poisson_mesher, P, "poisson mesher"),
      Bind("--PoissonMeshing.point_weight", &Base::poisson_point_weight, P, "poisson point weight"),
      Bind("--PoissonMeshing.depth", &Base::poisson_depth, P, "poisson depth"),
      Bind("--PoissonMeshing.color", &Base::poisson_color, P, "poisson color"),
      Bind("--PoissonMeshing.trim", &Base::poisson_trim, P, "poisson trim"),
      Bind("--PoissonMeshing.num_threads", &Base::poisson_num_threads, P, "Poisson CPU threads; -1 automatic"),
      Bind("--Colite.delaunay_mesher", &Base::delaunay_mesher, P, "delaunay mesher"),
      Bind("--DelaunayMeshing.max_proj_dist", &Base::delaunay_max_proj_dist, P, "delaunay max proj dist"),
      Bind("--DelaunayMeshing.max_depth_dist", &Base::delaunay_max_depth_dist, P, "delaunay max depth dist"),
      Bind("--DelaunayMeshing.visibility_sigma", &Base::delaunay_visibility_sigma, P, "delaunay visibility sigma"),
      Bind("--DelaunayMeshing.distance_sigma_factor", &Base::delaunay_distance_sigma_factor, P, "delaunay distance sigma factor"),
      Bind("--DelaunayMeshing.quality_regularization", &Base::delaunay_quality_regularization, P, "delaunay quality regularization"),
      Bind("--DelaunayMeshing.max_side_length_factor", &Base::delaunay_max_side_length_factor, P, "delaunay max side length factor"),
      Bind("--DelaunayMeshing.max_side_length_percentile", &Base::delaunay_max_side_length_percentile, P, "delaunay max side length percentile"),
      Bind("--DelaunayMeshing.num_threads", &Base::delaunay_num_threads, P, "Delaunay CPU threads; -1 automatic"),
      Bind("--Colite.mesh_simplifier", &Base::mesh_simplifier, P, "mesh simplifier"),
      Bind("--MeshSimplification.target_face_ratio", &Base::mesh_simplification_target_face_ratio, P, "fraction of mesh faces to retain"),
      Bind("--MeshSimplification.max_error", &Base::mesh_simplification_max_error, P, "mesh simplification max error"),
      Bind("--MeshSimplification.boundary_weight", &Base::mesh_simplification_boundary_weight, P, "quadric penalty for boundary displacement"),
      Bind("--MeshSimplification.interpolate_colors", &Base::mesh_simplification_interpolate_colors, P, "interpolate colors at collapse positions"),
      Bind("--MeshSimplification.num_threads", &Base::mesh_simplification_num_threads, P, "mesh initialization CPU threads; -1 automatic"),
      Bind("--Colite.mesh_texturer", &Base::mesh_texturer, P, "mesh texturer"),
      Bind("--Colite.texture_output_type", &Base::texture_output_type, P, "textured PLY output: BIN or TXT"),
      Bind("--Colite.texture_write_obj", &Base::texture_write_obj, P, "also write mesh.obj and mesh.mtl"),
      Bind("--Colite.texture_atlas_size", &Base::texture_atlas_size, P, "texture atlas size"),
      Bind("--Colite.texture_max_atlas_size", &Base::texture_max_atlas_size, P, "texture max atlas size"),
      Bind("--MeshTextureMapping.atlas_patch_padding", &Base::texture_patch_padding, P, "texture patch padding"),
      Bind("--MeshTextureMapping.inpaint_radius", &Base::texture_inpaint_radius, P, "texture inpaint radius"),
      Bind("--MeshTextureMapping.texture_scale_factor", &Base::texture_scale, P, "texture scale"),
      Bind("--MeshTextureMapping.min_cos_normal_angle", &Base::texture_min_cos_normal_angle, P, "texture min cos normal angle"),
      Bind("--MeshTextureMapping.min_visible_vertices", &Base::texture_min_visible_vertices, P, "texture min visible vertices"),
      Bind("--MeshTextureMapping.view_selection_smoothing_iterations", &Base::texture_view_smoothing_iterations, P, "texture view smoothing iterations"),
      Bind("--Colite.texture_view_selection_min_component_size", &Base::texture_view_min_component_size, P, "texture view min component size"),
      Bind("--Colite.texture_visibility_resolution", &Base::texture_visibility_resolution, P, "texture visibility resolution"),
      Bind("--Colite.texture_occlusion_relative_tolerance", &Base::texture_occlusion_relative_tolerance, P, "texture occlusion relative tolerance"),
      Bind("--Colite.texture_use_vertex_color_fallback", &Base::texture_use_vertex_color_fallback, P, "texture use vertex color fallback"),
      Bind("--MeshTextureMapping.apply_color_correction", &Base::texture_apply_color_correction, P, "texture apply color correction"),
      Bind("--MeshTextureMapping.color_correction_regularization", &Base::texture_color_correction_regularization, P, "texture color correction regularization"),
      Bind("--MeshTextureMapping.num_threads", &Base::texture_num_threads, P, "Texture CPU threads; -1 automatic"),
      Bind("--Colite.geometric_max_reprojection_error_pixels", &Base::geometric_max_reprojection_error_pixels, P | S, "geometric max reprojection error pixels"),
      Bind("--Colite.geometric_max_relative_depth_error", &Base::geometric_max_relative_depth_error, P | S, "geometric max relative depth error"),
      Bind("--Colite.write_debug_maps", &Base::write_debug_maps, P | S, "also write debug depth/normal/cost text maps"),
      Bind("--Colite.write_cost_maps", &Base::write_cost_maps, P | S, "write Colite cost binary maps"),
      Bind("--Colite.image_cache_size", &Base::image_cache_size, A, "image cache size"),
      Bind("--Colite.resolution_level", &Base::resolution_level, A, "resolution level"),
      Bind("--default_random_seed", &Base::random_seed, A, "random seed"),
      Bind("--Colite.image_undistorter", &Base::image_undistorter, P, "image undistorter"),
      Bind("--blank_pixels", &Base::undistort_blank_pixels, P | U, "undistort blank pixels"),
      Bind("--min_scale", &Base::undistort_min_scale, P | U, "undistort min scale"),
      Bind("--max_scale", &Base::undistort_max_scale, P | U, "undistort max scale"),
      Bind("--max_image_size", &Base::undistort_max_image_size, P | U, "undistort max image size"),
      Bind("--workspace_path", &ReconMvsCliOptions::workspace_path, S | F, "COLMAP dense workspace"),
      Bind("--workspace_format", &ReconMvsCliOptions::workspace_format, S | F, "Supported format: COLMAP"),
      Bind("--output_type", &ReconMvsCliOptions::undistort_output_type, P | U, "Supported type: COLMAP"),
  };
  return options;
}
const Option* FindOption(const std::string& name, MvsOptionScope scope) {
  for (const auto& option : Options()) {
    if (option.name == name && (option.scopes & Mask(scope))) return &option;
  }
  return nullptr;
}

void ValidateOptions(const ReconMvsCliOptions& options) {
  if (options.show_help || options.show_version) {
    return;
  }
  if (options.model_path.empty()) {
    throw std::invalid_argument("Missing required argument --input_path");
  }
  if (options.output_path.empty()) {
    throw std::invalid_argument("Missing required argument --output_path");
  }
  if (options.min_num_shared_points == 0) {
    throw std::invalid_argument("--Colite.min_num_shared_points must be positive");
  }
  if (!std::isfinite(options.source_min_triangulation_angle_degrees) ||
      !std::isfinite(options.source_max_triangulation_angle_degrees) ||
      options.source_min_triangulation_angle_degrees < 0.0 ||
      options.source_min_triangulation_angle_degrees >= 180.0 ||
      (options.source_selection_mode == "weighted" && options.source_max_triangulation_angle_degrees <
          options.source_min_triangulation_angle_degrees) ||
      options.source_max_triangulation_angle_degrees > 180.0) {
    throw std::invalid_argument(
        "Source triangulation angle range must be within [0, 180]");
  }
  if (!std::isfinite(options.source_triangulation_angle_weight) ||
      !std::isfinite(options.source_shared_points_weight) ||
      !std::isfinite(options.source_scale_consistency_weight) ||
      !std::isfinite(options.source_coverage_weight) ||
      options.source_triangulation_angle_weight < 0.0 ||
      options.source_shared_points_weight < 0.0 ||
      options.source_scale_consistency_weight < 0.0 ||
      options.source_coverage_weight < 0.0) {
    throw std::invalid_argument(
        "Source view score weights must be non-negative");
  }
  if (options.source_coverage_grid_size == 0) {
    throw std::invalid_argument(
        "--Colite.source_coverage_grid_size must be positive");
  }
  if (options.resolution_level > 30) {
    throw std::invalid_argument("--Colite.resolution_level must be between 0 and 30");
  }
  if (options.initialize_patch_match && options.image_path.empty()) {
    throw std::invalid_argument(
        "--Colite.initialize_patch_match requires --image_path");
  }
  if (options.image_undistorter && options.image_path.empty()) {
    throw std::invalid_argument(
        "--Colite.image_undistorter requires --image_path");
  }
  const bool undistortion_options_valid =
      std::isfinite(options.undistort_blank_pixels) &&
      options.undistort_blank_pixels >= 0.0 &&
      options.undistort_blank_pixels <= 1.0 &&
      std::isfinite(options.undistort_min_scale) &&
      std::isfinite(options.undistort_max_scale) &&
      options.undistort_min_scale > 0.0 &&
      options.undistort_min_scale <= options.undistort_max_scale &&
      options.undistort_max_image_size != 0;
  if (!undistortion_options_valid) {
    throw std::invalid_argument("Invalid image undistortion options");
  }
  if (options.run_patch_match && options.image_path.empty()) {
    throw std::invalid_argument("--Colite.run_patch_match requires --image_path");
  }
  if (options.patch_match_geometric_refinement_iterations > 0 &&
      !options.run_patch_match) {
    throw std::invalid_argument(
        "--Colite.patch_match_geometric_refinement_iterations requires "
        "--Colite.run_patch_match");
  }
  if (options.patch_match_window_radius <= 0 || options.patch_match_window_radius > 32) {
    throw std::invalid_argument(
        "--PatchMatchStereo.window_radius must be within [1, 32]");
  }
  if (options.patch_match_iterations == 0 || options.patch_match_source_samples == 0) {
    throw std::invalid_argument("PatchMatchStereo.num_iterations and num_samples must be positive");
  }
  if (!std::isfinite(
          options.patch_match_initial_normal_hemisphere_degrees) ||
      options.patch_match_initial_normal_hemisphere_degrees < 0.0F ||
      options.patch_match_initial_normal_hemisphere_degrees > 180.0F) {
    throw std::invalid_argument(
        "--Colite.patch_match_initial_normal_hemisphere_degrees must be in "
        "[0, 180]");
  }
  if (!std::isfinite(options.patch_match_normal_perturbation_degrees) ||
      options.patch_match_normal_perturbation_degrees < 0.0F ||
      options.patch_match_normal_perturbation_degrees > 180.0F) {
    throw std::invalid_argument(
        "--Colite.patch_match_normal_perturbation_degrees must be in [0, 180]");
  }
  if (!std::isfinite(options.patch_match_sigma_spatial) ||
      !std::isfinite(options.patch_match_sigma_color) ||
      options.patch_match_sigma_color <= 0.0F) {
    throw std::invalid_argument(
        "PatchMatch bilateral NCC sigmas must be finite and color sigma "
        "must be positive");
  }
  if (!std::isfinite(options.patch_match_ncc_sigma) ||
      options.patch_match_ncc_sigma <= 0.0F) {
    throw std::invalid_argument(
        "PatchMatchStereo.ncc_sigma must be finite and positive");
  }
  if (!std::isfinite(options.patch_match_min_selection_probability) ||
      !std::isfinite(options.patch_match_max_selection_probability) ||
      options.patch_match_min_selection_probability < 0.0F ||
      options.patch_match_max_selection_probability > 1.0F ||
      options.patch_match_min_selection_probability >
          options.patch_match_max_selection_probability) {
    throw std::invalid_argument(
        "PatchMatch dynamic source selection options are invalid");
  }
  try {
    (void)mvs::PatchMatchBackendFromName(options.patch_match_backend);
  } catch (const std::invalid_argument& error) {
    throw std::invalid_argument(error.what());
  }
  if (options.patch_match_gpu_index < -1) {
    throw std::invalid_argument("--PatchMatchStereo.gpu_index must be >= -1");
  }
  if (options.patch_match_pyramid_levels == 0) {
    throw std::invalid_argument(
        "--Colite.patch_match_pyramid_levels must be positive");
  }
  if (options.patch_match_perturbation_samples == 0) {
    throw std::invalid_argument(
        "--Colite.patch_match_perturbation_samples must be positive");
  }
  if (options.patch_match_min_num_valid_sources == 0) {
    throw std::invalid_argument(
        "--Colite.patch_match_min_num_valid_sources must be positive");
  }
  if (options.patch_match_propagation_schedule != "forward_backward" &&
      options.patch_match_propagation_schedule != "four_direction" &&
      options.patch_match_propagation_schedule != "checkerboard") {
    throw std::invalid_argument(
        "Invalid PatchMatch propagation schedule: " +
        options.patch_match_propagation_schedule);
  }
  if (options.patch_match_checkerboard_rounds == 0) {
    throw std::invalid_argument(
        "--Colite.patch_match_checkerboard_rounds must be positive");
  }
  if (!std::isfinite(options.patch_match_geometric_consistency_weight) ||
      options.patch_match_geometric_consistency_weight < 0.0F) {
    throw std::invalid_argument(
        "--PatchMatchStereo.geom_consistency_regularizer must be non-negative");
  }
  if (!std::isfinite(
          options.patch_match_geometric_max_relative_depth_error) ||
      options.patch_match_geometric_max_relative_depth_error <= 0.0F) {
    throw std::invalid_argument(
        "--Colite.patch_match_geometric_max_relative_depth_error must be positive");
  }
  if (!std::isfinite(
          options.normal_refinement_max_relative_depth_difference) ||
      options.normal_refinement_max_relative_depth_difference < 0.0F) {
    throw std::invalid_argument(
        "--Colite.normal_refinement_max_relative_depth_difference must be "
        "non-negative");
  }
  if (!std::isfinite(options.filter_photometric_max_cost) ||
      options.filter_photometric_max_cost < -1.0F) {
    throw std::invalid_argument(
        "--Colite.filter_photometric_max_cost must be non-negative when enabled");
  }
  if (!std::isfinite(options.patch_match_filter_min_ncc) ||
      options.patch_match_filter_min_ncc < -1.0F || options.patch_match_filter_min_ncc > 1.0F ||
      !std::isfinite(options.patch_match_filter_min_triangulation_angle_degrees) ||
      options.patch_match_filter_min_triangulation_angle_degrees < 0.0F ||
      options.patch_match_filter_min_triangulation_angle_degrees > 180.0F ||
      !std::isfinite(options.patch_match_filter_geom_consistency_max_cost) ||
      options.patch_match_filter_geom_consistency_max_cost < 0.0F) {
    throw std::invalid_argument("Invalid PatchMatchStereo final filter options");
  }
  if (options.patch_match_filter && options.filter_photometric_max_cost >= 0.0F) {
    throw std::invalid_argument("Do not combine NCC filtering with --Colite.filter_photometric_max_cost");
  }
  if (options.check_geometric_consistency && !options.run_patch_match &&
      !options.fuse_stereo) {
    throw std::invalid_argument(
        "--Colite.check_geometric_consistency requires --Colite.run_patch_match");
  }
  if ((options.filter_dense_cloud || options.mesh_dense_cloud) &&
      !options.fuse_stereo) {
    throw std::invalid_argument(
        "Dense cloud filtering and meshing require --Colite.fuse_stereo");
  }
  options.ValidateSurfaceOptions();
  if (options.fusion_min_num_observations == 0) {
    throw std::invalid_argument(
        "--StereoFusion.min_num_pixels must be positive");
  }
  if (!std::isfinite(options.fusion_cache_size_gb) || options.fusion_cache_size_gb <= 0.0) {
    throw std::invalid_argument("--StereoFusion.cache_size must be finite and positive");
  }
  if (options.fusion_max_num_observations <
      options.fusion_min_num_observations) {
    throw std::invalid_argument(
        "--StereoFusion.max_num_pixels must be >= "
        "--StereoFusion.min_num_pixels");
  }
  if (!std::isfinite(options.fusion_max_reprojection_error_pixels) ||
      options.fusion_max_reprojection_error_pixels < 0.0F) {
    throw std::invalid_argument(
        "--StereoFusion.max_reproj_error must be non-negative");
  }
  if (!std::isfinite(options.fusion_max_relative_depth_error) ||
      options.fusion_max_relative_depth_error < 0.0F) {
    throw std::invalid_argument(
        "--StereoFusion.max_depth_error must be non-negative");
  }
  if (!std::isfinite(options.fusion_max_normal_error_degrees) ||
      options.fusion_max_normal_error_degrees < 0.0F ||
      options.fusion_max_normal_error_degrees > 180.0F) {
    throw std::invalid_argument(
        "--StereoFusion.max_normal_error must be in [0, 180]");
  }
  if (!std::isfinite(options.fusion_max_view_angle_degrees) ||
      options.fusion_max_view_angle_degrees < 0.0F ||
      options.fusion_max_view_angle_degrees > 180.0F) {
    throw std::invalid_argument(
        "--Colite.fusion_max_view_angle_degrees must be in [0, 180]");
  }
  if (options.fusion_source_pixel_search_radius > 5) {
    throw std::invalid_argument(
        "--Colite.fusion_source_pixel_search_radius must be <= 5");
  }
  if (options.dense_cloud_min_track_observations == 0) {
    throw std::invalid_argument(
        "--Colite.dense_cloud_min_track_observations must be positive");
  }
  if (!std::isfinite(options.dense_cloud_voxel_size) ||
      options.dense_cloud_voxel_size < 0.0) {
    throw std::invalid_argument(
        "--Colite.dense_cloud_voxel_size must be non-negative");
  }
  if (!std::isfinite(options.dense_cloud_spatial_bounds_quantile) ||
      options.dense_cloud_spatial_bounds_quantile < 0.0 ||
      options.dense_cloud_spatial_bounds_quantile >= 0.5) {
    throw std::invalid_argument(
        "--Colite.dense_cloud_spatial_bounds_quantile must be in [0, 0.5)");
  }
  if (!std::isfinite(options.mesh_max_edge_length) ||
      options.mesh_max_edge_length < 0.0) {
    throw std::invalid_argument("--Colite.mesh_max_edge_length must be non-negative");
  }
  if (!std::isfinite(options.mesh_max_normal_angle_degrees) ||
      options.mesh_max_normal_angle_degrees < 0.0 ||
      options.mesh_max_normal_angle_degrees > 180.0) {
    throw std::invalid_argument(
        "--Colite.mesh_max_normal_angle_degrees must be in [0, 180]");
  }
  if (!std::isfinite(options.geometric_max_reprojection_error_pixels) ||
      options.geometric_max_reprojection_error_pixels < 0.0F) {
    throw std::invalid_argument(
        "--Colite.geometric_max_reprojection_error_pixels must be non-negative");
  }
  if (!std::isfinite(options.geometric_max_relative_depth_error) ||
      options.geometric_max_relative_depth_error < 0.0F) {
    throw std::invalid_argument(
        "--Colite.geometric_max_relative_depth_error must be non-negative");
  }
  if (options.source_selection_mode != "weighted" &&
      options.source_selection_mode != "colmap") {
    throw std::invalid_argument(
        "--Colite.source_selection_mode must be either weighted or colmap");
  }
  if (options.num_threads == 0) {
    throw std::invalid_argument("--PatchMatchStereo.num_threads must be positive");
  }
}


}  // namespace

std::vector<std::string> NormalizeMvsArguments(
    const std::vector<std::string>& args) {
  if (args.empty()) return args;
  std::vector<std::string> expanded{args.front()};
  std::set<std::string> seen;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const auto equal = args[i].find('=');
    const bool is_option = args[i].rfind("--", 0) == 0 || args[i] == "-h";
    const std::string name = args[i] == "-h" ? "--help" :
        (is_option && equal != std::string::npos ? args[i].substr(0, equal) : args[i]);
    if (is_option && !seen.insert(name).second) {
      throw std::invalid_argument("Duplicate option: " + name);
    }
    expanded.push_back(name);
    if (is_option && equal != std::string::npos) {
      expanded.push_back(args[i].substr(equal + 1));
    }
  }
  return expanded;
}

bool MvsOptionAppliesTo(const std::string& name, MvsOptionScope scope) {
  return FindOption(name, scope) != nullptr;
}

void PrintMvsOptionsHelp(MvsOptionScope scope, std::ostream& out) {
  out << "colite " << Command(scope) << " " << colite::Version() << "\n\nUsage:\n  "
      << "colite " << Command(scope);
  if (scope == MvsOptionScope::kPipeline || scope == MvsOptionScope::kUndistorter) {
    out << " --input_path MODEL --image_path DIR --output_path DIR";
  } else {
    out << " --workspace_path DIR";
    if (scope == MvsOptionScope::kFusion) out << " --output_path fused.ply";
  }
  out << " [options]\n\nOptions:\n";
  for (const auto& option : Options()) {
    if (!(option.scopes & Mask(scope))) continue;
    out << "  " << option.name << " <" << option.type << ">  " << option.description
        << "; default " << (option.default_value.empty() ? "(unset)" : option.default_value) << ".\n";
  }
  out << "  --help\n  --version\n";
  if (scope != MvsOptionScope::kPipeline) out << "  --resume | --force\n  --dry_run\n";
}

ReconMvsCliOptions ParseReconMvsArguments(const std::vector<std::string>& raw_args,
                                         MvsOptionScope scope) {
  const auto args = NormalizeMvsArguments(raw_args);
  ReconMvsCliOptions options;
  if (args.size() <= 1) options.show_help = true;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const auto& name = args[i];
    if (name == "--help") { options.show_help = true; continue; }
    if (name == "--version") { options.show_version = true; continue; }
    const auto* option = FindOption(name, scope);
    if (option == nullptr) throw std::invalid_argument("Unknown argument: " + name + " for " + Command(scope));
    if (i + 1 == args.size() || args[i + 1].rfind("--", 0) == 0) {
      throw std::invalid_argument("Missing value for " + name);
    }
    option->set(options, args[++i]);
    if (name == "--input_type") options.fusion_input_type_explicit = true;
  }
  if (options.show_help || options.show_version) return options;
  if (options.workspace_format != "COLMAP") throw std::invalid_argument("--workspace_format only supports COLMAP");
  if (options.undistort_output_type != "COLMAP") throw std::invalid_argument("--output_type only supports COLMAP");
  if (options.fusion_input_type != "geometric" && options.fusion_input_type != "photometric") {
    throw std::invalid_argument("--input_type must be photometric or geometric");
  }
  if (scope == MvsOptionScope::kFusion && options.workspace_path.empty()) {
    throw std::invalid_argument("Missing required argument --workspace_path");
  }
  if (!options.workspace_path.empty()) {
    options.require_workspace_config = true;
    if (options.model_path.empty()) options.model_path = options.workspace_path / "sparse";
    if (options.image_path.empty()) options.image_path = options.workspace_path / "images";
    if (options.output_path.empty()) options.output_path = options.workspace_path;
  }
  if (scope == MvsOptionScope::kFusion && options.fused_output_path.empty()) {
    throw std::invalid_argument("Missing required argument --output_path (point cloud file)");
  }
  if (scope == MvsOptionScope::kUndistorter) options.image_undistorter = true;
  if (options.undistort_num_patch_match_src_images == 0) {
    throw std::invalid_argument("--num_patch_match_src_images must be positive");
  }
  if (scope == MvsOptionScope::kPatchMatch) options.run_patch_match = true;
  if (scope == MvsOptionScope::kFusion) options.fuse_stereo = true;
  if (options.poisson_mesher || options.delaunay_mesher) options.fuse_stereo = true;
  if (options.mesh_dense_cloud || options.dense_cloud_spatial_bounds_quantile > 0) options.filter_dense_cloud = true;
  options.patch_match_dynamic_source_selection |= options.patch_match_directional_hmm;
  options.check_geometric_consistency |= options.filter_geometric_min_num_consistent_sources > 0;
  if (options.patch_match_geom_consistency && options.run_patch_match) {
    if (options.patch_match_geometric_refinement_iterations == 0) {
      options.patch_match_geometric_refinement_iterations = options.patch_match_iterations;
    }
  } else if (!options.patch_match_geom_consistency && options.patch_match_geometric_refinement_iterations > 0) {
    throw std::invalid_argument("Colite.patch_match_geometric_refinement_iterations requires PatchMatchStereo.geom_consistency=1");
  }
  for (int threads : {options.patch_match_num_threads, options.fusion_num_threads, options.undistort_num_threads}) {
    if (threads == 0 || threads < -1) throw std::invalid_argument("num_threads must be -1 or positive");
  }
  for (int size : {options.patch_match_max_image_size, options.fusion_max_image_size}) {
    if (size == 0 || size < -1) throw std::invalid_argument("max_image_size must be -1 or positive");
  }
  if (options.patch_match_window_step <= 0 || options.patch_match_window_step > 2) {
    throw std::invalid_argument("PatchMatchStereo.window_step must be 1 or 2");
  }
  if (options.patch_match_incident_angle_sigma <= 0 || options.patch_match_geom_consistency_max_cost < 0) {
    throw std::invalid_argument("PatchMatchStereo sigma must be positive and maximum cost nonnegative");
  }
  for (float depth : {options.patch_match_depth_min, options.patch_match_depth_max}) {
    if (depth != -1 && depth <= 0) throw std::invalid_argument("PatchMatchStereo depth bounds must be -1 or positive");
  }
  if (options.patch_match_depth_min > 0 && options.patch_match_depth_max > 0 &&
      options.patch_match_depth_min >= options.patch_match_depth_max) {
    throw std::invalid_argument("PatchMatchStereo.depth_min must be less than depth_max");
  }
  ValidateOptions(options);
  return options;
}

void ValidateReconMvsArguments(const std::vector<std::string>& args) {
  (void)ParseReconMvsArguments(args);
}
int RunReconMvs(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  try {
    const auto options = ParseReconMvsArguments(args);
    if (options.show_help) { PrintMvsOptionsHelp(MvsOptionScope::kPipeline, out); return 0; }
    if (options.show_version) { out << colite::Version() << "\n"; return 0; }
    return controllers::ReconMvsController(options).Run(out);
  } catch (const std::exception& exception) {
    err << "Error: " << exception.what() << "\n";
    return 1;
  }
}
}  // namespace colite::exe
