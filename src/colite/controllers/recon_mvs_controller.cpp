#include "colite/controllers/recon_mvs_controller.h"
#include "colite/core/progress.h"
#include "colite/core/parallel.h"
#include "colite/core/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <ios>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>

#include "colite/core/camera.h"
#include "colite/core/geometry.h"
#include "colite/core/point2d.h"
#include "colite/core/point3d.h"
#include "colite/core/reconstruction.h"
#include "colite/io/sfm_workspace.h"
#include "colite/mvs/dense_cloud.h"
#include "colite/mvs/dense_map.h"
#include "colite/mvs/image_workspace.h"
#include "colite/mvs/image_undistorter.h"
#include "colite/mvs/meshing.h"
#include "colite/mvs/mesh_texturer.h"
#include "colite/mvs/mvs_pipeline.h"
#include "colite/mvs/patch_match.h"
#include "colite/mvs/patch_match_cuda.h"

namespace colite::controllers {

ReconMvsController::ReconMvsController(ReconMvsControllerOptions options)
    : options_(std::move(options)) {}

const ReconMvsControllerOptions& ReconMvsController::Options() const {
  return options_;
}

namespace {
using ControllerClock = std::chrono::steady_clock;

std::filesystem::path PhotometricCachePath(const std::filesystem::path& output) {
  return output / ".colite/photometric_cache.txt";
}

// Hash resolved inputs rather than CLI spellings or the command name. This
// also covers library callers and an undistorter that changed camera geometry.
std::string PhotometricCacheKey(const ReconMvsControllerOptions& options,
                               const Reconstruction& reconstruction,
                               const mvs::MvsPipelineSummary& summary) {
  std::ostringstream config;
  config.imbue(std::locale::classic());
  config << std::setprecision(std::numeric_limits<double>::max_digits10);
#define PHOTO_OPTION(field) config << #field << '=' << options.field << '\n'
  PHOTO_OPTION(random_seed);
  PHOTO_OPTION(resolution_level);
  PHOTO_OPTION(max_image_size);
  PHOTO_OPTION(patch_match_backend);
  PHOTO_OPTION(patch_match_gpu_index);
  PHOTO_OPTION(patch_match_iterations);
  PHOTO_OPTION(patch_match_pyramid_levels);
  PHOTO_OPTION(patch_match_window_radius);
  PHOTO_OPTION(patch_match_window_step);
  PHOTO_OPTION(patch_match_randomize_initial_normals);
  PHOTO_OPTION(patch_match_initial_normal_hemisphere_degrees);
  PHOTO_OPTION(patch_match_normal_perturbation_degrees);
  PHOTO_OPTION(patch_match_bilateral_ncc);
  PHOTO_OPTION(patch_match_sigma_spatial);
  PHOTO_OPTION(patch_match_sigma_color);
  PHOTO_OPTION(patch_match_dynamic_source_selection);
  PHOTO_OPTION(patch_match_directional_hmm);
  PHOTO_OPTION(patch_match_colmap_source_priors);
  PHOTO_OPTION(source_min_triangulation_angle_degrees);
  PHOTO_OPTION(patch_match_incident_angle_sigma);
  PHOTO_OPTION(patch_match_source_samples);
  PHOTO_OPTION(patch_match_ncc_sigma);
  PHOTO_OPTION(patch_match_min_selection_probability);
  PHOTO_OPTION(patch_match_max_selection_probability);
  PHOTO_OPTION(patch_match_perturbation_samples);
  PHOTO_OPTION(patch_match_min_num_valid_sources);
  PHOTO_OPTION(patch_match_max_num_aggregated_costs);
  PHOTO_OPTION(patch_match_propagation_schedule);
  PHOTO_OPTION(patch_match_checkerboard_rounds);
  PHOTO_OPTION(patch_match_geometric_refinement_iterations);
  PHOTO_OPTION(refine_normals_from_depth);
  PHOTO_OPTION(normal_refinement_max_relative_depth_difference);
  PHOTO_OPTION(filter_photometric_max_cost);
  PHOTO_OPTION(patch_match_filter);
  PHOTO_OPTION(patch_match_filter_min_ncc);
  PHOTO_OPTION(patch_match_filter_min_triangulation_angle_degrees);
  PHOTO_OPTION(patch_match_filter_min_num_consistent);
  PHOTO_OPTION(patch_match_filter_geom_consistency_max_cost);
#undef PHOTO_OPTION
  std::vector<std::pair<std::string, std::string>> entries{
      {"implementation", mvs::kPatchMatchImplementationVersion},
      {"options", config.str()}};
  std::set<ImageId> required;
  for (const auto& image : summary.images) {
    if (image.source_images.empty()) continue;
    required.insert(image.image_id);
    std::ostringstream state;
    state.imbue(std::locale::classic());
    state << std::setprecision(std::numeric_limits<double>::max_digits10)
          << image.image_id << ' ' << image.min_depth << ' ' << image.max_depth << '\n';
    for (const auto& source : image.source_images) {
      required.insert(source.image_id);
      state << "source " << source.image_id << '\n';
    }
    for (const auto& sample : image.sparse_depth_map.samples) {
      state << "sparse " << sample.x << ' ' << sample.y << ' ' << sample.depth << '\n';
    }
    // Bound temporary storage to one view's sparse samples.
    entries.emplace_back("reference", StableConfigHash({{"state", state.str()}}));
  }
  for (const ImageId id : required) {
    const auto& image = reconstruction.GetImage(id);
    const auto& camera = reconstruction.GetCamera(image.camera_id);
    const auto path = image.file_path.empty() ? options.image_path / image.name
                                             : std::filesystem::path(image.file_path);
    std::ostringstream state;
    state.imbue(std::locale::classic());
    state << std::setprecision(std::numeric_limits<double>::max_digits10)
          << id << ' ' << static_cast<int>(camera.model_id) << ' ' << camera.width << ' ' << camera.height;
    for (const double parameter : camera.params) state << ' ' << parameter;
    if (image.cam_from_world) {
      state << '\n' << image.cam_from_world->q_cw.coeffs().transpose()
            << '\n' << image.cam_from_world->t_cw.transpose();
    }
    state << '\n' << path.lexically_normal().generic_string() << '\n'
          << std::filesystem::file_size(path) << ' '
          << std::filesystem::last_write_time(path).time_since_epoch().count();
    entries.emplace_back("view", state.str());
  }
  return StableConfigHash(entries);
}

std::string PhotometricMapsFingerprint(const std::filesystem::path& output,
                                      const Reconstruction& reconstruction,
                                      const mvs::MvsPipelineSummary& summary) {
  std::vector<std::pair<std::string, std::string>> entries;
  for (const auto& image : summary.images) {
    if (image.source_images.empty()) continue;
    const auto file = reconstruction.GetImage(image.image_id).name + ".photometric.bin";
    for (const auto* category : {"depth_maps", "normal_maps"}) {
      const auto relative = std::filesystem::path("stereo") / category / file;
      const auto path = output / relative;
      std::error_code error;
      std::string metadata = "missing";
      if (std::filesystem::is_regular_file(path, error)) {
        metadata = std::to_string(std::filesystem::file_size(path)) + ":" +
            std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count());
      }
      entries.emplace_back(relative.generic_string(), std::move(metadata));
    }
  }
  return StableConfigHash(entries);
}

bool PhotometricCacheIsCurrent(const std::filesystem::path& path,
                              const std::string& expected_key,
                              const std::string& expected_maps) {
  std::ifstream file(path);
  std::string magic, status, key, maps;
  int version = 0;
  return (file >> magic >> version >> status >> key >> maps) &&
      magic == "COLITE_PHOTOMETRIC_CACHE" && version == 1 &&
      status == "complete" && key == expected_key && maps == expected_maps;
}

void WritePhotometricCache(const std::filesystem::path& path,
                           const std::string& key, bool complete,
                           const std::string& maps = "-") {
  std::filesystem::create_directories(path.parent_path());
  const std::filesystem::path temporary = path.string() + ".tmp";
  std::ofstream file(temporary);
  file << "COLITE_PHOTOMETRIC_CACHE 1\n"
       << (complete ? "complete" : "running") << '\n' << key << '\n' << maps << '\n';
  file.close();
  if (!file) throw std::runtime_error("Failed to write photometric cache: " + path.string());
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(path);
    std::filesystem::rename(temporary, path);
  }
}

std::size_t UndistortWorkspaceImages(
    const ReconMvsControllerOptions& cli_options,
    const std::filesystem::path& output_path,
    Reconstruction* reconstruction) {
  mvs::ImageUndistortionOptions options;
  options.blank_pixels = cli_options.undistort_blank_pixels;
  options.min_scale = cli_options.undistort_min_scale;
  options.max_scale = cli_options.undistort_max_scale;
  options.max_image_size = cli_options.undistort_max_image_size;
  options.resolution_level = cli_options.resolution_level;
  const auto original_cameras = reconstruction->Cameras();
  const std::filesystem::path image_output =
      output_path / "images";
  std::filesystem::create_directories(image_output);
  std::ofstream diagnostics(output_path / "images.tsv");
  diagnostics << "image_id\tname\twidth\theight\tvalid_pixel_ratio\n";
  std::size_t count = 0;
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    Image& image = reconstruction->GetImage(image_id);
    const Camera& original_camera = original_cameras.at(image.camera_id);
    const std::filesystem::path source_path =
        image.file_path.empty() ? cli_options.image_path / image.name
                                : std::filesystem::path(image.file_path);
    cv::Mat source = cv::imread(source_path.string(), cv::IMREAD_COLOR);
    if (source.empty()) {
      throw std::runtime_error("Failed to read image for undistortion: " +
                               source_path.string());
    }
    const cv::Size expected_size(static_cast<int>(original_camera.width),
                                 static_cast<int>(original_camera.height));
    if (source.size() != expected_size) {
      cv::resize(source, source, expected_size, 0.0, 0.0, cv::INTER_AREA);
    }
    mvs::UndistortedImage undistorted =
        mvs::UndistortImage(original_camera, source, options);
    const std::filesystem::path image_output_path = image_output / image.name;
    std::filesystem::create_directories(image_output_path.parent_path());
    if (!cv::imwrite(image_output_path.string(), undistorted.image)) {
      throw std::runtime_error("Failed to write undistorted image: " +
                               image_output_path.string());
    }
    reconstruction->GetCamera(image.camera_id) = undistorted.camera;
    image.file_path = image_output_path.string();
    diagnostics << image.id << '\t' << image.name << '\t'
                << undistorted.camera.width << '\t'
                << undistorted.camera.height << '\t'
                << std::setprecision(17) << undistorted.valid_pixel_ratio
                << '\n';
    ++count;
  }
  for (const ImageId image_id : reconstruction->RegisteredImageIds()) {
    Image& image = reconstruction->GetImage(image_id);
    const Camera& camera = reconstruction->GetCamera(image.camera_id);
    const Camera& original_camera = original_cameras.at(image.camera_id);
    for (Point2D& point : image.points2d) {
      const std::optional<Eigen::Vector2d> undistorted =
          mvs::UndistortPixel(original_camera, camera, point.xy);
      if (undistorted.has_value()) point.xy = *undistorted;
      if (!point.HasPoint3D() ||
          !reconstruction->ExistsPoint3D(point.point3d_id)) {
        continue;
      }
      const std::optional<Eigen::Vector2d> projected = ProjectWorldPoint(
          camera, *image.cam_from_world,
          reconstruction->GetPoint3D(point.point3d_id).xyz);
      if (projected.has_value()) point.xy = *projected;
    }
  }
  WriteColmapReconstruction(output_path / "sparse",
                            *reconstruction,
                            ColmapModelFormat::kTextAndBinary);
  const auto stereo = output_path / "stereo";
  for (const char* category : {"depth_maps", "normal_maps", "consistency_graphs"}) {
    std::filesystem::create_directories(stereo / category);
  }
  std::ofstream patch_match(stereo / "patch-match.cfg");
  std::ofstream fusion(stereo / "fusion.cfg");
  for (const ImageId id : reconstruction->RegisteredImageIds()) {
    const auto& name = reconstruction->GetImage(id).name;
    patch_match << name << "\n__auto__, "
                << cli_options.undistort_num_patch_match_src_images << '\n';
    fusion << name << '\n';
  }
  if (!patch_match || !fusion) throw std::runtime_error("Failed to write stereo workspace configuration");
  return count;
}

struct PatchMatchSchedulingSummary {
  double cuda_warmup_time_ms = 0.0;
  double initial_wall_time_ms = 0.0;
  double initial_sum_cuda_time_ms = 0.0;
  double initial_overlap_saved_time_ms = 0.0;
  double initial_overlap_ratio = 0.0;
  std::size_t initial_deferred_photometric_stats_images = 0;
  double initial_source_setup_sum_time_ms = 0.0;
  double initial_stereo_run_sum_time_ms = 0.0;
  double initial_stereo_wrapper_overhead_sum_time_ms = 0.0;
  double initial_postprocess_sum_time_ms = 0.0;
  double refinement_wall_time_ms = 0.0;
  double refinement_sum_cuda_time_ms = 0.0;
  double refinement_overlap_saved_time_ms = 0.0;
  double refinement_overlap_ratio = 0.0;
  double consistency_graph_build_write_time_ms = 0.0;
  double consistency_graph_read_filter_time_ms = 0.0;
  double dense_map_output_time_ms = 0.0;
};

void WriteMvsSummary(const std::filesystem::path& output_path,
                     const mvs::MvsPipelineSummary& summary,
                     std::size_t num_undistorted_images,
                     std::size_t num_initialized_patch_match_images,
                     std::size_t num_patch_match_images,
                     std::size_t num_photometric_filtered_pixels,
                     std::size_t num_normal_refined_pixels,
                     std::size_t num_geometric_consistent_pixels,
                     std::size_t num_geometric_filtered_pixels,
                     const PatchMatchSchedulingSummary& scheduling_summary,
                     const mvs::StereoFusionSummary& fusion_summary) {
  std::filesystem::create_directories(output_path);
  std::ofstream file(output_path / "summary.txt");
  file << std::setprecision(17)
       << "success " << (summary.success ? 1 : 0) << "\n"
       << "num_images " << summary.num_images << "\n"
       << "num_sparse_depth_samples " << summary.num_sparse_depth_samples
       << "\n"
       << "undistorted_images " << num_undistorted_images << "\n"
       << "initialized_patch_match_images "
       << num_initialized_patch_match_images
       << "\n"
       << "patch_match_images "
       << num_patch_match_images
       << "\n"
       << "photometric_filtered_pixels "
       << num_photometric_filtered_pixels
       << "\n"
       << "normal_refined_pixels "
       << num_normal_refined_pixels
       << "\n"
       << "geometric_consistent_pixels "
       << num_geometric_consistent_pixels
       << "\n"
       << "geometric_filtered_pixels "
       << num_geometric_filtered_pixels
       << "\n"
       << "cuda_warmup_time_ms "
       << scheduling_summary.cuda_warmup_time_ms
       << "\n"
       << "patch_match_initial_wall_time_ms "
       << scheduling_summary.initial_wall_time_ms
       << "\n"
       << "patch_match_initial_sum_cuda_time_ms "
       << scheduling_summary.initial_sum_cuda_time_ms
       << "\n"
       << "patch_match_initial_overlap_saved_time_ms "
       << scheduling_summary.initial_overlap_saved_time_ms
       << "\n"
       << "patch_match_initial_overlap_ratio "
       << scheduling_summary.initial_overlap_ratio
       << "\n"
       << "patch_match_initial_deferred_photometric_stats_images "
       << scheduling_summary.initial_deferred_photometric_stats_images
       << "\n"
       << "patch_match_initial_source_setup_sum_time_ms "
       << scheduling_summary.initial_source_setup_sum_time_ms
       << "\n"
       << "patch_match_initial_stereo_run_sum_time_ms "
       << scheduling_summary.initial_stereo_run_sum_time_ms
       << "\n"
       << "patch_match_initial_stereo_wrapper_overhead_sum_time_ms "
       << scheduling_summary.initial_stereo_wrapper_overhead_sum_time_ms
       << "\n"
       << "patch_match_initial_postprocess_sum_time_ms "
       << scheduling_summary.initial_postprocess_sum_time_ms
       << "\n"
       << "patch_match_refinement_wall_time_ms "
       << scheduling_summary.refinement_wall_time_ms
       << "\n"
       << "patch_match_refinement_sum_cuda_time_ms "
       << scheduling_summary.refinement_sum_cuda_time_ms
       << "\n"
       << "patch_match_refinement_overlap_saved_time_ms "
       << scheduling_summary.refinement_overlap_saved_time_ms
       << "\n"
       << "patch_match_refinement_overlap_ratio "
       << scheduling_summary.refinement_overlap_ratio
       << "\n"
       << "consistency_graph_build_write_time_ms "
       << scheduling_summary.consistency_graph_build_write_time_ms
       << "\n"
       << "consistency_graph_read_filter_time_ms "
       << scheduling_summary.consistency_graph_read_filter_time_ms
       << "\n"
       << "dense_map_output_time_ms "
       << scheduling_summary.dense_map_output_time_ms
       << "\n"
       << "fused_points "
       << fusion_summary.num_fused_points
       << "\n"
       << "fusion_tracks "
       << fusion_summary.num_fusion_tracks
       << "\n"
       << "fused_observations "
       << fusion_summary.num_fused_observations
       << "\n"
       << "fusion_depth_pixels "
       << fusion_summary.num_depth_pixels
       << "\n"
       << "fusion_visited_pixels "
       << fusion_summary.num_visited_pixels
       << "\n"
       << "fusion_skipped_pixels "
       << fusion_summary.num_skipped_pixels
       << "\n"
       << "fusion_rejected_tracks "
       << fusion_summary.num_rejected_tracks
       << "\n"
       << "fusion_rejected_normal_observations "
       << fusion_summary.num_rejected_normal_observations
       << "\n"
       << "fusion_rejected_view_observations "
       << fusion_summary.num_rejected_view_observations
       << "\n";
}

void WriteSourceImages(const std::filesystem::path& output_path,
                       const mvs::MvsPipelineSummary& summary) {
  std::ofstream file(output_path / "source_images.tsv");
  file << "target_image_id\tsource_image_id\tnum_shared_points"
       << "\tmean_triangulation_angle_degrees"
       << "\ttriangulation_angle_percentile_degrees"
       << "\tmean_scale_ratio\tcoverage_score\tscore\n";
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    for (const mvs::SourceImageScore& source : image_summary.source_images) {
      file << std::setprecision(17) << image_summary.image_id << "\t"
           << source.image_id << "\t"
           << source.num_shared_points << "\t"
           << source.mean_triangulation_angle_degrees << "\t"
           << source.triangulation_angle_percentile_degrees << "\t"
           << source.mean_scale_ratio << "\t"
           << source.coverage_score << "\t"
           << source.score << "\n";
    }
  }
}

void WriteSparseDepthSamples(const std::filesystem::path& output_path,
                             const mvs::MvsPipelineSummary& summary) {
  std::ofstream file(output_path / "sparse_depth_samples.tsv");
  file << "image_id\tpoint3d_id\tx\ty\tdepth\n";
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    for (const mvs::SparseDepthSample& sample :
         image_summary.sparse_depth_map.samples) {
      file << std::setprecision(17) << image_summary.image_id << "\t"
           << sample.point3d_id << "\t" << sample.x << "\t" << sample.y
           << "\t" << sample.depth << "\n";
    }
  }
}

std::size_t InitializePatchMatchMapsForWorkspace(
    const std::filesystem::path& output_path,
    const ReconMvsControllerOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  workspace_options.resolution_level = cli_options.resolution_level;
  workspace_options.load_color = false;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  mvs::PatchMatchInitializationOptions init_options;
  init_options.random_seed = cli_options.random_seed;

  const std::filesystem::path depth_dir =
      output_path / "stereo" / "depth_maps";
  const std::filesystem::path normal_dir =
      output_path / "stereo" / "normal_maps";
  std::size_t num_initialized = 0;
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    if (image_summary.sparse_depth_map.samples.empty()) {
      continue;
    }
    const mvs::MvsImage& image = workspace.GetImage(image_summary.image_id);
    const mvs::PatchMatchMaps maps =
        mvs::InitializePatchMatchMaps(image, image_summary.sparse_depth_map,
                                      init_options);
    const std::string file_name = image.name + ".init.bin";
    mvs::WriteDenseDepthMapColmapBinary(depth_dir / file_name, maps.depth_map);
    mvs::WriteNormalMapColmapBinary(normal_dir / file_name, maps.normal_map);
    if (cli_options.write_debug_maps) {
      mvs::WriteDenseDepthMapText(depth_dir / (image.name + ".init.txt"), maps.depth_map);
      mvs::WriteNormalMapText(normal_dir / (image.name + ".init.txt"), maps.normal_map);
    }
    ++num_initialized;
  }
  return num_initialized;
}

mvs::PatchMatchView MakePatchMatchView(const Reconstruction& reconstruction,
                                       ImageId image_id,
                                       std::size_t width,
                                       std::size_t height) {
  const Image& image = reconstruction.GetImage(image_id);
  mvs::PatchMatchView view;
  view.image_id = image_id;
  view.camera = reconstruction.GetCamera(image.camera_id);
  if (!image.cam_from_world.has_value()) {
    throw std::runtime_error("PatchMatch image has no registered pose.");
  }
  view.cam_from_world = *image.cam_from_world;

  const double scale_x = static_cast<double>(width) / view.camera.width;
  const double scale_y = static_cast<double>(height) / view.camera.height;
  if (scale_x != 1.0 || scale_y != 1.0) {
    view.camera.width = static_cast<std::uint32_t>(width);
    view.camera.height = static_cast<std::uint32_t>(height);
    const CameraModelSpec& spec = GetCameraModelSpec(view.camera.model_id);
    if (spec.focal_length_idxs.size() == 1) {
      view.camera.params.at(spec.focal_length_idxs[0]) *= (scale_x + scale_y) / 2.0;
    } else {
      view.camera.params.at(spec.focal_length_idxs[0]) *= scale_x;
      view.camera.params.at(spec.focal_length_idxs[1]) *= scale_y;
    }
    view.camera.params.at(spec.principal_point_idxs[0]) *= scale_x;
    view.camera.params.at(spec.principal_point_idxs[1]) *= scale_y;
  }
  return view;
}

mvs::PatchMatchView MakePatchMatchView(const Reconstruction& reconstruction,
                                      ImageId image_id, const mvs::MvsImage& pixels) {
  return MakePatchMatchView(reconstruction, image_id, pixels.Width(), pixels.Height());
}

std::vector<mvs::MeshTextureView> LoadMeshTextureViews(
    const Reconstruction& reconstruction,
    const std::vector<mvs::PatchMatchView>& geometry_views,
    const std::filesystem::path& image_path) {
  std::vector<mvs::MeshTextureView> texture_views;
  texture_views.reserve(geometry_views.size());
  for (const mvs::PatchMatchView& geometry : geometry_views) {
    const Image& reconstruction_image =
        reconstruction.GetImage(geometry.image_id);
    texture_views.push_back(mvs::ReadMeshTextureView(
        geometry, image_path / reconstruction_image.name));
  }
  return texture_views;
}

std::filesystem::path FusedCloudPath(const std::filesystem::path& output_path,
                                      const ReconMvsControllerOptions& options) {
  return options.fused_output_path.empty() ? output_path / "fused.ply"
                                          : options.fused_output_path;
}

void RunSurfaceSteps(
    const std::filesystem::path& output_path,
    const ReconMvsControllerOptions& options,
    const Reconstruction& reconstruction,
    const std::vector<ImageId>& model_order) {
  const auto dense = output_path / "dense";
  const auto fused = FusedCloudPath(output_path, options);
  const auto visibility = std::filesystem::path(fused.string() + ".vis");
  std::vector<mvs::PatchMatchView> views;
  for (const auto id : model_order) {
    const auto& camera = reconstruction.GetCamera(reconstruction.GetImage(id).camera_id);
    views.push_back(MakePatchMatchView(reconstruction, id, camera.width, camera.height));
  }
  std::vector<mvs::MeshTextureView> texture_views;
  if (options.mesh_texturer) {
    texture_views = LoadMeshTextureViews(reconstruction, views, options.image_path);
  }
  const auto finish = [&](const std::string& base, const std::string& label) {
    auto mesh_path = dense / (base + ".ply");
    auto mesh = mvs::ReadDenseTriangleMeshPly(mesh_path);
    const auto stats = base == "mesh" ? "mesh_stats.txt" : base + "-stats.txt";
    mvs::WriteDenseTriangleMeshStats(dense / stats, mvs::ComputeDenseTriangleMeshStats(mesh));
    if (options.mesh_simplifier) {
      mesh = mvs::SimplifyMesh(mesh, options.SimplificationOptions());
      mesh_path = dense / (base + "-simplified.ply");
      mvs::WriteDenseTriangleMeshPly(mesh_path, mesh);
      mvs::WriteDenseTriangleMeshStats(dense / (base + "-simplified-stats.txt"),
                                      mvs::ComputeDenseTriangleMeshStats(mesh));
    }
    if (options.mesh_texturer) {
      // Match standalone commands, including the persisted PLY precision.
      const auto textured = mvs::TextureMesh(mvs::ReadDenseTriangleMeshPly(mesh_path),
                                              texture_views, options.TexturingOptions());
      mvs::WriteTexturedTriangleMesh(dense / ("textured-" + label), textured,
          mvs::TexturedMeshOutputTypeFromString(options.texture_output_type), options.texture_write_obj);
    }
  };
  if (options.poisson_mesher) {
    if (!mvs::PoissonMeshing(options.PoissonOptions(), fused, dense / "meshed-poisson.ply")) {
      throw std::runtime_error("Poisson meshing failed");
    }
    finish("meshed-poisson", "poisson");
  }
  if (options.delaunay_mesher) {
    const auto cloud = mvs::ReadDenseTriangleMeshPly(fused);
    const auto mesh = mvs::DelaunayMeshing(options.DelaunayOptions(), cloud.vertices,
        mvs::ReadPointsVisibilityColmapBinary(visibility), views);
    mvs::WriteDenseTriangleMeshPly(dense / "meshed-delaunay.ply", mesh);
    finish("meshed-delaunay", "delaunay");
  }
  if (options.mesh_dense_cloud) {
    const auto cloud = mvs::ReadDenseTriangleMeshPly(dense / "fused_clean.ply");
    mvs::DenseCloudMeshingOptions mesh_options;
    mesh_options.max_edge_length = options.mesh_max_edge_length;
    mesh_options.max_normal_angle_degrees = options.mesh_max_normal_angle_degrees;
    mvs::WriteDenseTriangleMeshPly(dense / "mesh.ply", mvs::MeshDenseCloud(cloud.vertices, mesh_options));
    finish("mesh", "mesh");
  }
}

std::string FormatStat(double value) {
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

template <typename T>
std::string JoinDiagnosticVector(const std::vector<T>& values) {
  std::ostringstream stream;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << values[i];
  }
  return stream.str();
}

std::string FormatDistributionValue(double value) {
  if (std::isinf(value)) {
    return value > 0.0 ? "inf" : "-inf";
  }
  return FormatStat(value);
}

double Percentile(std::vector<float> values, double q) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  if (values.size() == 1) {
    return static_cast<double>(values.front());
  }
  const double position =
      q * static_cast<double>(values.size() - 1U);
  const std::size_t lower_idx = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper_idx = static_cast<std::size_t>(std::ceil(position));
  const double alpha = position - static_cast<double>(lower_idx);
  return static_cast<double>(values[lower_idx]) * (1.0 - alpha) +
         static_cast<double>(values[upper_idx]) * alpha;
}

std::string DenseMapFileName(const std::string& image_name, bool geometric) {
  return image_name + (geometric ? ".geometric.bin" : ".photometric.bin");
}

struct FusionMapInput {
  std::filesystem::path depth;
  std::filesystem::path normal;
  bool binary = false;
};

FusionMapInput ResolveFusionMapInput(const std::filesystem::path& root,
                                     const std::string& name,
                                     const ReconMvsControllerOptions& options) {
  const std::string colmap_name = DenseMapFileName(name, options.fusion_input_type == "geometric");
  const auto depth = root / "stereo/depth_maps" / colmap_name;
  const auto normal = root / "stereo/normal_maps" / colmap_name;
  return {depth, normal, true};
}

void SampleFusionColorsAtDepthResolution(std::size_t width, std::size_t height,
                                         mvs::MvsImage* image) {
  if (width == image->Width() && height == image->Height()) return;
  const double scale_x = double(image->Width()) / width;
  const double scale_y = double(image->Height()) / height;
  cv::Mat colors(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      const auto source_row = std::min(image->color.rows - 1,
          static_cast<int>(std::lround(row * scale_y)));
      const auto source_col = std::min(image->color.cols - 1,
          static_cast<int>(std::lround(col * scale_x)));
      colors.at<cv::Vec3b>(row, col) = image->color.at<cv::Vec3b>(source_row, source_col);
    }
  }
  image->scale /= scale_x;
  image->color = std::move(colors);
  image->gray.release();
}

void WriteDenseMaps(const std::filesystem::path& root, const std::string& name,
                     bool geometric, const mvs::DenseDepthMap& depth,
                     const mvs::NormalMap& normal, const mvs::DenseCostMap& cost,
                     const ReconMvsControllerOptions& options) {
  const auto file = DenseMapFileName(name, geometric);
  mvs::WriteDenseDepthMapColmapBinary(root / "stereo/depth_maps" / file, depth);
  mvs::WriteNormalMapColmapBinary(root / "stereo/normal_maps" / file, normal);
  const auto cost_path = root / "stereo/cost_maps" / file;
  if (options.write_cost_maps) mvs::WriteDenseCostMapBinary(cost_path, cost);
  else std::filesystem::remove(cost_path);
  auto text_file = std::filesystem::path(file).replace_extension(".txt");
  for (const char* category : {"depth_maps", "normal_maps", "cost_maps"}) {
    const auto path = root / "stereo" / category / text_file;
    if (!options.write_debug_maps) { std::filesystem::remove(path); continue; }
    if (std::string_view(category) == "depth_maps") mvs::WriteDenseDepthMapText(path, depth);
    else if (std::string_view(category) == "normal_maps") mvs::WriteNormalMapText(path, normal);
    else mvs::WriteDenseCostMapText(path, cost);
  }
}

mvs::PatchMatchPropagationSchedule ParsePatchMatchPropagationSchedule(
    const std::string& name) {
  if (name == "forward_backward") {
    return mvs::PatchMatchPropagationSchedule::kForwardBackward;
  }
  if (name == "four_direction") {
    return mvs::PatchMatchPropagationSchedule::kFourDirection;
  }
  if (name == "checkerboard") {
    return mvs::PatchMatchPropagationSchedule::kCheckerboard;
  }
  throw std::invalid_argument("Invalid PatchMatch propagation schedule: " +
                              name);
}

struct PatchMatchWorkspaceResult {
  ImageId image_id = kInvalidImageId;
  mvs::PatchMatchView view;
  mvs::MvsImage image;
  std::vector<ImageId> source_image_ids;
  float min_depth = 0.0F;
  float max_depth = 0.0F;
  mvs::PatchMatchStereoResult result;
  mvs::ConsistencyGraph consistency_graph;
  mvs::PatchMatchPhotometricStats stats_before_filter;
  mvs::PatchMatchPhotometricStats stats_after_photometric_filter;
  std::size_t photometric_filtered_pixels = 0;
  bool photometric_stats_deferred = false;
  bool side_maps_persisted = false;
  bool reused_photometric = false;
  bool geometric_refined = false;
  bool final_filter_applied = false;
  double initial_source_setup_time_ms = 0.0;
  double initial_stereo_run_time_ms = 0.0;
  double initial_postprocess_time_ms = 0.0;
  std::size_t geometric_consistent_pixels = 0;
  std::size_t geometric_filtered_pixels = 0;
};

bool PersistFinalPatchMatchGraph(const std::filesystem::path& output,
                                const std::string& image_name,
                                mvs::PatchMatchStereoResult* result) {
  if (!result->filter_graph) return false;
  mvs::WriteConsistencyGraphBinary(
      output / "stereo" / ".colite_refinement" / (image_name + ".graph.bin"),
      *result->filter_graph);
  result->filter_graph.reset();
  return true;
}

void WriteInitialPatchMatchDiagnostics(
    const std::filesystem::path& output_path,
    const std::vector<PatchMatchWorkspaceResult>& results) {
  std::filesystem::create_directories(output_path / "stereo");
  std::ofstream diagnostics(output_path / "stereo" /
                            "patch_match_initial.tsv");
  diagnostics << "image_id\trequested_backend\tactual_backend"
              << "\tpyramid_levels\titerations\tvalid_cost_pixels"
              << "\tplane_propagation_candidates"
              << "\tperturbation_candidate_evaluations"
              << "\tsource_selection_probability_updates"
              << "\titeration_source_selection_probability_means"
              << "\tfinal_source_selection_probability_means"
              << "\tcuda_upload_time_ms"
              << "\tcuda_initialization_time_ms"
              << "\tcuda_evaluate_cost_time_ms"
              << "\tcuda_propagation_time_ms"
              << "\tcuda_perturbation_time_ms"
              << "\tcuda_download_time_ms"
              << "\tcuda_total_time_ms"
              << "\tcuda_wrapper_total_time_ms"
              << "\tcuda_wrapper_overhead_time_ms"
              << "\tcuda_request_validation_time_ms"
              << "\tcuda_pyramid_resize_time_ms"
              << "\tcuda_host_pack_time_ms"
              << "\tcuda_single_scale_dispatch_time_ms"
              << "\tcuda_result_materialization_time_ms"
              << "\tcuda_output_to_vector_time_ms"
              << "\tcuda_dense_map_fill_time_ms"
              << "\tcuda_normal_refinement_time_ms"
              << "\tcuda_valid_cost_count_time_ms"
              << "\tcuda_host_initial_normal_upload_normalizations"
              << "\tcuda_host_output_normalizations"
              << "\tcuda_device_count_queries"
              << "\tcuda_device_count_cache_hits"
              << "\tcuda_workspace_pool_size"
              << "\tcuda_workspace_pool_slot\treused_photometric\n";
  for (const PatchMatchWorkspaceResult& result : results) {
    const mvs::PatchMatchStereoSummary& summary = result.result.summary;
    const double wrapper_overhead_time_ms =
        std::max(0.0,
                 summary.cuda_wrapper_total_time_ms -
                     summary.cuda_total_time_ms);
    diagnostics << result.image_id << "\t"
                << mvs::PatchMatchBackendToName(summary.requested_backend)
                << "\t"
                << (result.reused_photometric ? "persisted" : mvs::PatchMatchBackendToName(summary.actual_backend))
                << "\t"
                << summary.num_pyramid_levels << "\t"
                << summary.num_iterations << "\t"
                << (result.reused_photometric ? std::string("unknown") : std::to_string(summary.num_valid_cost_pixels)) << "\t"
                << summary.num_plane_propagation_candidates << "\t"
                << summary.num_perturbation_candidate_evaluations << "\t"
                << summary.num_source_selection_probability_updates << "\t"
                << JoinDiagnosticVector(
                       summary.iteration_source_selection_probability_means)
                << "\t"
                << JoinDiagnosticVector(
                       summary.final_source_selection_probability_means)
                << "\t"
                << FormatStat(summary.cuda_upload_time_ms) << "\t"
                << FormatStat(summary.cuda_initialization_time_ms) << "\t"
                << FormatStat(summary.cuda_evaluate_cost_time_ms) << "\t"
                << FormatStat(summary.cuda_propagation_time_ms) << "\t"
                << FormatStat(summary.cuda_perturbation_time_ms) << "\t"
                << FormatStat(summary.cuda_download_time_ms) << "\t"
                << FormatStat(summary.cuda_total_time_ms) << "\t"
                << FormatStat(summary.cuda_wrapper_total_time_ms) << "\t"
                << FormatStat(wrapper_overhead_time_ms) << "\t"
                << FormatStat(summary.cuda_request_validation_time_ms)
                << "\t"
                << FormatStat(summary.cuda_pyramid_resize_time_ms) << "\t"
                << FormatStat(summary.cuda_host_pack_time_ms) << "\t"
                << FormatStat(summary.cuda_single_scale_dispatch_time_ms)
                << "\t"
                << FormatStat(summary.cuda_result_materialization_time_ms)
                << "\t"
                << FormatStat(summary.cuda_output_to_vector_time_ms) << "\t"
                << FormatStat(summary.cuda_dense_map_fill_time_ms) << "\t"
                << FormatStat(summary.cuda_normal_refinement_time_ms) << "\t"
                << FormatStat(summary.cuda_valid_cost_count_time_ms) << "\t"
                << summary.cuda_host_initial_normal_upload_normalizations
                << "\t"
                << summary.cuda_host_output_normalizations << "\t"
                << summary.cuda_device_count_queries << "\t"
                << summary.cuda_device_count_cache_hits << "\t"
                << summary.cuda_workspace_pool_size << "\t"
                << summary.cuda_workspace_pool_slot << "\t" << result.reused_photometric << "\n";
  }
}

using CudaWorkspacePool = std::vector<mvs::CudaPatchMatchWorkspacePtr>;

bool UsesCudaWorkspacePool(const mvs::PatchMatchStereoOptions& options) {
  if (options.backend == mvs::PatchMatchBackend::kCpu) {
    return false;
  }
  if (options.backend == mvs::PatchMatchBackend::kCuda) {
    return true;
  }
  return mvs::IsCudaPatchMatchAvailable();
}

CudaWorkspacePool CreateCudaWorkspacePool(
    const mvs::PatchMatchStereoOptions& options,
    std::size_t num_images) {
  CudaWorkspacePool pool;
  if (!UsesCudaWorkspacePool(options) || num_images == 0) {
    return pool;
  }
  // Bound each GPU to one set of layered textures and pinned staging buffers.
  // CPU preparation and consistency filtering retain their own parallelism.
  const std::size_t pool_size = 1;
  pool.reserve(pool_size);
  for (std::size_t i = 0; i < pool_size; ++i) {
    pool.push_back(mvs::MakeCudaPatchMatchWorkspace());
  }
  return pool;
}

void FillOverlapSummary(double wall_time_ms,
                        double sum_cuda_time_ms,
                        double* wall_time_out,
                        double* sum_cuda_time_out,
                        double* overlap_saved_time_out,
                        double* overlap_ratio_out) {
  *wall_time_out = wall_time_ms;
  *sum_cuda_time_out = sum_cuda_time_ms;
  *overlap_saved_time_out = std::max(0.0, sum_cuda_time_ms - wall_time_ms);
  *overlap_ratio_out =
      sum_cuda_time_ms <= 0.0 ? 0.0
                              : *overlap_saved_time_out / sum_cuda_time_ms;
}

std::string TrimConfigLine(const std::string& value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return {};
  return value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
}

std::vector<std::string> ReadWorkspaceConfig(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Failed to read stereo configuration: " + path.string());
  std::vector<std::string> lines;
  for (std::string line; std::getline(file, line);) {
    line = TrimConfigLine(line);
    if (!line.empty() && line.front() != '#') lines.push_back(std::move(line));
  }
  if (!file.eof()) throw std::runtime_error("Failed to read stereo configuration: " + path.string());
  return lines;
}

std::unordered_map<std::string, ImageId> RegisteredImageNames(const Reconstruction& reconstruction) {
  std::unordered_map<std::string, ImageId> names;
  for (const ImageId id : reconstruction.RegisteredImageIds()) {
    const auto& name = reconstruction.GetImage(id).name;
    if (!names.emplace(name, id).second) throw std::runtime_error("Duplicate registered image name: " + name);
  }
  return names;
}

ImageId ConfigImageId(const std::unordered_map<std::string, ImageId>& names,
                      const std::string& name) {
  const auto found = names.find(name);
  if (found == names.end()) throw std::invalid_argument("Unknown registered image in stereo configuration: " + name);
  return found->second;
}

// Configuration selects reference jobs independently of their source pool.
// Explicit source lists are authoritative, including images without sparse tracks.
void ApplyPatchMatchConfig(const Reconstruction& reconstruction,
                           const ReconMvsControllerOptions& options,
                           const mvs::MvsPipelineOptions& source_options,
                           mvs::MvsPipelineSummary* summary) {
  const auto path = options.patch_match_config_path.empty()
      ? options.output_path / "stereo/patch-match.cfg" : options.patch_match_config_path;
  if (!std::filesystem::exists(path) && !options.require_workspace_config &&
      options.patch_match_config_path.empty()) return;
  const auto lines = ReadWorkspaceConfig(path);
  if (lines.empty() || lines.size() % 2 != 0) {
    throw std::invalid_argument("PatchMatch configuration requires reference/source line pairs: " + path.string());
  }
  const auto names = RegisteredImageNames(reconstruction);
  std::unordered_map<ImageId, const mvs::ImageMvsSummary*> entries;
  for (const auto& image : summary->images) entries.emplace(image.image_id, &image);
  std::unordered_set<ImageId> seen_references;
  std::vector<mvs::ImageMvsSummary> selected;
  for (std::size_t line = 0; line < lines.size(); line += 2) {
    const ImageId id = ConfigImageId(names, lines[line]);
    if (!seen_references.insert(id).second) throw std::invalid_argument("Duplicate PatchMatch reference: " + lines[line]);
    mvs::ImageMvsSummary entry = *entries.at(id);
    entry.source_images.clear();
    std::vector<std::string> sources;
    std::istringstream csv(lines[line + 1]);
    for (std::string token; std::getline(csv, token, ',');) {
      token = TrimConfigLine(token);
      if (token.empty()) throw std::invalid_argument("Empty PatchMatch source name");
      sources.push_back(std::move(token));
    }
    if (lines[line + 1].back() == ',') throw std::invalid_argument("Empty PatchMatch source name");
    if (sources.size() == 2 && sources.front() == "__auto__") {
      std::size_t count = 0;
      const auto& text = sources.back();
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), count);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
          count == 0 || count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Invalid __auto__ source count: " + text);
      }
      auto automatic = source_options;
      automatic.source_selection_mode = mvs::SourceSelectionMode::kColmap;
      automatic.max_num_source_images = count;
      entry.source_images = mvs::SelectSourceImages(reconstruction, id, automatic);
    } else {
      std::vector<ImageId> source_ids;
      if (sources.size() == 1 && sources.front() == "__all__") {
        for (const ImageId candidate : reconstruction.RegisteredImageIds()) {
          if (candidate != id) source_ids.push_back(candidate);
        }
      } else {
        std::unordered_set<ImageId> seen_sources;
        for (const auto& name : sources) {
          const ImageId source = ConfigImageId(names, name);
          if (source == id || !seen_sources.insert(source).second) {
            throw std::invalid_argument("Self or duplicate PatchMatch source: " + name);
          }
          source_ids.push_back(source);
        }
      }
      for (const ImageId source : source_ids) {
        mvs::SourceImageScore score;
        score.image_id = source;
        entry.source_images.push_back(score);
      }
    }
    selected.push_back(std::move(entry));
  }
  summary->images = std::move(selected);
  summary->num_images = summary->images.size();
  summary->num_sparse_depth_samples = 0;
  for (const auto& entry : summary->images) summary->num_sparse_depth_samples += entry.sparse_depth_map.samples.size();
}

std::vector<const mvs::ImageMvsSummary*> CollectPatchMatchSummaries(
    const mvs::MvsPipelineSummary& summary) {
  std::vector<const mvs::ImageMvsSummary*> image_summaries;
  for (const mvs::ImageMvsSummary& image_summary : summary.images) {
    if (!(image_summary.min_depth > 0 && image_summary.min_depth < image_summary.max_depth) ||
        image_summary.source_images.empty()) {
      continue;
    }
    image_summaries.push_back(&image_summary);
  }
  return image_summaries;
}

std::vector<ImageId> CollectRequiredMvsImageIds(
    const std::vector<const mvs::ImageMvsSummary*>& image_summaries) {
  std::vector<ImageId> image_ids;
  for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
    image_ids.push_back(image_summary->image_id);
    for (const mvs::SourceImageScore& source : image_summary->source_images) {
      image_ids.push_back(source.image_id);
    }
  }
  std::sort(image_ids.begin(), image_ids.end());
  image_ids.erase(std::unique(image_ids.begin(), image_ids.end()),
                  image_ids.end());
  return image_ids;
}

std::unordered_map<ImageId, mvs::MvsImage> PreloadMvsImages(
    const std::vector<ImageId>& image_ids,
    mvs::MvsImageWorkspace* workspace) {
  std::unordered_map<ImageId, mvs::MvsImage> images;
  images.reserve(image_ids.size());
  for (const ImageId image_id : image_ids) {
    images.emplace(image_id, workspace->GetImage(image_id));
  }
  return images;
}

std::unordered_map<ImageId, mvs::PatchMatchView> BuildPatchMatchViews(
    const Reconstruction& reconstruction,
    const std::unordered_map<ImageId, mvs::MvsImage>& images) {
  std::unordered_map<ImageId, mvs::PatchMatchView> views;
  views.reserve(images.size());
  for (const auto& [image_id, image] : images) {
    views.emplace(image_id,
                  MakePatchMatchView(reconstruction, image_id, image));
  }
  return views;
}

PatchMatchWorkspaceResult RunInitialPatchMatchForImage(
    const mvs::ImageMvsSummary& image_summary,
    const std::unordered_map<ImageId, mvs::MvsImage>& images,
    const std::unordered_map<ImageId, mvs::PatchMatchView>& views,
    const mvs::PatchMatchStereoOptions& patch_match_options,
    const ReconMvsControllerOptions& cli_options,
    bool run_geometric_refinement,
    mvs::CudaPatchMatchWorkspace* cuda_workspace,
    std::size_t cuda_workspace_pool_size,
    std::size_t cuda_workspace_pool_slot) {
  const ControllerClock::time_point source_setup_start =
      ControllerClock::now();
  const mvs::MvsImage& reference_image = images.at(image_summary.image_id);
  const mvs::PatchMatchView& reference_view = views.at(image_summary.image_id);

  std::vector<mvs::PatchMatchView> source_views;
  std::vector<mvs::MvsImage> source_images;
  const std::size_t num_patch_match_sources =
      image_summary.source_images.size();
  source_views.reserve(num_patch_match_sources);
  source_images.reserve(num_patch_match_sources);
  for (std::size_t source_idx = 0;
       source_idx < num_patch_match_sources; ++source_idx) {
    const mvs::SourceImageScore& source =
        image_summary.source_images[source_idx];
    source_views.push_back(views.at(source.image_id));
    source_images.push_back(images.at(source.image_id));
  }
  const double source_setup_time_ms =
      std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                source_setup_start)
          .count();

  const ControllerClock::time_point stereo_run_start = ControllerClock::now();
  mvs::PatchMatchStereoOptions initial_options = patch_match_options;
  initial_options.filter.min_num_consistent = std::min(
      initial_options.filter.min_num_consistent, source_views.size());
  initial_options.filter.enabled &= !run_geometric_refinement;
  mvs::PatchMatchStereoResult result;
  const auto photo_file = DenseMapFileName(reference_image.name, false);
  const auto photo_depth = cli_options.output_path / "stereo/depth_maps" / photo_file;
  const auto photo_normal = cli_options.output_path / "stereo/normal_maps" / photo_file;
  bool reuse_photometric = run_geometric_refinement && !cli_options.force_recompute &&
      std::filesystem::exists(photo_depth) && std::filesystem::exists(photo_normal);
  if (reuse_photometric) {
    try {
      result.depth_map = mvs::ReadDenseDepthMapBinary(photo_depth);
      result.normal_map = mvs::ReadNormalMapBinary(photo_normal);
    } catch (const std::exception& error) {
      LOG(WARNING) << "Recomputing invalid photometric reference maps for "
                   << reference_image.name << ": " << error.what();
      reuse_photometric = false;
    }
  }
  if (reuse_photometric) {
    mvs::DownsizeDepthNormalMaps(reference_image.Width(), reference_image.Height(),
                                &result.depth_map, &result.normal_map);
    if (result.depth_map.Width() != reference_image.Width() ||
        result.depth_map.Height() != reference_image.Height()) {
      throw std::runtime_error("Photometric map/image resolution mismatch for " + reference_image.name +
          "; use the same PatchMatchStereo.max_image_size or --force to recompute");
    }
    // Standard files carry no photometric costs. Refinement recomputes them;
    // never present invented zero costs as measurements of an exchanged map.
    result.summary.requested_backend = patch_match_options.backend;
  } else {
    result = mvs::RunPatchMatchStereoForImageWithCudaWorkspace(
      reference_view, source_views, reference_image, source_images,
      static_cast<float>(image_summary.min_depth),
      static_cast<float>(image_summary.max_depth), cuda_workspace,
      initial_options);
  }
  if (result.summary.actual_backend == mvs::PatchMatchBackend::kCuda &&
      cuda_workspace_pool_size > 0) {
    result.summary.cuda_workspace_pool_size = cuda_workspace_pool_size;
    result.summary.cuda_workspace_pool_slot = cuda_workspace_pool_slot;
  }
  const double stereo_run_time_ms =
      std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                stereo_run_start)
          .count();

  const ControllerClock::time_point postprocess_start = ControllerClock::now();
  mvs::PatchMatchPhotometricFilterSummary filter_summary;
  mvs::PatchMatchPhotometricStats stats_before;
  mvs::PatchMatchPhotometricStats stats_after;
  const bool defer_photometric_stats = run_geometric_refinement;
  if (!defer_photometric_stats) {
    stats_before = mvs::ComputePatchMatchPhotometricStats(result.cost_map);
    if (cli_options.filter_photometric_max_cost >= 0.0F) {
      filter_summary = mvs::FilterPatchMatchPhotometricResults(
          cli_options.filter_photometric_max_cost, &result.depth_map,
          &result.normal_map, &result.cost_map);
    } else {
      filter_summary.num_pixels = result.cost_map.NumPixels();
      filter_summary.num_kept_pixels = result.cost_map.ValidPixelCount();
    }
    stats_after = mvs::ComputePatchMatchPhotometricStats(result.cost_map);
  } else {
    filter_summary.num_pixels = result.cost_map.NumPixels();
    filter_summary.num_kept_pixels = result.cost_map.ValidPixelCount();
  }

  PatchMatchWorkspaceResult workspace_result;
  workspace_result.reused_photometric = reuse_photometric;
  workspace_result.image_id = image_summary.image_id;
  workspace_result.view = reference_view;
  workspace_result.image = reference_image;
  workspace_result.min_depth = static_cast<float>(image_summary.min_depth);
  workspace_result.max_depth = static_cast<float>(image_summary.max_depth);
  workspace_result.source_image_ids.reserve(num_patch_match_sources);
  for (std::size_t source_idx = 0;
       source_idx < num_patch_match_sources; ++source_idx) {
    workspace_result.source_image_ids.push_back(
        image_summary.source_images[source_idx].image_id);
  }
  workspace_result.final_filter_applied = PersistFinalPatchMatchGraph(
      cli_options.output_path, reference_image.name, &result);
  workspace_result.result = std::move(result);
  workspace_result.stats_before_filter = stats_before;
  workspace_result.stats_after_photometric_filter = stats_after;
  workspace_result.photometric_filtered_pixels =
      filter_summary.num_filtered_pixels;
  workspace_result.photometric_stats_deferred = defer_photometric_stats;
  workspace_result.initial_source_setup_time_ms = source_setup_time_ms;
  workspace_result.initial_stereo_run_time_ms = stereo_run_time_ms;
  workspace_result.initial_postprocess_time_ms =
      std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                postprocess_start)
          .count();
  return workspace_result;
}

std::vector<PatchMatchWorkspaceResult> RunInitialPatchMatchForWorkspace(
    const std::vector<const mvs::ImageMvsSummary*>& image_summaries,
    const std::unordered_map<ImageId, mvs::MvsImage>& images,
    const std::unordered_map<ImageId, mvs::PatchMatchView>& views,
    const mvs::PatchMatchStereoOptions& patch_match_options,
    const ReconMvsControllerOptions& cli_options,
    bool run_geometric_refinement,
    const std::vector<mvs::CudaPatchMatchWorkspace*>& cuda_workspace_pool,
    PatchMatchSchedulingSummary* scheduling_summary) {
  const ControllerClock::time_point batch_start = ControllerClock::now();
  std::vector<PatchMatchWorkspaceResult> results;
  results.reserve(image_summaries.size());
  ReportProgress("patch_match_photometric", 0, image_summaries.size());
  const std::size_t num_threads =
      std::max<std::size_t>(1, cli_options.num_threads);
  if (!cuda_workspace_pool.empty()) {
    const std::size_t pool_size = cuda_workspace_pool.size();
    std::vector<std::future<PatchMatchWorkspaceResult>> futures;
    futures.reserve(pool_size);
    std::size_t next_slot = 0;
    const auto collect_oldest_future = [&results, &futures, &image_summaries]() {
      results.push_back(futures.front().get());
      futures.erase(futures.begin());
      ReportProgress("patch_match_photometric", results.size(), image_summaries.size(), results.back().image.name);
    };
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      if (futures.size() >= pool_size) {
        collect_oldest_future();
      }
      const std::size_t slot = next_slot;
      next_slot = (next_slot + 1) % pool_size;
      futures.push_back(std::async(
          std::launch::async,
          [&images, &views, &patch_match_options, &cli_options,
           run_geometric_refinement, image_summary,
           &cuda_workspace_pool, pool_size, slot]() {
            return RunInitialPatchMatchForImage(
                *image_summary, images, views, patch_match_options,
                cli_options, run_geometric_refinement,
                cuda_workspace_pool[slot], pool_size, slot);
          }));
    }
    while (!futures.empty()) {
      collect_oldest_future();
    }
  } else if (num_threads == 1 ||
      image_summaries.size() <= 1) {
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      results.push_back(RunInitialPatchMatchForImage(
          *image_summary, images, views, patch_match_options, cli_options,
          run_geometric_refinement, nullptr, 0, 0));
      ReportProgress("patch_match_photometric", results.size(), image_summaries.size(), results.back().image.name);
    }
  } else {
    std::vector<std::future<PatchMatchWorkspaceResult>> futures;
    futures.reserve(num_threads);
    const auto collect_oldest_future = [&results, &futures, &image_summaries]() {
      results.push_back(futures.front().get());
      futures.erase(futures.begin());
      ReportProgress("patch_match_photometric", results.size(), image_summaries.size(), results.back().image.name);
    };
    for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
      futures.push_back(std::async(
          std::launch::async,
          [&images, &views, &patch_match_options, &cli_options,
           run_geometric_refinement, image_summary]() {
            return RunInitialPatchMatchForImage(
                *image_summary, images, views, patch_match_options,
                cli_options, run_geometric_refinement,
                nullptr, 0, 0);
          }));
      if (futures.size() >= num_threads) {
        collect_oldest_future();
      }
    }
    while (!futures.empty()) {
      collect_oldest_future();
    }
  }

  std::sort(results.begin(), results.end(),
            [](const PatchMatchWorkspaceResult& lhs,
               const PatchMatchWorkspaceResult& rhs) {
              return lhs.image_id < rhs.image_id;
            });
  if (scheduling_summary != nullptr) {
    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                  batch_start)
            .count();
    double sum_cuda_time_ms = 0.0;
    for (const PatchMatchWorkspaceResult& result : results) {
      if (result.result.summary.actual_backend == mvs::PatchMatchBackend::kCuda) {
        sum_cuda_time_ms += result.result.summary.cuda_total_time_ms;
        scheduling_summary->initial_stereo_wrapper_overhead_sum_time_ms +=
            std::max(0.0, result.initial_stereo_run_time_ms -
                              result.result.summary.cuda_total_time_ms);
      }
      scheduling_summary->initial_source_setup_sum_time_ms +=
          result.initial_source_setup_time_ms;
      scheduling_summary->initial_stereo_run_sum_time_ms +=
          result.initial_stereo_run_time_ms;
      scheduling_summary->initial_postprocess_sum_time_ms +=
          result.initial_postprocess_time_ms;
    }
    FillOverlapSummary(
        wall_time_ms, sum_cuda_time_ms,
        &scheduling_summary->initial_wall_time_ms,
        &scheduling_summary->initial_sum_cuda_time_ms,
        &scheduling_summary->initial_overlap_saved_time_ms,
        &scheduling_summary->initial_overlap_ratio);
  }
  return results;
}

struct PatchMatchRefinementJob {
  std::size_t result_index = 0;
  mvs::PatchMatchView reference_view;
  mvs::MvsImage reference_image;
  std::vector<mvs::PatchMatchView> source_views;
  std::vector<mvs::MvsImage> source_images;
  std::vector<std::size_t> source_result_indices;
  mvs::NormalMap initial_normal_map;
  float min_depth = 0.0F;
  float max_depth = 0.0F;
};

struct PatchMatchRefinementResult {
  std::size_t result_index = 0;
  mvs::PatchMatchStereoResult result;
  bool side_maps_persisted = false;
  bool final_filter_applied = false;
  mvs::PatchMatchPhotometricStats stats_before_filter;
  mvs::PatchMatchPhotometricStats stats_after_filter;
  std::size_t photometric_filtered_pixels = 0;
};

std::filesystem::path RefinementSideMapPath(
    const std::filesystem::path& output, const std::string& image_name,
    const char* kind) {
  return output / "stereo" / ".colite_refinement" /
         (image_name + "." + kind + ".bin");
}

PatchMatchRefinementResult RunPatchMatchRefinementJob(
    PatchMatchRefinementJob job,
    const std::vector<mvs::DenseDepthMap>& initial_depth_maps,
    const mvs::PatchMatchStereoOptions& refinement_options,
    const ReconMvsControllerOptions& cli_options,
    mvs::CudaPatchMatchWorkspace* cuda_workspace,
    std::size_t cuda_workspace_pool_size,
    std::size_t cuda_workspace_pool_slot) {
  PatchMatchRefinementResult output;
  output.result_index = job.result_index;
  mvs::PatchMatchMaps initial_maps;
  initial_maps.depth_map = initial_depth_maps.at(job.result_index);
  initial_maps.normal_map = std::move(job.initial_normal_map);
  auto job_options = refinement_options;
  job_options.filter.min_num_consistent = std::min(
      job_options.filter.min_num_consistent, job.source_views.size());
  if (cuda_workspace != nullptr) {
    std::vector<const mvs::DenseDepthMap*> source_depth_maps;
    source_depth_maps.reserve(job.source_result_indices.size());
    for (const std::size_t source_result_index : job.source_result_indices) {
      source_depth_maps.push_back(
          &initial_depth_maps.at(source_result_index));
    }
    output.result = mvs::
        RunPatchMatchStereoForImageWithSourceDepthPointersAndCudaWorkspace(
            job.reference_view, job.source_views, job.reference_image,
            job.source_images, source_depth_maps, job.min_depth,
            job.max_depth, cuda_workspace, job_options,
            &initial_maps);
  } else {
    // CPU kernels currently use an owning contiguous depth-map vector.
    std::vector<mvs::DenseDepthMap> source_depth_maps;
    source_depth_maps.reserve(job.source_result_indices.size());
    for (const std::size_t source_result_index : job.source_result_indices) {
      source_depth_maps.push_back(initial_depth_maps.at(source_result_index));
    }
    output.result =
        mvs::RunPatchMatchStereoForImageWithSourceDepthsAndCudaWorkspace(
            job.reference_view, job.source_views, job.reference_image,
            job.source_images, source_depth_maps, job.min_depth,
            job.max_depth, nullptr, job_options, &initial_maps);
  }
  if (output.result.summary.actual_backend == mvs::PatchMatchBackend::kCuda &&
      cuda_workspace_pool_size > 0) {
    output.result.summary.cuda_workspace_pool_size =
        cuda_workspace_pool_size;
    output.result.summary.cuda_workspace_pool_slot =
        cuda_workspace_pool_slot;
  }
  output.final_filter_applied = PersistFinalPatchMatchGraph(
      cli_options.output_path, job.reference_image.name, &output.result);
  if (!cli_options.fuse_stereo) {
    // No subsequent refinement job reads these new normal/cost maps: all
    // source depths come from the immutable photometric snapshot. Filter and
    // spill a completed view now instead of accumulating every side map.
    output.stats_before_filter = mvs::ComputePatchMatchPhotometricStats(output.result.cost_map);
    if (cli_options.filter_photometric_max_cost >= 0.0F) {
      const auto filtered = mvs::FilterPatchMatchPhotometricResults(
          cli_options.filter_photometric_max_cost, &output.result.depth_map,
          &output.result.normal_map, &output.result.cost_map);
      output.photometric_filtered_pixels = filtered.num_filtered_pixels;
    }
    output.stats_after_filter = mvs::ComputePatchMatchPhotometricStats(output.result.cost_map);
    mvs::OrientNormalMapTowardsCamera(job.reference_view, &output.result.normal_map);
    mvs::WriteNormalMapBinary(RefinementSideMapPath(cli_options.output_path,
        job.reference_image.name, "normal"), output.result.normal_map);
    mvs::WriteDenseCostMapBinary(RefinementSideMapPath(cli_options.output_path,
        job.reference_image.name, "cost"), output.result.cost_map);
    output.result.normal_map = mvs::NormalMap();
    output.result.cost_map = mvs::DenseCostMap();
    output.side_maps_persisted = true;
  }
  return output;
}

std::vector<PatchMatchRefinementResult> RunPatchMatchRefinementJobs(
    std::vector<PatchMatchRefinementJob>* jobs,
    const std::vector<mvs::DenseDepthMap>& initial_depth_maps,
    const mvs::PatchMatchStereoOptions& refinement_options,
    const ReconMvsControllerOptions& cli_options,
    const std::vector<mvs::CudaPatchMatchWorkspace*>& cuda_workspace_pool,
    PatchMatchSchedulingSummary* scheduling_summary) {
  const ControllerClock::time_point batch_start = ControllerClock::now();
  std::vector<PatchMatchRefinementResult> outputs;
  outputs.reserve(jobs->size());
  ReportProgress("patch_match_geometric", 0, jobs->size());
  const std::size_t num_threads =
      std::max<std::size_t>(1, cli_options.num_threads);
  const auto collect_oldest_future =
      [&outputs, &cuda_workspace_pool, jobs](std::vector<std::future<PatchMatchRefinementResult>>*
                     futures) {
        outputs.push_back(futures->front().get());
        futures->erase(futures->begin());
        ReportProgress("patch_match_geometric", outputs.size(), jobs->size());
        if (cuda_workspace_pool.size() == 1 && outputs.size() % 8 == 0) {
          ReleaseUnusedMemory();
        }
      };

  if (!cuda_workspace_pool.empty()) {
    const std::size_t pool_size = cuda_workspace_pool.size();
    std::vector<std::future<PatchMatchRefinementResult>> futures;
    futures.reserve(pool_size);
    std::size_t next_slot = 0;
    for (PatchMatchRefinementJob& job : *jobs) {
      if (futures.size() >= pool_size) {
        collect_oldest_future(&futures);
      }
      const std::size_t slot = next_slot;
      next_slot = (next_slot + 1) % pool_size;
      futures.push_back(std::async(
          std::launch::async,
          [job = std::move(job), &initial_depth_maps, &refinement_options, &cli_options,
           &cuda_workspace_pool, pool_size, slot]() mutable {
            return RunPatchMatchRefinementJob(
                std::move(job), initial_depth_maps, refinement_options, cli_options,
                cuda_workspace_pool[slot], pool_size, slot);
          }));
    }
    while (!futures.empty()) {
      collect_oldest_future(&futures);
    }
  } else if (num_threads == 1 || jobs->size() <= 1) {
    for (PatchMatchRefinementJob& job : *jobs) {
      outputs.push_back(RunPatchMatchRefinementJob(
          std::move(job), initial_depth_maps, refinement_options, cli_options,
          nullptr, 0, 0));
      ReportProgress("patch_match_geometric", outputs.size(), jobs->size());
    }
  } else {
    std::vector<std::future<PatchMatchRefinementResult>> futures;
    futures.reserve(num_threads);
    for (PatchMatchRefinementJob& job : *jobs) {
      futures.push_back(std::async(
          std::launch::async,
          [job = std::move(job), &initial_depth_maps,
           &refinement_options, &cli_options]() mutable {
            return RunPatchMatchRefinementJob(
                std::move(job), initial_depth_maps, refinement_options, cli_options,
                nullptr, 0, 0);
          }));
      if (futures.size() >= num_threads) {
        collect_oldest_future(&futures);
      }
    }
    while (!futures.empty()) {
      collect_oldest_future(&futures);
    }
  }

  if (scheduling_summary != nullptr) {
    const double wall_time_ms =
        std::chrono::duration<double, std::milli>(ControllerClock::now() -
                                                  batch_start)
            .count();
    double sum_cuda_time_ms = 0.0;
    for (const PatchMatchRefinementResult& output : outputs) {
      if (output.result.summary.actual_backend == mvs::PatchMatchBackend::kCuda) {
        sum_cuda_time_ms += output.result.summary.cuda_total_time_ms;
      }
    }
    const double overlap_saved_time_ms =
        std::max(0.0, sum_cuda_time_ms - wall_time_ms);
    scheduling_summary->refinement_wall_time_ms += wall_time_ms;
    scheduling_summary->refinement_sum_cuda_time_ms += sum_cuda_time_ms;
    scheduling_summary->refinement_overlap_saved_time_ms +=
        overlap_saved_time_ms;
    scheduling_summary->refinement_overlap_ratio =
        scheduling_summary->refinement_sum_cuda_time_ms > 0.0
            ? scheduling_summary->refinement_overlap_saved_time_ms /
                  scheduling_summary->refinement_sum_cuda_time_ms
            : 0.0;
  }
  return outputs;
}

void WriteFusedPointCloudPly(const std::filesystem::path& path,
                             const std::vector<mvs::FusedPoint>& points) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write fused point cloud: " +
                             path.string());
  }

  file << "ply\n"
       << "format binary_little_endian 1.0\n"
       << "element vertex " << points.size() << "\n"
       << "property float x\n"
       << "property float y\n"
       << "property float z\n"
       << "property float nx\n"
       << "property float ny\n"
       << "property float nz\n"
       << "property uchar red\n"
       << "property uchar green\n"
       << "property uchar blue\n"
       << "end_header\n";
  for (const mvs::FusedPoint& point : points) {
    const std::array<float, 6> values = {
        static_cast<float>(point.xyz.x()),
        static_cast<float>(point.xyz.y()),
        static_cast<float>(point.xyz.z()),
        point.normal.x(),
        point.normal.y(),
        point.normal.z(),
    };
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(sizeof(values)));
    file.write(reinterpret_cast<const char*>(point.color.data()),
               static_cast<std::streamsize>(3 * sizeof(std::uint8_t)));
    if (!file) {
      throw std::runtime_error("Failed while writing fused point cloud: " +
                               path.string());
    }
  }
}

void WriteDenseCloudFilterStats(
    const std::filesystem::path& path,
    const mvs::DenseCloudFilterStats& stats) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write dense cloud filter stats: " +
                             path.string());
  }

  file << "num_input_points " << stats.num_input_points << "\n"
       << "num_filtered_points " << stats.num_filtered_points << "\n"
       << "num_removed_by_track_observations "
       << stats.num_removed_by_track_observations << "\n"
       << "num_removed_by_spatial_bounds "
       << stats.num_removed_by_spatial_bounds << "\n"
       << "num_removed_by_voxel " << stats.num_removed_by_voxel << "\n";
}

void WriteFusionTracksDiagnostics(
    const std::filesystem::path& path,
    const std::vector<mvs::StereoFusionTrack>& tracks) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion tracks diagnostics: " +
                             path.string());
  }

  file << "point_idx\tseed_image_id\tseed_row\tseed_col"
       << "\tnum_observations\tnum_unique_images\tmax_traversal_depth"
       << "\tmax_normal_angle_degrees\tmax_view_angle_degrees\n";
  file << std::setprecision(17);
  for (const mvs::StereoFusionTrack& track : tracks) {
    file << track.point_idx << "\t"
         << track.seed_image_id << "\t"
         << track.seed_row << "\t"
         << track.seed_col << "\t"
         << track.num_observations << "\t"
         << track.num_unique_images << "\t"
         << track.max_traversal_depth << "\t"
         << track.max_normal_angle_degrees << "\t"
         << track.max_view_angle_degrees << "\n";
  }
}

void WriteFusionImagePairsDiagnostics(
    const std::filesystem::path& path,
    const std::vector<mvs::StereoFusionPairDiagnostics>& diagnostics) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion image pair diagnostics: " +
                             path.string());
  }

  file << "reference_image_id\tsource_image_id\tnum_candidates\tnum_accepted"
       << "\tnum_rejected_normal\tnum_rejected_view"
       << "\tnum_rejected_projection\tnum_rejected_depth"
       << "\tnum_rejected_mutual_consistency\tnum_rejected_invalid"
       << "\tnum_rejected_visited\tnum_rejected_duplicate"
       << "\tnum_rejected_incomplete\tnum_rejected_other\n";
  for (const mvs::StereoFusionPairDiagnostics& pair : diagnostics) {
    file << pair.reference_image_id << "\t"
         << pair.source_image_id << "\t"
         << pair.num_candidates << "\t"
         << pair.num_accepted << "\t"
         << pair.num_rejected_normal << "\t"
         << pair.num_rejected_view << "\t"
         << pair.num_rejected_projection << "\t"
         << pair.num_rejected_depth << "\t"
         << pair.num_rejected_mutual_consistency << "\t"
         << pair.num_rejected_invalid << "\t"
         << pair.num_rejected_visited << "\t"
         << pair.num_rejected_duplicate << "\t"
         << pair.num_rejected_incomplete << "\t"
         << pair.num_rejected_other << "\n";
  }
}

void WriteFusionTrackLengthHistogram(
    const std::filesystem::path& path,
    const std::vector<mvs::StereoFusionTrack>& tracks) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion track length histogram: " +
                             path.string());
  }

  std::map<std::size_t, std::size_t> histogram;
  for (const mvs::StereoFusionTrack& track : tracks) {
    histogram[track.num_observations]++;
  }

  file << "num_observations\tnum_tracks\n";
  for (const auto& [num_observations, num_tracks] : histogram) {
    file << num_observations << "\t" << num_tracks << "\n";
  }
}

void WriteFusionMetricDistribution(
    std::ofstream* file,
    std::string_view metric,
    const std::vector<float>& samples,
    const std::vector<std::pair<double, double>>& bins) {
  const std::array<std::pair<std::string_view, double>, 5> percentiles = {
      std::make_pair(std::string_view("p50"), 0.50),
      std::make_pair(std::string_view("p75"), 0.75),
      std::make_pair(std::string_view("p90"), 0.90),
      std::make_pair(std::string_view("p95"), 0.95),
      std::make_pair(std::string_view("p99"), 0.99),
  };

  *file << metric << "\tsummary\tcount\t\t\t" << samples.size() << "\t"
        << samples.size() << "\n";
  for (const auto& [label, q] : percentiles) {
    *file << metric << "\tpercentile\t" << label << "\t\t\t"
          << samples.size() << "\t";
    if (!samples.empty()) {
      *file << Percentile(samples, q);
    }
    *file << "\n";
  }

  for (const auto& [lower, upper] : bins) {
    std::size_t count = 0;
    for (const float sample : samples) {
      if (!std::isfinite(sample)) {
        continue;
      }
      const double value = static_cast<double>(sample);
      if (value >= lower && value < upper) {
        count++;
      }
    }
    const std::string label =
        "[" + FormatDistributionValue(lower) + "," +
        FormatDistributionValue(upper) + ")";
    *file << metric << "\thistogram\t" << label << "\t"
          << FormatDistributionValue(lower) << "\t"
          << FormatDistributionValue(upper) << "\t" << count << "\t\n";
  }
}

void WriteFusionErrorDistribution(
    const std::filesystem::path& path,
    const mvs::StereoFusionErrorSamples& samples) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to write fusion error distribution: " +
                             path.string());
  }

  constexpr double kInf = std::numeric_limits<double>::infinity();
  const std::vector<std::pair<double, double>> relative_depth_bins = {
      {0.0, 0.0025}, {0.0025, 0.005}, {0.005, 0.01},
      {0.01, 0.02},  {0.02, 0.05},    {0.05, 0.1},
      {0.1, 0.2},    {0.2, kInf},
  };
  const std::vector<std::pair<double, double>> reprojection_bins = {
      {0.0, 0.25}, {0.25, 0.5}, {0.5, 1.0}, {1.0, 2.0},
      {2.0, 3.0},  {3.0, 5.0},  {5.0, 10.0}, {10.0, kInf},
  };
  const std::vector<std::pair<double, double>> angle_bins = {
      {0.0, 1.0},   {1.0, 2.0},   {2.0, 5.0},   {5.0, 10.0},
      {10.0, 20.0}, {20.0, 45.0}, {45.0, 90.0}, {90.0, 180.0},
      {180.0, kInf},
  };

  file << std::setprecision(17);
  file << "metric\trecord\tlabel\tlower\tupper\tcount\tvalue\n";
  WriteFusionMetricDistribution(&file, "relative_depth_error",
                                samples.relative_depth_errors,
                                relative_depth_bins);
  WriteFusionMetricDistribution(&file, "reprojection_error_pixels",
                                samples.reprojection_errors_pixels,
                                reprojection_bins);
  WriteFusionMetricDistribution(&file, "normal_angle_degrees",
                                samples.normal_angles_degrees, angle_bins);
  WriteFusionMetricDistribution(&file, "view_angle_degrees",
                                samples.view_angles_degrees, angle_bins);
}

std::size_t RunPatchMatchForWorkspace(
    const std::filesystem::path& output_path,
    const ReconMvsControllerOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary,
    const std::vector<ImageId>& image_ids_in_model_order,
    std::size_t* num_photometric_filtered_pixels,
    std::size_t* num_normal_refined_pixels,
    std::size_t* num_geometric_consistent_pixels,
    std::size_t* num_geometric_filtered_pixels,
    PatchMatchSchedulingSummary* scheduling_summary_out) {
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  workspace_options.resolution_level = cli_options.resolution_level;
  // PatchMatch consumes normalized grayscale only. Persisted Fusion owns a
  // separate RGB-only workspace, so retaining color for every source view in
  // this phase wastes three bytes per pixel across the full image set.
  workspace_options.load_color = false;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  mvs::PatchMatchStereoOptions patch_match_options;
  patch_match_options.backend =
      mvs::PatchMatchBackendFromName(cli_options.patch_match_backend);
  patch_match_options.cuda_gpu_index = cli_options.patch_match_gpu_index;
  patch_match_options.initialization.random_seed = cli_options.random_seed;
  patch_match_options.initialization.randomize_initial_normals =
      cli_options.patch_match_randomize_initial_normals;
  patch_match_options.initialization.initial_normal_hemisphere_degrees =
      cli_options.patch_match_initial_normal_hemisphere_degrees;
  patch_match_options.perturbation.random_seed = cli_options.random_seed + 7919;
  patch_match_options.perturbation.normal_radius_degrees =
      cli_options.patch_match_normal_perturbation_degrees;
  patch_match_options.num_iterations = cli_options.patch_match_iterations;
  patch_match_options.pyramid_levels = cli_options.patch_match_pyramid_levels;
  patch_match_options.cost.window_radius = cli_options.patch_match_window_radius;
  patch_match_options.cost.window_step = cli_options.patch_match_window_step;
  patch_match_options.cost.bilateral_weighting =
      cli_options.patch_match_bilateral_ncc;
  patch_match_options.cost.sigma_spatial =
      cli_options.patch_match_sigma_spatial;
  patch_match_options.cost.sigma_color = cli_options.patch_match_sigma_color;
  patch_match_options.source_selection.dynamic_probability =
      cli_options.patch_match_dynamic_source_selection;
  patch_match_options.source_selection.directional_hmm =
      cli_options.patch_match_directional_hmm ||
      cli_options.patch_match_dynamic_source_selection;
  patch_match_options.source_selection.geometric_priors =
      cli_options.patch_match_colmap_source_priors;
  patch_match_options.source_selection.num_samples =
      cli_options.patch_match_source_samples;
  patch_match_options.source_selection.ncc_sigma =
      cli_options.patch_match_ncc_sigma;
  patch_match_options.source_selection.incident_angle_sigma =
      cli_options.patch_match_incident_angle_sigma;
  patch_match_options.source_selection.anneal_history = true;
  patch_match_options.source_selection.min_probability =
      cli_options.patch_match_min_selection_probability;
  patch_match_options.source_selection.max_probability =
      cli_options.patch_match_max_selection_probability;
  patch_match_options.perturbation.num_samples_per_pixel =
      cli_options.patch_match_perturbation_samples;
  patch_match_options.cost.min_num_valid_sources =
      cli_options.patch_match_min_num_valid_sources;
  patch_match_options.cost.max_num_aggregated_costs =
      cli_options.patch_match_max_num_aggregated_costs;
  patch_match_options.normal_refinement.enabled =
      cli_options.refine_normals_from_depth;
  patch_match_options.normal_refinement.max_relative_depth_difference =
      cli_options.normal_refinement_max_relative_depth_difference;
  patch_match_options.propagation_schedule =
      ParsePatchMatchPropagationSchedule(
          cli_options.patch_match_propagation_schedule);
  patch_match_options.checkerboard_propagation_rounds =
      cli_options.patch_match_checkerboard_rounds;
  patch_match_options.collect_diagnostics =
      cli_options.patch_match_diagnostics;
  patch_match_options.filter.enabled = cli_options.patch_match_filter;
  patch_match_options.filter.min_ncc = cli_options.patch_match_filter_min_ncc;
  patch_match_options.filter.min_triangulation_angle_degrees =
      cli_options.patch_match_filter_min_triangulation_angle_degrees;
  patch_match_options.filter.min_num_consistent =
      cli_options.patch_match_filter_min_num_consistent;
  patch_match_options.filter.geom_consistency_max_cost =
      cli_options.patch_match_filter_geom_consistency_max_cost;
  mvs::ValidatePatchMatchFilterOptions(patch_match_options.filter);

  const bool run_geometric_refinement =
      cli_options.patch_match_geometric_refinement_iterations > 0;
  mvs::PatchMatchStereoOptions refinement_options = patch_match_options;
  refinement_options.num_iterations =
      cli_options.patch_match_geometric_refinement_iterations;
  // Keep the requested photometric pyramid unchanged, but use a small
  // coarse-to-fine hierarchy for geometric refinement. At a single scale,
  // red-black propagation only travels a few full-resolution pixels and
  // stalls on large weakly textured surfaces.
  refinement_options.pyramid_levels =
      std::max<std::size_t>(3, patch_match_options.pyramid_levels);
  refinement_options.cost.geometric_consistency_weight =
      cli_options.patch_match_geometric_consistency_weight;
  refinement_options.cost.robust_geometric_aggregation =
      cli_options.patch_match_robust_geometric_aggregation;
  refinement_options.cost.geometric_max_reprojection_error =
      cli_options.patch_match_geom_consistency_max_cost;
  refinement_options.cost.geometric_max_relative_depth_error =
      cli_options.patch_match_geometric_max_relative_depth_error;
  // COLMAP treats geometry as a soft forward-backward reprojection cost. It
  // does not reject every candidate with an occluded or conflicting source.
  refinement_options.cost.geometric_reject_occluded_candidates = false;
  refinement_options.cost.geometric_reject_depth_conflicts = false;

  const std::filesystem::path depth_dir =
      output_path / "stereo" / "depth_maps";
  const std::filesystem::path normal_dir =
      output_path / "stereo" / "normal_maps";
  const std::filesystem::path cost_dir = output_path / "stereo" / "cost_maps";
  const std::filesystem::path consistency_dir = output_path / "stereo/.colite_refinement";
  std::filesystem::remove_all(consistency_dir);  // Private scratch from an interrupted attempt.
  std::filesystem::create_directories(output_path / "stereo");
  for (const auto& directory : {depth_dir, normal_dir, cost_dir, consistency_dir}) {
    std::filesystem::create_directories(directory);
  }
  if (cli_options.patch_match_write_consistency_graph) {
    std::filesystem::create_directories(output_path / "stereo/consistency_graphs");
  }

  std::size_t num_patch_match_images = 0;
  std::size_t total_photometric_filtered_pixels = 0;
  std::size_t total_normal_refined_pixels = 0;
  const std::vector<const mvs::ImageMvsSummary*> image_summaries =
      CollectPatchMatchSummaries(summary);
  const std::vector<ImageId> required_image_ids =
      CollectRequiredMvsImageIds(image_summaries);
  const std::unordered_map<ImageId, mvs::MvsImage> images =
      PreloadMvsImages(required_image_ids, &workspace);
  const std::unordered_map<ImageId, mvs::PatchMatchView> views =
      BuildPatchMatchViews(reconstruction, images);
  CudaWorkspacePool cuda_workspaces =
      CreateCudaWorkspacePool(patch_match_options, image_summaries.size());
  std::vector<mvs::CudaPatchMatchWorkspace*> cuda_workspace_ptrs;
  cuda_workspace_ptrs.reserve(cuda_workspaces.size());
  for (const mvs::CudaPatchMatchWorkspacePtr& workspace_ptr :
       cuda_workspaces) {
    cuda_workspace_ptrs.push_back(workspace_ptr.get());
  }
  if (scheduling_summary_out != nullptr &&
      patch_match_options.backend != mvs::PatchMatchBackend::kCpu &&
      !image_summaries.empty()) {
    std::string unavailable_reason;
    scheduling_summary_out->cuda_warmup_time_ms =
        mvs::WarmupCudaPatchMatch(patch_match_options.cuda_gpu_index,
                                  &unavailable_reason);
    if (!unavailable_reason.empty() &&
        patch_match_options.backend == mvs::PatchMatchBackend::kCuda) {
      throw std::runtime_error(unavailable_reason);
    }
  }

  std::vector<PatchMatchWorkspaceResult> results =
      RunInitialPatchMatchForWorkspace(
          image_summaries, images, views, patch_match_options, cli_options,
          run_geometric_refinement, cuda_workspace_ptrs,
          scheduling_summary_out);
  WriteInitialPatchMatchDiagnostics(output_path, results);
  std::unordered_map<ImageId, std::size_t> result_index_by_image_id;
  for (std::size_t i = 0; i < results.size(); ++i) {
    total_photometric_filtered_pixels +=
        results[i].photometric_filtered_pixels;
    total_normal_refined_pixels +=
        results[i].result.summary.num_normal_refined_pixels;
    result_index_by_image_id[results[i].image_id] = i;
    if (scheduling_summary_out != nullptr &&
        results[i].photometric_stats_deferred) {
      ++scheduling_summary_out->initial_deferred_photometric_stats_images;
    }
  }
  num_patch_match_images = results.size();
  std::unordered_map<ImageId, mvs::DenseDepthMap> external_depths;
  std::vector<ImageId> external_source_ids;
  std::unordered_map<ImageId, std::size_t> external_source_indices;
  if (run_geometric_refinement) {
    for (const auto& result : results) {
      if (!result.reused_photometric) {
        auto normal = result.result.normal_map;
        mvs::OrientNormalMapTowardsCamera(result.view, &normal);
        WriteDenseMaps(output_path, result.image.name, false, result.result.depth_map,
                       normal, result.result.cost_map, cli_options);
      }
      const auto graph = output_path / "stereo/consistency_graphs" / DenseMapFileName(result.image.name, false);
      if (cli_options.patch_match_write_consistency_graph &&
          (!result.reused_photometric || !std::filesystem::exists(graph))) {
        mvs::WriteConsistencyGraphColmapBinary(graph,
            mvs::ConsistencyGraph(result.result.depth_map.Width(), result.result.depth_map.Height()),
            image_ids_in_model_order);
      } else if (!cli_options.patch_match_write_consistency_graph && !result.reused_photometric) {
        std::filesystem::remove(graph);
      }
    }
    std::vector<PatchMatchRefinementJob> refinement_jobs;
    refinement_jobs.reserve(results.size());
    std::vector<std::uint8_t> refinement_job_by_result(results.size(), 0);
    for (std::size_t result_idx = 0; result_idx < results.size();
         ++result_idx) {
      const PatchMatchWorkspaceResult& result = results[result_idx];
      PatchMatchRefinementJob job;
      job.result_index = result_idx;
      job.reference_view = result.view;
      job.reference_image = result.image;
      job.min_depth = result.min_depth;
      job.max_depth = result.max_depth;
      for (const ImageId source_id : result.source_image_ids) {
        const auto it = result_index_by_image_id.find(source_id);
        if (it == result_index_by_image_id.end()) {
          if (!external_source_indices.count(source_id)) {
            const auto& source_image = images.at(source_id);
            const auto file = DenseMapFileName(source_image.name, false);
            const auto depth_path = output_path / "stereo/depth_maps" / file;
            const auto normal_path = output_path / "stereo/normal_maps" / file;
            if (!std::filesystem::exists(depth_path) || !std::filesystem::exists(normal_path)) {
              if (!cli_options.patch_match_allow_missing_files) {
                throw std::runtime_error("Missing photometric depth/normal dependency for " + source_image.name);
              }
              LOG(WARNING) << "Skipping missing geometric source: " << source_image.name;
              continue;
            }
            auto depth = mvs::ReadDenseDepthMapBinary(depth_path);
            auto normal = mvs::ReadNormalMapBinary(normal_path);
            mvs::DownsizeDepthNormalMaps(source_image.Width(), source_image.Height(), &depth, &normal);
            if (depth.Width() != source_image.Width() || depth.Height() != source_image.Height()) {
              throw std::runtime_error("Photometric source map/image resolution mismatch for " + source_image.name);
            }
            external_source_indices.emplace(source_id, results.size() + external_source_ids.size());
            external_source_ids.push_back(source_id);
            external_depths.emplace(source_id, std::move(depth));
          }
          job.source_views.push_back(views.at(source_id));
          job.source_images.push_back(images.at(source_id));
          job.source_result_indices.push_back(external_source_indices.at(source_id));
          continue;
        }
        const PatchMatchWorkspaceResult& source_result = results[it->second];
        job.source_views.push_back(source_result.view);
        job.source_images.push_back(source_result.image);
        job.source_result_indices.push_back(it->second);
      }
      if (job.source_views.empty()) {
        LOG(WARNING) << "No geometric sources for reference: " << result.image.name;
        continue;
      }
      // Only an actual refinement job recomputes these costs. A view whose
      // selected sources have no dense result must keep its photometric map.
      results[result_idx].result.cost_map = mvs::DenseCostMap();
      // Each reference normal map is used by exactly one refinement job.
      // Transfer ownership into the queued job so completed jobs release the
      // old map while their replacement result is accumulated. Keeping a
      // separate all-view normal snapshot doubles the 12-byte-per-pixel
      // normal working set at large resolutions.
      job.initial_normal_map =
          std::move(results[result_idx].result.normal_map);
      refinement_job_by_result[result_idx] = 1;
      refinement_jobs.push_back(std::move(job));
    }
    if (!results.empty() && refinement_jobs.empty()) {
      throw std::runtime_error("No geometric reference has usable photometric sources");
    }
    ReleaseUnusedMemory();

    // A single inherited geometric pass already runs the configured inner
    // iterations over a three-level coarse-to-fine pyramid. A second outer
    // synchronization pass has little accuracy benefit but nearly doubles the
    // large-image refinement time and transfers, so keep the photometric
    // result as the immutable source snapshot for one inherited pass.
    constexpr std::size_t kGeometricRefinementPasses = 1;
    for (std::size_t pass = 0; pass < kGeometricRefinementPasses; ++pass) {
      std::vector<mvs::DenseDepthMap> source_depth_snapshot;
      source_depth_snapshot.reserve(results.size() + external_source_ids.size());
      for (std::size_t result_idx = 0; result_idx < results.size();
           ++result_idx) {
        PatchMatchWorkspaceResult& result = results[result_idx];
        // Refinement consumes the previous pass only as source depth. Move
        // those maps into the immutable pass snapshot instead of keeping a
        // second full-resolution copy beside the newly produced maps.
        if (refinement_job_by_result[result_idx] != 0) {
          source_depth_snapshot.push_back(
              std::move(result.result.depth_map));
        } else {
          source_depth_snapshot.push_back(result.result.depth_map);
        }
      }
      for (const ImageId id : external_source_ids) source_depth_snapshot.push_back(std::move(external_depths.at(id)));
      std::vector<PatchMatchRefinementResult> refinement_results =
          RunPatchMatchRefinementJobs(
              &refinement_jobs, source_depth_snapshot,
              refinement_options, cli_options,
              cuda_workspace_ptrs, scheduling_summary_out);
      for (PatchMatchRefinementResult& refinement_result :
           refinement_results) {
        auto& result = results[refinement_result.result_index];
        result.geometric_refined = true;
        result.result = std::move(refinement_result.result);
        result.final_filter_applied = refinement_result.final_filter_applied;
        result.side_maps_persisted = refinement_result.side_maps_persisted;
        if (result.side_maps_persisted) {
          result.stats_before_filter = refinement_result.stats_before_filter;
          result.stats_after_photometric_filter = refinement_result.stats_after_filter;
          result.photometric_filtered_pixels = refinement_result.photometric_filtered_pixels;
        }
      }
      for (const ImageId id : external_source_ids) {
        external_depths.at(id) = std::move(source_depth_snapshot.at(external_source_indices.at(id)));
      }
    }

    total_photometric_filtered_pixels = 0;
    total_normal_refined_pixels = 0;
    for (PatchMatchWorkspaceResult& result : results) {
      if (!result.geometric_refined) continue;
      total_normal_refined_pixels +=
          result.result.summary.num_normal_refined_pixels;
      if (result.side_maps_persisted) {
        total_photometric_filtered_pixels += result.photometric_filtered_pixels;
        continue;
      }
      result.stats_before_filter =
          mvs::ComputePatchMatchPhotometricStats(result.result.cost_map);
      mvs::PatchMatchPhotometricFilterSummary filter_summary;
      if (cli_options.filter_photometric_max_cost >= 0.0F) {
        filter_summary = mvs::FilterPatchMatchPhotometricResults(
            cli_options.filter_photometric_max_cost, &result.result.depth_map,
            &result.result.normal_map, &result.result.cost_map);
      } else {
        filter_summary.num_pixels = result.result.cost_map.NumPixels();
        filter_summary.num_kept_pixels =
            result.result.cost_map.ValidPixelCount();
      }
      result.photometric_filtered_pixels =
          filter_summary.num_filtered_pixels;
      total_photometric_filtered_pixels +=
          result.photometric_filtered_pixels;
      result.stats_after_photometric_filter =
          mvs::ComputePatchMatchPhotometricStats(result.result.cost_map);
    }
  }

  // No CUDA workspace is used after photometric/geometric refinement. Release
  // texture arrays, device buffers, and pinned staging memory before CPU
  // consistency-graph construction so graph workers do not overlap the
  // largest GPU-host working set.
  cuda_workspace_ptrs.clear();
  cuda_workspaces.clear();
  ReleaseUnusedMemory();

  // Plane homographies are invariant to flipping the plane normal, so
  // PatchMatch cost minimization alone cannot enforce a stable sign. Persist
  // the standard camera-facing orientation before geometric filtering,
  // Fusion, and dense-map output.
  for (PatchMatchWorkspaceResult& result : results) {
    if (result.side_maps_persisted) continue;
    mvs::OrientNormalMapTowardsCamera(
        result.view, &result.result.normal_map);
  }

  std::size_t total_geometric_consistent_pixels = 0;
  std::size_t total_geometric_filtered_pixels = 0;
  if (cli_options.check_geometric_consistency) {
    mvs::PatchMatchGeometricConsistencyOptions geometric_options;
    geometric_options.max_reprojection_error_pixels =
        cli_options.geometric_max_reprojection_error_pixels;
    geometric_options.max_relative_depth_error =
        cli_options.geometric_max_relative_depth_error;

    // Compute and persist one compact graph at a time while every depth map is
    // still unfiltered. A second streaming pass applies the graph masks. This
    // keeps the result order-independent without retaining a second copy of
    // all depth maps or all consistency graphs in memory.
    const ControllerClock::time_point graph_build_start =
        ControllerClock::now();
    // A compact graph still needs a full size_t offset array plus its source
    // ids while it is being built. At High resolution eight concurrent
    // graphs can add hundreds of MiB without improving the pixel-parallel
    // PatchMatch kernels. Bound graph materialization independently from the
    // user CPU thread count; each worker continues to use all source views.
    const std::size_t graph_num_threads = std::min<std::size_t>(
        std::max<std::size_t>(1, cli_options.num_threads),
        2);
    std::vector<std::future<void>> graph_futures;
    graph_futures.reserve(graph_num_threads);
    const auto collect_oldest_graph_future = [&graph_futures]() {
      graph_futures.front().get();
      graph_futures.erase(graph_futures.begin());
    };
    for (std::size_t result_idx = 0; result_idx < results.size();
         ++result_idx) {
      if (graph_futures.size() >= graph_num_threads) {
        collect_oldest_graph_future();
      }
      graph_futures.push_back(std::async(
          std::launch::async,
          [result_idx, &results, &result_index_by_image_id,
           &geometric_options, &consistency_dir, &external_depths, &views,
           run_geometric_refinement]() {
            const PatchMatchWorkspaceResult& result = results[result_idx];
            if (run_geometric_refinement && !result.geometric_refined) return;
            if (result.final_filter_applied) return;
            std::vector<mvs::PatchMatchView> source_views;
            std::vector<const mvs::DenseDepthMap*> source_depth_maps;
            source_views.reserve(result.source_image_ids.size());
            source_depth_maps.reserve(result.source_image_ids.size());
            for (const ImageId source_id : result.source_image_ids) {
              const auto it = result_index_by_image_id.find(source_id);
              if (it == result_index_by_image_id.end()) {
                if (external_depths.count(source_id)) {
                  source_views.push_back(views.at(source_id));
                  source_depth_maps.push_back(&external_depths.at(source_id));
                }
                continue;
              }
              const PatchMatchWorkspaceResult& source_result =
                  results[it->second];
              source_views.push_back(source_result.view);
              source_depth_maps.push_back(&source_result.result.depth_map);
            }
            const mvs::ConsistencyGraph graph =
                mvs::ComputeGeometricConsistencyGraph(
                    result.view, source_views, result.result.depth_map,
                    source_depth_maps, geometric_options);
            mvs::WriteConsistencyGraphBinary(
                consistency_dir / (result.image.name + ".graph.bin"), graph);
          }));
    }
    while (!graph_futures.empty()) {
      collect_oldest_graph_future();
    }
    if (scheduling_summary_out != nullptr) {
      scheduling_summary_out->consistency_graph_build_write_time_ms =
          std::chrono::duration<double, std::milli>(
              ControllerClock::now() - graph_build_start)
              .count();
    }

    const ControllerClock::time_point graph_filter_start =
        ControllerClock::now();
    std::vector<std::future<std::pair<std::size_t, std::size_t>>>
        filter_futures;
    filter_futures.reserve(graph_num_threads);
    const auto collect_oldest_filter_future =
        [&filter_futures, &total_geometric_consistent_pixels,
         &total_geometric_filtered_pixels]() {
          const auto [consistent_pixels, filtered_pixels] =
              filter_futures.front().get();
          filter_futures.erase(filter_futures.begin());
          total_geometric_consistent_pixels += consistent_pixels;
          total_geometric_filtered_pixels += filtered_pixels;
        };
    for (std::size_t result_idx = 0; result_idx < results.size();
         ++result_idx) {
      if (filter_futures.size() >= graph_num_threads) {
        collect_oldest_filter_future();
      }
      filter_futures.push_back(std::async(
          std::launch::async,
          [result_idx, &results, &consistency_dir, &cli_options, &output_path,
           run_geometric_refinement]() {
            PatchMatchWorkspaceResult& result = results[result_idx];
            if (run_geometric_refinement && !result.geometric_refined) return std::make_pair(std::size_t{0}, std::size_t{0});
            if (result.side_maps_persisted) {
              result.result.normal_map = mvs::ReadNormalMapBinary(
                  RefinementSideMapPath(output_path, result.image.name, "normal"));
              result.result.cost_map = mvs::ReadDenseCostMapBinary(
                  RefinementSideMapPath(output_path, result.image.name, "cost"));
            }
            mvs::ConsistencyGraph graph = mvs::ReadConsistencyGraphBinary(
                consistency_dir / (result.image.name + ".graph.bin"));
            result.geometric_consistent_pixels =
                graph.NumConsistentPixels(1);
            if (cli_options
                    .filter_geometric_min_num_consistent_sources > 0) {
              const mvs::PatchMatchGeometricFilterSummary geometric_filter =
                  mvs::FilterPatchMatchGeometricResults(
                      graph,
                      cli_options
                          .filter_geometric_min_num_consistent_sources,
                      &result.result.depth_map, &result.result.normal_map,
                      &result.result.cost_map);
              result.geometric_filtered_pixels =
                  geometric_filter.num_filtered_pixels;
            }
            // CSR is private scratch; public graphs use COLMAP pixel records.
            if (result.side_maps_persisted) {
              mvs::WriteNormalMapBinary(RefinementSideMapPath(output_path,
                  result.image.name, "normal"), result.result.normal_map);
              mvs::WriteDenseCostMapBinary(RefinementSideMapPath(output_path,
                  result.image.name, "cost"), result.result.cost_map);
              result.result.normal_map = mvs::NormalMap();
              result.result.cost_map = mvs::DenseCostMap();
            }
            return std::make_pair(result.geometric_consistent_pixels,
                                  result.geometric_filtered_pixels);
          }));
    }
    while (!filter_futures.empty()) {
      collect_oldest_filter_future();
    }
    if (scheduling_summary_out != nullptr) {
      scheduling_summary_out->consistency_graph_read_filter_time_ms =
          std::chrono::duration<double, std::milli>(
              ControllerClock::now() - graph_filter_start)
              .count();
    }
  }

  const ControllerClock::time_point dense_output_start =
      ControllerClock::now();
  std::ofstream diagnostics(output_path / "stereo" / "patch_match.tsv");
  diagnostics << "image_id\trequested_backend\tactual_backend"
              << "\tpyramid_levels\titerations"
              << "\tbilateral_ncc\tsigma_spatial\tsigma_color"
              << "\tdynamic_source_selection\tncc_sigma"
              << "\tselection_history_schedule"
              << "\tmin_selection_probability"
              << "\tmax_selection_probability"
              << "\tvalid_cost_pixels\tcandidate_evaluations"
              << "\tplane_propagation_candidates"
              << "\tperturbation_candidate_evaluations"
              << "\tpropagation_updates\tperturbation_updates"
              << "\tsource_selection_tests"
              << "\tsource_selection_acceptances"
              << "\tsource_selection_rejections"
              << "\tsource_selection_probability_updates"
              << "\tperturbation_mean_abs_depth_update"
              << "\tperturbation_max_abs_depth_update"
              << "\tperturbation_mean_normal_angle_degrees"
              << "\tperturbation_max_normal_angle_degrees"
              << "\tperturbation_geometric_candidate_rejections"
              << "\tperturbation_geometric_candidate_acceptances"
              << "\tforward_sweeps\tbackward_sweeps"
              << "\ttop_down_sweeps\tbottom_up_sweeps"
              << "\tcheckerboard_passes"
              << "\tcuda_upload_time_ms"
              << "\tcuda_initialization_time_ms"
              << "\tcuda_evaluate_cost_time_ms"
              << "\tcuda_propagation_time_ms"
              << "\tcuda_perturbation_time_ms"
              << "\tcuda_download_time_ms"
              << "\tcuda_total_time_ms"
              << "\tcuda_wrapper_total_time_ms"
              << "\tcuda_request_validation_time_ms"
              << "\tcuda_pyramid_resize_time_ms"
              << "\tcuda_host_pack_time_ms"
              << "\tcuda_single_scale_dispatch_time_ms"
              << "\tcuda_result_materialization_time_ms"
              << "\tcuda_output_to_vector_time_ms"
              << "\tcuda_dense_map_fill_time_ms"
              << "\tcuda_normal_refinement_time_ms"
              << "\tcuda_valid_cost_count_time_ms"
              << "\tcuda_host_initial_normal_upload_normalizations"
              << "\tcuda_host_output_normalizations"
              << "\tcuda_device_count_queries"
              << "\tcuda_device_count_cache_hits"
              << "\tcuda_shared_reference_tile_pixels"
              << "\tcuda_workspace_texture_allocations"
              << "\tcuda_workspace_texture_reuses"
              << "\tcuda_workspace_buffer_allocations"
              << "\tcuda_workspace_buffer_reuses"
              << "\tcuda_iteration_counter_snapshots"
              << "\tcuda_iteration_counter_readbacks"
              << "\tcuda_packed_output_pixels"
              << "\tcuda_output_map_readbacks"
              << "\tcuda_async_host_uploads"
              << "\tcuda_async_texture_uploads"
              << "\tcuda_async_counter_readbacks"
              << "\tcuda_async_output_readbacks"
              << "\tcuda_pinned_host_upload_bytes"
              << "\tcuda_pinned_texture_upload_bytes"
              << "\tcuda_pinned_host_staging_bytes"
              << "\tcuda_stream_kernel_launches"
              << "\tcuda_workspace_pool_size"
              << "\tcuda_workspace_pool_slot"
              << "\titeration_levels"
              << "\titeration_indices"
              << "\titeration_candidate_evaluations"
              << "\titeration_propagation_updates"
              << "\titeration_perturbation_updates"
              << "\titeration_source_selection_tests"
              << "\titeration_source_selection_acceptances"
              << "\titeration_source_selection_rejections"
              << "\titeration_source_selection_probability_means"
              << "\tfinal_source_selection_probability_means"
              << "\tnormal_refinement_pixels"
              << "\tnormal_refined_pixels"
              << "\tnormal_refinement_skipped_pixels"
              << "\tvalid_cost_pixels_before_filter"
              << "\tvalid_cost_ratio_before_filter"
              << "\tmean_cost_before_filter"
              << "\tmedian_cost_before_filter"
              << "\tmax_cost_before_filter"
              << "\tphotometric_filtered_pixels"
              << "\tvalid_cost_pixels_after_filter"
              << "\tvalid_cost_ratio_after_filter"
              << "\tmean_cost_after_filter"
              << "\tmedian_cost_after_filter"
              << "\tmax_cost_after_filter"
              << "\tgeometric_consistent_pixels"
              << "\tgeometric_filtered_pixels"
              << "\tfinal_valid_cost_pixels"
              << "\tdepth_min\tdepth_max\tfinal_filter_applied\tselection_history_weights\n";
  for (std::size_t result_idx = 0; result_idx < results.size();
       ++result_idx) {
    const PatchMatchWorkspaceResult& result = results[result_idx];
    if (run_geometric_refinement && !result.geometric_refined) continue;
    mvs::NormalMap persisted_normal;
    mvs::DenseCostMap persisted_cost;
    if (result.side_maps_persisted) {
      persisted_normal = mvs::ReadNormalMapBinary(
          RefinementSideMapPath(output_path, result.image.name, "normal"));
      persisted_cost = mvs::ReadDenseCostMapBinary(
          RefinementSideMapPath(output_path, result.image.name, "cost"));
    }
    const auto& output_depth_map = result.result.depth_map;
    const auto& output_normal_map = result.side_maps_persisted ? persisted_normal : result.result.normal_map;
    const auto& output_cost_map = result.side_maps_persisted ? persisted_cost : result.result.cost_map;
    const mvs::PatchMatchPhotometricStats final_stats =
        mvs::ComputePatchMatchPhotometricStats(output_cost_map);
    WriteDenseMaps(output_path, result.image.name, run_geometric_refinement,
                   output_depth_map, output_normal_map, output_cost_map, cli_options);
    const auto public_graph = output_path / "stereo/consistency_graphs" /
        DenseMapFileName(result.image.name, run_geometric_refinement);
    if (cli_options.patch_match_write_consistency_graph) {
      const auto private_graph = consistency_dir / (result.image.name + ".graph.bin");
      const auto graph = std::filesystem::exists(private_graph)
          ? mvs::ReadConsistencyGraphBinary(private_graph)
          : mvs::ConsistencyGraph(output_depth_map.Width(), output_depth_map.Height());
      mvs::WriteConsistencyGraphColmapBinary(public_graph, graph, image_ids_in_model_order);
    } else std::filesystem::remove(public_graph);
    if (result.side_maps_persisted) {
      std::filesystem::remove(RefinementSideMapPath(output_path, result.image.name, "normal"));
      std::filesystem::remove(RefinementSideMapPath(output_path, result.image.name, "cost"));
    }
    diagnostics << result.image_id << "\t"
                << mvs::PatchMatchBackendToName(
                       result.result.summary.requested_backend)
                << "\t"
                << mvs::PatchMatchBackendToName(
                       result.result.summary.actual_backend)
                << "\t"
                << result.result.summary.num_pyramid_levels << "\t"
                << result.result.summary.num_iterations << "\t"
                << (cli_options.patch_match_bilateral_ncc ? 1 : 0) << "\t"
                << FormatStat(
                       cli_options.patch_match_sigma_spatial > 0.0F
                           ? cli_options.patch_match_sigma_spatial
                           : static_cast<float>(std::max(
                                 1, cli_options.patch_match_window_radius)))
                << "\t"
                << FormatStat(cli_options.patch_match_sigma_color) << "\t"
                << (cli_options.patch_match_dynamic_source_selection ? 1 : 0)
                << "\t"
                << FormatStat(cli_options.patch_match_ncc_sigma) << "\t"
                << (cli_options.patch_match_dynamic_source_selection ? "linear" : "disabled")
                << "\t"
                << FormatStat(
                       cli_options.patch_match_min_selection_probability)
                << "\t"
                << FormatStat(
                       cli_options.patch_match_max_selection_probability)
                << "\t"
                << result.result.summary.num_valid_cost_pixels << "\t"
                << result.result.summary.num_candidate_evaluations << "\t"
                << result.result.summary.num_plane_propagation_candidates
                << "\t"
                << result.result.summary
                       .num_perturbation_candidate_evaluations
                << "\t"
                << result.result.summary.num_propagation_updates << "\t"
                << result.result.summary.num_perturbation_updates << "\t"
                << result.result.summary.num_source_selection_tests << "\t"
                << result.result.summary.num_source_selection_acceptances
                << "\t"
                << result.result.summary.num_source_selection_rejections
                << "\t"
                << result.result.summary
                       .num_source_selection_probability_updates
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_mean_abs_depth_update)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_max_abs_depth_update)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_mean_normal_angle_degrees)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .perturbation_max_normal_angle_degrees)
                << "\t"
                << result.result.summary
                       .num_perturbation_geometric_candidate_rejections
                << "\t"
                << result.result.summary
                       .num_perturbation_geometric_candidate_acceptances
                << "\t"
                << result.result.summary.num_forward_sweeps << "\t"
                << result.result.summary.num_backward_sweeps << "\t"
                << result.result.summary.num_top_down_sweeps << "\t"
                << result.result.summary.num_bottom_up_sweeps << "\t"
                << result.result.summary.num_checkerboard_passes << "\t"
                << FormatStat(result.result.summary.cuda_upload_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_initialization_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_evaluate_cost_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_propagation_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_perturbation_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_download_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_total_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_wrapper_total_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_request_validation_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_pyramid_resize_time_ms)
                << "\t"
                << FormatStat(result.result.summary.cuda_host_pack_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .cuda_single_scale_dispatch_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary
                           .cuda_result_materialization_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_output_to_vector_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_dense_map_fill_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_normal_refinement_time_ms)
                << "\t"
                << FormatStat(
                       result.result.summary.cuda_valid_cost_count_time_ms)
                << "\t"
                << result.result.summary
                       .cuda_host_initial_normal_upload_normalizations
                << "\t"
                << result.result.summary.cuda_host_output_normalizations
                << "\t"
                << result.result.summary.cuda_device_count_queries << "\t"
                << result.result.summary.cuda_device_count_cache_hits
                << "\t"
                << result.result.summary.cuda_shared_reference_tile_pixels
                << "\t"
                << result.result.summary.cuda_workspace_texture_allocations
                << "\t"
                << result.result.summary.cuda_workspace_texture_reuses << "\t"
                << result.result.summary.cuda_workspace_buffer_allocations
                << "\t"
                << result.result.summary.cuda_workspace_buffer_reuses
                << "\t"
                << result.result.summary.cuda_iteration_counter_snapshots
                << "\t"
                << result.result.summary.cuda_iteration_counter_readbacks
                << "\t"
                << result.result.summary.cuda_packed_output_pixels
                << "\t"
                << result.result.summary.cuda_output_map_readbacks
                << "\t"
                << result.result.summary.cuda_async_host_uploads
                << "\t"
                << result.result.summary.cuda_async_texture_uploads
                << "\t"
                << result.result.summary.cuda_async_counter_readbacks
                << "\t"
                << result.result.summary.cuda_async_output_readbacks
                << "\t"
                << result.result.summary.cuda_pinned_host_upload_bytes
                << "\t"
                << result.result.summary.cuda_pinned_texture_upload_bytes
                << "\t"
                << result.result.summary.cuda_pinned_host_staging_bytes
                << "\t"
                << result.result.summary.cuda_stream_kernel_launches
                << "\t"
                << result.result.summary.cuda_workspace_pool_size
                << "\t"
                << result.result.summary.cuda_workspace_pool_slot
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_pyramid_levels)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_indices)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_candidate_evaluations)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_propagation_updates)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_perturbation_updates)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary.iteration_source_selection_tests)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_source_selection_acceptances)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_source_selection_rejections)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .iteration_source_selection_probability_means)
                << "\t"
                << JoinDiagnosticVector(
                       result.result.summary
                           .final_source_selection_probability_means)
                << "\t"
                << result.result.summary.num_normal_refinement_pixels << "\t"
                << result.result.summary.num_normal_refined_pixels << "\t"
                << result.result.summary.num_normal_refinement_skipped_pixels
                << "\t"
                << result.stats_before_filter.num_valid_cost_pixels << "\t"
                << FormatStat(result.stats_before_filter.valid_ratio) << "\t"
                << FormatStat(result.stats_before_filter.mean_cost) << "\t"
                << FormatStat(result.stats_before_filter.median_cost) << "\t"
                << FormatStat(result.stats_before_filter.max_cost) << "\t"
                << result.photometric_filtered_pixels << "\t"
                << result.stats_after_photometric_filter.num_valid_cost_pixels
                << "\t"
                << FormatStat(
                       result.stats_after_photometric_filter.valid_ratio)
                << "\t"
                << FormatStat(result.stats_after_photometric_filter.mean_cost)
                << "\t"
                << FormatStat(
                       result.stats_after_photometric_filter.median_cost)
                << "\t"
                << FormatStat(result.stats_after_photometric_filter.max_cost)
                << "\t" << result.geometric_consistent_pixels << "\t"
                << result.geometric_filtered_pixels << "\t"
                << final_stats.num_valid_cost_pixels << "\t"
                << FormatStat(result.min_depth) << "\t"
                << FormatStat(result.max_depth) << "\t"
                << (result.final_filter_applied ? 1 : 0) << "\t"
                << JoinDiagnosticVector(result.result.summary.selection_history_weights) << "\n";
  }
  std::filesystem::remove_all(output_path / "stereo" / ".colite_refinement");
  if (scheduling_summary_out != nullptr) {
    scheduling_summary_out->dense_map_output_time_ms =
        std::chrono::duration<double, std::milli>(
            ControllerClock::now() - dense_output_start)
            .count();
  }

  if (num_photometric_filtered_pixels != nullptr) {
    *num_photometric_filtered_pixels = total_photometric_filtered_pixels;
  }
  if (num_normal_refined_pixels != nullptr) {
    *num_normal_refined_pixels = total_normal_refined_pixels;
  }
  if (num_geometric_consistent_pixels != nullptr) {
    *num_geometric_consistent_pixels = total_geometric_consistent_pixels;
  }
  if (num_geometric_filtered_pixels != nullptr) {
    *num_geometric_filtered_pixels = total_geometric_filtered_pixels;
  }
  return num_patch_match_images;
}

mvs::StereoFusionSummary FusePersistedStereoMaps(
    const std::filesystem::path& output_path,
    const ReconMvsControllerOptions& cli_options,
    const Reconstruction& reconstruction,
    const mvs::MvsPipelineSummary& summary,
    const std::vector<ImageId>& image_ids_in_model_order,
    const std::unordered_map<ImageId, std::vector<ImageId>>* supplied_overlaps = nullptr) {
  const auto total_start = ControllerClock::now();
  const auto elapsed_ms = [](const ControllerClock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               ControllerClock::now() - start)
        .count();
  };
  mvs::MvsImageWorkspaceOptions workspace_options;
  workspace_options.image_path = cli_options.image_path;
  workspace_options.max_cache_entries = cli_options.image_cache_size;
  workspace_options.max_image_size = cli_options.max_image_size;
  workspace_options.resolution_level = cli_options.resolution_level;
  workspace_options.load_gray = false;
  workspace_options.load_color = true;
  mvs::MvsImageWorkspace workspace(reconstruction, workspace_options);

  const auto computed_overlaps = supplied_overlaps == nullptr
      ? mvs::ComputeOverlappingImages(reconstruction,
            cli_options.fusion_max_num_source_images, cli_options.min_num_shared_points)
      : std::unordered_map<ImageId, std::vector<ImageId>>{};
  const auto& fusion_source_images = supplied_overlaps == nullptr
      ? computed_overlaps : *supplied_overlaps;
  std::vector<const mvs::ImageMvsSummary*> image_summaries;
  for (const auto& image : summary.images) image_summaries.push_back(&image);
  const auto fusion_config = output_path / "stereo/fusion.cfg";
  std::unordered_set<ImageId> configured_images;
  const bool use_config = cli_options.require_workspace_config || std::filesystem::exists(fusion_config);
  if (use_config) {
    const auto names = RegisteredImageNames(reconstruction);
    for (const auto& name : ReadWorkspaceConfig(fusion_config)) {
      const ImageId id = ConfigImageId(names, name);
      if (!configured_images.insert(id).second) throw std::invalid_argument("Duplicate Fusion image: " + name);
    }
    if (configured_images.empty()) throw std::invalid_argument("Fusion configuration must contain images");
  }
  std::vector<ImageId> image_ids;
  std::vector<std::uint32_t> visibility_image_indices;
  std::unordered_map<ImageId, std::uint32_t> model_indices;
  for (const ImageId id : image_ids_in_model_order) {
    model_indices.emplace(id, static_cast<std::uint32_t>(model_indices.size()));
  }
  std::sort(image_summaries.begin(), image_summaries.end(),
      [&](const auto* lhs, const auto* rhs) {
        return model_indices.at(lhs->image_id) < model_indices.at(rhs->image_id);
      });
  std::vector<mvs::PatchMatchView> fusion_views;
  std::vector<FusionMapInput> map_inputs;
  std::vector<std::pair<std::size_t, std::size_t>> map_shapes;
  image_ids.reserve(image_summaries.size());
  fusion_views.reserve(image_summaries.size());
  for (const mvs::ImageMvsSummary* image_summary : image_summaries) {
    if (use_config && !configured_images.count(image_summary->image_id)) continue;
    const mvs::MvsImage image = workspace.GetImage(image_summary->image_id);
    const auto input = ResolveFusionMapInput(output_path, image.name, cli_options);
    const bool has_depth = std::filesystem::exists(input.depth);
    const bool has_normal = std::filesystem::exists(input.normal);
    if (!has_depth && !has_normal) {
      LOG(WARNING) << "Ignoring fusion image without " << cli_options.fusion_input_type
                   << " maps: " << image.name;
      continue;
    }
    if (!has_depth || !has_normal) {
      throw std::runtime_error("Incomplete depth/normal map pair for fusion image: " + image.name);
    }
    image_ids.push_back(image_summary->image_id);
    visibility_image_indices.push_back(model_indices.at(image_summary->image_id));
    const auto [map_width, map_height] = mvs::ReadDenseDepthMapSize(input.depth, input.binary);
    // COLMAP caps oversized maps at the working model dimensions, while
    // retaining smaller maps and their independently scaled intrinsics.
    const auto [width, height] = mvs::DenseMapDownsizeShape(
        map_width, map_height, image.Width(), image.Height());
    map_inputs.push_back(input);
    map_shapes.emplace_back(width, height);
    fusion_views.push_back(MakePatchMatchView(
        reconstruction, image_summary->image_id, width, height));
  }
  if (fusion_views.empty()) {
    throw std::runtime_error("No complete " + cli_options.fusion_input_type + " depth/normal maps to fuse");
  }
  // COLMAP Fusion considers every sparse-overlap candidate and only uses the
  // triangulation-angle gate for PatchMatch source selection.
  const double image_loading_ms = elapsed_ms(total_start);
  // The stereo-view LRU owns RGB images together with their dense maps. Do not
  // retain a second RGB cache, or mutate its LRU from parallel view loaders.
  workspace.ClearCache();

  struct CacheEntry {
    std::shared_ptr<const mvs::StereoFusionViewData> data;
    std::list<std::size_t>::iterator lru_it;
    std::size_t bytes = 0;
  };
  std::mutex cache_mutex;
  std::list<std::size_t> cache_lru;
  std::unordered_map<std::size_t, CacheEntry> view_cache;
  std::atomic<std::size_t> num_view_loads{0};
  std::atomic<std::int64_t> rgb_load_ns{0}, map_load_ns{0}, normal_orient_ns{0};
  std::unordered_map<
      std::size_t,
      std::weak_ptr<const mvs::StereoFusionViewData>>
      loaded_views;
  const double cache_budget = cli_options.fusion_cache_size_gb * 1024 * 1024 * 1024;
  const std::size_t view_cache_capacity_bytes =
      !cli_options.fusion_use_cache ||
              cache_budget >= static_cast<double>(std::numeric_limits<std::size_t>::max())
          ? std::numeric_limits<std::size_t>::max()
          : static_cast<std::size_t>(std::max(1.0, cache_budget));
  std::size_t view_cache_bytes = 0;
  const auto view_bytes = [](const mvs::StereoFusionViewData& data) {
    return data.depth_map.NumPixels() * sizeof(float) +
           data.normal_map.NumPixels() * sizeof(Eigen::Vector3f) +
           data.image.color.total() * data.image.color.elemSize() +
           data.image.gray.total() * data.image.gray.elemSize();
  };
  std::vector<std::unique_ptr<std::mutex>> view_load_mutexes;
  view_load_mutexes.reserve(image_ids.size());
  for (std::size_t image_idx = 0; image_idx < image_ids.size(); ++image_idx) {
    view_load_mutexes.push_back(std::make_unique<std::mutex>());
  }
  const auto trim_view_cache = [&]() {
    while (view_cache_bytes > view_cache_capacity_bytes && view_cache.size() > 1) {
      const std::size_t evicted_idx = cache_lru.back();
      cache_lru.pop_back();
      view_cache_bytes -= view_cache.at(evicted_idx).bytes;
      view_cache.erase(evicted_idx);
    }
  };
  const auto find_cached_view =
      [&](std::size_t image_idx)
      -> std::shared_ptr<const mvs::StereoFusionViewData> {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto cached = view_cache.find(image_idx);
    if (cached != view_cache.end()) {
      cache_lru.splice(cache_lru.begin(), cache_lru,
                       cached->second.lru_it);
      return cached->second.data;
    }
    const auto loaded = loaded_views.find(image_idx);
    if (loaded == loaded_views.end()) {
      return nullptr;
    }
    std::shared_ptr<const mvs::StereoFusionViewData> active =
        loaded->second.lock();
    if (!active) {
      return nullptr;
    }
    cache_lru.push_front(image_idx);
    const std::size_t bytes = view_bytes(*active);
    view_cache.emplace(
        image_idx, CacheEntry{active, cache_lru.begin(), bytes});
    view_cache_bytes += bytes;
    trim_view_cache();
    return active;
  };
  const auto load_view =
      [&](std::size_t image_idx)
      -> std::shared_ptr<const mvs::StereoFusionViewData> {
    if (std::shared_ptr<const mvs::StereoFusionViewData> cached =
            find_cached_view(image_idx)) {
      return cached;
    }

    // Serialize only duplicate loads of the same view. Dense-map and graph
    // I/O must not hold the global LRU mutex, otherwise one cache miss stalls
    // every Fusion worker for the full read/parse duration.
    std::lock_guard<std::mutex> view_load_lock(
        *view_load_mutexes.at(image_idx));
    if (std::shared_ptr<const mvs::StereoFusionViewData> cached =
            find_cached_view(image_idx)) {
      return cached;
    }

    auto data = std::make_shared<mvs::StereoFusionViewData>();
    const auto rgb_start = ControllerClock::now();
    data->image = workspace.ReadImage(image_ids.at(image_idx));
    rgb_load_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        ControllerClock::now() - rgb_start).count(), std::memory_order_relaxed);
    data->fallback_source_image_ids = fusion_source_images.at(image_ids.at(image_idx));
    const auto& input = map_inputs.at(image_idx);
    const auto maps_start = ControllerClock::now();
    data->depth_map =
        input.binary ? mvs::ReadDenseDepthMapBinary(input.depth)
                     : mvs::ReadDenseDepthMapText(input.depth);
    data->normal_map =
        input.binary ? mvs::ReadNormalMapBinary(input.normal)
                     : mvs::ReadNormalMapText(input.normal);
    const auto [width, height] = map_shapes.at(image_idx);
    mvs::DownsizeDepthNormalMaps(width, height, &data->depth_map, &data->normal_map);
    SampleFusionColorsAtDepthResolution(width, height, &data->image);
    map_load_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        ControllerClock::now() - maps_start).count(), std::memory_order_relaxed);
    const auto orientation_start = ControllerClock::now();
    mvs::OrientNormalMapTowardsCamera(
        fusion_views.at(image_idx), &data->normal_map);
    normal_orient_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        ControllerClock::now() - orientation_start).count(), std::memory_order_relaxed);
    // Geometric filtering has already applied the per-pixel consistency
    // graph and synchronized the invalid mask into depth and normal maps.
    // Persisted Fusion deliberately traverses the independently selected
    // 50-view overlap list below, so loading the graph here cannot affect any
    // acceptance decision. On High-resolution House data the unused CSR
    // payload is one of the largest members of every cached view and also
    // adds avoidable parsing on each cache miss.
    data->consistency_graph = mvs::ConsistencyGraph();
    num_view_loads.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(cache_mutex);
      cache_lru.push_front(image_idx);
      const std::size_t bytes = view_bytes(*data);
      view_cache.emplace(
          image_idx, CacheEntry{data, cache_lru.begin(), bytes});
      view_cache_bytes += bytes;
      loaded_views[image_idx] = data;
      trim_view_cache();
    }
    return data;
  };
  const double dense_map_loading_ms = 0.0;
  const double consistency_graph_loading_ms = 0.0;

  mvs::StereoFusionOptions options;
  options.min_num_consistent_sources =
      cli_options.filter_geometric_min_num_consistent_sources > 0
          ? cli_options.filter_geometric_min_num_consistent_sources
          : 1;
  options.min_num_fused_observations = cli_options.fusion_min_num_observations;
  options.max_num_fused_observations = cli_options.fusion_max_num_observations;
  options.max_traversal_depth = cli_options.fusion_max_traversal_depth;
  options.max_reprojection_error_pixels =
      cli_options.fusion_max_reprojection_error_pixels;
  options.max_relative_depth_error = cli_options.fusion_max_relative_depth_error;
  options.max_normal_error_degrees = cli_options.fusion_max_normal_error_degrees;
  options.max_view_angle_degrees = cli_options.fusion_max_view_angle_degrees;
  options.source_pixel_search_radius =
      cli_options.fusion_source_pixel_search_radius;
  // COLMAP's default photometric PatchMatch does not write a consistency
  // graph. Persisted fusion falls back to the ordered MVS source list and
  // applies the same reprojection/depth/normal checks at traversal time.
  options.require_consistency_graph = false;
  // Geometric filtering has already invalidated unsupported pixels. Reusing
  // the per-pixel graph here applies the same support constraint twice and
  // prevents valid cross-view tracks that COLMAP's fuser accepts.
  options.prefer_fallback_source_images = true;
  options.skip_completed_reference_views = true;
  // Geometric filtering persists invalid pixels consistently in depth,
  // normal, and cost maps. Depth and normal are sufficient here, so avoid
  // reparsing the large text cost maps during standalone fusion.
  options.require_cost_map = false;
  options.collect_diagnostics = cli_options.fusion_diagnostics;
  options.compact_visibility = fusion_views.size() <= 64;
  options.preload_views = !cli_options.fusion_use_cache;
  const auto fusion_start = ControllerClock::now();
  const mvs::StereoFusionResult fusion =
      mvs::FuseStereoDepthMapsWithViewCache(
          fusion_views, load_view, options);
  const double fusion_core_ms = elapsed_ms(fusion_start);
  const auto output_start = ControllerClock::now();
  std::filesystem::create_directories(output_path / "dense");
  const auto fused_path = FusedCloudPath(output_path, cli_options);
  const auto visibility_path = std::filesystem::path(fused_path.string() + ".vis");
  WriteFusedPointCloudPly(fused_path, fusion.points);
  if (!fusion.points_visibility_masks.empty()) {
    mvs::WritePointsVisibilityColmapBinary(
        visibility_path,
        fusion.points_visibility_masks, visibility_image_indices);
  } else {
    mvs::WritePointsVisibilityColmapBinary(
        visibility_path,
        fusion.points_visibility, visibility_image_indices);
  }
  if (cli_options.fusion_diagnostics) {
    WriteFusionTracksDiagnostics(
        output_path / "dense" / "fusion_tracks.tsv", fusion.tracks);
    WriteFusionImagePairsDiagnostics(
        output_path / "dense" / "fusion_image_pairs.tsv",
        fusion.pair_diagnostics);
    WriteFusionErrorDistribution(
        output_path / "dense" / "fusion_error_distribution.tsv",
        fusion.error_samples);
  }
  WriteFusionTrackLengthHistogram(
      output_path / "dense" / "fusion_track_lengths.tsv", fusion.tracks);
  mvs::WriteFusionTrackQualityStats(
      output_path / "dense" / "fusion_summary.txt",
      mvs::ComputeFusionTrackQualityStats(fusion.tracks));
  if (cli_options.filter_dense_cloud) {
    mvs::DenseCloudFilterOptions filter_options;
    filter_options.min_track_observations =
        cli_options.dense_cloud_min_track_observations;
    filter_options.max_normal_angle_degrees =
        cli_options.fusion_max_normal_error_degrees;
    filter_options.voxel_size = cli_options.dense_cloud_voxel_size;
    filter_options.spatial_bounds_quantile =
        cli_options.dense_cloud_spatial_bounds_quantile;
    const mvs::DenseCloudFilterResult clean_cloud =
        mvs::FilterDenseCloudWithStats(fusion.points, fusion.tracks,
                                       filter_options);
    WriteFusedPointCloudPly(output_path / "dense" / "fused_clean.ply",
                            clean_cloud.points);
    WriteDenseCloudFilterStats(
        output_path / "dense" / "fused_clean_stats.txt", clean_cloud.stats);
  }
  const double output_writing_ms = elapsed_ms(output_start);
  std::ofstream timing_file(output_path / "dense" / "fusion_timing.txt");
  if (!timing_file) {
    throw std::runtime_error("Failed to write fusion timing summary.");
  }
  timing_file << std::setprecision(17)
              << "image_loading_ms " << image_loading_ms << "\n"
              << "dense_map_loading_ms " << dense_map_loading_ms << "\n"
              << "consistency_graph_loading_ms "
              << consistency_graph_loading_ms << "\n"
              << "fusion_core_ms " << fusion_core_ms << "\n"
              << "dense_view_loads " << num_view_loads.load() << "\n"
              << "view_rgb_load_sum_ms " << rgb_load_ns.load() * 1e-6 << "\n"
              << "view_map_load_sum_ms " << map_load_ns.load() * 1e-6 << "\n"
              << "normal_orient_sum_ms " << normal_orient_ns.load() * 1e-6 << "\n"
              << "output_writing_ms " << output_writing_ms << "\n"
              << "total_ms " << elapsed_ms(total_start) << "\n";
  return fusion.summary;
}

}  // namespace

mvs::PoissonMeshingOptions ReconMvsControllerOptions::PoissonOptions() const {
  mvs::PoissonMeshingOptions options;
  options.point_weight = poisson_point_weight;
  options.depth = poisson_depth;
  options.color = poisson_color;
  options.trim = poisson_trim;
  options.num_threads = poisson_num_threads;
  return options;
}

mvs::DelaunayMeshingOptions ReconMvsControllerOptions::DelaunayOptions() const {
  mvs::DelaunayMeshingOptions options;
  options.max_proj_dist = delaunay_max_proj_dist;
  options.max_depth_dist = delaunay_max_depth_dist;
  options.visibility_sigma = delaunay_visibility_sigma;
  options.distance_sigma_factor = delaunay_distance_sigma_factor;
  options.quality_regularization = delaunay_quality_regularization;
  options.max_side_length_factor = delaunay_max_side_length_factor;
  options.max_side_length_percentile = delaunay_max_side_length_percentile;
  options.num_threads = delaunay_num_threads;
  return options;
}

mvs::MeshSimplificationOptions ReconMvsControllerOptions::SimplificationOptions() const {
  mvs::MeshSimplificationOptions options;
  options.target_face_ratio = mesh_simplification_target_face_ratio;
  options.max_error = mesh_simplification_max_error;
  options.boundary_weight = mesh_simplification_boundary_weight;
  options.interpolate_colors = mesh_simplification_interpolate_colors;
  options.num_threads = mesh_simplification_num_threads;
  return options;
}

mvs::MeshTexturingOptions ReconMvsControllerOptions::TexturingOptions() const {
  mvs::MeshTexturingOptions options;
  options.atlas_size = texture_atlas_size;
  options.max_atlas_size = texture_max_atlas_size;
  options.atlas_patch_padding = texture_patch_padding;
  options.inpaint_radius = texture_inpaint_radius;
  options.texture_scale = texture_scale;
  options.min_cos_normal_angle = texture_min_cos_normal_angle;
  options.min_visible_vertices = texture_min_visible_vertices;
  options.view_selection_smoothing_iterations =
      texture_view_smoothing_iterations;
  options.view_selection_min_component_size =
      texture_view_min_component_size;
  options.visibility_resolution = texture_visibility_resolution;
  options.occlusion_relative_tolerance = texture_occlusion_relative_tolerance;
  options.use_vertex_color_fallback = texture_use_vertex_color_fallback;
  options.apply_color_correction = texture_apply_color_correction;
  options.color_correction_regularization =
      texture_color_correction_regularization;
  options.num_threads = texture_num_threads;
  return options;
}

void ReconMvsControllerOptions::ValidateSurfaceOptions() const {
  (void)mvs::TexturedMeshOutputTypeFromString(texture_output_type);
  if (!PoissonOptions().Check()) throw std::invalid_argument("Invalid Poisson meshing options");
  if (!DelaunayOptions().Check()) throw std::invalid_argument("Invalid Delaunay meshing options");
  if (!SimplificationOptions().Check()) throw std::invalid_argument("Invalid mesh simplification options");
  if (!TexturingOptions().Check()) throw std::invalid_argument("Invalid mesh texturing options");
  if ((poisson_mesher || delaunay_mesher || mesh_dense_cloud) && !fuse_stereo) {
    throw std::invalid_argument("Surface generation requires --Colite.fuse_stereo");
  }
  if ((mesh_simplifier || mesh_texturer) && !poisson_mesher && !delaunay_mesher && !mesh_dense_cloud) {
    throw std::invalid_argument("Mesh simplification and texturing require a mesh generation option");
  }
  if (poisson_mesher && !mvs::IsPoissonMeshingAvailable()) {
    throw std::invalid_argument("Poisson meshing is unavailable in this build");
  }
  if (delaunay_mesher && !mvs::IsDelaunayMeshingAvailable()) {
    throw std::invalid_argument("Delaunay meshing requires a build configured with CGAL");
  }
}

int ReconMvsController::Run(std::ostream& out) const {
  ReconMvsControllerOptions cli_options = options_;
  cli_options.ValidateSurfaceOptions();
  if (cli_options.run_patch_match && cli_options.patch_match_geom_consistency &&
      cli_options.patch_match_geometric_refinement_iterations == 0) {
    cli_options.patch_match_geometric_refinement_iterations = cli_options.patch_match_iterations;
  }
  struct ThreadScope {
    int previous = parallel::getNumThreads();
    int previous_cv = cv::getNumThreads();
    explicit ThreadScope(std::size_t requested) {
      const int count = static_cast<int>(std::min<std::size_t>(
          requested, std::numeric_limits<int>::max()));
      parallel::setNumThreads(count);
      cv::setNumThreads(count);
    }
    ~ThreadScope() {
      parallel::setNumThreads(previous);
      cv::setNumThreads(previous_cv);
    }
  } thread_scope(cli_options.num_threads);
  const auto phase_options = [&](int threads, int max_size) {
    ReconMvsControllerOptions phase = cli_options;
    if (threads != 0) {
      phase.num_threads = threads < 0
          ? std::max(1U, std::thread::hardware_concurrency())
          : static_cast<std::size_t>(threads);
    }
    if (max_size != 0) phase.max_image_size = max_size;
    return phase;
  };
  LOG(INFO) << "MVS pipeline: model_path=" << cli_options.model_path.string()
            << " image_path=" << cli_options.image_path.string()
            << " output_path=" << cli_options.output_path.string()
            << " undistort=" << (cli_options.image_undistorter ? 1 : 0)
            << " patch_match=" << (cli_options.run_patch_match ? 1 : 0)
            << " fusion=" << (cli_options.fuse_stereo ? 1 : 0)
            << " backend=" << cli_options.patch_match_backend
            << " threads=" << cli_options.num_threads;
  std::vector<ImageId> image_ids_in_model_order;
  Reconstruction reconstruction = ReadColmapReconstruction(
      cli_options.model_path, ColmapModelFormat::kAuto, &image_ids_in_model_order);
  const std::filesystem::path output_path(cli_options.output_path);
  std::size_t num_undistorted_images = 0;
  if (cli_options.image_undistorter) {
    const auto undistort_options = phase_options(cli_options.undistort_num_threads, 0);
    ThreadScope phase_threads(undistort_options.num_threads);
    num_undistorted_images =
        UndistortWorkspaceImages(undistort_options, output_path, &reconstruction);
    cli_options.image_path = output_path / "images";
    cli_options.max_image_size = -1;
    cli_options.resolution_level = 0;
    // The undistorter writes both model encodings in numeric-ID order.
    image_ids_in_model_order.assign(reconstruction.RegisteredImageIds().begin(),
                                     reconstruction.RegisteredImageIds().end());
  }

  mvs::MvsPipelineOptions mvs_options;
  mvs_options.max_num_source_images = cli_options.max_num_source_images;
  mvs_options.min_num_shared_points = cli_options.min_num_shared_points;
  mvs_options.source_selection_mode =
      cli_options.source_selection_mode == "colmap"
          ? mvs::SourceSelectionMode::kColmap
          : mvs::SourceSelectionMode::kWeighted;
  mvs_options.min_triangulation_angle_degrees =
      cli_options.source_min_triangulation_angle_degrees;
  mvs_options.max_triangulation_angle_degrees =
      cli_options.source_max_triangulation_angle_degrees;
  mvs_options.triangulation_angle_weight =
      cli_options.source_triangulation_angle_weight;
  mvs_options.shared_points_weight =
      cli_options.source_shared_points_weight;
  mvs_options.scale_consistency_weight =
      cli_options.source_scale_consistency_weight;
  mvs_options.coverage_weight = cli_options.source_coverage_weight;
  mvs_options.coverage_grid_size = cli_options.source_coverage_grid_size;
  const bool fusion_only = cli_options.fuse_stereo &&
      !cli_options.run_patch_match && !cli_options.initialize_patch_match &&
      !cli_options.image_undistorter;
  mvs::MvsPipelineSummary summary;
  std::unordered_map<ImageId, std::vector<ImageId>> fusion_overlaps;
  if (fusion_only) {
    // Fusion consumes persisted maps and sparse overlap, not photometric
    // source scores, angle gates or depth-range diagnostics. In particular,
    // a PatchMatch angle option must not silently exclude valid fusion maps.
    fusion_overlaps = mvs::ComputeOverlappingImages(reconstruction,
        cli_options.fusion_max_num_source_images, cli_options.min_num_shared_points);
    for (const ImageId id : reconstruction.RegisteredImageIds()) {
      const auto& image = reconstruction.GetImage(id);
      if (!image.cam_from_world || !reconstruction.ExistsCamera(image.camera_id)) continue;
      mvs::ImageMvsSummary entry;
      entry.image_id = id;
      entry.sparse_depth_map = mvs::BuildSparseDepthMap(reconstruction, id);
      summary.num_sparse_depth_samples += entry.sparse_depth_map.samples.size();
      summary.images.push_back(std::move(entry));
    }
    summary.num_images = summary.images.size();
    summary.success = summary.num_images > 0;
  } else {
    summary = mvs::RunMvsPipeline(reconstruction, mvs_options);
  }
  mvs::MvsPipelineSummary fusion_input_summary;
  if (cli_options.fuse_stereo) {
    for (const auto& image : summary.images) {
      mvs::ImageMvsSummary entry;
      entry.image_id = image.image_id;
      fusion_input_summary.images.push_back(std::move(entry));
    }
  }
  if (cli_options.run_patch_match) {
    ApplyPatchMatchConfig(reconstruction, cli_options, mvs_options, &summary);
  }
  for (auto& image : summary.images) {
    if (cli_options.patch_match_depth_min > 0) image.min_depth = cli_options.patch_match_depth_min;
    if (cli_options.patch_match_depth_max > 0) image.max_depth = cli_options.patch_match_depth_max;
    if ((cli_options.patch_match_depth_min > 0 || cli_options.patch_match_depth_max > 0 ||
         (cli_options.run_patch_match && !image.source_images.empty() &&
          (cli_options.require_workspace_config || !cli_options.patch_match_config_path.empty() ||
           std::filesystem::exists(output_path / "stereo/patch-match.cfg")))) &&
        !(image.min_depth > 0 && image.min_depth < image.max_depth)) {
      throw std::invalid_argument("PatchMatchStereo depth bounds are invalid for image " + std::to_string(image.image_id) +
          "; supply depth_min and depth_max when no sparse depth range is available");
    }
  }

  std::size_t num_initialized_patch_match_images = 0;
  if (cli_options.initialize_patch_match) {
    const auto initialization_options = phase_options(cli_options.patch_match_num_threads,
                                                      cli_options.patch_match_max_image_size);
    ThreadScope phase_threads(initialization_options.num_threads);
    num_initialized_patch_match_images = InitializePatchMatchMapsForWorkspace(
        output_path, initialization_options, reconstruction, summary);
  }
  std::size_t num_patch_match_images = 0;
  std::size_t num_photometric_filtered_pixels = 0;
  std::size_t num_normal_refined_pixels = 0;
  std::size_t num_geometric_consistent_pixels = 0;
  std::size_t num_geometric_filtered_pixels = 0;
  PatchMatchSchedulingSummary scheduling_summary;
  mvs::StereoFusionSummary fusion_summary;
  if (cli_options.run_patch_match) {
    auto patch_match_options = phase_options(cli_options.patch_match_num_threads,
                                            cli_options.patch_match_max_image_size);
    ThreadScope phase_threads(patch_match_options.num_threads);
    // Persist PatchMatch outputs first, then let all per-view CUDA/host
    // workspaces and map buffers leave scope before standalone fusion loads
    // its bounded working set.
    patch_match_options.fuse_stereo = false;
    const auto cache_path = PhotometricCachePath(output_path);
    const auto cache_key = PhotometricCacheKey(patch_match_options, reconstruction, summary);
    const bool recorded = std::filesystem::exists(cache_path);
    const bool legacy_owned =
        std::filesystem::exists(output_path / "stereo/patch_match_initial.tsv") ||
        std::filesystem::exists(output_path / ".colite/stages/patch_match_stereo.json");
    if ((recorded && !PhotometricCacheIsCurrent(cache_path, cache_key,
            PhotometricMapsFingerprint(output_path, reconstruction, summary))) ||
        (!recorded && legacy_owned)) {
      patch_match_options.force_recompute = true;
    }
    // Mark incomplete before writing any maps. A crash, exception or parameter
    // change cannot bless old or partially replaced maps with the new key.
    WritePhotometricCache(cache_path, cache_key, false);
    num_patch_match_images = RunPatchMatchForWorkspace(
        output_path, patch_match_options, reconstruction, summary, image_ids_in_model_order,
        &num_photometric_filtered_pixels, &num_normal_refined_pixels,
        &num_geometric_consistent_pixels, &num_geometric_filtered_pixels,
        &scheduling_summary);
    WritePhotometricCache(cache_path, cache_key, true,
        PhotometricMapsFingerprint(output_path, reconstruction, summary));
  }
  if (cli_options.fuse_stereo) {
    auto fusion_options = phase_options(cli_options.fusion_num_threads,
                                              cli_options.fusion_max_image_size);
    if (!cli_options.fusion_input_type_explicit && cli_options.run_patch_match &&
        cli_options.patch_match_geometric_refinement_iterations == 0) {
      fusion_options.fusion_input_type = "photometric";
    }
    ThreadScope phase_threads(fusion_options.num_threads);
    fusion_summary = FusePersistedStereoMaps(
        output_path, fusion_options, reconstruction, fusion_input_summary,
        image_ids_in_model_order,
        fusion_only ? &fusion_overlaps : nullptr);
  }

  if (cli_options.poisson_mesher || cli_options.delaunay_mesher || cli_options.mesh_dense_cloud) {
    RunSurfaceSteps(output_path, cli_options, reconstruction, image_ids_in_model_order);
  }

  WriteMvsSummary(output_path, summary, num_undistorted_images,
                  num_initialized_patch_match_images, num_patch_match_images,
                  num_photometric_filtered_pixels, num_normal_refined_pixels,
                  num_geometric_consistent_pixels,
                  num_geometric_filtered_pixels, scheduling_summary,
                  fusion_summary);
  if (!fusion_only) {
    WriteSourceImages(output_path, summary);
    WriteSparseDepthSamples(output_path, summary);
  }

  out << "Images: " << summary.num_images << "\n"
      << "Undistorted images: " << num_undistorted_images << "\n"
      << "Sparse depth samples: " << summary.num_sparse_depth_samples << "\n"
      << "Initialized PatchMatch images: "
      << num_initialized_patch_match_images << "\n"
      << "PatchMatch images: " << num_patch_match_images << "\n"
      << "Photometric filtered pixels: "
      << num_photometric_filtered_pixels << "\n"
      << "Geometric consistent pixels: "
      << num_geometric_consistent_pixels << "\n"
      << "Geometric filtered pixels: "
      << num_geometric_filtered_pixels << "\n"
      << "Fused points: " << fusion_summary.num_fused_points << "\n"
      << "Fusion tracks: " << fusion_summary.num_fusion_tracks << "\n"
      << "Fused observations: "
      << fusion_summary.num_fused_observations << "\n"
      << "Output: " << output_path.string() << "\n";
  LOG(INFO) << "MVS complete: success=" << (summary.success ? 1 : 0)
            << " images=" << summary.num_images
            << " patch_match_images=" << num_patch_match_images
            << " fused_points=" << fusion_summary.num_fused_points
            << " fused_observations="
            << fusion_summary.num_fused_observations;
  return summary.success ? 0 : 2;
}


}  // namespace colite::controllers
