#include "colite/controllers/mapper.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "colite/controllers/recon_sfm_controller.h"
#include "colite/core/geometry.h"
#include "colite/core/memory.h"
#include "colite/core/parallel.h"
#include "colite/io/sfm_workspace.h"
#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/controllers/sfm_options.h"
#include "colite/sfm/incremental_mapper.h"
#include "colite/sfm/reconstruction_manager.h"
#include "colite/sfm/similarity_transform.h"

namespace colite::controllers {
namespace {

std::string CurrentStageHash(const std::filesystem::path& workspace_path,
                             const std::string& stage) {
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, stage);
  if (!metadata.config_hash.empty() &&
      WorkspaceCacheIsValid(workspace_path, stage, metadata.config_hash)) {
    return metadata.config_hash;
  }
  return "uncached_" + stage;
}

std::string MapperConfigHash(const MapperControllerOptions& options) {
  std::vector<std::pair<std::string, std::string>> entries =
      {{"stage", "sparse"},
       {"geometry_hash", CurrentStageHash(options.workspace_path, "geometry")},
       {"algorithm_version", SfmMappingImplementationKey()},
       {"ba_diagnostics", VLOG_IS_ON(1) ? "1" : "0"},
       {"output_model_format",
        ColmapModelFormatToString(options.output_model_format)},
       {"multiple_models", options.multiple_models ? "1" : "0"},
       {"merge_models", options.merge_models ? "1" : "0"},
       {"max_num_models", std::to_string(options.max_num_models)},
       {"min_num_images_per_model",
        std::to_string(options.min_num_images_per_model)},
       {"ba_refine_focal_length",
        HasAny(options.ba_refine_camera_params,
               CameraParameterGroup::kFocalLength)
            ? "1"
            : "0"},
       {"ba_refine_principal_point",
        HasAny(options.ba_refine_camera_params,
               CameraParameterGroup::kPrincipalPoint)
            ? "1"
            : "0"},
       {"ba_refine_extra_params",
        HasAny(options.ba_refine_camera_params,
               CameraParameterGroup::kDistortion)
            ? "1"
            : "0"},
       {"min_pair_median_triangulation_angle_deg", "0"},
       {"local_retriangulation_min_angle_deg", "1.5"},
       {"use_position_priors", options.use_position_priors ? "1" : "0"},
       {"position_prior_stddev", SfmNumberKey(options.position_prior_stddev)},
       {"position_prior_loss_scale",
        SfmNumberKey(options.position_prior_loss_scale)},
       {"position_prior_alignment_max_error",
        SfmNumberKey(options.position_prior_alignment_max_error)}};
  for (const auto& [name, value] : options.mapper_options) entries.emplace_back(name, value);
  if (options.global_mapping) {
    const auto& g = options.global_options;
    entries.emplace_back("mapper_method", "global_v1");
    entries.emplace_back("global_min_inliers", std::to_string(g.view_graph.min_num_inliers));
    entries.emplace_back("global_rotation_cycle", SfmNumberKey(g.view_graph.max_rotation_cycle_error_deg));
    entries.emplace_back("global_rotation_iterations", std::to_string(g.rotation_averaging.max_num_iterations));
    entries.emplace_back("global_rotation_error", SfmNumberKey(g.rotation_averaging.max_relative_rotation_error_deg));
    entries.emplace_back("global_direction_error", SfmNumberKey(g.positioning.max_direction_residual));
    entries.emplace_back("global_filter_outliers", g.positioning.filter_outliers ? "1" : "0");
  }
  if (!options.input_model_path.empty()) {
    entries.emplace_back("input_model", options.input_model_path.lexically_normal().generic_string());
    entries.emplace_back("input_format", ColmapModelFormatToString(options.input_model_format));
    entries.emplace_back("input_model_fingerprint",
                         ColmapModelFingerprint(options.input_model_path, options.input_model_format));
  }
  if (!options.image_path.empty()) {
    entries.emplace_back("image_path", options.image_path.lexically_normal().generic_string());
  }
  for (const auto& name : options.ba_refine_camera_param_names) entries.emplace_back("ba_param_" + name, "1");
  return StableConfigHash(entries);
}

std::optional<MapperControllerSummary> ReadCachedMapperSummary(
    const std::filesystem::path& workspace_path,
    const std::string& config_hash) {
  if (!WorkspaceCacheIsValid(workspace_path, "sparse", config_hash)) {
    return std::nullopt;
  }
  const WorkspaceCacheMetadata metadata =
      ReadWorkspaceCacheMetadata(workspace_path, "sparse");
  MapperControllerSummary summary;
  summary.success = metadata.values.at("success") == "1";
  summary.cache_hit = true;
  summary.num_registered_images = static_cast<std::size_t>(
      std::stoull(metadata.values.at("registered_images")));
  summary.num_points3d =
      static_cast<std::size_t>(std::stoull(metadata.values.at("points3d")));
  summary.num_models = static_cast<std::size_t>(
      std::stoull(metadata.values.at("models")));
  summary.num_unregistered_images = static_cast<std::size_t>(
      std::stoull(metadata.values.at("unregistered_images")));
  if (const auto it = metadata.values.find("position_prior_aligned_models");
      it != metadata.values.end()) {
    summary.num_position_prior_aligned_models = std::stoull(it->second);
  }
  if (const auto it = metadata.values.find("final_mean_position_error");
      it != metadata.values.end()) {
    summary.final_mean_position_error = std::stod(it->second);
  }
  return summary;
}

std::vector<std::filesystem::path> SparseRequiredFiles(
    const std::size_t num_models, const ColmapModelFormat format) {
  std::vector<std::filesystem::path> files = {
      "features", "cameras.txt", "images.tsv", "matches/verified_matches.tsv",
      "sparse/points3D.ply",
      "sparse/summary.txt",
      "sparse/models.txt"};
  const auto append_model_files = [&](const std::filesystem::path& model) {
    if (format == ColmapModelFormat::kText ||
        format == ColmapModelFormat::kTextAndBinary) {
      files.push_back(model / "cameras.txt");
      files.push_back(model / "images.txt");
      files.push_back(model / "points3D.txt");
    }
    if (format == ColmapModelFormat::kBinary ||
        format == ColmapModelFormat::kTextAndBinary) {
      files.push_back(model / "cameras.bin");
      files.push_back(model / "images.bin");
      files.push_back(model / "points3D.bin");
    }
  };
  append_model_files("sparse");
  for (std::size_t model_idx = 0; model_idx < num_models; ++model_idx) {
    const std::filesystem::path model =
        std::filesystem::path("sparse/models") / std::to_string(model_idx);
    append_model_files(model);
    files.push_back(model / "points3D.ply");
  }
  return files;
}

void ExportSparseModels(const MapperControllerOptions& options,
                        const MapperControllerSummary& summary) {
  if (!summary.success || options.output_path.empty()) return;
  // COLMAP exports a continued reconstruction directly into output_path.
  // Keep additional workspace components cached, but export only the input
  // reconstruction in this mode, as COLMAP's mapper does.
  const bool continuing = !options.input_model_path.empty();
  const std::size_t num_exports =
      continuing ? std::min<std::size_t>(summary.num_models, 1) : summary.num_models;
  for (std::size_t idx = 0; idx < num_exports; ++idx) {
    const auto model = ReadColmapReconstruction(
        WorkspaceSparsePath(options.workspace_path) / "models" /
        std::to_string(idx));
    const auto output = continuing ? options.output_path
                                   : options.output_path / std::to_string(idx);
    WriteColmapReconstruction(output, model, options.output_model_format);
    WritePlyPointCloud(output / "points3D.ply", model);
  }
}

}  // namespace

MapperController::MapperController(MapperControllerOptions options)
    : options_(std::move(options)) {}

MapperControllerSummary MapperController::Run() const {
  if (options_.global_mapping && !options_.input_model_path.empty())
    throw std::invalid_argument("Global mapping cannot continue an input model");
  if (options_.global_mapping) {
    for (const auto& entry : options_.mapper_options) {
      const auto& name = entry.first;
      if (name.rfind("--Mapper.tri_", 0) != 0 && name.rfind("--Mapper.filter_", 0) != 0 &&
          name != "--Mapper.num_threads" && name != "--Mapper.ba_global_max_num_iterations" &&
          name != "--Mapper.ba_global_function_tolerance" && name != "--Mapper.ba_min_num_residuals_for_cpu_multi_threading")
        throw std::invalid_argument(name + " is not applicable to global mapping");
    }
  }
  IncrementalMappingOptions mapping_options = ColmapSfmMappingDefaults();
  mapping_options.mapper.triangulation.min_pair_median_triangulation_angle_deg =
      0.0;
  mapping_options.mapper.local_retriangulation_filter
      .min_triangulation_angle_deg = 1.5;
  mapping_options.mapper.local_bundle_adjustment.refine_camera_params =
      options_.ba_refine_camera_params;
  mapping_options.mapper.local_bundle_adjustment.refine_camera_param_names =
      options_.ba_refine_camera_param_names;
  mapping_options.global_bundle_adjustment.refine_camera_params =
      options_.ba_refine_camera_params;
  mapping_options.global_bundle_adjustment.refine_camera_param_names =
      options_.ba_refine_camera_param_names;
  ApplySfmMapperOptions(options_.mapper_options, &mapping_options);
  if (!options_.image_path.empty() &&
      !std::filesystem::is_directory(options_.image_path)) {
    throw std::invalid_argument("--image_path must be an existing image directory");
  }
  LOG(INFO) << "Mapper: workspace=" << options_.workspace_path.string()
            << " multiple_models=" << (options_.multiple_models ? 1 : 0)
            << " merge_models=" << (options_.merge_models ? 1 : 0)
            << " input_model=" << options_.input_model_path.string();
  const std::string config_hash = MapperConfigHash(options_);
  if (const auto cached_summary =
          ReadCachedMapperSummary(options_.workspace_path, config_hash)) {
    LOG(INFO) << "Mapper cache hit: registered_images="
              << cached_summary->num_registered_images
              << " points3d=" << cached_summary->num_points3d
              << " models=" << cached_summary->num_models;
    ExportSparseModels(options_, *cached_summary);
    return *cached_summary;
  }

  const std::vector<Camera> cameras =
      ReadWorkspaceCameras(options_.workspace_path);
  const std::vector<Image> images = ReadWorkspaceImages(options_.workspace_path);
  if (cameras.empty() || images.size() < 2) {
    throw std::runtime_error("Workspace must contain cameras and images");
  }

  MatchGraph graph;
  ReadWorkspaceFeatureSets(options_.workspace_path, &graph,
                           WorkspaceFeatureLoading::kDeferredDescriptors);
  // Mapping consumes verified correspondences; raw descriptor candidates are
  // only needed by the matcher and need not be parsed or retained here.
  ReadWorkspaceVerifiedGeometries(options_.workspace_path, &graph);

  Reconstruction reconstruction = ReadWorkspaceReconstruction(
      options_.workspace_path);
  if (!options_.input_model_path.empty()) {
    reconstruction = ApplyColmapModel(reconstruction,
                                      options_.input_model_path,
                                      options_.input_model_format);
  }
  if (!options_.image_path.empty()) {
    for (const auto& [id, image] : reconstruction.Images()) {
      reconstruction.GetImage(id).file_path =
          (options_.image_path / image.name).string();
    }
  }

  struct ThreadScope {
    int previous = parallel::getNumThreads();
    explicit ThreadScope(int count) { parallel::setNumThreads(count <= 0 ? -1 : count); }
    ~ThreadScope() { parallel::setNumThreads(previous); }
  } mapper_threads(mapping_options.num_threads);
  LOG(INFO) << "Mapper CPU threads=" << parallel::getNumThreads();
  MultiModelMappingOptions multi_options;
  multi_options.mapping = mapping_options;
  multi_options.max_num_models =
      options_.multiple_models ? options_.max_num_models : 1;
  multi_options.min_num_images_per_model = options_.min_num_images_per_model;
  multi_options.merge_models = options_.merge_models;
  std::map<ImageId, Camera> cameras_by_image;
  for (const auto& image : images) {
    cameras_by_image.emplace(image.id, reconstruction.GetCamera(image.camera_id));
  }
  const auto recovered_poses = RecoverMatchGraphRelativePoses(cameras_by_image, &graph);
  LOG(INFO) << "Recovered missing relative poses=" << recovered_poses;
  MultiModelMappingResult result;
  if (options_.global_mapping) {
    auto global = options_.global_options;
    global.triangulation = mapping_options.mapper.triangulation;
    global.bundle_adjustment = mapping_options.global_bundle_adjustment;
    global.fixed_rotation_bundle_adjustment = mapping_options.global_bundle_adjustment;
    global.observation_filter = mapping_options.global_observation_filter;
    const auto summary = RunGlobalMapping(graph, global, &reconstruction);
    if (!summary.success) throw std::runtime_error("Global mapping failed: " + summary.message);
    IncrementalMappingSummary compatible;
    compatible.success = summary.success;
    compatible.num_registered_images = summary.num_registered_images;
    compatible.num_points3d = summary.num_points3d;
    compatible.global_bundle_adjustment = summary.bundle_adjustment;
    compatible.global_track_completion = summary.triangulation;
    compatible.global_observation_filter = summary.observation_filter;
    compatible.message = summary.message;
    result.num_unregistered_images = images.size() - summary.num_registered_images;
    result.models.push_back(std::move(reconstruction));
    result.mapping_summaries.push_back(std::move(compatible));
  } else {
    result = RunIncrementalMappingMultiple(graph, multi_options, std::move(reconstruction));
  }
  if (result.models.empty()) {
    throw std::runtime_error("Mapper did not produce a valid sparse model");
  }
  // The graph is no longer consumed after mapping (including descriptor-
  // guided completion). Release it before BA priors, image colors and export.
  graph = MatchGraph{};
  ReleaseUnusedMemory();
  std::vector<std::size_t> order(result.models.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(),
            [&](std::size_t lhs, std::size_t rhs) {
              return result.models[lhs].RegisteredImageIds().size() >
                     result.models[rhs].RegisteredImageIds().size();
            });
  std::size_t position_prior_aligned_models = 0;
  double position_error_sum = 0.0;
  std::size_t position_error_count = 0;
  if (options_.use_position_priors) {
    for (Reconstruction& model : result.models) {
      const PositionPriorAlignmentSummary alignment =
          AlignReconstructionToPositionPriors(
              options_.position_prior_alignment_max_error, &model);
      if (!alignment.success) continue;
      GlobalBundleAdjustmentOptions ba =
          mapping_options.global_bundle_adjustment;
      ba.gauge_fixing = BundleGaugeFixing::kNone;
      ba.use_position_priors = true;
      ba.position_prior_stddev = options_.position_prior_stddev;
      ba.position_prior_loss_scale = options_.position_prior_loss_scale;
      const GlobalBundleAdjustmentSummary ba_summary =
          AdjustGlobalBundle(ba, &model);
      if (!ba_summary.success) continue;
      ++position_prior_aligned_models;
      position_error_sum += ba_summary.final_mean_position_error *
                            ba_summary.num_position_priors;
      position_error_count += ba_summary.num_position_priors;
      LOG(INFO) << "Mapper position-prior refinement: priors="
                << ba_summary.num_position_priors
                << " inliers=" << alignment.num_inliers
                << " final_error=" << ba_summary.final_mean_position_error;
    }
  }
  for (Reconstruction& model : result.models) {
    ExtractPointColors(&model);
    UpdatePoint3DReprojectionErrors(&model);
  }
  Reconstruction& primary = result.models[order.front()];
  const IncrementalMappingSummary& mapping_summary =
      result.mapping_summaries[order.front()];
  WriteWorkspaceSparseReconstruction(options_.workspace_path, primary,
                                     options_.output_model_format);
  WritePlyPointCloud(WorkspaceSparsePath(options_.workspace_path) /
                         "points3D.ply",
                     primary);
  const std::filesystem::path models_path =
      WorkspaceSparsePath(options_.workspace_path) / "models";
  for (std::size_t output_idx = 0; output_idx < order.size(); ++output_idx) {
    const std::filesystem::path model_path =
        models_path / std::to_string(output_idx);
    WriteColmapReconstruction(model_path, result.models[order[output_idx]],
                              options_.output_model_format);
    WritePlyPointCloud(model_path / "points3D.ply",
                       result.models[order[output_idx]]);
  }
  std::ofstream models_file(WorkspaceSparsePath(options_.workspace_path) /
                            "models.txt");
  models_file << "# MODEL_INDEX REGISTERED_IMAGES POINTS3D PATH\n";
  for (std::size_t output_idx = 0; output_idx < order.size(); ++output_idx) {
    const Reconstruction& model = result.models[order[output_idx]];
    models_file << output_idx << " " << model.RegisteredImageIds().size()
                << " " << model.Points3D().size() << " models/"
                << output_idx << "\n";
  }
  WriteRegistrationDiagnostics(WorkspaceSparsePath(options_.workspace_path),
                               mapping_summary);
  std::ofstream summary_file(WorkspaceSparsePath(options_.workspace_path) /
                             "summary.txt");
  summary_file << "mapper_initialization_seconds " << mapping_summary.initialization_seconds << "\n"
               << "mapper_candidate_selection_seconds " << mapping_summary.candidate_selection_seconds << "\n"
               << "mapper_global_track_completion_seconds " << mapping_summary.global_track_completion_seconds << "\n"
               << "mapper_pose_guided_track_completion_seconds " << mapping_summary.pose_guided_track_completion_seconds << "\n"
               << "mapper_global_retriangulation_seconds " << mapping_summary.global_retriangulation_seconds << "\n"
               << "mapper_global_ba_seconds " << mapping_summary.global_bundle_adjustment_seconds << "\n"
               << "mapper_global_filter_seconds " << mapping_summary.global_observation_filter_seconds << "\n"
               << "mapper_final_global_ba_seconds " << mapping_summary.final_global_bundle_adjustment_seconds << "\n";
  summary_file << "mapper_method " << (options_.global_mapping ? "global" : "incremental") << "\n";
  IncrementalMapperSummary timings;
  for (const auto& registration : mapping_summary.registrations) {
    timings.registration_seconds += registration.registration_seconds;
    timings.track_completion_seconds += registration.track_completion_seconds;
    timings.triangulation_seconds += registration.triangulation_seconds;
    timings.local_retriangulation_seconds += registration.local_retriangulation_seconds;
    timings.local_bundle_adjustment_seconds += registration.local_bundle_adjustment_seconds;
    timings.observation_filter_seconds += registration.observation_filter_seconds;
  }
  summary_file << "mapper_registration_seconds " << timings.registration_seconds << "\n"
               << "mapper_track_completion_seconds " << timings.track_completion_seconds << "\n"
               << "mapper_triangulation_seconds " << timings.triangulation_seconds << "\n"
               << "mapper_local_retriangulation_seconds " << timings.local_retriangulation_seconds << "\n"
               << "mapper_local_ba_seconds " << timings.local_bundle_adjustment_seconds << "\n"
               << "mapper_local_filter_seconds " << timings.observation_filter_seconds << "\n";
  summary_file << "success " << (mapping_summary.success ? 1 : 0) << "\n"
               << "registered_images "
               << mapping_summary.num_registered_images << "\n"
               << "points3d " << mapping_summary.num_points3d << "\n"
               << "models " << result.models.size() << "\n"
               << "unregistered_images " << result.num_unregistered_images
               << "\n"
               << "position_prior_aligned_models "
               << position_prior_aligned_models << "\n"
               << "final_mean_position_error "
               << (position_error_count == 0
                       ? -1.0
                       : position_error_sum / position_error_count)
               << "\n"
               << "initialization_attempts "
               << mapping_summary.initialization.num_initialization_attempts
               << "\n"
               << "global_completed_observations "
               << mapping_summary.global_track_completion
                      .num_completed_observations
               << "\n"
               << "global_merged_points "
               << mapping_summary.global_track_completion.num_merged_points
               << "\n"
               << "global_merged_observations "
               << mapping_summary.global_track_completion
                      .num_merged_observations
               << "\n"
               << "global_filtered_observations "
               << mapping_summary.global_observation_filter.num_removed_observations
               << "\n"
               << "global_ba_refinements " << mapping_summary.num_global_refinements
               << "\n"
               << "global_ba_growth_refinements " << mapping_summary.num_growth_global_refinements
               << "\n"
               << "registration_recovery_attempts " << mapping_summary.num_registration_recovery_attempts
               << "\n"
               << "registration_recovery_successes " << mapping_summary.num_registration_recovery_successes
               << "\n"
               << "global_ba_invalid_steps " << mapping_summary.num_global_ba_invalid_steps
               << "\n"
               << "global_ba_last_refinement_change " << mapping_summary.last_global_refinement_change
               << "\n"
               << "global_filtered_points "
               << mapping_summary.global_observation_filter.num_removed_points
               << "\n"
               << "message " << mapping_summary.message << "\n";
  if (!mapping_summary.initialization.message.empty()) {
    summary_file << "initialization_message "
                 << mapping_summary.initialization.message << "\n"
                 << "initialization_pair "
                 << mapping_summary.initialization.initial_pair.image_id1
                 << " "
                 << mapping_summary.initialization.initial_pair.image_id2
                 << "\n"
                 << "initialization_points "
                 << mapping_summary.initialization.num_triangulated_points
                 << "\ninitialization_geometry_reestimated "
                 << mapping_summary.initialization.geometry_reestimated
                 << "\ninitialization_geometry_inliers "
                 << mapping_summary.initialization.num_geometry_inliers
                 << "\ninitialization_median_triangulation_angle "
                 << mapping_summary.initialization.median_triangulation_angle
                 << "\n";
  }

  const auto saved_stats = ComputeReconstructionQualityStats(primary);
  summary_file << std::setprecision(17)
               << "num_observations " << saved_stats.num_observations << '\n'
               << "mean_reprojection_error " << saved_stats.mean_reprojection_error << '\n'
               << "max_reprojection_error " << saved_stats.max_reprojection_error << '\n';
  WriteGlobalBundleAdjustmentSummary(summary_file, mapping_summary);
  WriteInitializationRefinements(summary_file, mapping_summary);
  MapperControllerSummary summary;
  summary.success = mapping_summary.success;
  summary.num_registered_images = mapping_summary.num_registered_images;
  summary.num_points3d = mapping_summary.num_points3d;
  models_file.close();
  summary_file.close();
  if (!models_file || !summary_file) {
    throw std::runtime_error("Failed to write mapper summaries");
  }
  summary.num_models = result.models.size();
  summary.num_unregistered_images = result.num_unregistered_images;
  summary.num_position_prior_aligned_models = position_prior_aligned_models;
  summary.final_mean_position_error =
      position_error_count == 0 ? -1.0
                                : position_error_sum / position_error_count;
  WriteWorkspaceCacheMetadata(
      options_.workspace_path, "sparse", config_hash,
      {{"success", mapping_summary.success ? "1" : "0"},
       {"implementation_version", kSfmMappingImplementationVersion},
       {"pose_solver_backend", std::string(PoseSolverBackend())},
       {"registered_images", std::to_string(mapping_summary.num_registered_images)},
       {"points3d", std::to_string(mapping_summary.num_points3d)},
       {"models", std::to_string(result.models.size())},
       {"unregistered_images",
        std::to_string(result.num_unregistered_images)},
       {"position_prior_aligned_models",
        std::to_string(position_prior_aligned_models)},
       {"final_mean_position_error",
        std::to_string(summary.final_mean_position_error)}},
      SparseRequiredFiles(result.models.size(), options_.output_model_format));
  LOG(INFO) << "Mapper complete: success=" << (summary.success ? 1 : 0)
            << " registered_images=" << summary.num_registered_images
            << " points3d=" << summary.num_points3d
            << " models=" << summary.num_models
            << " unregistered_images=" << summary.num_unregistered_images;
  ExportSparseModels(options_, summary);
  return summary;
}

}  // namespace colite::controllers
