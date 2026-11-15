#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "colite/matching/two_view_geometry_verifier.h"
#include "colite/core/version.h"
#include "colite/sfm/incremental_mapper.h"

namespace colite::controllers {

inline constexpr const char* kSfmMappingImplementationVersion =
    "incremental_mapper_v67_opt_in_model_merge";

inline std::string SfmMappingImplementationKey() {
  return std::string(kSfmMappingImplementationVersion) + ":pose_solver=" +
         std::string(PoseSolverBackend());
}

inline void WriteInitializationRefinements(
    std::ostream& out, const IncrementalMappingSummary& summary) {
  out << "initialization_reconstruction_trials " << summary.initialization_refinements.size() << '\n'
      << "mapper_initial_ba_seconds " << summary.initial_bundle_adjustment_seconds << '\n';
  for (std::size_t i = 0; i < summary.initialization_refinements.size(); ++i) {
    const auto& trial = summary.initialization_refinements[i];
    const auto prefix = "initialization_trial_" + std::to_string(i) + "_";
    out << prefix << "pair " << trial.pair.image_id1 << ' ' << trial.pair.image_id2 << '\n'
        << prefix << "success " << trial.success << '\n'
        << prefix << "points_before " << trial.num_points_before << '\n'
        << prefix << "points_after " << trial.num_points_after << '\n'
        << prefix << "ba_attempted " << trial.bundle_adjustment_attempted << '\n'
        << prefix << "ba_success " << trial.bundle_adjustment.success << '\n'
        << prefix << "ba_max_num_iterations " << trial.bundle_adjustment.solver_max_num_iterations << '\n'
        << prefix << "ba_gradient_tolerance " << trial.bundle_adjustment.solver_gradient_tolerance << '\n'
        << prefix << "normalized " << trial.normalized << '\n'
        << prefix << "normalization_scale " << trial.normalization_scale << '\n'
        << prefix << "message " << trial.message << '\n';
  }
}

inline GlobalBundleAdjustmentOptions ColmapStandaloneBundleAdjustmentDefaults() {
  GlobalBundleAdjustmentOptions options;
  options.max_num_iterations = 100;
  options.refine_camera_params = CameraParameterGroup::kFocalLength |
                                 CameraParameterGroup::kDistortion;
  return options;
}

// Costs from different rounds can use different observations. Saved-model
// reprojection statistics are reported separately from these objectives.
inline void WriteGlobalBundleAdjustmentSummary(
    std::ostream& out, const IncrementalMappingSummary& summary) {
  const auto& first = summary.global_bundle_adjustment;
  const auto& last = summary.num_global_refinements > 1
      ? summary.final_global_bundle_adjustment : first;
  out << "global_ba_first_initial_cost " << first.initial_cost << '\n'
      << "global_ba_first_final_cost " << first.final_cost << '\n'
      << "global_ba_last_initial_cost " << last.initial_cost << '\n'
      << "global_ba_last_final_cost " << last.final_cost << '\n'
      << "global_ba_first_initial_mean_error " << first.initial_mean_reprojection_error << '\n'
      << "global_ba_first_final_mean_error " << first.final_mean_reprojection_error << '\n'
      << "global_ba_last_initial_mean_error " << last.initial_mean_reprojection_error << '\n'
      << "global_ba_last_final_mean_error " << last.final_mean_reprojection_error << '\n';
}

inline IncrementalMappingOptions ColmapSfmMappingDefaults() {
  IncrementalMappingOptions options;
  options.initialization.min_num_inliers = 100;
  options.initialization.min_triangulation_angle_deg = 16.0;
  options.max_num_initialization_trials = 200;
  options.initialization.min_num_triangulated_points = 1;
  options.initialization.reestimate_geometry = true;
  options.initialization.random_seed = -1;
  options.mapper.registration.random_seed = -1;
  options.relax_initialization_constraints = true;
  options.mapper.registration.max_reprojection_error_px = 12.0;
  options.mapper.registration.min_num_inliers = 30;
  options.mapper.registration.min_num_correspondences = 30;
  options.mapper.registration.min_inlier_ratio = 0.25;
  options.mapper.use_structureless_registration = true;
  // Angular triangulation gates first; pixel filtering follows local BA.
  // Filtering these new points before their pose/structure refinement can
  // discard valid support. COLMAP also filters after local bundle adjustment.
  options.mapper.run_local_retriangulation_filter = false;
  options.mapper.observation_filter.min_triangulation_angle_deg = 1.5;
  options.mapper.local_retriangulation_filter.min_triangulation_angle_deg = 1.5;
  options.global_observation_filter.min_triangulation_angle_deg = 1.5;
  // Solver controls are shared. Standalone/local/global iteration budgets
  // remain 100/25/50, matching their corresponding COLMAP workflows.
  options.registration_retry_reprojection_error_multiplier = 1.0;
  options.registration_retry_max_reprojection_error_px = 12.0;
  return options;
}

inline double SfmDouble(const std::string& value) {
  std::size_t consumed = 0;
  const double number = std::stod(value, &consumed);
  if (consumed != value.size() || !std::isfinite(number)) {
    throw std::invalid_argument("Expected a finite number: " + value);
  }
  return number;
}

inline std::size_t SfmCount(const std::string& value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a non-negative integer: " + value);
  }
  const auto number = std::stoull(value);
  if (number > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("Integer is too large: " + value);
  }
  return static_cast<std::size_t>(number);
}

inline int SfmInt(const std::string& value) {
  std::size_t consumed = 0;
  const int number = std::stoi(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument("Expected an integer: " + value);
  }
  return number;
}

inline bool SfmBool(const std::string& value) {
  if (value == "1" || value == "true" || value == "TRUE") return true;
  if (value == "0" || value == "false" || value == "FALSE") return false;
  throw std::invalid_argument("Expected 0/1 or true/false: " + value);
}

inline std::vector<std::string> SfmBundleAdjustmentOptionNames() {
  return {"--BundleAdjustment.refine_focal_length",
          "--BundleAdjustment.refine_principal_point",
          "--BundleAdjustment.refine_extra_params",
          "--BundleAdjustment.refine_rig_from_world",
          "--BundleAdjustment.refine_sensor_from_rig",
          "--BundleAdjustment.refine_points3D",
          "--BundleAdjustment.constant_rig_from_world_rotation",
          "--BundleAdjustment.min_track_length",
          "--BundleAdjustmentCeres.max_num_iterations",
          "--BundleAdjustmentCeres.max_linear_solver_iterations",
          "--BundleAdjustmentCeres.function_tolerance",
          "--BundleAdjustmentCeres.gradient_tolerance",
          "--BundleAdjustmentCeres.parameter_tolerance",
          "--BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading",
          "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver",
          "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver",
          "--BundleAdjustmentCeres.use_gpu"};
}

inline bool ApplySfmBundleAdjustmentOption(const std::string& name,
    const std::string& value, GlobalBundleAdjustmentOptions* options) {
  auto& solver = options->solver;
  auto nonnegative = [&] {
    const double v = SfmDouble(value);
    if (v < 0) throw std::invalid_argument(name + " must be non-negative");
    return v;
  };
  auto count = [&] {
    const int v = SfmInt(value);
    if (v < 0) throw std::invalid_argument(name + " must be non-negative");
    return v;
  };
  if (name == "--BundleAdjustment.refine_focal_length" ||
      name == "--BundleAdjustment.refine_principal_point" ||
      name == "--BundleAdjustment.refine_extra_params") {
    const auto group = name == "--BundleAdjustment.refine_focal_length"
        ? CameraParameterGroup::kFocalLength : name == "--BundleAdjustment.refine_principal_point"
        ? CameraParameterGroup::kPrincipalPoint : CameraParameterGroup::kDistortion;
    const int bits = static_cast<int>(options->refine_camera_params);
    options->refine_camera_params = static_cast<CameraParameterGroup>(SfmBool(value)
        ? bits | static_cast<int>(group) : bits & ~static_cast<int>(group));
  } else if (name == "--BundleAdjustment.refine_rig_from_world") options->refine_poses = SfmBool(value);
  else if (name == "--BundleAdjustment.refine_sensor_from_rig") {
    // Supported model I/O admits only one reference sensor per trivial rig.
    // Its sensor-from-rig transform is fixed identity in COLMAP as well.
    (void)SfmBool(value);
  } else if (name == "--BundleAdjustment.refine_points3D") options->refine_points = SfmBool(value);
  else if (name == "--BundleAdjustment.constant_rig_from_world_rotation") options->constant_pose_rotation = SfmBool(value);
  else if (name == "--BundleAdjustment.min_track_length") options->min_track_length = static_cast<std::size_t>(count());
  else if (name == "--BundleAdjustmentCeres.max_num_iterations") options->max_num_iterations = count();
  else if (name == "--BundleAdjustmentCeres.max_linear_solver_iterations") {
    solver.max_linear_solver_iterations = count();
    if (solver.max_linear_solver_iterations == 0) throw std::invalid_argument(name + " must be positive");
  } else if (name == "--BundleAdjustmentCeres.function_tolerance") solver.function_tolerance = nonnegative();
  else if (name == "--BundleAdjustmentCeres.gradient_tolerance") solver.gradient_tolerance = nonnegative();
  else if (name == "--BundleAdjustmentCeres.parameter_tolerance") solver.parameter_tolerance = nonnegative();
  else if (name == "--BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading") solver.min_num_residuals_for_cpu_multi_threading = count();
  else if (name == "--BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver") solver.max_num_images_direct_dense_cpu_solver = count();
  else if (name == "--BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver") solver.max_num_images_direct_sparse_cpu_solver = count();
  else if (name == "--BundleAdjustmentCeres.use_gpu") {
    if (SfmBool(value)) throw std::invalid_argument("GPU BA is not implemented; use BundleAdjustmentCeres.use_gpu=0");
  } else return false;
  return true;
}

// This registry contains only controls that are actually implemented. Similar
// names with different semantics (e.g. init_max_reg_trials) are not aliases.
inline bool ApplySfmMapperOption(const std::string& name,
                                 const std::string& value,
                                 IncrementalMappingOptions* options) {
  auto positive = [&]() {
    const double number = SfmDouble(value);
    if (number <= 0) throw std::invalid_argument(name + " must be positive");
    return number;
  };
  auto angle = [&]() {
    const double number = SfmDouble(value);
    if (number < 0 || number > 180) throw std::invalid_argument(name + " must be in [0, 180] degrees");
    return number;
  };
  auto count = [&]() {
    const auto number = SfmCount(value);
    if (number == 0) throw std::invalid_argument(name + " must be positive");
    return number;
  };
  auto iterations = [&]() {
    const int number = SfmInt(value);
    if (number <= 0) throw std::invalid_argument(name + " must be positive");
    return number;
  };
  auto& mapper = options->mapper;
  auto& tri = mapper.triangulation;
  if (name == "--Mapper.init_image_id1" || name == "--Mapper.init_image_id2") {
    const int id = SfmInt(value);
    if (id < -1) throw std::invalid_argument(name + " must be -1 or a non-negative image ID");
    const auto image_id = id == -1 ? kInvalidImageId : static_cast<ImageId>(id);
    if (name == "--Mapper.init_image_id1") options->initialization.image_id1 = image_id;
    else options->initialization.image_id2 = image_id;
  } else if (name == "--Mapper.init_max_reg_trials") options->initialization.max_num_reg_trials = iterations();
  else if (name == "--Mapper.num_threads") {
    options->num_threads = SfmInt(value);
    if (options->num_threads < -1) throw std::invalid_argument(name + " must be >= -1");
  } else if (name == "--Mapper.random_seed") {
    const int seed = SfmInt(value);
    if (seed < -1) throw std::invalid_argument(name + " must be >= -1");
    options->initialization.random_seed = seed;
    mapper.registration.random_seed = seed;
  } else if (name == "--Mapper.init_min_num_inliers") options->initialization.min_num_inliers = count();
  else if (name == "--Mapper.init_max_error") options->initialization.max_reprojection_error_px = positive();
  else if (name == "--Mapper.init_max_forward_motion") {
    const double number = SfmDouble(value);
    if (number < 0 || number > 1) throw std::invalid_argument(name + " must be in [0, 1]");
    options->initialization.max_forward_motion = number;
  }
  else if (name == "--Mapper.init_min_tri_angle") options->initialization.min_triangulation_angle_deg = angle();
  else if (name == "--Mapper.abs_pose_max_error") mapper.registration.max_reprojection_error_px = positive();
  else if (name == "--Mapper.abs_pose_min_num_inliers") {
    mapper.registration.min_num_inliers = count();
    mapper.registration.min_num_correspondences =
        std::max<std::size_t>(6, mapper.registration.min_num_inliers);
  }
  else if (name == "--Mapper.abs_pose_min_inlier_ratio") {
    const double number = SfmDouble(value);
    if (number < 0 || number > 1) throw std::invalid_argument(name + " must be in [0, 1]");
    mapper.registration.min_inlier_ratio = number;
  }
  else if (name == "--Mapper.max_reg_trials") options->max_num_registration_trials_per_image = count();
  else if (name == "--Mapper.init_num_trials") options->max_num_initialization_trials = count();
  else if (name == "--Mapper.structure_less_registration_fallback") mapper.use_structureless_registration = SfmBool(value);
  else if (name == "--Colite.registration_retry_error_multiplier") {
    const double number = positive();
    if (number < 1) throw std::invalid_argument(name + " must be >= 1");
    options->registration_retry_reprojection_error_multiplier = number;
  } else if (name == "--Colite.registration_retry_max_error") options->registration_retry_max_reprojection_error_px = positive();
  else if (name == "--Mapper.filter_max_reproj_error") {
    mapper.observation_filter.max_reprojection_error_px = positive();
    mapper.local_retriangulation_filter.max_reprojection_error_px = positive();
    options->global_observation_filter.max_reprojection_error_px = positive();
  } else if (name == "--Mapper.filter_min_tri_angle") {
    mapper.observation_filter.min_triangulation_angle_deg = angle();
    mapper.local_retriangulation_filter.min_triangulation_angle_deg = angle();
    options->global_observation_filter.min_triangulation_angle_deg = angle();
  } else if (name == "--Mapper.ba_local_num_images") {
    const auto number = count();
    if (number < 2) throw std::invalid_argument(name + " must be >= 2");
    mapper.local_bundle_adjustment.max_num_images = number;
  } else if (name == "--Mapper.ba_local_max_refinements") mapper.ba_local_max_refinements = iterations();
  else if (name == "--Mapper.ba_local_max_refinement_change") {
    const double number = SfmDouble(value);
    if (number < 0) throw std::invalid_argument(name + " must be non-negative");
    mapper.ba_local_max_refinement_change = number;
  } else if (name == "--Mapper.ba_local_max_num_iterations") mapper.local_bundle_adjustment.max_num_iterations = iterations();
  else if (name == "--Mapper.ba_global_max_num_iterations") options->global_bundle_adjustment.max_num_iterations = iterations();
  else if (name == "--Mapper.ba_local_function_tolerance" || name == "--Mapper.ba_global_function_tolerance") {
    const double number = SfmDouble(value);
    if (number < 0) throw std::invalid_argument(name + " must be non-negative");
    if (name == "--Mapper.ba_local_function_tolerance") mapper.local_bundle_adjustment.solver.function_tolerance = number;
    else options->global_bundle_adjustment.solver.function_tolerance = number;
  } else if (name == "--Mapper.ba_min_num_residuals_for_cpu_multi_threading") {
    const int number = SfmInt(value);
    if (number < 0) throw std::invalid_argument(name + " must be non-negative");
    mapper.local_bundle_adjustment.solver.min_num_residuals_for_cpu_multi_threading = number;
    options->global_bundle_adjustment.solver.min_num_residuals_for_cpu_multi_threading = number;
  }
  else if (name == "--Mapper.ba_global_frames_ratio" || name == "--Mapper.ba_global_points_ratio") {
    const double number = SfmDouble(value);
    if (number <= 1.0) throw std::invalid_argument(name + " must be > 1");
    if (name == "--Mapper.ba_global_frames_ratio") options->ba_global_frames_ratio = number;
    else options->ba_global_points_ratio = number;
  } else if (name == "--Mapper.ba_global_frames_freq" || name == "--Mapper.ba_global_points_freq") {
    const int number = iterations();
    if (name == "--Mapper.ba_global_frames_freq") options->ba_global_frames_freq = number;
    else options->ba_global_points_freq = number;
  }
  else if (name == "--Mapper.ba_global_max_refinements") options->max_num_global_refinements = SfmCount(value);
  else if (name == "--Mapper.ba_global_max_refinement_change") {
    const double number = SfmDouble(value);
    if (number < 0) throw std::invalid_argument(name + " must be non-negative");
    options->global_refinement_change = number;
  }
  else if (name == "--Mapper.tri_max_transitivity" || name == "--Mapper.tri_complete_max_transitivity") {
    const int number = SfmInt(value);
    if (number < 0) throw std::invalid_argument(name + " must be non-negative");
    if (name == "--Mapper.tri_max_transitivity") tri.max_transitivity = number;
    else tri.complete_max_transitivity = number;
  }
  else if (name == "--Mapper.tri_create_max_angle_error") tri.create_max_angle_error_deg = angle();
  else if (name == "--Mapper.tri_continue_max_angle_error") tri.continue_max_angle_error_deg = angle();
  else if (name == "--Mapper.tri_merge_max_reproj_error") tri.merge_max_reprojection_error_px = positive();
  else if (name == "--Mapper.tri_complete_max_reproj_error") tri.complete_max_reprojection_error_px = positive();
  else if (name == "--Mapper.tri_min_angle") tri.min_triangulation_angle_deg = angle();
  else if (name == "--Mapper.tri_ignore_two_view_tracks") tri.ignore_two_view_tracks = SfmBool(value);
  else return false;
  return true;
}

inline void ApplySfmMapperOptions(const std::map<std::string, std::string>& values,
                                  IncrementalMappingOptions* options) {
  for (const auto& [name, value] : values) {
    if (!ApplySfmMapperOption(name, value, options)) {
      throw std::invalid_argument("Unsupported mapper option: " + name);
    }
  }
  const auto& initialization = options->initialization;
  if (initialization.image_id1 != kInvalidImageId &&
      initialization.image_id1 == initialization.image_id2)
    throw std::invalid_argument("Mapper.init_image_id1 and init_image_id2 must refer to different images");
}

inline bool ApplySfmGeometryOption(const std::string& name,
                                   const std::string& value,
                                   TwoViewGeometryOptions* options) {
  if (name == "--TwoViewGeometry.max_error") options->max_error_px = SfmDouble(value);
  else if (name == "--TwoViewGeometry.confidence") options->confidence = SfmDouble(value);
  else if (name == "--TwoViewGeometry.min_inlier_ratio") options->min_inlier_ratio = SfmDouble(value);
  else if (name == "--TwoViewGeometry.min_num_inliers") options->min_num_inliers = SfmInt(value);
  else if (name == "--TwoViewGeometry.max_num_trials") options->max_num_iterations = SfmInt(value);
  else if (name == "--TwoViewGeometry.random_seed") options->random_seed = SfmInt(value);
  else if (name == "--TwoViewGeometry.multiple_models") options->multiple_models = SfmBool(value);
  else if (name == "--TwoViewGeometry.compute_relative_pose") options->compute_relative_pose = SfmBool(value);
  else if (name == "--TwoViewGeometry.detect_watermark") options->detect_watermark = SfmBool(value);
  else if (name == "--TwoViewGeometry.multiple_ignore_watermark") options->multiple_ignore_watermark = SfmBool(value);
  else if (name == "--TwoViewGeometry.filter_stationary_matches") options->filter_stationary_matches = SfmBool(value);
  else if (name == "--TwoViewGeometry.watermark_detection_max_error") options->watermark_detection_max_error = SfmDouble(value);
  else if (name == "--TwoViewGeometry.stationary_matches_max_error") options->stationary_matches_max_error = SfmDouble(value);
  else return false;
  options->Check();
  return true;
}

inline std::string SfmGeometryKey(const TwoViewGeometryOptions& options) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << PoseSolverBackend() << ';'
         << options.max_error_px << ',' << options.confidence << ','
         << options.min_num_inliers << ',' << options.min_num_iterations << ','
         << options.max_num_iterations << ',' << options.min_E_F_inlier_ratio << ','
         << options.max_H_inlier_ratio << ',' << options.min_inlier_ratio << ',' << options.random_seed << ','
         << options.compute_relative_pose << ',' << options.detect_watermark << ',' << options.multiple_models << ','
         << options.multiple_ignore_watermark << ',' << options.watermark_min_inlier_ratio << ','
         << options.watermark_border_size << ',' << options.watermark_detection_max_error << ','
         << options.filter_stationary_matches << ',' << options.stationary_matches_max_error << ',' << options.force_H_use;
  return stream.str();
}

inline std::string SfmNumberKey(double number) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(std::numeric_limits<double>::max_digits10) << number;
  return stream.str();
}

}  // namespace colite::controllers
