#pragma once

#include <algorithm>
#include <cctype>
#include <iterator>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "colite/controllers/sfm_options.h"
#include "colite/controllers/sfm_frontend_options.h"
#include "colite/controllers/sfm_camera_setup.h"
#include "colite/matching/descriptor_matcher.h"

namespace colite::exe {

enum class SfmOptionScope { kFeatures, kMatching, kMapper, kPipeline, kModel };

inline bool ApplySfmFeatureExecutionOption(const std::string& name, const std::string& value,
    controllers::SfmFeatureExecutionOptions* options) {
  auto positive_int = [&] { const int n = controllers::SfmInt(value);
    if (n <= 0) throw std::invalid_argument(name + " must be positive"); return n; };
  auto& sift = options->sift;
  if (name == "--ImageReader.mask_path") options->masks.mask_path = value;
  else if (name == "--ImageReader.camera_mask_path") options->masks.camera_mask_path = value;
  else if (name == "--FeatureExtraction.use_gpu") options->use_gpu = controllers::SfmBool(value);
  else if (name == "--FeatureExtraction.gpu_index") {
    options->gpu_index = controllers::SfmInt(value);
    if (options->gpu_index < -1) throw std::invalid_argument(name + " must be >= -1 (single device)");
  } else if (name == "--FeatureExtraction.max_image_size") options->max_image_size = controllers::SfmInt(value);
  else if (name == "--SiftExtraction.first_octave") {
    const int n = controllers::SfmInt(value);
    if (n != -1 && n != 0) throw std::invalid_argument(name + " currently supports -1 or 0");
    sift.upsample_first_octave = n == -1;
  } else if (name == "--SiftExtraction.num_octaves") sift.num_octaves = positive_int();
  else if (name == "--SiftExtraction.octave_resolution") sift.num_octave_layers = positive_int();
  else if (name == "--SiftExtraction.peak_threshold" || name == "--SiftExtraction.edge_threshold") {
    const double n = controllers::SfmDouble(value);
    if (n <= 0) throw std::invalid_argument(name + " must be positive");
    if (name == "--SiftExtraction.peak_threshold") sift.peak_threshold = n;
    else sift.edge_threshold = n;
  } else if (name == "--SiftExtraction.max_num_orientations") {
    sift.max_num_orientations = positive_int();
    if (sift.max_num_orientations > 4) throw std::invalid_argument(name + " must be <= 4");
  } else if (name == "--descriptor_normalization") {
    std::string normalization = value;
    std::transform(normalization.begin(), normalization.end(), normalization.begin(),
        [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (normalization != "L1_ROOT" && normalization != "L2")
      throw std::invalid_argument(name + " expects l1_root or l2");
    sift.use_root_sift = normalization == "L1_ROOT";
  } else return false;
  return true;
}

inline bool ApplySfmMatchingExecutionOption(const std::string& name, const std::string& value,
    controllers::SfmMatchingExecutionOptions* options) {
  if (name == "--FeatureMatching.use_gpu") options->use_gpu = controllers::SfmBool(value);
  else if (name == "--FeatureMatching.gpu_index") {
    options->gpu_index = controllers::SfmInt(value);
    if (options->gpu_index < -1) throw std::invalid_argument(name + " must be >= -1 (single device)");
  } else if (name == "--FeatureMatching.num_threads") {
    options->num_threads = controllers::SfmInt(value);
    if (options->num_threads == 0 || options->num_threads < -1) throw std::invalid_argument(name + " must be -1 or positive");
  } else return false;
  return true;
}

// This is the public CLI contract. There is no legacy-name translation layer.
inline std::vector<std::string> NormalizeSfmArguments(
    const std::vector<std::string>& input, SfmOptionScope scope) {
  if (input.empty()) return input;
  std::vector<std::string> expanded{input.front()};
  for (std::size_t i = 1; i < input.size(); ++i) {
    const auto equal = input[i].find('=');
    if (input[i].rfind("--", 0) == 0 && equal != std::string::npos) {
      expanded.push_back(input[i].substr(0, equal));
      expanded.push_back(input[i].substr(equal + 1));
    } else expanded.push_back(input[i]);
  }
  std::set<std::string> seen;
  std::vector<std::string> result{input.front()};
  std::size_t camera_modes = 0;
  std::string feature_type = "sift";
  std::string matcher_type = "SIFT_BRUTEFORCE";
  for (std::size_t i = 1; i < expanded.size(); ++i) {
    std::string name = expanded[i];
    if (name == "-h") name = "--help";
    if (name.rfind("--", 0) != 0) throw std::invalid_argument("Unexpected argument: " + name);
    if (!seen.insert(name).second) throw std::invalid_argument("Duplicate option: " + name);
    const bool no_value = name == "--help" || name == "--version" ||
                          name == "--resume" || name == "--force" || name == "--dry_run";
    std::string value;
    if (!no_value) {
      if (i + 1 == expanded.size() || expanded[i + 1].rfind("--", 0) == 0) {
        throw std::invalid_argument("Missing value for " + name);
      }
      value = expanded[++i];
    }
    if (name == "--ImageReader.single_camera" ||
        name == "--ImageReader.single_camera_per_folder" ||
        name == "--ImageReader.single_camera_per_image") {
      camera_modes += controllers::SfmBool(value) ? 1 : 0;
      if (camera_modes > 1) throw std::invalid_argument("Conflicting ImageReader camera sharing options");
    }
    if (name == "--FeatureExtraction.type") {
      std::transform(value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      feature_type = value;
    }
    if (name == "--FeatureMatching.type") {
      std::transform(value.begin(), value.end(), value.begin(),
          [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
      matcher_type = value;
    }
    if (name == "--SpatialMatching.max_distance" && controllers::SfmDouble(value) <= 0) {
      throw std::invalid_argument(name + " must be positive; min_num_neighbors is not implemented");
    }
    result.push_back(name);
    if (!no_value) result.push_back(value);
  }
  if (scope == SfmOptionScope::kFeatures || scope == SfmOptionScope::kPipeline) {
    for (const auto& name : seen) {
      const bool sift_option = name.rfind("--SiftExtraction.", 0) == 0;
      const bool aliked_option = name.rfind("--AlikedExtraction.", 0) == 0;
      if ((sift_option && feature_type != "sift") || (aliked_option && feature_type == "sift")) {
        throw std::invalid_argument(name + " does not apply to FeatureExtraction.type " + feature_type);
      }
    }
    if ((seen.count("--AlikedExtraction.n16rot_model_path") && feature_type != "aliked_n16rot") ||
        (seen.count("--AlikedExtraction.n32_model_path") && feature_type != "aliked_n32")) {
      throw std::invalid_argument("ALIKED model path does not match FeatureExtraction.type");
    }
  }
  if (scope == SfmOptionScope::kMatching || scope == SfmOptionScope::kPipeline) {
    for (const auto& name : seen) {
      const bool sift = name.rfind("--SiftMatching.", 0) == 0;
      const bool aliked = name.rfind("--AlikedMatching.", 0) == 0;
      const bool lightglue = name.find(".lightglue_") != std::string::npos;
      if ((sift && matcher_type.rfind("SIFT_", 0) != 0) ||
          (aliked && matcher_type.rfind("ALIKED_", 0) != 0) ||
          ((sift || aliked) && lightglue && matcher_type.find("LIGHTGLUE") == std::string::npos) ||
          ((sift || aliked) && !lightglue && matcher_type.find("BRUTEFORCE") == std::string::npos)) {
        throw std::invalid_argument(name + " does not apply to FeatureMatching.type " + matcher_type);
      }
    }
  }
  return result;
}

inline bool ApplySfmCameraSharingFlag(const std::string& name,
                                      const std::string& value,
                                      controllers::CameraSharingMode* mode) {
  using Mode = controllers::CameraSharingMode;
  Mode selected;
  if (name == "--ImageReader.single_camera") selected = Mode::kSingle;
  else if (name == "--ImageReader.single_camera_per_folder") selected = Mode::kPerFolder;
  else if (name == "--ImageReader.single_camera_per_image") selected = Mode::kPerImage;
  else return false;
  if (controllers::SfmBool(value)) *mode = selected;
  return true;
}

inline FeatureMatcherType ParseSfmMatcherType(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (value == "SIFT_BRUTEFORCE") return FeatureMatcherType::kSiftBruteForce;
  if (value == "SIFT_LIGHTGLUE") return FeatureMatcherType::kSiftLightGlue;
  if (value == "ALIKED_BRUTEFORCE") return FeatureMatcherType::kAlikedBruteForce;
  if (value == "ALIKED_LIGHTGLUE") return FeatureMatcherType::kAlikedLightGlue;
  throw std::invalid_argument("Unsupported FeatureMatching.type: " + value);
}

inline void PrintSfmOptionsHelp(std::ostream& out, SfmOptionScope scope) {
  if (scope == SfmOptionScope::kModel) return;
  out << "\nCOLMAP 4.0 options (no legacy aliases):\n";
  if (scope == SfmOptionScope::kFeatures || scope == SfmOptionScope::kPipeline) {
    out << "  --FeatureExtraction.use_gpu <0|1> Default available GPU.\n"
        << "  --FeatureExtraction.gpu_index <index> Single GPU; -1 selects device 0.\n"
        << "  --ImageReader.mask_path <dir> Per-image masks: relative/image.jpg.png or image.png.\n"
        << "  --ImageReader.camera_mask_path <file> Global mask; black pixels excluded.\n"
        << "  --FeatureExtraction.max_image_size <px> Non-positive: SIFT 3200 / ALIKED 1600.\n"
        << "  --SiftExtraction.first_octave <-1|0> Default -1.\n"
        << "  --SiftExtraction.num_octaves <n> Default 4.\n"
        << "  --SiftExtraction.octave_resolution <n> Default 3.\n"
        << "  --SiftExtraction.peak_threshold <value> Default 0.02/3.\n"
        << "  --SiftExtraction.edge_threshold <value> Default 10.\n"
        << "  --SiftExtraction.max_num_orientations <1..4> Default 2.\n"
        << "  --descriptor_normalization <l1_root|l2> Default l1_root.\n"
        << "  --ImageReader.single_camera <0|1>\n"
        << "  --ImageReader.single_camera_per_folder <0|1>\n"
        << "  --ImageReader.single_camera_per_image <0|1>\n"
        << "  --AlikedExtraction.max_num_features <n>\n"
        << "  --AlikedExtraction.n32_model_path <path>\n";
  }
  if (scope == SfmOptionScope::kMatching || scope == SfmOptionScope::kPipeline) {
    out << "  --FeatureMatching.use_gpu <0|1> Default available GPU for SIFT.\n"
        << "  --FeatureMatching.gpu_index <index> Single GPU; -1 selects device 0.\n"
        << "  --FeatureMatching.num_threads <-1|positive> Matching and geometry thread budget.\n"
        << "  --FeatureMatching.type SIFT_BRUTEFORCE|SIFT_LIGHTGLUE|ALIKED_BRUTEFORCE|ALIKED_LIGHTGLUE\n"
        << "  --AlikedMatching.brute_force_max_ratio <value>\n"
        << "  --AlikedMatching.brute_force_cross_check <0|1>\n"
        << "  --AlikedMatching.lightglue_min_score <value>\n"
        << "  --AlikedMatching.lightglue_model_path <path>\n"
        << "  --TwoViewGeometry.max_error <px>\n"
        << "  --TwoViewGeometry.min_num_inliers <n>\n"
        << "  --TwoViewGeometry.confidence <probability>\n"
        << "  --TwoViewGeometry.max_num_trials <n> Default 10000 (>=100).\n"
        << "  --TwoViewGeometry.min_inlier_ratio <0..1> RANSAC trial prior; default 0.25.\n"
        << "  --TwoViewGeometry.random_seed <-1|nonnegative> Default -1 (random).\n"
        << "  --TwoViewGeometry.compute_relative_pose <0|1> Default 0; mapper recovers on demand.\n"
        << "  --TwoViewGeometry.detect_watermark <0|1> Default 1.\n"
        << "  --TwoViewGeometry.watermark_detection_max_error <px> Default 4.\n"
        << "  --TwoViewGeometry.filter_stationary_matches <0|1> Default 0.\n"
        << "  --TwoViewGeometry.stationary_matches_max_error <px> Default 4.\n"
        << "  --TwoViewGeometry.multiple_models <0|1> Default 0.\n"
        << "  --TwoViewGeometry.multiple_ignore_watermark <0|1> Default 1.\n";
  }
  if (scope == SfmOptionScope::kMatching) {
    out << "  --SiftMatching.max_distance <value> Default 0.7.\n"
        << "  --SiftMatching.cross_check <0|1> Default 1.\n";
  }
  if (scope == SfmOptionScope::kMapper || scope == SfmOptionScope::kPipeline) {
    out << "  --Mapper.init_image_id1 <int> Seed image ID; default -1 (automatic).\n"
        << "  --Mapper.init_image_id2 <int> Seed image ID; default -1. A single specified ID fixes the first image.\n"
        << "  --Mapper.init_max_reg_trials <int> Automatic first-seed registration limit per image; default 2.\n";
    out << "  --Mapper.num_threads <-1|0|positive> Mapper CPU budget; non-positive selects hardware.\n";
    out << "  --Mapper.ba_local_max_refinements <int> Local BA/refine passes; default 2, positive.\n"
        << "  --Mapper.ba_local_max_refinement_change <number> Stop below changed/adjusted observations; default 0.001.\n";
    out << "  --Mapper.ba_global_frames_ratio <number> Global BA frame growth ratio; default 1.1, > 1.\n"
        << "  --Mapper.ba_global_points_ratio <number> Global BA point growth ratio; default 1.1, > 1.\n"
        << "  --Mapper.ba_global_frames_freq <int> Global BA absolute frame growth; default 500, positive.\n"
        << "  --Mapper.ba_global_points_freq <int> Global BA absolute point growth; default 250000, positive.\n";
    for (const char* name : {"ba_refine_focal_length", "ba_refine_principal_point", "ba_refine_extra_params", "max_reg_trials", "init_num_trials", "structure_less_registration_fallback",
         "random_seed", "init_min_num_inliers", "init_max_error", "init_max_forward_motion", "init_min_tri_angle", "abs_pose_max_error",
         "abs_pose_min_num_inliers", "abs_pose_min_inlier_ratio", "filter_max_reproj_error",
         "filter_min_tri_angle", "ba_local_num_images", "ba_local_max_num_iterations",
         "ba_global_max_num_iterations", "ba_global_max_refinements", "ba_global_max_refinement_change",
         "ba_local_function_tolerance", "ba_global_function_tolerance",
         "ba_min_num_residuals_for_cpu_multi_threading",
         "tri_max_transitivity", "tri_create_max_angle_error",
         "tri_continue_max_angle_error", "tri_merge_max_reproj_error", "tri_complete_max_reproj_error",
         "tri_complete_max_transitivity", "tri_min_angle", "tri_ignore_two_view_tracks"}) {
      if (scope == SfmOptionScope::kPipeline &&
          (std::string(name).rfind("ba_refine_", 0) == 0 ||
           std::string(name) == "max_reg_trials" || std::string(name) == "init_num_trials" ||
           std::string(name) == "structure_less_registration_fallback")) continue;
      out << "  --Mapper." << name << " <value>\n";
    }
  }
}

inline void PrintBundleAdjustmentOptionsHelp(std::ostream& out) {
  const auto defaults = controllers::ColmapStandaloneBundleAdjustmentDefaults();
  const auto& solver = defaults.solver;
  out << "  --BundleAdjustment.refine_focal_length <0|1> Default 1.\n"
      << "  --BundleAdjustment.refine_principal_point <0|1> Default 0.\n"
      << "  --BundleAdjustment.refine_extra_params <0|1> Default 1.\n"
      << "  --BundleAdjustment.refine_rig_from_world <0|1> Default 1; single-camera image poses.\n"
      << "  --BundleAdjustment.refine_sensor_from_rig <0|1> Default 1; trivial reference sensor is fixed.\n"
      << "  --BundleAdjustment.refine_points3D <0|1> Default 1.\n"
      << "  --BundleAdjustment.constant_rig_from_world_rotation <0|1> Default 0.\n"
      << "  --BundleAdjustment.min_track_length <integer> Default 0.\n"
      << "  --BundleAdjustmentCeres.max_num_iterations <integer> Default " << defaults.max_num_iterations << ".\n"
      << "  --BundleAdjustmentCeres.max_linear_solver_iterations <integer> Default " << solver.max_linear_solver_iterations << ".\n"
      << "  --BundleAdjustmentCeres.function_tolerance <number> Default " << solver.function_tolerance << ".\n"
      << "  --BundleAdjustmentCeres.gradient_tolerance <number> Default " << solver.gradient_tolerance << ".\n"
      << "  --BundleAdjustmentCeres.parameter_tolerance <number> Default " << solver.parameter_tolerance << ".\n"
      << "  --BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading <integer> Default " << solver.min_num_residuals_for_cpu_multi_threading << ".\n"
      << "  --BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver <integer> Default " << solver.max_num_images_direct_dense_cpu_solver << ".\n"
      << "  --BundleAdjustmentCeres.max_num_images_direct_sparse_cpu_solver <integer> Default " << solver.max_num_images_direct_sparse_cpu_solver << ".\n"
      << "  --BundleAdjustmentCeres.use_gpu <0> CPU BA only; requesting GPU is rejected.\n"
      << "  --Colite.ref_images_path <file> Optional camera-center priors.\n"
      << "  --Colite.position_prior_stddev <number> Default 5.\n"
      << "  --Colite.position_prior_loss_scale <number> Default 2.\n"
      << "  --Colite.position_prior_alignment_max_error <number> Default 20.\n"
      << "  --Colite.input_format auto|bin|txt; --Colite.output_format both|bin|txt\n";
}

}  // namespace colite::exe
