#include "colite/exe/recon_auto.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "colite/exe/mvs_stages.h"
#include "colite/exe/mvs_tools.h"
#include "colite/exe/recon_sfm.h"
#include "colite/exe/recon_mvs.h"
#include "colite/controllers/sfm_options.h"
#include "colite/features/feature_mask.h"
#include "colite/features/sift_extractor.h"
#include "colite/exe/stage.h"

namespace colite::exe {
namespace {

std::string Require(const std::map<std::string, std::string>& values,
                    const std::string& name) {
  const auto it = values.find(name);
  if (it == values.end()) throw std::invalid_argument("Missing required option " + name);
  return it->second;
}

bool BooleanValue(const std::map<std::string, std::string>& values,
                  const std::string& name, bool fallback) {
  const auto it = values.find(name);
  if (it == values.end()) return fallback;
  if (it->second == "1" || it->second == "true" || it->second == "yes") {
    return true;
  }
  if (it->second == "0" || it->second == "false" || it->second == "no") {
    return false;
  }
  throw std::invalid_argument("Invalid boolean value for " + name + ": " +
                              it->second);
}

void AppendOptionsWithPrefix(
    const std::map<std::string, std::string>& values,
    const std::string& prefix, std::vector<std::string>* arguments,
    bool remove_prefix) {
  for (const auto& [name, value] : values) {
    if (name.rfind(prefix, 0) != 0) continue;
    arguments->push_back(remove_prefix ? "--" + name.substr(prefix.size())
                                       : name);
    arguments->push_back(value);
  }
}

void AppendMvsOptions(const std::map<std::string, std::string>& values,
                      MvsOptionScope scope, std::vector<std::string>* args) {
  for (const auto& [name, value] : values) {
    if (name.rfind("--mvs.", 0) != 0) continue;
    const std::string option = "--" + name.substr(6);
    if (MvsOptionAppliesTo(option, scope)) {
      args->insert(args->end(), {option, value});
    }
  }
}

void AppendTextureOptions(const std::map<std::string, std::string>& values,
                           std::vector<std::string>* args) {
  AppendOptionsWithPrefix(values, "--MeshTextureMapping.", args, false);
  for (const auto& [name, value] : values) {
    if (name.rfind("--Colite.texture_", 0) != 0) continue;
    args->insert(args->end(), {name == "--Colite.texture_output_type" ? "--output_type" : name, value});
  }
}

}  // namespace

int RunReconAuto(const std::vector<std::string>& raw_args,
                 std::ostream& out, std::ostream& err) {
  try {
    const auto args = NormalizeMvsArguments(raw_args);
    std::map<std::string, std::string> values;
    StageOptions stage;
    bool help = false;
    for (std::size_t i = 1; i < args.size(); ++i) {
      if (args[i] == "--help" || args[i] == "-h") help = true;
      else if (args[i] == "--resume") stage.resume = true;
      else if (args[i] == "--force") stage.force = true;
      else if (args[i] == "--dry_run") stage.dry_run = true;
      else {
        const std::string option = args[i];
        if (option.rfind("--", 0) != 0 || ++i >= args.size()) {
          throw std::invalid_argument("Expected --option value");
        }
        values[option] = args[i];
      }
    }
    if (help) {
      out << "Usage: recon_auto --image_path DIR --output_path DIR "
             "[--start_from sfm|undistort|stereo|fusion|mesh|simplify|texture] "
             "[--stop_after sfm|undistort|stereo|fusion|mesh|simplify|texture] "
             "[--mesher poisson|delaunay|none] [--simplify 0|1] "
             "[--texture 0|1] "
             "[--sfm.ImageReader.camera_model SIMPLE_RADIAL] "
             "[--sfm.ImageReader.camera_params CSV] "
             "[--sfm.ImageReader.single_camera 0|1] "
             "[--sfm.<option> value] [--mvs.<option> value] "
             "[--resume|--force] [--dry_run]\n";
      return 0;
    }
    if (stage.resume && stage.force) {
      throw std::invalid_argument("--resume and --force are mutually exclusive");
    }
    for (const auto& [name, value] : values) {
      if (name == "--image_path" || name == "--output_path" ||
          name == "--start_from" || name == "--stop_after" ||
          name == "--mesher" || name == "--simplify" || name == "--texture" ||
          name.rfind("--sfm.", 0) == 0 || name.rfind("--PoissonMeshing.", 0) == 0 ||
          name.rfind("--DelaunayMeshing.", 0) == 0 || name.rfind("--MeshSimplification.", 0) == 0 ||
          name.rfind("--MeshTextureMapping.", 0) == 0 || name.rfind("--Colite.texture_", 0) == 0) continue;
      if (name.rfind("--mvs.", 0) != 0) throw std::invalid_argument("Unknown argument: " + name);
      const std::string option = "--" + name.substr(6);
      if (option == "--input_path" || option == "--output_path" || option == "--image_path" ||
          option == "--workspace_path" ||
          !(MvsOptionAppliesTo(option, MvsOptionScope::kUndistorter) ||
            MvsOptionAppliesTo(option, MvsOptionScope::kPatchMatch) ||
            MvsOptionAppliesTo(option, MvsOptionScope::kFusion))) {
        throw std::invalid_argument("Unknown or reserved pipeline argument: " + name);
      }
    }
    const std::filesystem::path images = Require(values, "--image_path");
    const std::filesystem::path root = Require(values, "--output_path");
    const std::vector<std::string> names{"sfm", "undistort", "stereo",
                                         "fusion", "mesh", "simplify",
                                         "texture"};
    const std::string start = values.count("--start_from") ? values["--start_from"] : "sfm";
    const std::string stop = values.count("--stop_after") ? values["--stop_after"] : "texture";
    const std::string mesher = values.count("--mesher")
                                   ? values["--mesher"]
                                   : "poisson";
    const bool simplify = BooleanValue(values, "--simplify", true);
    const bool texture = BooleanValue(values, "--texture", true);
    if (mesher != "poisson" && mesher != "delaunay" && mesher != "none") {
      throw std::invalid_argument("--mesher must be poisson, delaunay, or none");
    }
    const auto start_it = std::find(names.begin(), names.end(), start);
    const auto stop_it = std::find(names.begin(), names.end(), stop);
    if (start_it == names.end() || stop_it == names.end() || start_it > stop_it) {
      throw std::invalid_argument("Invalid --start_from/--stop_after stage range");
    }
    auto controls = [&] {
      std::vector<std::string> result;
      if (stage.resume) result.push_back("--resume");
      if (stage.force) result.push_back("--force");
      if (stage.dry_run) result.push_back("--dry_run");
      return result;
    };
    auto enabled = [&](const std::string& name) {
      const auto it = std::find(names.begin(), names.end(), name);
      return it >= start_it && it <= stop_it;
    };
    const auto sfm = root / "sfm";
    const auto mvs = root / "mvs";
    const auto dense_workspace = mvs;
    // Validate every MVS option before starting SfM, including disabled stages.
    for (const auto scope : {MvsOptionScope::kUndistorter, MvsOptionScope::kPatchMatch,
                             MvsOptionScope::kFusion}) {
      std::vector<std::string> validation{args.front(), "--input_path", sfm.string(),
          "--image_path", images.string(), "--output_path",
          (scope == MvsOptionScope::kFusion ? mvs / "fused.ply" : mvs).string()};
      if (scope == MvsOptionScope::kFusion) {
        validation.insert(validation.end(), {"--workspace_path", mvs.string()});
      }
      AppendMvsOptions(values, scope, &validation);
      (void)ParseReconMvsArguments(validation, scope);
    }
    // Reject all surface configuration errors before SfM, even when the
    // corresponding stage is outside the selected execution range.
    for (const auto& [tool, prefix] :
         std::vector<std::pair<std::string, std::string>>{
             {"poisson_mesher", "--PoissonMeshing."},
             {"delaunay_mesher", "--DelaunayMeshing."},
             {"mesh_simplifier", "--MeshSimplification."},
             {"mesh_texturer", "--MeshTextureMapping."}}) {
      std::vector<std::string> validation{args.front(), "--input_path",
          (root / "surface/input.ply").string(), "--output_path",
          (root / "surface/output.ply").string()};
      if (tool == "mesh_texturer") {
        validation.insert(validation.end(), {"--workspace_path", mvs.string()});
        AppendTextureOptions(values, &validation);
      } else {
        AppendOptionsWithPrefix(values, prefix, &validation, false);
      }
      ValidateSurfaceArguments(tool, validation);
    }
    if (enabled("sfm")) {
      std::vector<std::string> sfm_args{args.front(), "--image_path", images.string(),
                                        "--output_path", sfm.string()};
      AppendOptionsWithPrefix(values, "--sfm.", &sfm_args, true);
      auto identity_arguments = sfm_args;
      identity_arguments.push_back(std::string("@implementation=recon_auto_sfm_cache_v2:") +
          controllers::SfmMappingImplementationKey());
      if (stage.resume) sfm_args.push_back("--resume");
      if (stage.force) sfm_args.push_back("--force");
      ValidateReconSfmArguments(sfm_args);
      ImageMaskOptions masks;
      if (values.count("--sfm.ImageReader.mask_path")) masks.mask_path = values.at("--sfm.ImageReader.mask_path");
      if (values.count("--sfm.ImageReader.camera_mask_path")) masks.camera_mask_path = values.at("--sfm.ImageReader.camera_mask_path");
      if (!masks.Empty()) {
        identity_arguments.push_back("@image_masks=" +
            FeatureMasks(images, ListImageFiles(images), masks).Fingerprint());
      }
      const ColmapModelFormat sfm_format = ColmapModelFormatFromString(
          values.count("--sfm.Colite.output_format")
              ? values.at("--sfm.Colite.output_format")
              : "both");
      std::vector<std::filesystem::path> sfm_outputs{
          sfm / "points3D.ply", sfm / "models", sfm / "models.txt", sfm / "summary.txt",
          sfm / "workspace/features", sfm / "workspace/cameras.txt", sfm / "workspace/images.tsv",
          sfm / "workspace/matches/raw_matches.tsv", sfm / "workspace/matches/verified_matches.tsv",
          sfm / "workspace/cache/features.txt", sfm / "workspace/cache/matching.txt",
          sfm / "workspace/cache/geometry.txt", sfm / "workspace/cache/recon_sfm.txt"};
      if (sfm_format == ColmapModelFormat::kText ||
          sfm_format == ColmapModelFormat::kTextAndBinary) {
        sfm_outputs.insert(sfm_outputs.end(),
                           {sfm / "cameras.txt", sfm / "images.txt",
                            sfm / "points3D.txt"});
      }
      if (sfm_format == ColmapModelFormat::kBinary ||
          sfm_format == ColmapModelFormat::kTextAndBinary) {
        sfm_outputs.insert(sfm_outputs.end(),
                           {sfm / "cameras.bin", sfm / "images.bin",
                            sfm / "points3D.bin"});
      }
      StageSpec spec{"recon_sfm", root, {images}, sfm_outputs, identity_arguments};
      const int result = RunStage(spec, stage,
          [&] { return RunReconSfm(sfm_args, out, err); }, out, err);
      if (result) return result;
    }
    if (enabled("undistort")) {
      std::vector<std::string> command{args.front(), "--image_path", images.string(),
          "--input_path", sfm.string(), "--output_path", mvs.string()};
      AppendMvsOptions(values, MvsOptionScope::kUndistorter, &command);
      const auto extra = controls(); command.insert(command.end(), extra.begin(), extra.end());
      const int result = RunImageUndistorter(command, out, err);
      if (result) return result;
    }
    if (enabled("stereo")) {
      std::vector<std::string> command{args.front(), "--input_path",
          (dense_workspace / "sparse").string(), "--image_path",
          (dense_workspace / "images").string(), "--output_path", mvs.string()};
      AppendMvsOptions(values, MvsOptionScope::kPatchMatch, &command);
      const auto extra = controls(); command.insert(command.end(), extra.begin(), extra.end());
      const int result = RunPatchMatchStereo(command, out, err);
      if (result) return result;
    }
    if (enabled("fusion")) {
      std::vector<std::string> command{args.front(), "--input_path",
          (dense_workspace / "sparse").string(), "--image_path",
          (dense_workspace / "images").string(), "--workspace_path", mvs.string(),
          "--output_path", (mvs / "fused.ply").string()};
      AppendMvsOptions(values, MvsOptionScope::kFusion, &command);
      const auto extra = controls(); command.insert(command.end(), extra.begin(), extra.end());
      const int result = RunStereoFusion(command, out, err);
      if (result) return result;
    }
    const auto surface = root / "surface";
    const auto mesh_path = surface /
        (mesher == "delaunay" ? "meshed-delaunay.ply"
                              : "meshed-poisson.ply");
    const auto simplified_path = surface / "mesh-simplified.ply";
    if (enabled("mesh") && mesher != "none") {
      std::vector<std::string> command{args.front(), "--stage_root", root.string()};
      if (mesher == "poisson") {
        command.insert(command.end(), {"--input_path",
            (mvs / "fused.ply").string(), "--output_path",
            mesh_path.string()});
        AppendOptionsWithPrefix(values, "--PoissonMeshing.", &command, false);
      } else {
        command.insert(command.end(), {"--input_path", dense_workspace.string(),
            "--input_type", "dense", "--output_path", mesh_path.string()});
        AppendOptionsWithPrefix(values, "--DelaunayMeshing.", &command, false);
      }
      const auto extra = controls();
      command.insert(command.end(), extra.begin(), extra.end());
      const int result = mesher == "poisson"
                             ? RunPoissonMesher(command, out, err)
                             : RunDelaunayMesher(command, out, err);
      if (result) return result;
    }
    if (enabled("simplify") && mesher != "none" && simplify) {
      std::vector<std::string> command{
          args.front(), "--input_path", mesh_path.string(), "--output_path",
          simplified_path.string(), "--stage_root", root.string()};
      AppendOptionsWithPrefix(values, "--MeshSimplification.", &command, false);
      const auto extra = controls();
      command.insert(command.end(), extra.begin(), extra.end());
      const int result = RunMeshSimplifier(command, out, err);
      if (result) return result;
    }
    if (enabled("texture") && mesher != "none" && texture) {
      const auto texture_input = simplify ? simplified_path : mesh_path;
      std::vector<std::string> command{
          args.front(), "--input_path", texture_input.string(), "--output_path",
          (surface / "textured").string(), "--workspace_path",
          dense_workspace.string(), "--stage_root", root.string()};
      AppendTextureOptions(values, &command);
      const auto extra = controls();
      command.insert(command.end(), extra.begin(), extra.end());
      const int result = RunMeshTexturer(command, out, err);
      if (result) return result;
    }
    return 0;
  } catch (const std::exception& exception) {
    err << "ERROR: " << exception.what() << "\n";
    return 1;
  }
}

}  // namespace colite::exe
