#include "colite/exe/mvs_stages.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "colite/core/version.h"
#include "colite/exe/recon_mvs.h"
#include "colite/exe/stage.h"
#include "colite/mvs/patch_match.h"

namespace colite::exe {
namespace {
int RunMvsStage(const std::vector<std::string>& argv, MvsOptionScope scope,
                const std::string& name, std::ostream& out, std::ostream& err) {
  try {
    const auto expanded = NormalizeMvsArguments(argv);
    StageOptions control;
    std::vector<std::string> arguments;
    if (!expanded.empty()) arguments.push_back(expanded.front());
    for (std::size_t i = 1; i < expanded.size(); ++i) {
      if (expanded[i] == "--resume") control.resume = true;
      else if (expanded[i] == "--force") control.force = true;
      else if (expanded[i] == "--dry_run") control.dry_run = true;
      else arguments.push_back(expanded[i]);
    }
    if (control.resume && control.force) throw std::invalid_argument("--resume and --force are mutually exclusive");
    auto options = ParseReconMvsArguments(arguments, scope);
    options.force_recompute = control.force;
    if (options.show_help) { PrintMvsOptionsHelp(scope, out); return 0; }
    if (options.show_version) { out << colite::Version() << "\n"; return 0; }
    StageSpec spec{name, options.output_path,
                   {options.model_path, options.image_path}, {}, arguments};
    if (scope == MvsOptionScope::kUndistorter) {
      spec.outputs = {options.output_path / "images", options.output_path / "sparse",
                      options.output_path / "stereo/patch-match.cfg",
                      options.output_path / "stereo/fusion.cfg"};
      spec.arguments.push_back("@implementation=image_undistorter_v4");
    } else if (scope == MvsOptionScope::kPatchMatch) {
      spec.outputs.push_back(options.output_path / ".colite/photometric_cache.txt");
      for (const char* category : {"depth_maps", "normal_maps"}) {
        spec.outputs.push_back(options.output_path / "stereo" / category);
      }
      if (options.write_cost_maps || options.write_debug_maps) spec.outputs.push_back(options.output_path / "stereo/cost_maps");
      if (options.patch_match_write_consistency_graph) spec.outputs.push_back(options.output_path / "stereo/consistency_graphs");
      spec.inputs.push_back(options.patch_match_config_path.empty()
          ? options.output_path / "stereo/patch-match.cfg" : options.patch_match_config_path);
      spec.arguments.push_back(std::string("@implementation=") + mvs::kPatchMatchImplementationVersion);
    } else {
      spec.inputs.push_back(options.output_path / "stereo");
      spec.outputs = {options.fused_output_path,
                      options.fused_output_path.string() + ".vis"};
      spec.arguments.push_back("@implementation=stereo_fusion_v10");
    }
    // The controller owns photometric cache validation for every entry point.
    // A stage manifest governs whole-stage resume, not borrowed photo maps.
    return RunStage(spec, control,
                    [&] { return controllers::ReconMvsController(options).Run(out); }, out, err);
  } catch (const std::exception& exception) {
    err << "ERROR: " << exception.what() << "\n";
    return 1;
  }
}
}  // namespace
int RunImageUndistorter(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  return RunMvsStage(args, MvsOptionScope::kUndistorter, "image_undistorter", out, err);
}
int RunPatchMatchStereo(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  return RunMvsStage(args, MvsOptionScope::kPatchMatch, "patch_match_stereo", out, err);
}
int RunStereoFusion(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
  return RunMvsStage(args, MvsOptionScope::kFusion, "stereo_fusion", out, err);
}
}  // namespace colite::exe
