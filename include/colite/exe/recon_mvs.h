#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "colite/controllers/recon_mvs_controller.h"

namespace colite::exe {

enum class MvsOptionScope { kPipeline, kUndistorter, kPatchMatch, kFusion };

struct ReconMvsCliOptions : controllers::ReconMvsControllerOptions {
  ReconMvsCliOptions() {
    patch_match_num_threads = fusion_num_threads = undistort_num_threads = -1;
    patch_match_max_image_size = fusion_max_image_size = -1;
  }
  bool show_help = false;
  bool show_version = false;
  std::filesystem::path workspace_path;
  std::string workspace_format = "COLMAP";
  std::string undistort_output_type = "COLMAP";
};

// Shared semantic validation, without reading inputs or creating outputs.
void ValidateReconMvsArguments(const std::vector<std::string>& args);
ReconMvsCliOptions ParseReconMvsArguments(
    const std::vector<std::string>& args,
    MvsOptionScope scope = MvsOptionScope::kPipeline);
void PrintMvsOptionsHelp(MvsOptionScope scope, std::ostream& out);
bool MvsOptionAppliesTo(const std::string& name, MvsOptionScope scope);
std::vector<std::string> NormalizeMvsArguments(
    const std::vector<std::string>& args);

int RunReconMvs(const std::vector<std::string>& args,
                std::ostream& out,
                std::ostream& err);

}  // namespace colite::exe
