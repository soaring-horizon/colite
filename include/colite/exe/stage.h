#pragma once

#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

struct StageOptions {
  bool resume = false;
  bool force = false;
  bool dry_run = false;
};

struct StageSpec {
  std::string name;
  std::filesystem::path output_root;
  std::vector<std::filesystem::path> inputs;
  std::vector<std::filesystem::path> outputs;
  std::vector<std::string> arguments;
};

enum class StageInputState { kUnrecorded, kMatching, kChanged };
// Distinguish external inputs from outputs owned by an older configuration.
StageInputState InspectStageInputs(const StageSpec& spec);

int RunStage(const StageSpec& spec, const StageOptions& options,
             const std::function<int()>& runner, std::ostream& out,
             std::ostream& err);

}  // namespace colite::exe
