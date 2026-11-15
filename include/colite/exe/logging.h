#pragma once

#include <functional>
#include <string>
#include <vector>

namespace colite::exe {

// Initializes glog and removes Colite-wide logging options from argv before
// command-specific parsing. Supported options may appear before or after the
// subcommand: --log_target, --log_path, --log_level, --log_severity, --log_color.
std::vector<std::string> InitializeLoggingAndBuildArgs(int argc, char** argv);

void ShutdownLogging();

// Guard every installed entry point, including logging initialization failures.
int RunLoggedCommand(int argc, char** argv,
    const std::function<int(const std::vector<std::string>&)>& command);

}  // namespace colite::exe
