#include "colite/exe/logging.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <glog/logging.h>

namespace colite::exe {
namespace {

bool ParseBool(const std::string& value, const std::string& option) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (normalized == "1" || normalized == "true") return true;
  if (normalized == "0" || normalized == "false") return false;
  throw std::invalid_argument(option + " expects 0/1 or true/false");
}

int ParseInt(const std::string& value, const std::string& option) {
  std::size_t consumed = 0;
  const int number = std::stoi(value, &consumed);
  if (consumed != value.size()) throw std::invalid_argument(option + " expects an int32");
  return number;
}

}  // namespace

std::vector<std::string> InitializeLoggingAndBuildArgs(const int argc, char** argv) {
  if (argc <= 0 || argv == nullptr || argv[0] == nullptr)
    throw std::invalid_argument("Invalid process arguments");

  // COLMAP 4 distinguishes verbose detail from minimum severity.
  std::string target = "stderr_and_file", path;
  int verbosity = 0, severity = 0;
  bool color = true;
  std::vector<std::string> args{argv[0]};
  args.reserve(static_cast<std::size_t>(argc));
  for (int idx = 1; idx < argc; ++idx) {
    const std::string token = argv[idx];
    const auto equals = token.find('=');
    const auto option = token.substr(0, equals);
    const bool known = option == "--log_target" || option == "--log_path" ||
        option == "--log_level" || option == "--log_severity" || option == "--log_color";
    if (!known) { args.push_back(token); continue; }
    const auto value = [&]() -> std::string {
      if (equals != std::string::npos) return token.substr(equals + 1);
      if (++idx >= argc) throw std::invalid_argument("Missing value for " + option);
      return argv[idx];
    }();
    if (option == "--log_target") target = value;
    else if (option == "--log_path") path = value;
    else if (option == "--log_level") verbosity = ParseInt(value, option);
    else if (option == "--log_severity") severity = ParseInt(value, option);
    else color = ParseBool(value, option);
  }
  if (target != "stderr" && target != "stdout" && target != "file" && target != "stderr_and_file")
    throw std::invalid_argument("--log_target expects stderr, stdout, file or stderr_and_file");
  if (severity < 0 || severity > 3)
    throw std::invalid_argument("--log_severity must be in [0, 3]");
  if (!path.empty() && (target == "file" || target == "stderr_and_file"))
    std::filesystem::create_directories(path);

  FLAGS_log_dir = path;
  FLAGS_logtostderr = target == "stderr";
  FLAGS_alsologtostderr = target == "stderr_and_file";
#ifdef COLITE_GLOG_HAS_STDOUT
  FLAGS_logtostdout = target == "stdout";
  FLAGS_colorlogtostdout = color;
#else
  if (target == "stdout") FLAGS_logtostderr = true;
#endif
  FLAGS_colorlogtostderr = color;
  FLAGS_minloglevel = severity;
  FLAGS_v = verbosity;
  google::InitGoogleLogging(argv[0]);
#ifndef COLITE_GLOG_HAS_STDOUT
  if (target == "stdout") LOG(WARNING) << "log_target=stdout requires glog >= 0.6; using stderr";
#endif
  return args;
}

void ShutdownLogging() { google::ShutdownGoogleLogging(); }

int RunLoggedCommand(int argc, char** argv,
    const std::function<int(const std::vector<std::string>&)>& command) {
  bool initialized = false;
  try {
    const auto args = InitializeLoggingAndBuildArgs(argc, argv);
    initialized = true;
    const int result = command(args);
    ShutdownLogging();
    return result;
  } catch (const std::exception& error) {
    if (initialized) ShutdownLogging();
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace colite::exe
