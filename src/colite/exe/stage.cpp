#include "colite/exe/stage.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <glog/logging.h>

namespace colite::exe {
namespace {

void HashBytes(std::uint64_t* hash, const std::string& value) {
  for (const unsigned char byte : value) {
    *hash ^= byte;
    *hash *= 1099511628211ULL;
  }
}

std::string Hex(std::uint64_t value) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}

std::string FingerprintPaths(
    const std::vector<std::filesystem::path>& roots) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto& root : roots) {
    std::vector<std::filesystem::path> paths{root};
    std::error_code error;
    if (std::filesystem::is_directory(root, error)) {
      for (std::filesystem::recursive_directory_iterator it(root, error), end;
           !error && it != end; it.increment(error)) {
        paths.push_back(it->path());
      }
    }
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) {
      HashBytes(&hash, path.lexically_normal().generic_string());
      if (std::filesystem::is_regular_file(path, error)) {
        HashBytes(&hash, std::to_string(std::filesystem::file_size(path, error)));
        HashBytes(&hash, std::to_string(
                             std::filesystem::last_write_time(path, error)
                                 .time_since_epoch().count()));
      }
    }
  }
  return Hex(hash);
}

std::string FingerprintArguments(const std::vector<std::string>& arguments) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto& argument : arguments) HashBytes(&hash, argument + "\n");
  return Hex(hash);
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  for (const char c : value) {
    if (c == '\\' || c == '"') escaped.push_back('\\');
    escaped.push_back(c);
  }
  return escaped;
}

std::string ReadJsonString(const std::filesystem::path& path,
                           const std::string& key) {
  std::ifstream file(path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  const std::string marker = "\"" + key + "\": \"";
  const std::size_t begin = text.find(marker);
  if (begin == std::string::npos) return {};
  const std::size_t value_begin = begin + marker.size();
  const std::size_t end = text.find('"', value_begin);
  return end == std::string::npos ? std::string{}
                                  : text.substr(value_begin, end - value_begin);
}

bool OutputsExist(const StageSpec& spec) {
  std::error_code error;
  return !spec.outputs.empty() &&
         std::all_of(spec.outputs.begin(), spec.outputs.end(),
                     [&](const auto& path) {
                       if (std::filesystem::is_regular_file(path, error)) {
                         return std::filesystem::file_size(path, error) > 0 &&
                                !error;
                       }
                       return std::filesystem::is_directory(path, error) &&
                              !std::filesystem::is_empty(path, error) && !error;
                     });
}

void WriteManifest(const StageSpec& spec, const std::string& status,
                   const std::string& input_fingerprint,
                   const std::string& options_fingerprint,
                   double elapsed_seconds) {
  const auto path = spec.output_root / ".colite" / "stages" /
                    (spec.name + ".json");
  std::filesystem::create_directories(path.parent_path());
  const std::filesystem::path temporary = path.string() + ".tmp";
  std::ofstream file(temporary);
  file << "{\n"
       << "  \"schema_version\": 2,\n"
       << "  \"stage\": \"" << JsonEscape(spec.name) << "\",\n"
       << "  \"status\": \"" << status << "\",\n"
       << "  \"input_fingerprint\": \"" << input_fingerprint << "\",\n"
       << "  \"options_fingerprint\": \"" << options_fingerprint
       << "\",\n"
       << "  \"output_fingerprint\": \""
       << (status == "complete" ? FingerprintPaths(spec.outputs) : "")
       << "\",\n"
       << "  \"elapsed_seconds\": " << std::setprecision(9)
       << elapsed_seconds << ",\n  \"outputs\": [";
  for (std::size_t i = 0; i < spec.outputs.size(); ++i) {
    if (i) file << ", ";
    file << "\"" << JsonEscape(spec.outputs[i].generic_string()) << "\"";
  }
  file << "]\n}\n";
  file.close();
  if (!file) {
    throw std::runtime_error("Failed to write stage manifest: " + path.string());
  }
  // Publish only a complete manifest. Windows needs an explicit replacement
  // when rename cannot overwrite the old file; an absent manifest is safe.
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(path);
    std::filesystem::rename(temporary, path);
  }
}

}  // namespace

StageInputState InspectStageInputs(const StageSpec& spec) {
  const auto manifest = spec.output_root / ".colite" / "stages" / (spec.name + ".json");
  if (!std::filesystem::exists(manifest)) return StageInputState::kUnrecorded;
  return ReadJsonString(manifest, "input_fingerprint") == FingerprintPaths(spec.inputs) &&
                 ReadJsonString(manifest, "options_fingerprint") == FingerprintArguments(spec.arguments)
      ? StageInputState::kMatching : StageInputState::kChanged;
}

int RunStage(const StageSpec& spec, const StageOptions& options,
             const std::function<int()>& runner, std::ostream& out,
             std::ostream& err) {
  LOG(INFO) << "Stage " << spec.name << ": output="
            << spec.output_root.string() << " resume="
            << (options.resume ? 1 : 0) << " force="
            << (options.force ? 1 : 0) << " dry_run="
            << (options.dry_run ? 1 : 0);
  const std::string input_fingerprint = FingerprintPaths(spec.inputs);
  const std::string options_fingerprint = FingerprintArguments(spec.arguments);
  const auto manifest = spec.output_root / ".colite" / "stages" /
                        (spec.name + ".json");
  if (options.resume && !options.force && OutputsExist(spec) &&
      ReadJsonString(manifest, "status") == "complete" &&
      ReadJsonString(manifest, "input_fingerprint") == input_fingerprint &&
      ReadJsonString(manifest, "options_fingerprint") == options_fingerprint &&
      ReadJsonString(manifest, "output_fingerprint") ==
          FingerprintPaths(spec.outputs)) {
    out << "SKIP " << spec.name << " (manifest and outputs are current)\n";
    LOG(INFO) << "Stage " << spec.name << " skipped: cache current";
    return 0;
  }
  if (options.dry_run) {
    out << "PLAN " << spec.name << "\n";
    LOG(INFO) << "Stage " << spec.name << " planned only";
    return 0;
  }
  out << "RUN " << spec.name << "\n";
  const auto begin = std::chrono::steady_clock::now();
  try {
    int result = runner();
    if (result == 0 && !OutputsExist(spec)) {
      err << "Stage did not produce all required outputs: " << spec.name << "\n";
      result = 1;
    }
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
    WriteManifest(spec, result == 0 ? "complete" : "failed",
                  input_fingerprint, options_fingerprint, elapsed);
    LOG(INFO) << "Stage " << spec.name << " complete: exit_code=" << result
              << " elapsed_seconds=" << elapsed;
    if (result != 0) err << "Stage failed: " << spec.name << "\n";
    return result;
  } catch (...) {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - begin)
                               .count();
    WriteManifest(spec, "failed", input_fingerprint, options_fingerprint,
                  elapsed);
    LOG(ERROR) << "Stage " << spec.name
               << " failed with exception after " << elapsed << " seconds";
    throw;
  }
}

}  // namespace colite::exe
