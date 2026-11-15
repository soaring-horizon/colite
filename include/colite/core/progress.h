#pragma once
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace colite {
// Optional, versioned progress events. Default CLI output remains unchanged.
inline void ReportProgress(const std::string& phase, std::size_t done,
                           std::size_t total, const std::string& detail = {}) {
  static const bool enabled = [] {
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, "COLITE_PROGRESS")) return false;
    const bool result = value && std::string(value) == "1";
    std::free(value);
    return result;
#else
    const char* value = std::getenv("COLITE_PROGRESS");
    return value && std::string(value) == "1";
#endif
  }();
  if (!enabled) return;
  auto quoted = [](const std::string& value) {
    std::ostringstream out;
    out << '"';
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
      if (c == '"' || c == '\\') out << '\\' << char(c);
      else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
      else out << char(c);
    }
    out << '"';
    return out.str();
  };
  std::ostringstream event;
  event << "@colite.progress {\"schema\":1,\"phase\":" << quoted(phase)
        << ",\"done\":" << done << ",\"total\":" << total
        << ",\"detail\":" << quoted(detail) << "}\n";
  static std::mutex mutex;
  const std::lock_guard<std::mutex> guard(mutex);
  std::cerr << event.str() << std::flush;
}

// Serialize count assignment and emission so concurrent workers cannot report
// completion out of order. Limit high-volume pair progress to ten events/sec.
class ProgressCounter {
 public:
  ProgressCounter(std::string phase, std::size_t total)
      : phase_(std::move(phase)), total_(total), last_(std::chrono::steady_clock::now()) {
    ReportProgress(phase_, 0, total_);
  }
  void Advance() {
    const std::lock_guard<std::mutex> guard(mutex_);
    ++done_;
    const auto now = std::chrono::steady_clock::now();
    if (done_ == total_ || now - last_ >= std::chrono::milliseconds(100)) {
      ReportProgress(phase_, done_, total_);
      last_ = now;
    }
  }
 private:
  std::string phase_;
  std::size_t total_, done_ = 0;
  std::chrono::steady_clock::time_point last_;
  std::mutex mutex_;
};
}
