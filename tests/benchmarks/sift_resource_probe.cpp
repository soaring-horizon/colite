// Benchmark-only wrapper around the production command. CUDA's async pool
// exposes allocation high-water marks without polling or instrumenting kernels.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include "colite/exe/command.h"
#include "colite/exe/logging.h"

int main(int argc, char** argv) {
  auto args = colite::exe::InitializeLoggingAndBuildArgs(argc, argv);
  try {
    std::string stats_path;
    bool gpu = false;
    int device = 0;
    for (std::size_t i = 1; i < args.size(); ++i) {
      if (args[i] == "--probe_stats_path") {
        if (i + 1 >= args.size()) throw std::invalid_argument("Missing probe stats path");
        stats_path = args[i + 1];
        args.erase(args.begin() + i, args.begin() + i + 2);
        --i;
      } else if (args[i] == "--Colite.feature_extractor" && i + 1 < args.size()) {
        gpu = args[i + 1] == "cuda_vlfeat" || args[i + 1] == "cuda_opencv";
      } else if (args[i] == "--FeatureExtraction.gpu_index" && i + 1 < args.size()) {
        device = std::max(0, std::stoi(args[i + 1]));
      }
    }
    if (stats_path.empty()) throw std::invalid_argument("Expected --probe_stats_path");
    const int result = colite::exe::RunColite(args, std::cout, std::cerr);
    if (result != 0) {
      colite::exe::ShutdownLogging();
      return result;
    }
    std::uint64_t used = 0, reserved = 0;
    if (gpu) {
      cudaMemPool_t pool;
      if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess ||
          cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used) != cudaSuccess ||
          cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved) != cudaSuccess) {
        throw std::runtime_error("Cannot query CUDA allocation high-water marks");
      }
    }
    std::ofstream file(stats_path);
    if (!file) throw std::runtime_error("Cannot write benchmark stats");
    file << "{\"cuda_async_pool_used_high_bytes\":";
    if (gpu) file << used; else file << "null";
    file << ",\"cuda_async_pool_reserved_high_bytes\":";
    if (gpu) file << reserved; else file << "null";
    file << ",\"scope\":\"CUDA async allocation pool only; excludes driver context, libraries and other processes\"}\n";
    colite::exe::ShutdownLogging();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    colite::exe::ShutdownLogging();
    return 1;
  }
}
