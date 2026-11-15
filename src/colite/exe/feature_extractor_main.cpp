#include <iostream>

#include "colite/exe/feature_extractor.h"
#include "colite/exe/logging.h"

int main(int argc, char** argv) {
  return colite::exe::RunLoggedCommand(argc, argv, [](const auto& args) {
    return colite::exe::RunFeatureExtractor(args, std::cout, std::cerr);
  });
}
