#include <iostream>

#include "colite/exe/compare_colmap.h"
#include "colite/exe/logging.h"

int main(int argc, char** argv) {
  return colite::exe::RunLoggedCommand(argc, argv, [](const auto& args) {
    return colite::exe::RunCompareColmap(args, std::cout, std::cerr);
  });
}
