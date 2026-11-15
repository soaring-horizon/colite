#include <iostream>
#include "colite/exe/command.h"
#include "colite/exe/logging.h"
#include "colite_release.h"

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--build-info") {
    std::cout << COLITE_BUILD_JSON << '\n';
    return 0;
  }
  return colite::exe::RunLoggedCommand(argc, argv, [](const auto& args) {
    return colite::exe::RunColite(args, std::cout, std::cerr);
  });
}
