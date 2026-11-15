#include <iostream>

#include "colite/exe/logging.h"
#include "colite/exe/mapper.h"

int main(int argc, char** argv) {
  return colite::exe::RunLoggedCommand(argc, argv, [](const auto& args) {
    return colite::exe::RunMapper(args, std::cout, std::cerr);
  });
}
