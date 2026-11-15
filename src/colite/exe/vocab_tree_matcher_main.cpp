#include <iostream>

#include "colite/exe/logging.h"
#include "colite/exe/sequential_matcher.h"

int main(int argc, char** argv) {
  return colite::exe::RunLoggedCommand(argc, argv, [](const auto& args) {
    return colite::exe::RunVocabTreeMatcher(args, std::cout, std::cerr);
  });
}
