#include <iostream>

#include "colite/exe/logging.h"
#include "colite/exe/recon_sfm.h"

int main(int argc, char** argv) {
  return colite::exe::RunLoggedCommand(argc, argv, [](const auto& args) {
    return colite::exe::RunReconSfm(args, std::cout, std::cerr);
  });
}
