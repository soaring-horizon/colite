#include "colite/core/version.h"

namespace colite {

std::string_view Version() {
  return COLITE_VERSION_STRING;
}

std::string_view PoseSolverBackend() {
#ifdef COLITE_HAS_POSELIB
  return "poselib";
#else
  return "builtin_p3p";
#endif
}

}  // namespace colite

