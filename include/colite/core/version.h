#pragma once

#include <string_view>

namespace colite {

std::string_view Version();

// Read from the compiled core library, not the caller's private build macros.
// Geometry estimation and generalized registration depend on this backend.
std::string_view PoseSolverBackend();

}  // namespace colite

