#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace colite::exe {

// Parse paths and typed options without accessing inputs or creating outputs.
void ValidateSurfaceArguments(const std::string& tool,
                               const std::vector<std::string>& args);

int RunPoissonMesher(const std::vector<std::string>& args,
                     std::ostream& out, std::ostream& err);
int RunDelaunayMesher(const std::vector<std::string>& args,
                      std::ostream& out, std::ostream& err);
int RunMeshSimplifier(const std::vector<std::string>& args,
                      std::ostream& out, std::ostream& err);
int RunMeshTexturer(const std::vector<std::string>& args,
                    std::ostream& out, std::ostream& err);

}  // namespace colite::exe
