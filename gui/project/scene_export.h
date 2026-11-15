#pragma once
#include <QString>
#include <functional>
namespace colite::gui {
// Exports into a new directory, committing only after all files are copied.
void exportScene(const QString& source, const QString& destination,
                 const std::function<bool()>& cancelled = {});
}
