#pragma once
#include "project/project.h"
#include <QStringList>

namespace colite::gui {
enum class OutputState { Missing, Unverified, Current, Stale };
QStringList stageOutputFiles(const Project& project, int stage);
OutputState outputState(const Project& project, int stage);
QString outputStateText(OutputState state);
void invalidateStages(const Project& project, int first);
QString stageInputStamp(const Project& project, int stage);
void recordStage(const Project& project, int stage, const QString& expectedInput = {});
void validateParameters(const Project& project);
void checkStageInputs(const Project& project, int first);
}
