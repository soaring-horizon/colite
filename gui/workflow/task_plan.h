#pragma once
#include "project/project.h"
#include "task_runner.h"

namespace colite::gui {
enum class Stage { Features = 1, Matching, Sparse, Dense, Surface, Texture };
struct Workflow {
  QVector<Stage> stages;
  static Workflow from(int first, bool follow);
};
QString stageTitle(Stage stage);
QVector<TaskStep> cliSteps(const Project& project, const Workflow& workflow);
// The GUI and installed pipeline verification use exactly the same commands.
QVector<TaskStep> buildTaskPlan(const Project& project, int firstStage, bool all);
}
