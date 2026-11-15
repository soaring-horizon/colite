#pragma once
#include <QFile>
#include <QObject>
#include <QProcess>
#include <QTimer>
#include <QLockFile>
#include <QElapsedTimer>
#include <functional>
#include <memory>
#include <vector>

#include "project/project.h"

namespace colite::gui {
struct Workflow;
struct TaskStep {
  QString title;
  QStringList arguments;
  std::function<void()> validate;
  int stage = 0;
};
class TaskRunner : public QObject {
  Q_OBJECT
 public:
  enum class State {
    Idle,
    Running,
    PauseRequested,
    Paused,
    Succeeded,
    Failed,
    Aborting,
    Aborted
  };
  explicit TaskRunner(QObject* parent = nullptr);
  ~TaskRunner() override;
  void startWorkflow(const Project& project, const Workflow& workflow);
  void start(const Project& project, const QVector<TaskStep>& plan,
             const QString& bridge);
  bool restoreHistory(const Project& project);
  void pauseOrResume();
  void abort();
  bool busy() const;
  State state() const { return state_; }
  int completed() const { return index_; }
  int total() const { return steps_.size(); }
  bool validating() const { return validating_; }
  QString title() const;
  int currentStage() const { return index_ < steps_.size() ? steps_[index_].stage : 0; }
  bool containsStage(int stage) const;
  bool completedStage(int stage) const;
  QString resultPath() const { return resultPath_; }
  qint64 elapsedMilliseconds() const { return busy() && clock_.isValid() ? clock_.elapsed() : elapsed_; }
  qint64 progressDone() const { return progressDone_; }
  qint64 progressTotal() const { return progressTotal_; }
  QString progressDetail() const { return progressDetail_; }
  QString progressPhase() const { return progressPhase_; }
  QString lastError() const { return lastError_; }
  QString logPath() const { return logPath_; }
 signals:
  void changed();
  void output(const QString& text);
  void finished(bool success);
  void progressChanged();

 private:
  QProcess* process_ = nullptr;
  QTimer heartbeat_, deadline_;
  QFile log_;
  Project project_;
  QVector<TaskStep> steps_;
  int index_ = 0;
  int generation_ = 0;
  bool validating_ = false;
  bool persistState_ = false;
  State state_ = State::Idle;
  QString bridge_, control_, logPath_;
  QString resultPath_, startedAt_, lastError_;
  QByteArray pending_;
  QByteArray diagnosticTail_;
  QTimer elapsedTimer_;
  QElapsedTimer clock_;
  qint64 elapsed_ = 0, progressDone_ = 0, progressTotal_ = 0;
  QString progressPhase_, progressDetail_, stageInput_;
  std::vector<std::unique_ptr<QLockFile>> locks_;
  void advance();
  void consume();
  void fail(const QString& error);
  void end(State state);
  void writeControl(const char* state);
  bool persistResult() const;
};
}  // namespace colite::gui
