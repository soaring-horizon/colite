#include "task_runner.h"
#include "task_plan.h"
#include "workflow_state.h"
#include "project/workspace.h"
#include <algorithm>
#include <cmath>

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QSaveFile>
#include <QSet>
#include <QStandardPaths>
#include <QLockFile>
#include <QUuid>
#include <QtConcurrent>
#include <vector>
#ifdef Q_OS_UNIX
#include <signal.h>
#include <unistd.h>
#endif
#ifdef Q_OS_WIN
#define NOMINMAX
#include <windows.h>
#endif

namespace colite::gui {
namespace {
class OwnedProcess : public QProcess {
 public:
#ifdef Q_OS_WIN
  HANDLE job = nullptr;
  STARTUPINFOEXW startup{};
  std::vector<unsigned char> attributes;
  bool ready = false;
  explicit OwnedProcess(QObject* parent) : QProcess(parent) {
    job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                         &info, sizeof(info))) return;
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    attributes.resize(size);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &size)) {
      startup.lpAttributeList = nullptr;
      return;
    }
    if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0,
                                   PROC_THREAD_ATTRIBUTE_JOB_LIST, &job,
                                   sizeof(job), nullptr, nullptr)) return;
    ready = true;
    setCreateProcessArgumentsModifier([this](CreateProcessArguments* args) {
      startup.StartupInfo = *args->startupInfo;
      startup.StartupInfo.cb = sizeof(startup);
      args->startupInfo = &startup.StartupInfo;
      args->flags |= EXTENDED_STARTUPINFO_PRESENT;
    });
  }
  void own() {}
  void killOwned() {
    if (job)
      TerminateJobObject(job, 1);
    else
      kill();
  }
  ~OwnedProcess() override {
    if (startup.lpAttributeList) DeleteProcThreadAttributeList(startup.lpAttributeList);
    if (job) CloseHandle(job);
  }
#else
  using QProcess::QProcess;
  const bool ready = true;
  void own() {}
  void killOwned() {
    if (processId() > 0) ::kill(-pid_t(processId()), SIGKILL);
    kill();
  }

 protected:
  void setupChildProcess() override { ::setpgid(0, 0); }
#endif
};
}  // namespace
TaskRunner::TaskRunner(QObject* parent) : QObject(parent) {
  elapsedTimer_.setInterval(1000);
  connect(&elapsedTimer_, &QTimer::timeout, this, &TaskRunner::progressChanged);
  connect(this, &TaskRunner::changed, this, [this] {
    if (persistState_ && !resultPath_.isEmpty() && !persistResult() && busy())
      fail(QString::fromUtf8("无法保存任务状态；任务已停止。"));
  });
  heartbeat_.setInterval(1000);
  connect(&heartbeat_, &QTimer::timeout, this, [this] {
    writeControl(state_ == State::Aborting ? "abort" : "alive");
  });
  deadline_.setSingleShot(true);
  connect(&deadline_, &QTimer::timeout, this, [this] {
    if (process_) static_cast<OwnedProcess*>(process_)->killOwned();
  });
}
TaskRunner::~TaskRunner() {
  if (busy() || (process_ && process_->state() != QProcess::NotRunning)) {
    writeControl("abort");
    if (process_) {
      if (project_.backendMode != "wsl")
        static_cast<OwnedProcess*>(process_)->killOwned();
      process_->waitForFinished(1500);
    }
  }
}
bool TaskRunner::busy() const {
  return state_ == State::Running || state_ == State::PauseRequested ||
         state_ == State::Paused || state_ == State::Aborting;
}
QString TaskRunner::title() const {
  if (steps_.isEmpty()) return QString::fromUtf8("尚无任务");
  return index_ < steps_.size() ? steps_[index_].title
                                : QString::fromUtf8("任务结束");
}
bool TaskRunner::containsStage(int stage) const {
  for (const auto& step : steps_) if (step.stage == stage) return true;
  return false;
}
bool TaskRunner::completedStage(int stage) const {
  bool found = false;
  for (int i = 0; i < steps_.size(); ++i) if (steps_[i].stage == stage) {
    if (i >= index_) return false;
    found = true;
  }
  return found;
}
void TaskRunner::writeControl(const char* state) {
  if (control_.isEmpty()) return;
  QSaveFile file(control_);
  if (file.open(QIODevice::WriteOnly)) {
    file.write(state);
    file.commit();
  }
}
void TaskRunner::startWorkflow(const Project& project, const Workflow& workflow) {
  if (workflow.stages.isEmpty()) return;
  checkStageInputs(project, static_cast<int>(workflow.stages.first()));
  if (workflow.stages.size() > 1 && Workspace::read(project).images.size() < 2)
    throw std::invalid_argument("Reconstruction needs at least two source photos");
  const QDir bin(QCoreApplication::applicationDirPath());
#ifdef Q_OS_WIN
  const QString program = bin.filePath("colite.exe");
#else
  const QString program = bin.filePath("colite");
#endif
  if (!QFileInfo(program).isExecutable()) throw std::runtime_error("Missing bundled Colite reconstruction engine");
  auto execution = project;
  execution.backend = program;
  execution.backendMode = "native";
  start(execution, cliSteps(execution, workflow), {});
}
void TaskRunner::start(const Project& project, const QVector<TaskStep>& plan,
                       const QString& bridge) {
  if (busy() || plan.isEmpty() ||
      (process_ && process_->state() != QProcess::NotRunning))
    return;
  ++generation_;
  clock_.start(); elapsed_ = 0; elapsedTimer_.start();
  validating_ = false;
  project_ = project;
  if (project_.backendMode == "native") {
    if (project_.backend.isEmpty()) {
      project_.backend = QDir(QCoreApplication::applicationDirPath()).filePath(
#ifdef Q_OS_WIN
          "colite.exe"
#else
          "colite"
#endif
          );
    } else if (project_.backend.contains('/') || project_.backend.contains('\\')) {
      project_.backend = QFileInfo(project_.file).absoluteDir().absoluteFilePath(localPath(project_.backend));
    } else {
      const auto executable = QStandardPaths::findExecutable(project_.backend);
      if (!executable.isEmpty()) project_.backend = executable;
    }
  }
  persistState_ = true;
  resultPath_.clear();
  lastError_.clear();
  startedAt_ = QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs);
  steps_ = plan;
  index_ = 0;
  bridge_ = bridge;
  pending_.clear();
  const auto id = QUuid::createUuid().toString(QUuid::WithoutBraces);
  const auto folder =
      QDir(project.historyRoot()).filePath(id);
  if (project.historyRoot().isEmpty() || !QDir().mkpath(folder)) {
    fail(QString::fromUtf8("无法创建任务日志目录"));
    return;
  }
  control_ = QDir(folder).filePath("control");
  logPath_ = QDir(folder).filePath("task.log");
  resultPath_ = QDir(folder).filePath("result.json");
  log_.setFileName(logPath_);
  if (!log_.open(QIODevice::WriteOnly)) {
    fail(log_.errorString());
    return;
  }
  QJsonArray commands;
  for (const auto& step : plan)
    commands.append(
        QJsonObject{{"title", step.title}, {"stage", step.stage},
                    {"arguments", QJsonArray::fromStringList(step.arguments)}});
  QSaveFile request(QDir(folder).filePath("request.json"));
  const auto requestBytes = QJsonDocument(QJsonObject{{"schema", 1},
                                              {"project", project.file}, {"project_id", project.id},
                                              {"started_at", startedAt_},
                                              {"backend", project_.backend},
                                              {"mode", project.backendMode},
                                              {"distribution", project.distribution},
                                              {"steps", commands}}).toJson();
  if (!request.open(QIODevice::WriteOnly) ||
      request.write(requestBytes) != requestBytes.size() ||
      !request.commit()) {
    fail("Cannot persist task request");
    return;
  }
  QStringList outputRoots;
  for (const auto& path : {project.workspace, project.dense,
                           project.sparse, project.mesh.isEmpty()
                               ? QString() : QFileInfo(project.mesh).absolutePath()}) {
    if (!path.isEmpty()) outputRoots << QDir::cleanPath(QFileInfo(path).absoluteFilePath());
  }
  outputRoots.removeDuplicates();
  outputRoots.sort();
  locks_.clear();
  QSet<QString> lockedPaths;
  for (const auto& root : outputRoots) {
    if (!QDir().mkpath(root)) {
      fail("Cannot create output directory: " + root);
      return;
    }
    const auto canonical = QFileInfo(root).canonicalFilePath();
    const auto identity =
#ifdef Q_OS_WIN
        canonical.toCaseFolded();
#else
        canonical;
#endif
    if (lockedPaths.contains(identity)) continue;
    lockedPaths.insert(identity);
    auto lock = std::make_unique<QLockFile>(QDir(canonical).filePath(".colite-gui.lock"));
    lock->setStaleLockTime(0);
    if (!lock->tryLock(0)) {
      fail(QString::fromUtf8("输出目录正被另一任务使用或不可写：") + root);
      return;
    }
    locks_.push_back(std::move(lock));
  }
  state_ = State::Running;
  advance();
}
void TaskRunner::advance() {
  if (index_ >= steps_.size()) {
    end(State::Succeeded);
    return;
  }
  if (state_ == State::PauseRequested) {
    state_ = State::Paused;
    emit changed();
    return;
  }
  progressDone_ = progressTotal_ = 0; progressPhase_.clear(); progressDetail_.clear();
  diagnosticTail_.clear();
  auto* process = new OwnedProcess(this);
  if (steps_[index_].stage > 0 && (index_ == 0 || steps_[index_ - 1].stage != steps_[index_].stage)) {
    try {
      invalidateStages(project_, steps_[index_].stage);
      stageInput_ = stageInputStamp(project_, steps_[index_].stage);
    }
    catch (const std::exception& e) { delete process; fail(QString::fromUtf8(e.what())); return; }
  }
  process_ = process;
  if (!process->ready) {
    fail("Cannot create an owned process tree on this Windows host");
    process_ = nullptr;
    process->deleteLater();
    return;
  }
  process->setProcessChannelMode(QProcess::MergedChannels);
  connect(process, &QProcess::readyReadStandardOutput, this,
          &TaskRunner::consume);
  connect(process, &QProcess::started, this, [this, process] {
    process->own();
    emit changed();
  });
  connect(process, &QProcess::errorOccurred, this,
          [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) fail(process_->errorString());
          });
  connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
          this, [this, process](int code, QProcess::ExitStatus status) {
            consume();
            heartbeat_.stop();
            deadline_.stop();
            process_ = nullptr;
            process->deleteLater();
            if (state_ == State::Aborting) {
              end(State::Aborted);
              return;
            }
            if (!busy()) {
              QFile::remove(control_);
              locks_.clear();
              return;
            }
            if (code != 0 || status != QProcess::NormalExit) {
              const auto tail = QString::fromUtf8(diagnosticTail_);
              const auto lower = tail.toLower();
              QString advice = tr("计算未完成。请查看下方错误记录，修正原因后从当前步骤重试。");
              if (lower.contains("no space left") || lower.contains("disk full") || lower.contains("磁盘空间不足"))
                advice = tr("工作磁盘空间不足。释放空间后从当前步骤重试，已有结果会重新校验。");
              else if (lower.contains("out of memory") || lower.contains("bad_alloc") || lower.contains("cudaerrormemoryallocation"))
                advice = tr("计算内存不足。降低图像分辨率或并发线程数后重试；显存不足时可改用 CPU。");
              else if (lower.contains("cuda") && (lower.contains("not available") || lower.contains("unavailable") ||
                       lower.contains("insufficient") || lower.contains("no cuda-capable")))
                advice = tr("CUDA 计算不可用。请在任务参数中选择 CPU，或检查与发行包匹配的显卡及驱动。");
              else if (lower.contains("permission denied") || lower.contains("access is denied") || lower.contains("failed to write"))
                advice = tr("无法写入计算结果。请检查工作目录权限和磁盘可用空间，再从当前步骤重试。");
              fail(advice + tr("\n\n后端退出码：%1\n日志：%2\n\n最近输出：\n%3")
                  .arg(code).arg(logPath_, tail.right(3000)));
              return;
            }
            validating_ = true;
            emit changed();
            const int token = generation_;
            const auto validate = steps_[index_].validate;
            auto* watcher = new QFutureWatcher<QString>(this);
            connect(watcher, &QFutureWatcher<QString>::finished, this,
                    [this, watcher, token] {
                      const auto error = watcher->result();
                      watcher->deleteLater();
                      if (token != generation_ || !busy()) return;
                      validating_ = false;
                      if (state_ == State::Aborting) {
                        end(State::Aborted);
                        return;
                      }
                      if (!error.isEmpty()) {
                        fail(QString::fromUtf8("产物检查失败：") + error);
                        return;
                      }
                      const int stage = steps_[index_].stage;
                      if (stage > 0 && (index_ + 1 == steps_.size() || steps_[index_ + 1].stage != stage)) {
                        try { recordStage(project_, stage, stageInput_); }
                        catch (const std::exception& e) { fail(QString::fromUtf8(e.what())); return; }
                      }
                      ++index_;
                      emit changed();
                      advance();
                    });
            watcher->setFuture(QtConcurrent::run([validate] {
              try {
                if (validate) validate();
                return QString();
              } catch (const std::exception& e) {
                return QString::fromUtf8(e.what());
              }
            }));
          });
  emit output(QString::fromUtf8("开始：") + title() + "\n");
  auto arguments = steps_[index_].arguments;
  if (project_.backendMode == "wsl") {
    if (!QFileInfo::exists(bridge_)) {
      fail("Missing installed worker_bridge.py");
      return;
    }
    try {
      for (int i = 1; i < arguments.size(); ++i)
        if (arguments[i - 1].endsWith("_path"))
          arguments[i] = wslPath(arguments[i]);
      QStringList launch{"-d",
                         project_.distribution,
                         "--exec",
                         "/usr/bin/python3",
                         wslPath(bridge_),
                         "--control",
                         wslPath(control_),
                         "--",
                         project_.backend};
      launch += arguments;
      writeControl("alive");
      heartbeat_.start();
      process->start("wsl.exe", launch);
    } catch (const std::exception& e) {
      fail(QString::fromUtf8(e.what()));
    }
  } else {
    auto environment = QProcessEnvironment::systemEnvironment();
    environment.insert("COLITE_PROGRESS", "1");
    process->setProcessEnvironment(environment);
    process->start(project_.backend, arguments);
  }
  emit changed();
}
void TaskRunner::consume() {
  if (!process_) return;
  const auto bytes = process_->readAllStandardOutput();
  diagnosticTail_ = (diagnosticTail_ + bytes).right(16384);
  if (log_.isOpen() && (log_.write(bytes) != bytes.size() || !log_.flush())) {
    fail(QString::fromUtf8("日志写入失败；中止任务以免丢失诊断。"));
    return;
  }
  pending_ += bytes;
  int end;
  while ((end = pending_.indexOf('\n')) >= 0) {
    const auto line = pending_.left(end + 1);
    if (line.startsWith("@colite.progress ")) {
      const auto value = QJsonDocument::fromJson(line.mid(17)).object();
      const double done = value["done"].toDouble(-1), total = value["total"].toDouble(-1);
      const auto phase = value["phase"].toString();
      if (value["schema"].toInt() == 1 && !phase.isEmpty() && total > 0 && total <= 1e12 &&
          done >= 0 && done <= total && done == std::floor(done) && total == std::floor(total) &&
          (phase != progressPhase_ || done >= progressDone_)) {
        progressPhase_ = phase; progressDone_ = qint64(done); progressTotal_ = qint64(total);
        progressDetail_ = value["detail"].toString().left(300);
        emit progressChanged();
      }
    } else emit output(QString::fromUtf8(line));
    pending_.remove(0, end + 1);
  }
  if (process_->state() == QProcess::NotRunning && !pending_.isEmpty()) {
    emit output(QString::fromUtf8(pending_));
    pending_.clear();
  }
}
void TaskRunner::pauseOrResume() {
  if (state_ == State::Running) {
    state_ = State::PauseRequested;
    emit output(QString::fromUtf8(
        "已请求暂停后续队列；当前 CLI 步骤继续计算，结束后确认暂停。\n"));
  } else if (state_ == State::PauseRequested)
    state_ = State::Running;
  else if (state_ == State::Paused) {
    state_ = State::Running;
    advance();
  }
  emit changed();
}
void TaskRunner::abort() {
  if (!busy()) return;
  state_ = State::Aborting;
  writeControl("abort");
  if (process_) {
    if (project_.backendMode == "wsl")
      deadline_.start(10000);
    else
      static_cast<OwnedProcess*>(process_)->killOwned();
  } else if (!validating_)
    end(State::Aborted);
  emit changed();
}
void TaskRunner::fail(const QString& error) {
  lastError_ = error;
  emit output(error + "\n");
  if (log_.isOpen()) {
    log_.write((error + "\n").toUtf8());
    log_.flush();
  }
  writeControl("abort");
  if (process_ && process_->state() != QProcess::NotRunning) {
    if (project_.backendMode != "wsl")
      static_cast<OwnedProcess*>(process_)->killOwned();
    else
      deadline_.start(10000);
  }
  end(State::Failed);
}
void TaskRunner::end(State state) {
  elapsed_ = clock_.isValid() ? clock_.elapsed() : 0;
  elapsedTimer_.stop();
  state_ = state;
  validating_ = false;
  heartbeat_.stop();
  if (!process_ || process_->state() == QProcess::NotRunning) deadline_.stop();
  log_.close();
  if (!control_.isEmpty() &&
      (!process_ || process_->state() == QProcess::NotRunning))
    QFile::remove(control_);
  if (!process_ || process_->state() == QProcess::NotRunning) locks_.clear();
  emit changed();
  emit finished(state == State::Succeeded);
}
bool TaskRunner::persistResult() const {
  static const char* names[] = {"idle", "running", "pause_requested", "paused",
      "succeeded", "failed", "aborting", "aborted"};
  const QJsonObject result{{"schema", 1}, {"state", names[int(state_)]},
      {"project", project_.file}, {"project_id", project_.id}, {"backend", project_.backend},
      {"backend_mode", project_.backendMode}, {"started_at", startedAt_},
      {"updated_at", QDateTime::currentDateTimeUtc().toString(Qt::ISODateWithMs)},
      {"completed", index_}, {"total", steps_.size()}, {"validating", validating_},
      {"elapsed_ms", double(elapsedMilliseconds())},
      {"process_id", process_ ? double(process_->processId()) : 0.0},
      {"log", logPath_}, {"error", lastError_}};
  QSaveFile file(resultPath_);
  const auto bytes = QJsonDocument(result).toJson();
  return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
}
bool TaskRunner::restoreHistory(const Project& project) {
  if (busy() || (process_ && process_->state() != QProcess::NotRunning)) return false;
  ++generation_;
  persistState_ = false;
  elapsedTimer_.stop(); clock_.invalidate(); elapsed_ = 0;
  progressDone_ = progressTotal_ = 0; progressPhase_.clear(); progressDetail_.clear();
  project_ = project;
  steps_.clear();
  index_ = 0;
  state_ = State::Idle;
  validating_ = false;
  resultPath_.clear();
  control_.clear();
  logPath_.clear();
  lastError_.clear();
  if (project.historyRoot().isEmpty()) { emit changed(); return true; }
  const QDir tasks(project.historyRoot());
  auto entries = tasks.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
  if (!project.file.isEmpty()) {
    const QDir legacy(QFileInfo(project.file).dir().filePath("gui/tasks"));
    if (legacy != tasks) {
      entries.append(legacy.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot));
      std::stable_sort(entries.begin(), entries.end(), [](const QFileInfo& a, const QFileInfo& b) {
        return a.lastModified() > b.lastModified();
      });
    }
  }
  for (const auto& entry : entries) {
    QFile result(QDir(entry.absoluteFilePath()).filePath("result.json"));
    if (!result.open(QIODevice::ReadOnly) || result.size() > 1024 * 1024) continue;
    const auto record = QJsonDocument::fromJson(result.readAll()).object();
    const bool sameProject = record.contains("project_id")
        ? record["project_id"].toString() == project.id
        : !project.file.isEmpty() && record["project"].toString() == project.file;
    if (record["schema"].toInt() != 1 || !sameProject) continue;
    QFile request(QDir(entry.absoluteFilePath()).filePath("request.json"));
    if (!request.open(QIODevice::ReadOnly) || request.size() > 4 * 1024 * 1024) continue;
    const auto commands = QJsonDocument::fromJson(request.readAll()).object()["steps"].toArray();
    if (commands.isEmpty() || commands.size() > 100) continue;
    const QMap<QString, int> legacyStages{{"feature_extractor", 1}, {"exhaustive_matcher", 2},
        {"sequential_matcher", 2}, {"spatial_matcher", 2}, {"mapper", 3}, {"model_converter", 3},
        {"image_undistorter", 4}, {"patch_match_stereo", 4}, {"stereo_fusion", 4},
        {"poisson_mesher", 5}, {"mesh_texturer", 6}};
    for (const auto& command : commands) {
      const auto object = command.toObject();
      const auto arguments = object["arguments"].toArray();
      const int stage = object.contains("stage") ? object["stage"].toInt()
          : arguments.isEmpty() ? 0 : legacyStages.value(arguments.first().toString());
      steps_.push_back({object["title"].toString(), {}, {}, qBound(0, stage, 6)});
    }
    index_ = qBound(0, record["completed"].toInt(), steps_.size());
    const auto status = record["state"].toString();
    state_ = status == "succeeded" ? State::Succeeded
        : status == "aborted" ? State::Aborted : State::Failed;
    if (state_ == State::Succeeded && index_ != steps_.size()) state_ = State::Failed;
    resultPath_ = result.fileName();
    logPath_ = QDir(entry.absoluteFilePath()).filePath("task.log");
    lastError_ = record["error"].toString();
    elapsed_ = qint64(qBound(0.0, record["elapsed_ms"].toDouble(), 1e12));
    QFile log(logPath_);
    if (log.open(QIODevice::ReadOnly)) {
      log.seek(qMax(qint64(0), log.size() - 65536));
      emit output(QString::fromUtf8(log.readAll()));
    }
    if (status != "succeeded" && status != "failed" && status != "aborted")
      emit output(QString::fromUtf8("上次任务未记录结束；请检查日志。重新启动时会校验已有缓存。\n"));
    break;
  }
  emit changed();
  return true;
}
}  // namespace colite::gui
