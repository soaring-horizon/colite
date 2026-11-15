#include "recovery_store.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDateTime>
#include <QStandardPaths>
#include <stdexcept>

namespace colite::gui {
RecoveryStore::RecoveryStore(QString root) : root_(std::move(root)) {
  if (root_.isEmpty()) root_ = QDir(QStandardPaths::writableLocation(
      QStandardPaths::AppLocalDataLocation)).filePath("recovery");
  root_ = QDir(root_).absolutePath();
  path_ = QDir(root_).filePath(QUuid::createUuid().toString(QUuid::WithoutBraces) + ".colite.json");
}
void RecoveryStore::save(const Project& project) {
  if (!QDir().mkpath(root_)) throw std::runtime_error("Cannot create recovery directory");
  if (!lock_) {
    auto lock = std::make_unique<QLockFile>(path_ + ".lock");
    lock->setStaleLockTime(0);
    if (!lock->tryLock()) throw std::runtime_error("Cannot lock recovery snapshot");
    lock_ = std::move(lock);
  }
  auto snapshot = project;
  snapshot.parameters["gui_recovery_original_file"] = project.file;
  snapshot.parameters["gui_recovery_time"] = QDateTime::currentDateTimeUtc().toString(Qt::ISODate);
  snapshot.save(path_);
}
void RecoveryStore::clear() {
  if (QFileInfo::exists(path_) && !QFile::remove(path_))
    throw std::runtime_error("Cannot remove saved recovery snapshot");
}
QStringList RecoveryStore::available() const {
  QStringList result;
  for (const auto& entry : QDir(root_).entryInfoList({"*.colite.json"}, QDir::Files, QDir::Time)) {
    if (entry.absoluteFilePath() == path_) continue;
    QLockFile lock(entry.absoluteFilePath() + ".lock");
    lock.setStaleLockTime(0);
    if (lock.tryLock()) result << entry.absoluteFilePath();
  }
  return result;
}
Project RecoveryStore::restore(const QString& path) {
  if (QFileInfo(path).absolutePath() != root_ || !available().contains(path))
    throw std::runtime_error("Recovery snapshot is not available");
  QLockFile sourceLock(path + ".lock");
  sourceLock.setStaleLockTime(0);
  if (!sourceLock.tryLock()) throw std::runtime_error("Snapshot is being recovered by another window");
  auto project = Project::load(path);
  project.file = project.parameters.take("gui_recovery_original_file").toString();
  project.parameters.remove("gui_recovery_time");
  save(project); // Commit this session's copy before removing the abandoned one.
  if (!QFile::remove(path)) throw std::runtime_error("Cannot retire recovered snapshot");
  return project;
}
}
