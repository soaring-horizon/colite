#pragma once
#include "project.h"
#include <QLockFile>
#include <QStringList>
#include <memory>

namespace colite::gui {
// One locked snapshot per running window. Other live sessions are never offered.
class RecoveryStore {
 public:
  explicit RecoveryStore(QString root = {});
  void save(const Project& project);
  void clear();
  QStringList available() const;
  Project restore(const QString& path);
 private:
  QString root_, path_;
  std::unique_ptr<QLockFile> lock_;
};
}
