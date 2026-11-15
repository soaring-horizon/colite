#pragma once
#include <QJsonObject>
#include <QString>
#include <QMap>
#include <QUuid>
#include <functional>

namespace colite::gui {
struct Project {
  QString id = QUuid::createUuid().toString(QUuid::WithoutBraces);
  QString workDirectory;
  QMap<QString, QString> photoFiles; // Logical dataset name -> original file.
  QString file, name, imageRoot, workspace, sparse, dense, mesh, backend;
  QString backendMode = "native", distribution = "Ubuntu-22.04";
  QJsonObject parameters;
  QString historyRoot() const;
  QString activeSparse() const;
  static QString defaultWorkDirectory();
  void ensureOutputs();
  void assignOutputs(const QString& directory);
  void prepareInput(const std::function<void(qint64, qint64, const QString&)>& progress = {},
                    const std::function<bool()>& cancelled = {});
  static Project load(const QString& path);
  void save(const QString& path);
};
QString localPath(
    const QString& path);  // Explicit /mnt/<drive> interoperability.
QString wslPath(const QString& path);  // Reject UNC/non-drive paths.
}  // namespace colite::gui
