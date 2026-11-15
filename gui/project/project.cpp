#include "project.h"

#include <QDir>
#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTemporaryDir>
#include <QStorageInfo>
#include <limits>
#include <stdexcept>

namespace colite::gui {
namespace {
void fail(const QString& text) {
  throw std::runtime_error(text.toUtf8().constData());
}
}  // namespace
QString localPath(const QString& path) {
#ifdef Q_OS_WIN
  const auto match = QRegularExpression("^/mnt/([a-zA-Z])/(.*)$").match(path);
  if (match.hasMatch())
    return match.captured(1).toUpper() + ":/" + match.captured(2);
#endif
  return QDir::cleanPath(path);
}
QString wslPath(const QString& path) {
  if (path.startsWith('/') && !path.startsWith("//")) return path;
  const QString normalized =
      QDir::fromNativeSeparators(QFileInfo(path).absoluteFilePath());
  const auto match = QRegularExpression("^([a-zA-Z]):/(.*)$").match(normalized);
  if (!match.hasMatch())
    fail(QString::fromUtf8(
             "WSL 仅支持明确的 /mnt 盘符映射，请使用本地磁盘路径：") +
         path);
  return "/mnt/" + match.captured(1).toLower() + "/" + match.captured(2);
}
QString Project::historyRoot() const {
  if (!workDirectory.isEmpty()) return QDir(workDirectory).filePath("gui/tasks");
  return file.isEmpty() ? QString() : QFileInfo(file).dir().filePath("gui/tasks");
}
QString Project::activeSparse() const {
  const int model = parameters.value("sparse_model").toInt(-1);
  return model < 0 ? sparse : QDir(sparse).filePath(QString("models/%1").arg(model));
}
QString Project::defaultWorkDirectory() {
  return QDir(QCoreApplication::applicationDirPath()).filePath("output");
}
void Project::ensureOutputs() {
  if (!workspace.isEmpty()) return; // Preserve existing/legacy result locations.
  const auto root = workDirectory.isEmpty() ? defaultWorkDirectory() : workDirectory;
  if (!imageRoot.isEmpty()) {
    const auto relative = QDir(imageRoot).relativeFilePath(root);
    if (QDir::isRelativePath(relative) && relative != ".." && !relative.startsWith("../"))
      fail(QString::fromUtf8("工作目录不能位于原图目录内，请在“文件 → 项目数据”中修改。"));
  }
  assignOutputs(root);
}
void Project::assignOutputs(const QString& directory) {
  workDirectory = QDir::cleanPath(directory);
  const QDir run(QDir(workDirectory).filePath("runs/" + QUuid::createUuid().toString(QUuid::WithoutBraces)));
  workspace = run.filePath("workspace");
  sparse = QDir(workspace).filePath("sparse");
  dense = run.filePath("mvs");
  mesh = run.filePath("surface/mesh.ply");
}
void Project::prepareInput(const std::function<void(qint64, qint64, const QString&)>& progress,
                          const std::function<bool()>& cancelled) {
  if (photoFiles.isEmpty() || !imageRoot.isEmpty()) return;
  if (workDirectory.isEmpty() || !QDir().mkpath(workDirectory))
    fail(QString::fromUtf8("无法创建计算工作目录。"));
  qint64 total = 0, copied = 0;
  for (const auto& source : photoFiles) {
    const QFileInfo info(source);
    if (!info.isFile() || !info.isReadable()) fail(QString::fromUtf8("照片不存在或无法读取：") + source);
    if (info.size() > std::numeric_limits<qint64>::max() - total) fail("Photo input size overflow");
    total += info.size();
  }
  const QStorageInfo storage(workDirectory);
  if (storage.isValid() && storage.isReady() && storage.bytesAvailable() < total)
    fail(QString::fromUtf8("工作目录可用空间不足以准备照片。"));
  QTemporaryDir input(QDir(workDirectory).filePath("photos-XXXXXX"));
  if (!input.isValid()) fail(input.errorString());
  for (auto it = photoFiles.cbegin(); it != photoFiles.cend(); ++it) {
    if (cancelled && cancelled()) fail(QString::fromUtf8("已取消照片准备。"));
    const auto target = QDir(input.path()).filePath(it.key());
    QFile source(it.value()), output(target);
    if (!QDir().mkpath(QFileInfo(target).absolutePath()) || !source.open(QIODevice::ReadOnly) ||
        !output.open(QIODevice::WriteOnly))
      fail(QString::fromUtf8("无法准备照片：") + it.value());
    const auto sourceSize = source.size();
    qint64 fileBytes = 0;
    while (!source.atEnd()) {
      if (cancelled && cancelled()) fail(QString::fromUtf8("已取消照片准备。"));
      const auto bytes = source.read(1024 * 1024);
      if (bytes.isEmpty() || output.write(bytes) != bytes.size()) fail(QString::fromUtf8("照片复制失败：") + it.value());
      copied += bytes.size(); fileBytes += bytes.size();
      if (progress) progress(copied, total, it.key());
    }
    if (fileBytes != sourceSize || !output.flush()) fail(QString::fromUtf8("照片复制过程中发生变化或写入失败：") + it.value());
  }
  if (cancelled && cancelled()) fail(QString::fromUtf8("已取消照片准备。"));
  imageRoot = input.path();
  input.setAutoRemove(false);
}
Project Project::load(const QString& path) {
  QFile input(path);
  if (!input.open(QIODevice::ReadOnly)) fail(input.errorString());
  if (input.size() > 4 * 1024 * 1024) fail("Project document exceeds 4 MiB");
  QJsonParseError error;
  const auto document = QJsonDocument::fromJson(input.readAll(), &error);
  const auto object = document.object();
  if (error.error != QJsonParseError::NoError ||
      object["format"] != "colite-project" || object["schema"].toInt() != 1)
    fail(QString::fromUtf8("无法读取项目格式或 schema；原文件不会被覆盖。"));
  Project p;
  p.file = QFileInfo(path).absoluteFilePath();
  p.name = object["name"].toString();
  const QDir base = QFileInfo(p.file).dir();
  auto resolve = [&](const char* key) {
    const auto value = object[key].toString();
    return value.isEmpty()
               ? QString()
               : QDir::cleanPath(base.absoluteFilePath(localPath(value)));
  };
  const auto savedId = object["project_id"].toString();
  if (!savedId.isEmpty()) {
    if (QUuid(savedId).isNull()) fail("Invalid project ID");
    p.id = savedId;
  } else {
    // A legacy document acquires the same identity on each open until saved.
    p.id = QUuid::createUuidV5(QUuid("{ef9df227-a249-484d-9ebb-1f06bbbcfe84}"), p.file).toString(QUuid::WithoutBraces);
  }
  p.workDirectory = resolve("work_directory");
  const auto photos = object["photo_files"].toObject();
  for (auto it = photos.begin(); it != photos.end(); ++it) {
    const auto key = it.key();
    if (key.isEmpty() || key.contains(':') || key.contains('\\') || key.startsWith('/') ||
        key.split('/').contains("..") || QDir::cleanPath(key) != key || it.value().toString().isEmpty())
      fail("Invalid project photo reference");
    p.photoFiles.insert(key, QDir::cleanPath(base.absoluteFilePath(localPath(it.value().toString()))));
  }
  p.imageRoot = resolve("image_root");
  p.workspace = resolve("workspace");
  p.sparse = resolve("sparse");
  p.dense = resolve("dense");
  p.mesh = resolve("mesh");
  const auto backend = object["backend"].toObject();
  p.backend = backend["program"].toString();
  p.backendMode = backend["mode"].toString("native");
  if (p.backendMode == "native" && !p.backend.isEmpty() &&
      (p.backend.contains('/') || p.backend.contains('\\')))
    p.backend = QDir::cleanPath(base.absoluteFilePath(localPath(p.backend)));
  p.distribution = backend["distribution"].toString("Ubuntu-22.04");
  p.parameters = object["parameters"].toObject();
  const auto legacyWork = p.parameters.value("work_directory").toString();
  if (p.workDirectory.isEmpty() && !legacyWork.isEmpty())
    p.workDirectory = QDir::cleanPath(base.absoluteFilePath(localPath(legacyWork)));
  p.parameters.remove("work_directory");
  if (p.name.isEmpty() || (savedId.isEmpty() && (p.imageRoot.isEmpty() || p.workspace.isEmpty())) ||
      (p.backendMode != "native" && p.backendMode != "wsl"))
    fail("Invalid project fields");
  return p;
}
void Project::save(const QString& path) {
  if (name.trimmed().isEmpty() ||
      (backendMode != "native" && backendMode != "wsl"))
    fail("Project name and supported backend mode are required");
  const QDir base = QFileInfo(path).absoluteDir();
  auto relative = [&](const QString& value) {
    return value.isEmpty() ? QString() : base.relativeFilePath(value);
  };
  QJsonObject photos;
  for (auto it = photoFiles.begin(); it != photoFiles.end(); ++it) photos.insert(it.key(), relative(it.value()));
  QJsonObject object{{"project_id", id}, {"work_directory", relative(workDirectory)}, {"photo_files", photos}, {"format", "colite-project"},
                     {"schema", 1},
                     {"name", name},
                     {"image_root", relative(imageRoot)},
                     {"workspace", relative(workspace)},
                     {"sparse", relative(sparse)},
                     {"dense", relative(dense)},
                     {"mesh", relative(mesh)},
                     {"backend", QJsonObject{{"program", backend},
                                             {"mode", backendMode},
                                             {"distribution", distribution}}},
                     {"parameters", parameters}};
  QSaveFile output(path);
  const auto bytes = QJsonDocument(object).toJson();
  if (bytes.size() > 4 * 1024 * 1024) fail("Project document exceeds 4 MiB");
  if (!output.open(QIODevice::WriteOnly) ||
      output.write(bytes) != bytes.size() || !output.commit())
    fail(output.errorString());
  file = QFileInfo(path).absoluteFilePath();
}
}  // namespace colite::gui
