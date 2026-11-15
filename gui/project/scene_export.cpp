#include "scene_export.h"
#include "workspace.h"
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>
#include <stdexcept>

namespace colite::gui {
void exportScene(const QString& source, const QString& destination,
                 const std::function<bool()>& cancelled) {
  const QFileInfo target(destination);
  if (target.exists() || !target.dir().exists())
    throw std::runtime_error("Choose a new export directory inside an existing folder");
  const auto check = [&] {
    if (cancelled && cancelled()) throw std::runtime_error("Export cancelled");
  };
  check();
  if (!QFileInfo::exists(source)) throw std::runtime_error("Model does not exist");
  QTemporaryDir staging(target.dir().filePath(".colite-export-XXXXXX"));
  if (!staging.isValid()) throw std::runtime_error("Cannot create export directory");
  const auto copy = [&](const QString& path, const QString& name, bool ply) {
    QFile input(path), output(QDir(staging.path()).filePath(name));
    if (!input.open(QIODevice::ReadOnly) || !output.open(QIODevice::WriteOnly))
      throw std::runtime_error("Cannot open export file");
    if (ply) {
      bool ended = false;
      for (int i = 0; i < 10000 && !input.atEnd(); ++i) {
        auto line = input.readLine(16384);
        if (line.startsWith("comment TextureFile ")) {
          const auto original = QString::fromLatin1(line.mid(20)).trimmed();
          line = "comment TextureFile texture." + QFileInfo(original).suffix().toLatin1() + "\n";
        }
        if (output.write(line) != line.size()) throw std::runtime_error("Cannot write export header");
        if (line.trimmed() == "end_header") { ended = true; break; }
      }
      if (!ended) throw std::runtime_error("Invalid PLY export header");
    }
    while (!input.atEnd()) {
      check();
      const auto bytes = input.read(1024 * 1024);
      if (bytes.isEmpty() || output.write(bytes) != bytes.size()) throw std::runtime_error("Export copy failed");
    }
    if (!output.flush()) throw std::runtime_error("Cannot flush export file");
  };
  if (QFileInfo(source).isDir()) {
    for (const auto& name : {"cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt", "points3D.ply"}) {
      const auto file = QDir(source).filePath(name);
      if (QFileInfo(file).isFile()) copy(file, name, false);
    }
  } else {
    copy(source, "model.ply", true);
    QFile header(source);
    if (!header.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot read model header");
    for (int i = 0; i < 10000 && !header.atEnd(); ++i) {
      const auto line = header.readLine(16384).trimmed();
      if (line.startsWith("comment TextureFile ")) {
        const auto texture = QFileInfo(source).dir().filePath(QString::fromLatin1(line.mid(20)));
        copy(texture, "texture." + QFileInfo(texture).suffix(), false);
      }
      if (line == "end_header") break;
    }
  }
  check();
  const auto result = validateScene(QFileInfo(source).isDir() ? staging.path() : QDir(staging.path()).filePath("model.ply"), cancelled);
  if (!result.points) throw std::runtime_error("Exported model contains no points");
  check();
  if (!QDir().rename(staging.path(), target.absoluteFilePath()))
    throw std::runtime_error("Cannot commit export directory (destination may already exist)");
  staging.setAutoRemove(false);
}
}
