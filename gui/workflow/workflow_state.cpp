#include "workflow_state.h"
#include "project/workspace.h"
#include "parameter_schema.h"
#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSaveFile>
#include <cmath>
#include <stdexcept>

namespace colite::gui {
namespace {
QString receiptPath(const Project& project) { return QDir(project.workspace).filePath("gui-stage-state.json"); }
QJsonObject receipts(const Project& project) {
  QFile file(receiptPath(project));
  if (project.workspace.isEmpty() || !file.open(QIODevice::ReadOnly) || file.size() > 1024 * 1024) return {};
  return QJsonDocument::fromJson(file.readAll()).object();
}
void writeReceipts(const Project& project, const QJsonObject& value) {
  if (project.workspace.isEmpty() || !QDir().mkpath(project.workspace))
    throw std::runtime_error("Cannot create stage state directory");
  QSaveFile file(receiptPath(project));
  const auto bytes = QJsonDocument(value).toJson();
  if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() || !file.commit())
    throw std::runtime_error("Cannot persist stage validity");
}
QByteArray fileStamp(const QString& path) {
  const QFileInfo info(path);
  return info.absoluteFilePath().toUtf8() + '\0' + QByteArray::number(info.size()) + ':' +
      QByteArray::number(info.lastModified().toMSecsSinceEpoch()) + ':' + QByteArray::number(info.exists());
}
QString filesStamp(const QStringList& paths) {
  QCryptographicHash hash(QCryptographicHash::Sha256);
  for (const auto& path : paths) { hash.addData(fileStamp(path)); hash.addData("\n"); }
  return QString::fromLatin1(hash.result().toHex());
}
int parameterStage(const QString& key) {
  for (const auto& spec : parameterSpecs()) if (spec.key == key) return qMax(1, spec.page);
  if (key.startsWith("SiftExtraction.") || key.startsWith("FeatureExtraction.") || key.startsWith("camera_")) return 1;
  if (key.startsWith("SiftMatching.") || key.startsWith("FeatureMatching.") || key == "matching_strategy") return 2;
  if (key.startsWith("Mapper.")) return 3;
  if (key == "sparse_model" || key.startsWith("Undistortion.") || key.startsWith("PatchMatchStereo.") || key.startsWith("StereoFusion.")) return 4;
  if (key.startsWith("PoissonMeshing.")) return 5;
  if (key.startsWith("MeshTextureMapping.")) return 6;
  return key == "device" ? 1 : 99;
}
QString inputStamp(const Project& project, int stage) {
  QJsonObject params;
  const auto effective = effectiveParameters(project.parameters);
  for (auto it = effective.begin(); it != effective.end(); ++it)
    if (parameterStage(it.key()) <= stage) params[it.key()] = it.value();
  QCryptographicHash hash(QCryptographicHash::Sha256);
  hash.addData(QJsonDocument(params).toJson(QJsonDocument::Compact));
  // Model/mask replacements at the same path must invalidate their consumers.
  QStringList resources;
  for (const auto& spec : parameterSpecs()) {
    if (qMax(1, spec.page) > stage || !effective.contains(spec.key)) continue;
    const auto path = effective.value(spec.key).toString();
    if (spec.kind == ParameterKind::File) resources << path;
    else if (spec.kind == ParameterKind::Directory) {
      resources << path;
      QDirIterator iterator(path, QDir::Files, QDirIterator::Subdirectories);
      while (iterator.hasNext()) resources << iterator.next();
    }
  }
  resources.sort(); hash.addData(filesStamp(resources).toUtf8());
  auto sources = project;
  sources.workspace.clear();
  if (!sources.imageRoot.isEmpty()) sources.photoFiles.clear();
  QStringList paths;
  for (const auto& image : Workspace::read(sources).images) paths << image.path;
  paths.sort();
  hash.addData(filesStamp(paths).toUtf8());
  for (int i = 1; i < stage; ++i) hash.addData(filesStamp(stageOutputFiles(project, i)).toUtf8());
  if (stage >= 4 && project.activeSparse() != project.sparse) {
    auto selected = project; selected.sparse = project.activeSparse();
    hash.addData(filesStamp(stageOutputFiles(selected, 3)).toUtf8());
  }
  return QString::fromLatin1(hash.result().toHex());
}
}
QStringList stageOutputFiles(const Project& project, int stage) {
  if (project.workspace.isEmpty()) return {};
  const QDir work(project.workspace);
  if (stage == 1) {
    QStringList files{work.filePath("images.tsv")};
    for (const auto& image : Workspace::read(project).images)
      files << work.filePath(QString("features/%1.bin").arg(image.id));
    return files;
  }
  if (stage == 2) return {work.filePath("matches/raw_matches.tsv"), work.filePath("matches/verified_matches.tsv")};
  if (stage == 3) {
    const QDir root(project.sparse);
    const QString extension = root.exists("points3D.bin") ? ".bin" : ".txt";
    return {root.filePath("cameras" + extension), root.filePath("images" + extension), root.filePath("points3D" + extension)};
  }
  if (stage == 4) return {QDir(project.dense).filePath("fused.ply")};
  if (stage == 5) return {project.mesh};
  if (stage == 6) return {QFileInfo(project.mesh).dir().filePath("textured/mesh.ply"),
                          QFileInfo(project.mesh).dir().filePath("textured/texture.png")};
  return {};
}
OutputState outputState(const Project& project, int stage) {
  const auto files = stageOutputFiles(project, stage);
  if (files.isEmpty()) return OutputState::Missing;
  for (const auto& file : files) if (!QFileInfo(file).isFile() || !QFileInfo(file).size()) return OutputState::Missing;
  const auto record = receipts(project).value(QString::number(stage)).toObject();
  if (record.isEmpty()) return OutputState::Unverified;
  if (record["invalidated"].toBool() || record["input"] != inputStamp(project, stage) ||
      record["output"] != filesStamp(files)) return OutputState::Stale;
  return OutputState::Current;
}
QString outputStateText(OutputState state) {
  switch (state) {
    case OutputState::Missing: return QString::fromUtf8("缺少产物");
    case OutputState::Unverified: return QString::fromUtf8("已有产物 · 未核验当前配置");
    case OutputState::Current: return QString::fromUtf8("产物有效 · 可复用");
    case OutputState::Stale: return QString::fromUtf8("产物已过期 · 需要重新计算");
  }
  return {};
}
void invalidateStages(const Project& project, int first) {
  auto records = receipts(project);
  for (int i = first; i <= 6; ++i) records[QString::number(i)] = QJsonObject{{"invalidated", true}};
  writeReceipts(project, records);
}
QString stageInputStamp(const Project& project, int stage) { return inputStamp(project, stage); }
void recordStage(const Project& project, int stage, const QString& expectedInput) {
  const auto input = inputStamp(project, stage);
  if (!expectedInput.isEmpty() && input != expectedInput)
    throw std::runtime_error("Inputs changed while processing; rerun the current stage");
  auto records = receipts(project);
  records[QString::number(stage)] = QJsonObject{{"input", input},
      {"output", filesStamp(stageOutputFiles(project, stage))}};
  writeReceipts(project, records);
}
void validateParameters(const Project& project) {
  validateParameterValues(project.parameters);
}
void checkStageInputs(const Project& project, int first) {
  validateParameters(project);
  const auto images = Workspace::read(project).images;
  if (images.size() < (first == 1 ? 1 : 2))
    throw std::invalid_argument("The selected step needs more source photos");
  for (const auto& image : images) if (!QFileInfo(image.path).isFile())
    throw std::invalid_argument((QString::fromUtf8("缺少照片，请重新定位：") + image.path).toStdString());
  const QVector<QVector<int>> dependencies{{}, {}, {1}, {1, 2}, {3}, {4}, {4, 5}};
  for (int i : dependencies.value(first)) {
    const auto state = outputState(project, i);
    if (state == OutputState::Missing || state == OutputState::Stale)
      throw std::invalid_argument((QString::fromUtf8("前置步骤 %1：%2。请从该步骤重新运行。").arg(i).arg(outputStateText(state))).toStdString());
  }
  if (first == 4 && project.activeSparse() != project.sparse) {
    auto selected = project; selected.sparse = project.activeSparse();
    for (const auto& file : stageOutputFiles(selected, 3))
      if (!QFileInfo(file).isFile()) throw std::invalid_argument("Selected sparse model is missing; choose another model");
  }
}
}
