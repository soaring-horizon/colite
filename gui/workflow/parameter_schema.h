#pragma once
#include <QJsonObject>
#include <QStringList>
#include <QVector>

namespace colite::gui {
enum class ParameterKind { Number, Integer, Boolean, Choice, File, Directory, Text };
struct ParameterSpec {
  QString key, label, fallback;
  int page;
  ParameterKind kind;
  bool advanced = false;
  QString condition;
  QStringList values, labels;
  double minimum = -1e100, maximum = 1e100;
};
const QVector<ParameterSpec>& parameterSpecs();
QString parameterText(const QJsonValue& value, const QString& fallback = {});
QString parameterDefault(const ParameterSpec& spec, const QJsonObject& parameters);
bool parameterActive(const ParameterSpec& spec, const QJsonObject& parameters);
QJsonObject effectiveParameters(const QJsonObject& parameters);
void validateParameterValues(const QJsonObject& parameters);
QStringList workflowNames();
QJsonObject engineCapabilities();
}
