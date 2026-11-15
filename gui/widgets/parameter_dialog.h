#pragma once
#include <QtWidgets>
#include "project/project.h"

namespace colite::gui {
class ParameterDialog : public QDialog {
 public:
  ParameterDialog(const Project& project, int stage, bool singleStage,
                  bool canRun, QWidget* parent = nullptr);
  QJsonObject parameters() const { return acceptedParameters_; }
  Project project() const { return acceptedProject_; }
  int selectedStage() const { return navigation_->currentRow(); }
  bool runRequested() const { return launch_; }
  bool followStages() const { return follow_->isChecked(); }
 private:
  Project original_;
  Project acceptedProject_;
  QJsonObject acceptedParameters_;
  QListWidget* navigation_;
  QStackedWidget* pages_;
  QCheckBox* follow_;
  QLabel* error_;
  QLineEdit* workDirectory_;
  QString initialDirectory_;
  QString directoryValue() const;
  bool directoryChanged() const;
  struct Field { QString key, initial; QWidget* editor; int stage; QWidget* row; QWidget* label; };
  QVector<Field> fields_;
  QVector<QWidget*> advancedAreas_;
  QVector<QToolButton*> advancedToggles_;
  void updateFields();
  QJsonObject draftParameters() const;
  bool launch_ = false;
  void acceptParameters(bool run);
};
}
