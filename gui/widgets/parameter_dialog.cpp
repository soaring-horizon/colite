#include "parameter_dialog.h"
#include "workflow/workflow_state.h"
#include "workflow/parameter_schema.h"
#include <algorithm>
#include <stdexcept>

namespace colite::gui {
ParameterDialog::ParameterDialog(const Project& project, int stage, bool singleStage,
                                 bool canRun, QWidget* parent)
    : QDialog(parent), original_(project), acceptedProject_(project), acceptedParameters_(project.parameters) {
  setObjectName("parameterDialog");
  const auto names = workflowNames();
  stage = qBound(0, stage, 6);
  setWindowTitle(singleStage ? tr("%1 · 参数").arg(names[stage]) : tr("流程参数"));
  resize(QSize(singleStage ? 700 : 880, 680).boundedTo(
      QGuiApplication::primaryScreen()->availableGeometry().size() - QSize(40, 60)));
  auto* layout = new QVBoxLayout(this);
  layout->setContentsMargins(20, 18, 20, 18);
  layout->setSpacing(14);
  auto* content = new QHBoxLayout;
  navigation_ = new QListWidget;
  navigation_->setObjectName("parameterStages");
  navigation_->addItems(names);
  navigation_->setFixedWidth(150);
  navigation_->setVisible(!singleStage);
  content->addWidget(navigation_);
  pages_ = new QStackedWidget;
  pages_->setObjectName("parameterPages");
  content->addWidget(pages_, 1);
  layout->addLayout(content, 1);
  QVector<QVBoxLayout*> forms, advancedForms;
  const QStringList descriptions{
    tr("相机配置影响后续特征提取。"),
    tr("从照片提取可用于匹配的特征点。较大的图像尺寸与特征数量通常需要更多时间。"),
    tr("建立照片间的对应关系，并进行几何验证。"),
    tr("估计相机位置和稀疏三维点，优化相机内参。"),
    tr("依次完成去畸变、深度估计和点云融合。"),
    tr("从稠密点云生成连续三角网格。深度越大，计算和内存开销通常越高。"),
    tr("把照片颜色映射到网格，生成带纹理的模型。")};
  for (int i = 0; i < names.size(); ++i) {
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame);
    auto* page = new QWidget;
    auto* body = new QVBoxLayout(page);
    auto* heading = new QLabel(names[i]); heading->setObjectName("parameterHeading");
    body->addWidget(heading);
    auto* description = new QLabel(descriptions[i]);
    description->setWordWrap(true); description->setObjectName("parameterDescription");
    body->addWidget(description);
    auto* form = new QVBoxLayout;
    form->setSpacing(14);
    body->addLayout(form);
    auto* toggle = new QToolButton;
    toggle->setText(tr("高级参数")); toggle->setCheckable(true);
    toggle->setObjectName("advancedParameters" + QString::number(i));
    toggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon); toggle->setArrowType(Qt::RightArrow);
    auto* advanced = new QWidget;
    auto* advancedForm = new QVBoxLayout(advanced);
    advancedForm->setContentsMargins(0, 6, 0, 0); advancedForm->setSpacing(14);
    advanced->hide();
    connect(toggle, &QToolButton::toggled, this, [=](bool checked) {
      advanced->setVisible(checked); toggle->setArrowType(checked ? Qt::DownArrow : Qt::RightArrow);
    });
    body->addWidget(toggle); body->addWidget(advanced); body->addStretch();
    advancedAreas_ << advanced; advancedToggles_ << toggle; advancedForms << advancedForm;
    forms << form; scroll->setWidget(page); pages_->addWidget(scroll);
  }
  auto* source = new QLabel(project.imageRoot.isEmpty() ? tr("已添加的照片 / 文件夹") : project.imageRoot);
  source->setWordWrap(true); source->setTextFormat(Qt::PlainText);
  source->setTextInteractionFlags(Qt::TextSelectableByMouse);
  forms[0]->addWidget(new QLabel(tr("照片来源")));
  forms[0]->addWidget(source);
  for (const auto& spec : parameterSpecs()) {
    const auto initial = parameterText(project.parameters.value(spec.key), parameterDefault(spec, project.parameters));
    QWidget* editor = nullptr;
    QWidget* row = nullptr;
    if (spec.kind == ParameterKind::Choice) {
      auto* combo = new QComboBox;
      for (int i = 0; i < spec.values.size(); ++i) {
        combo->addItem(spec.labels.value(i, spec.values[i]), spec.values[i]);
        const auto capabilities = engineCapabilities();
        const bool unavailable = (spec.key == "FeatureExtraction.type" && i > 0 && !capabilities["learned_features"].toBool()) ||
            (spec.key.startsWith("learned_") && spec.values[i] == "cuda" && !capabilities["learned_cuda"].toBool()) ||
            (spec.key == "matching_method" && i > 0 && !capabilities["learned_features"].toBool()) ||
            (spec.key == "surface_method" && i > 0 && !capabilities["delaunay"].toBool()) ||
            (spec.key == "mapper_method" && i > 0 && !capabilities["global_mapping"].toBool());
        if (unavailable) {
          auto* item = static_cast<QStandardItemModel*>(combo->model())->item(i);
          item->setEnabled(false); item->setToolTip(tr("当前引擎未包含此功能"));
          item->setText(combo->itemText(i) + tr("（不可用）"));
        }
      }
      int index = combo->findData(initial);
      if (index < 0) { combo->addItem(initial, initial); index = combo->count() - 1; }
      combo->setCurrentIndex(index); editor = combo;
      connect(combo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this] { updateFields(); });
    } else if (spec.kind == ParameterKind::Boolean) {
      auto* check = new QCheckBox;
      check->setChecked(initial == "1"); editor = check;
      connect(check, &QCheckBox::toggled, this, [this] { updateFields(); });
    } else {
      auto* edit = new QLineEdit(initial); editor = edit;
      if (spec.kind == ParameterKind::File || spec.kind == ParameterKind::Directory) {
        row = new QWidget;
        auto* pathRow = new QHBoxLayout(row); pathRow->setContentsMargins(0,0,0,0); pathRow->setSpacing(6);
        auto* browse = new QPushButton(tr("选择…")); browse->setAutoDefault(false);
        pathRow->addWidget(edit, 1); pathRow->addWidget(browse);
        connect(browse, &QPushButton::clicked, this, [=] {
          const auto path = spec.kind == ParameterKind::Directory
              ? QFileDialog::getExistingDirectory(this, spec.label, edit->text())
              : QFileDialog::getOpenFileName(this, spec.label, edit->text(), spec.key.contains("model_path") ? tr("ONNX 模型 (*.onnx)") : tr("图像 (*.png *.jpg *.jpeg *.tif *.tiff)"));
          if (!path.isEmpty()) edit->setText(path);
        });
      }
    }
    editor->setObjectName(spec.key); editor->setAccessibleName(spec.label);
    if (spec.key == "PatchMatchStereo.max_image_size")
      editor->setToolTip(tr("50%：每张照片按宽高的一半计算。-1：原尺寸。也可输入最长边的像素数。"));
    else if (spec.kind == ParameterKind::Number || spec.kind == ParameterKind::Integer)
      editor->setToolTip(tr("有效范围：%1 至 %2").arg(spec.minimum).arg(spec.maximum));
    if (!row) row = editor;
    auto* form = spec.advanced ? advancedForms[spec.page] : forms[spec.page];
    // Hide a complete layout item so Qt 5 does not retain the spacing of
    // inactive QFormLayout rows when changing algorithms.
    auto* container = new QWidget;
    auto* rowLayout = new QHBoxLayout(container);
    rowLayout->setContentsMargins(0, 0, 0, 0); rowLayout->setSpacing(10);
    auto* label = new QLabel(spec.label); label->setWordWrap(true);
    label->setFixedWidth(220); label->setBuddy(editor);
    rowLayout->addWidget(label); rowLayout->addWidget(row, 1);
    form->addWidget(container);
    fields_.append({spec.key, initial, editor, spec.page, container, label});
  }
  updateFields();

  follow_ = new QCheckBox(tr("从当前节点继续运行到纹理映射"));
  follow_->setObjectName("followStages");
  layout->addWidget(follow_);
  auto* summary = new QLabel;
  summary->setObjectName("parameterRunSummary"); summary->setWordWrap(true);
  layout->addWidget(summary);
  error_ = new QLabel; error_->setObjectName("parameterError"); error_->setWordWrap(true);
  error_->hide(); layout->addWidget(error_);
  auto* directoryPanel = new QWidget;
  directoryPanel->setObjectName("parameterDirectoryPanel");
  auto* directoryRow = new QHBoxLayout(directoryPanel);
  directoryRow->setContentsMargins(0, 16, 0, 12);
  directoryRow->setSpacing(12);
  auto* directoryLabel = new QLabel(tr("工作目录"));
  initialDirectory_ = QDir::cleanPath(project.workDirectory.isEmpty()
      ? Project::defaultWorkDirectory() : project.workDirectory);
  workDirectory_ = new QLineEdit(QDir::toNativeSeparators(initialDirectory_));
  workDirectory_->setObjectName("parameterWorkDirectory");
  workDirectory_->setAccessibleName(tr("工作目录"));
  workDirectory_->setPlaceholderText(QDir::toNativeSeparators(Project::defaultWorkDirectory()));
  directoryLabel->setBuddy(workDirectory_);
  auto* browse = new QPushButton(QIcon(":/ui/FolderOpen.png"), tr("更改…"));
  browse->setObjectName("browseParameterWorkDirectory");
  browse->setAutoDefault(false);
  browse->setToolTip(tr("选择工作目录"));
  directoryRow->addWidget(directoryLabel);
  directoryRow->addWidget(workDirectory_, 1);
  directoryRow->addWidget(browse);
  layout->addWidget(directoryPanel);
  connect(browse, &QPushButton::clicked, this, [this] {
    const auto path = QFileDialog::getExistingDirectory(this, tr("选择工作目录"), directoryValue());
    if (!path.isEmpty()) workDirectory_->setText(QDir::toNativeSeparators(path));
  });
  auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
  buttons->button(QDialogButtonBox::Save)->setText(tr("保存参数"));
  buttons->button(QDialogButtonBox::Save)->setObjectName("saveParameters");
  buttons->button(QDialogButtonBox::Cancel)->setText(tr("取消"));
  auto* run = buttons->addButton(tr("运行节点"), QDialogButtonBox::ActionRole);
  run->setObjectName("runStage"); run->setDefault(false); run->setAutoDefault(false);
  layout->addWidget(buttons);
  QVector<bool> previouslyRun(7, false);
  for (int i = 1; i <= 6; ++i) {
    const auto outputs = stageOutputFiles(project, i);
    previouslyRun[i] = !outputs.isEmpty() && std::all_of(outputs.begin(), outputs.end(),
        [](const QString& path) { return QFileInfo(path).isFile() && QFileInfo(path).size() > 0; });
  }
  auto update = [=] {
    const int current = navigation_->currentRow();
    pages_->setCurrentIndex(current);
    follow_->setVisible(!singleStage && current > 0 && current < 6);
    if (!follow_->isVisible()) follow_->setChecked(false);
    const bool newDirectory = directoryChanged();
    run->setVisible(current > 0);
    run->setEnabled(canRun && current > 0 && (!newDirectory || current == 1));
    run->setText(!newDirectory && previouslyRun.value(current) ? tr("重新运行节点") : tr("运行节点"));
    summary->setText(newDirectory ? tr("工作目录已更改，原结果保留。请从特征提取开始运行。")
        : current == 0 ? tr("图像配置会在下次特征提取时使用；照片路径请通过“项目数据”调整。")
        : tr("运行范围：%1。使用当前页面中更新后的参数；受影响的后续结果会标记为待更新。")
              .arg(follow_->isChecked() ? names[current] + tr(" → 纹理映射") : names[current]));
  };
  connect(navigation_, &QListWidget::currentRowChanged, this, [=] { update(); error_->hide(); });
  connect(follow_, &QCheckBox::toggled, this, update);
  connect(workDirectory_, &QLineEdit::textChanged, this, [=] { update(); error_->hide(); });
  navigation_->setCurrentRow(stage);
  connect(run, &QPushButton::clicked, this, [this] { acceptParameters(true); });
  connect(buttons->button(QDialogButtonBox::Save), &QPushButton::clicked, this, [this] { acceptParameters(false); });
  connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
}
QJsonObject ParameterDialog::draftParameters() const {
  auto result = original_.parameters;
  for (const auto& field : fields_) {
    QString value;
    if (auto* edit = qobject_cast<QLineEdit*>(field.editor)) value = edit->text().trimmed();
    else if (auto* combo = qobject_cast<QComboBox*>(field.editor)) value = combo->currentData().toString();
    else value = static_cast<QCheckBox*>(field.editor)->isChecked() ? "1" : "0";
    if (value != field.initial) result[field.key] = value;
  }
  return result;
}
void ParameterDialog::updateFields() {
  const auto parameters = draftParameters();
  for (auto& field : fields_) {
    const auto& specs = parameterSpecs();
    auto found = std::find_if(specs.begin(), specs.end(), [&](const ParameterSpec& s) { return s.key == field.key; });
    if (found == specs.end()) continue;
    const bool active = parameterActive(*found, parameters);
    field.row->setVisible(active); if (field.label) field.label->setVisible(active);
    if (field.key == "FeatureExtraction.max_image_size" && !original_.parameters.contains(field.key)) {
      auto* edit = static_cast<QLineEdit*>(field.editor);
      if (edit->text() == field.initial) { field.initial = parameterDefault(*found, parameters); edit->setText(field.initial); }
    }
  }
}
QString ParameterDialog::directoryValue() const {
  const auto text = workDirectory_->text().trimmed();
  return QDir::cleanPath(text.isEmpty() ? Project::defaultWorkDirectory() : QDir::fromNativeSeparators(text));
}
bool ParameterDialog::directoryChanged() const {
  return QDir(directoryValue()) != QDir(initialDirectory_);
}
void ParameterDialog::acceptParameters(bool run) {
  auto candidate = original_;
  candidate.parameters = draftParameters();
  try {
    if (directoryChanged()) {
      const auto root = directoryValue();
      if (!QDir::isAbsolutePath(root))
        throw std::runtime_error(tr("工作目录需要使用绝对路径。").toStdString());
      auto existing = QFileInfo(root);
      while (!existing.exists() && existing.absoluteFilePath() != existing.dir().absolutePath())
        existing = QFileInfo(existing.dir().absolutePath());
      if (existing.exists() && !existing.isDir())
        throw std::runtime_error(tr("工作目录或其上级路径是文件，请选择文件夹。").toStdString());
      candidate.workspace.clear();
      candidate.workDirectory = root;
      candidate.ensureOutputs();
      candidate.parameters.remove("sparse_model");
      if (run && selectedStage() != 1)
        throw std::runtime_error(tr("工作目录已更改，请从特征提取开始运行。").toStdString());
    }
    validateParameters(candidate);
  }
  catch (const std::exception& e) {
    const auto message = QString::fromUtf8(e.what());
    for (const auto& field : fields_) if (message.contains(field.key)) {
      navigation_->setCurrentRow(field.stage); advancedToggles_[field.stage]->setChecked(true); field.editor->setFocus(); break;
    }
    error_->setText(message); error_->show(); return;
  }
  acceptedParameters_ = candidate.parameters;
  acceptedProject_ = candidate;
  launch_ = run; accept();
}
}
