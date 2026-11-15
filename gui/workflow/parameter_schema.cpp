#include "parameter_schema.h"
#include <QFileInfo>
#include <QCoreApplication>
#include <QDir>
#include <QProcess>
#include <QJsonDocument>
#include <QDateTime>
#include <cmath>
#include <stdexcept>

namespace colite::gui {
QJsonObject engineCapabilities() {
  const auto path = QDir(QCoreApplication::applicationDirPath()).filePath(
#ifdef Q_OS_WIN
      "colite.exe"
#else
      "colite"
#endif
      );
  static QString identity;
  static QJsonObject cached;
  const QFileInfo info(path);
  const auto stamp = path + QString::number(info.size()) + QString::number(info.lastModified().toMSecsSinceEpoch());
  if (identity == stamp) return cached;
  identity = stamp; cached = {};
  QProcess probe; probe.start(path, {"--capabilities"});
  if (probe.waitForFinished(3000) && probe.exitCode() == 0)
    cached = QJsonDocument::fromJson(probe.readAllStandardOutput()).object();
  else { probe.kill(); probe.waitForFinished(1000); }
  return cached;
}
QStringList workflowNames() {
  return {QString::fromUtf8("图像导入"), QString::fromUtf8("特征提取"), QString::fromUtf8("特征匹配"),
          QString::fromUtf8("稀疏重建"), QString::fromUtf8("稠密重建"), QString::fromUtf8("表面重建"), QString::fromUtf8("纹理映射")};
}
QString parameterText(const QJsonValue& value, const QString& fallback) {
  return value.isBool() ? (value.toBool() ? "1" : "0") : value.isDouble()
      ? QString::number(value.toDouble(), 'g', 17) : value.toString(fallback);
}
const QVector<ParameterSpec>& parameterSpecs() {
  static const auto specs = [] {
    QVector<ParameterSpec> result;
    using K = ParameterKind;
    auto add = [&](int page, const char* key, const char* label, const char* value,
                   K kind, bool advanced = false, const char* condition = "",
                   double minimum = -1e100, double maximum = 1e100) -> ParameterSpec& {
      result.append({key, QString::fromUtf8(label), value, page, kind, advanced, condition, {}, {}, minimum, maximum});
      return result.last();
    };
    auto choice = [&](int page, const char* key, const char* label, const char* value,
                      const QStringList& values, const QStringList& labels, bool advanced = false, const char* condition = "") {
      auto& spec = add(page, key, label, value, K::Choice, advanced, condition);
      spec.values = values; spec.labels = labels;
    };
    choice(0,"camera_model","相机模型","SIMPLE_RADIAL",{"SIMPLE_RADIAL","PINHOLE","SIMPLE_PINHOLE","RADIAL","OPENCV"},{});
    choice(0,"camera_strategy","相机分组","Auto",{"Auto","Single","Per-folder","Per-image"},{"自动","共用相机","按文件夹","每张照片独立"});
    choice(0,"device","计算设备","auto",{"auto","cpu","cuda"},{"自动","CPU","CUDA"});
    add(0,"ImageReader.camera_params","相机内参（逗号分隔）","",K::Text,true);
    add(0,"ImageReader.default_focal_length_factor","缺少 EXIF 时的焦距系数","1.2",K::Number,true,"",0.001,100);
    add(0,"ImageReader.mask_path","逐图遮罩目录","",K::Directory,true);
    add(0,"ImageReader.camera_mask_path","公共遮罩图像","",K::File,true);
    choice(1,"FeatureExtraction.type","特征方法","sift",{"sift","aliked_n16rot","aliked_n32"},{"SIFT","ALIKED N16Rot","ALIKED N32"});
    add(1,"FeatureExtraction.max_image_size","图像最长边（像素）","3200",K::Integer,false,"",1,100000);
    add(1,"SiftExtraction.max_num_features","每张图像最大特征数","8192",K::Integer,false,"sift",1,1000000);
    add(1,"SiftExtraction.peak_threshold","特征响应阈值","0.006666666666666667",K::Number,false,"sift",1e-12,1);
    add(1,"AlikedExtraction.max_num_features","每张图像最大特征数","2048",K::Integer,false,"learned",1,1000000);
    add(1,"AlikedExtraction.min_score","检测置信度阈值","0.2",K::Number,false,"learned",0,1);
    add(1,"AlikedExtraction.n16rot_model_path","ALIKED N16Rot 模型","",K::File,true,"aliked_n16rot");
    add(1,"AlikedExtraction.n32_model_path","ALIKED N32 模型","",K::File,true,"aliked_n32");
    choice(1,"learned_extraction_device","模型推理设备","auto",{"auto","cpu","cuda"},{"自动","CPU","CUDA"},true,"learned");
    choice(1,"Colite.feature_extractor","SIFT 实现","auto",{"auto","vlsift","cvsift","cuda_vlfeat","cuda_opencv"},{"自动","VLFeat · CPU","OpenCV · CPU","VLFeat · CUDA","OpenCV · CUDA"},true,"sift");
    add(1,"SiftExtraction.edge_threshold","边缘阈值","10",K::Number,true,"sift",0.001,10000);
    choice(1,"SiftExtraction.first_octave","起始尺度","-1",{"-1","0"},{"上采样首层","原始首层"},true,"sift");
    add(1,"SiftExtraction.num_octaves","尺度组数","4",K::Integer,true,"sift",1,16);
    add(1,"SiftExtraction.octave_resolution","每组尺度层数","3",K::Integer,true,"sift",1,16);
    add(1,"SiftExtraction.max_num_orientations","最大方向数","2",K::Integer,true,"sift",1,4);
    choice(1,"descriptor_normalization","描述子归一化","l1_root",{"l1_root","l2"},{"RootSIFT","L2"},true,"sift");
    add(1,"FeatureExtraction.num_threads","线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    choice(2,"matching_strategy","照片配对策略","exhaustive_matcher",{"exhaustive_matcher","sequential_matcher","spatial_matcher"},{"穷举匹配","序列匹配","空间匹配"});
    choice(2,"matching_method","描述子匹配方法","bruteforce",{"bruteforce","lightglue"},{"暴力匹配","LightGlue"});
    choice(2,"learned_matching_device","模型推理设备","auto",{"auto","cpu","cuda"},{"自动","CPU","CUDA"},true,"lightglue");
    add(2,"SequentialMatching.overlap","相邻照片数量","10",K::Integer,false,"sequential",1,10000);
    add(2,"SpatialMatching.max_num_neighbors","最大邻居数量","50",K::Integer,false,"spatial",1,10000);
    add(2,"SpatialMatching.max_distance","最大空间距离","100",K::Number,false,"spatial",0.001,1e9);
    add(2,"SiftMatching.max_ratio","最近邻比值阈值","0.8",K::Number,false,"sift_bf",0.000001,1);
    add(2,"SiftMatching.max_distance","最大描述子距离","0.7",K::Number,true,"sift_bf",0.000001,100);
    add(2,"SiftMatching.cross_check","双向一致性检查","1",K::Boolean,true,"sift_bf");
    add(2,"AlikedMatching.brute_force_min_cossim","最小余弦相似度","0.85",K::Number,false,"aliked_bf",-1,1);
    add(2,"AlikedMatching.brute_force_cross_check","双向一致性检查","1",K::Boolean,true,"aliked_bf");
    add(2,"SiftMatching.lightglue_model_path","SIFT LightGlue 模型","",K::File,true,"sift_lg");
    add(2,"AlikedMatching.lightglue_model_path","ALIKED LightGlue 模型","",K::File,true,"aliked_lg");
    add(2,"SiftMatching.lightglue_min_score","匹配置信度阈值","0.1",K::Number,false,"sift_lg",0,1);
    add(2,"AlikedMatching.lightglue_min_score","匹配置信度阈值","0.1",K::Number,false,"aliked_lg",0,1);
    add(2,"FeatureMatching.guided_matching","引导匹配","0",K::Boolean,false,"sift_bf");
    add(2,"TwoViewGeometry.max_error","几何验证最大误差（像素）","4",K::Number,false,"",0.001,10000);
    add(2,"TwoViewGeometry.min_num_inliers","最少几何内点","15",K::Integer,true,"",0,1000000);
    add(2,"TwoViewGeometry.confidence","RANSAC 置信度","0.999",K::Number,true,"",0,1);
    add(2,"TwoViewGeometry.max_num_trials","RANSAC 最大尝试次数","10000",K::Integer,true,"",100,10000000);
    add(2,"TwoViewGeometry.random_seed","随机种子（-1 随机）","-1",K::Integer,true,"",-1,2147483647);
    add(2,"SequentialMatching.quadratic_overlap","增加远距离序列配对","1",K::Boolean,true,"sequential");
    add(2,"FeatureMatching.num_threads","线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    choice(3,"mapper_method","重建方式","incremental",{"incremental","global"},{"增量式","全局式"});
    add(3,"Mapper.ba_refine_focal_length","优化焦距","1",K::Boolean);
    add(3,"Mapper.ba_refine_principal_point","优化主点","0",K::Boolean);
    add(3,"Mapper.ba_refine_extra_params","优化畸变参数","1",K::Boolean);
    add(3,"Mapper.filter_max_reproj_error","点过滤误差（像素）","4",K::Number,false,"",0.001,10000);
    add(3,"Mapper.filter_min_tri_angle","点过滤最小三角化角度（°）","1.5",K::Number,true,"",0,180);
    add(3,"Mapper.init_image_id1","初始图像 ID 1（-1 自动）","-1",K::Integer,true,"incremental",-1,2147483647);
    add(3,"Mapper.init_image_id2","初始图像 ID 2（-1 自动）","-1",K::Integer,true,"incremental",-1,2147483647);
    add(3,"Mapper.init_min_num_inliers","初始化最少内点","100",K::Integer,true,"incremental",1,1000000);
    add(3,"Mapper.init_min_tri_angle","初始化最小三角化角度（°）","16",K::Number,true,"incremental",0,180);
    add(3,"Mapper.abs_pose_max_error","相机注册最大误差（像素）","12",K::Number,true,"incremental",0.001,10000);
    add(3,"Mapper.abs_pose_min_num_inliers","相机注册最少内点","30",K::Integer,true,"incremental",1,1000000);
    add(3,"Mapper.multiple_models","保留多个独立模型","1",K::Boolean,true,"incremental");
    add(3,"Mapper.ba_local_num_images","局部优化窗口图像数","6",K::Integer,true,"incremental",2,10000);
    add(3,"Mapper.ba_local_max_num_iterations","局部 BA 最大迭代次数","25",K::Integer,true,"incremental",1,100000);
    add(3,"GlobalMapper.min_num_inliers","图像对最少内点","15",K::Integer,true,"global",1,1000000);
    add(3,"GlobalMapper.max_rotation_cycle_error","旋转环路最大误差（°）","10",K::Number,true,"global",0,180);
    add(3,"GlobalMapper.rotation_max_iterations","旋转平均最大迭代次数","100",K::Integer,true,"global",1,100000);
    add(3,"GlobalMapper.max_relative_rotation_error","相对旋转最大误差（°）","10",K::Number,true,"global",0.001,180);
    add(3,"GlobalMapper.max_direction_residual","位置估计最大方向残差","0.25",K::Number,true,"global",0.000001,2);
    add(3,"GlobalMapper.filter_outliers","过滤位置估计异常值","1",K::Boolean,true,"global");
    add(3,"Mapper.tri_min_angle","三角化最小角度（°）","1.5",K::Number,true,"",0,180);
    add(3,"Mapper.ba_global_max_num_iterations","全局 BA 最大迭代次数","50",K::Integer,true,"",1,100000);
    add(3,"Mapper.num_threads","线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    add(4,"PatchMatchStereo.max_image_size","深度图尺寸","50%",K::Integer,false,"",-1,100000);
    add(4,"PatchMatchStereo.geom_consistency","几何一致性","1",K::Boolean);
    add(4,"PatchMatchStereo.num_iterations","深度估计迭代次数","5",K::Integer,false,"",1,1000);
    add(4,"StereoFusion.min_num_pixels","融合最小支持像素数","5",K::Integer,false,"",1,100000);
    add(4,"Undistortion.max_image_size","去畸变最长边（-1 原尺寸）","-1",K::Integer,true,"",-1,100000);
    add(4,"PatchMatchStereo.window_radius","匹配窗口半径","5",K::Integer,true,"",1,32);
    add(4,"PatchMatchStereo.filter_min_ncc","最小 NCC 相似度","0.1",K::Number,true,"",-1,1);
    add(4,"PatchMatchStereo.filter_min_num_consistent","最少一致源图数量","2",K::Integer,true,"",0,100000);
    add(4,"StereoFusion.max_reproj_error","融合最大重投影误差（像素）","2",K::Number,true,"",0.001,10000);
    add(4,"StereoFusion.max_depth_error","融合最大相对深度误差","0.01",K::Number,true,"",0.000001,10);
    add(4,"StereoFusion.max_normal_error","融合最大法线夹角（°）","10",K::Number,true,"",0,180);
    add(4,"Undistortion.num_threads","去畸变线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    add(4,"PatchMatchStereo.num_threads","深度估计线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    add(4,"StereoFusion.num_threads","融合线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    choice(5,"surface_method","表面算法","poisson",{"poisson","delaunay"},{"Poisson","Delaunay"});
    add(5,"PoissonMeshing.depth","重建深度","9",K::Integer,false,"poisson",1,16);
    add(5,"PoissonMeshing.trim","低密度区域裁剪阈值","10",K::Number,false,"poisson",0,1000);
    add(5,"PoissonMeshing.point_weight","点权重","1",K::Number,true,"poisson",0,10000);
    add(5,"PoissonMeshing.color","保留顶点颜色","1",K::Boolean,true,"poisson");
    add(5,"DelaunayMeshing.quality_regularization","质量正则化","1",K::Number,false,"delaunay",0,10000);
    add(5,"DelaunayMeshing.max_side_length_factor","最大边长系数","25",K::Number,true,"delaunay",0.001,10000);
    add(5,"PoissonMeshing.num_threads","线程数（-1 自动）","-1",K::Integer,true,"poisson",-1,4096);
    add(5,"DelaunayMeshing.num_threads","线程数（-1 自动）","-1",K::Integer,true,"delaunay",-1,4096);
    add(5,"simplify_mesh","简化网格","0",K::Boolean);
    add(5,"MeshSimplification.target_face_ratio","保留三角形比例","0.1",K::Number,false,"simplify",0.000001,1);
    add(5,"MeshSimplification.boundary_weight","边界保留权重","1000",K::Number,true,"simplify",0,1e9);
    add(6,"MeshTextureMapping.texture_scale_factor","纹理缩放比例","1",K::Number,false,"",0.000001,1);
    add(6,"Colite.texture_max_atlas_size","最大纹理图集边长","32768",K::Integer,false,"",4096,32768);
    add(6,"MeshTextureMapping.apply_color_correction","颜色校正","1",K::Boolean);
    add(6,"MeshTextureMapping.min_cos_normal_angle","法线夹角余弦阈值","0.1",K::Number,true,"",-1,1);
    add(6,"MeshTextureMapping.view_selection_smoothing_iterations","视图选择平滑次数","3",K::Integer,true,"",0,1000);
    add(6,"MeshTextureMapping.atlas_patch_padding","纹理图块边距（像素）","2",K::Integer,true,"",0,1024);
    add(6,"MeshTextureMapping.inpaint_radius","纹理孔洞填补半径","5",K::Integer,true,"",0,1024);
    add(6,"Colite.texture_visibility_resolution","可见性计算分辨率","1024",K::Integer,true,"",16,32768);
    add(6,"MeshTextureMapping.num_threads","线程数（-1 自动）","-1",K::Integer,true,"",-1,4096);
    return result;
  }();
  return specs;
}
QString parameterDefault(const ParameterSpec& spec, const QJsonObject& p) {
  const QMap<QString, QString> models{
      {"AlikedExtraction.n16rot_model_path", "aliked-n16rot.onnx"},
      {"AlikedExtraction.n32_model_path", "aliked-n32.onnx"},
      {"AlikedMatching.lightglue_model_path", "aliked-lightglue.onnx"},
      {"SiftMatching.lightglue_model_path", "sift-lightglue.onnx"}};
  if (models.contains(spec.key)) {
    for (const auto& folder : {QString("../models"), QString("models")}) {
      const auto path = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(folder + '/' + models.value(spec.key));
      if (QFileInfo(path).isFile()) return QDir::cleanPath(path);
    }
  }
  if (spec.key == "FeatureExtraction.max_image_size" && p.value("FeatureExtraction.type").toString("sift") != "sift") return "1600";
  return spec.fallback;
}
bool parameterActive(const ParameterSpec& s, const QJsonObject& p) {
  const auto condition = s.condition;
  const auto feature = p.value("FeatureExtraction.type").toString("sift");
  const bool sift = feature == "sift", bf = p.value("matching_method").toString("bruteforce") == "bruteforce";
  if (condition.isEmpty()) return true;
  if (condition == "sift") return sift;
  if (condition == "learned") return !sift;
  if (condition == "lightglue") return !bf;
  if (condition.startsWith("aliked_n")) return condition == feature;
  if (condition == "sift_bf") return sift && bf;
  if (condition == "aliked_bf") return !sift && bf;
  if (condition == "sift_lg") return sift && !bf;
  if (condition == "aliked_lg") return !sift && !bf;
  if (condition == "sequential" || condition == "spatial") return p.value("matching_strategy").toString() == condition + "_matcher";
  if (condition == "incremental" || condition == "global") return p.value("mapper_method").toString("incremental") == condition;
  if (condition == "poisson" || condition == "delaunay") return p.value("surface_method").toString("poisson") == condition;
  if (condition == "simplify") return parameterText(p.value("simplify_mesh"), "0") == "1";
  return false;
}
QJsonObject effectiveParameters(const QJsonObject& p) {
  auto effective = p;
  for (const auto& s : parameterSpecs()) {
    if (!parameterActive(s, p)) { effective.remove(s.key); continue; }
    const auto text = parameterText(p.value(s.key), parameterDefault(s, p));
    if (text.isEmpty()) effective.remove(s.key);
    else effective[s.key] = text;
  }
  return effective;
}
void validateParameterValues(const QJsonObject& parameters) {
  const auto p = effectiveParameters(parameters);
  for (const auto& s : parameterSpecs()) {
    if (!parameterActive(s, parameters)) continue;
    const auto text = parameterText(p.value(s.key));
    auto fail = [&](const QString& detail) { throw std::invalid_argument((s.label + "：" + detail + " (" + s.key + ")").toStdString()); };
    if (s.kind == ParameterKind::Choice) { if (!s.values.contains(text)) fail(QString::fromUtf8("不支持的选项")); }
    else if (s.kind == ParameterKind::Boolean) { if (text != "0" && text != "1") fail(QString::fromUtf8("需要开关值")); }
    else if (s.kind == ParameterKind::Number || s.kind == ParameterKind::Integer) {
      if (s.key == "PatchMatchStereo.max_image_size" && text == "50%") continue;
      bool ok; const double value = text.toDouble(&ok);
      if (!ok || !std::isfinite(value) || value < s.minimum || value > s.maximum ||
          (s.kind == ParameterKind::Integer && std::floor(value) != value) ||
          ((s.key.endsWith("num_threads") || s.key.endsWith("max_image_size")) && value == 0))
        fail(QString::fromUtf8("数值超出有效范围"));
    } else if (s.kind == ParameterKind::File || s.kind == ParameterKind::Directory) {
      if (text.isEmpty() && s.key.contains("model_path")) fail(QString::fromUtf8("请选择模型文件"));
      if (!text.isEmpty() && !(s.kind == ParameterKind::File ? QFileInfo(text).isFile() : QFileInfo(text).isDir()))
        fail(QString::fromUtf8("路径不存在或类型不正确"));
    }
  }
  if (p.value("mapper_method") == "incremental" && p.value("Mapper.init_image_id1") != "-1" &&
      p.value("Mapper.init_image_id1") == p.value("Mapper.init_image_id2"))
    throw std::invalid_argument("Mapper.init_image_id2: initial image IDs must differ");
}
}
