# Colite 命令行参考

本文对应 Colite 0.2.0。参数解析器和当前构建的帮助是最终依据：

```bash
COLITE=colite
$COLITE --help
$COLITE <command> --help
```

统一使用 `colite <command>`。少数命令同时生成独立可执行文件，但参数相同。所有示例输出到新的 `output/<功能>_<场景>_<日期>/`，不要写入构建目录。

## 命令总览

| 命令 | 输入 | 输出/职责 |
|---|---|---|
| `feature_extractor` | 原图 | SfM workspace、特征缓存 |
| `exhaustive_matcher` | SfM workspace | 全 pair 匹配和几何 |
| `sequential_matcher` | SfM workspace | 时序邻近匹配和几何 |
| `spatial_matcher` | 带位置先验 workspace | 空间邻近匹配和几何 |
| `vocab_tree_builder` | 特征 workspace | 持久层次词树 |
| `vocab_tree_matcher` | workspace、可选词树 | 检索候选匹配和几何 |
| `transitive_matcher` | 已有 verified workspace | 沿已验证图生成受控新 pair，再执行真实匹配和几何验证 |
| `mapper` | 已匹配 workspace | 稀疏模型和带 RGB PLY |
| `compare_colmap` | 两个 COLMAP 模型 | Sim(3) 位姿/结构差异报告 |
| `model_converter` | COLMAP TXT/BIN 模型 | 保留模型字段，转换为 TXT、BIN 或两种格式 |
| `model_analyzer` | COLMAP 稀疏模型 | 相机、图像、点、观测、track、误差和边界统计 |
| `model_transformer` | 稀疏模型、3×4/4×4 矩阵 | 应用 Sim(3) 并写兼容模型 |
| `model_aligner` | 稀疏模型、按图像名 XYZ | 鲁棒位置对齐 |
| `model_cropper` | 稀疏模型、三维边界 | 裁剪点及其反向观测 |
| `model_splitter` | 稀疏模型 | 按 track 连通分量输出多个模型 |
| `bundle_adjuster` | 稀疏模型 | 独立全局 BA，可选位置先验 |
| `point_triangulator` | workspace、已有模型 | 独立重三角化、complete/merge |
| `image_registrator` | workspace、已有模型 | 独立注册指定或全部未注册图像 |
| `recon_sfm` | 原图 | 完整稀疏重建 |
| `image_undistorter` | 原图、稀疏模型 | 去畸变 MVS workspace |
| `patch_match_stereo` | 去畸变 workspace | 深度、法线、代价、一致性图 |
| `stereo_fusion` | dense maps | 带 RGB/法线融合点云、可见性 |
| `poisson_mesher` | 有向点云 | Poisson PLY 网格 |
| `delaunay_mesher` | 点云、可见性、模型 | Delaunay PLY 网格 |
| `mesh_simplifier` | 三角 PLY | 简化 PLY |
| `mesh_texturer` | 网格、标准工作区 | textured PLY（BIN/TXT）、PNG；可选 OBJ/MTL |
| `recon_mvs` | 稀疏模型、图像 | 兼容组合 MVS 入口 |
| `recon_auto` | 原图 | SfM 到纹理网格完整编排 |

## Stage 通用参数

`image_undistorter` 到 `mesh_texturer` 的独立 stage，以及 `recon_auto`，支持：

- `--resume`：manifest 输入/参数及输出清单指纹一致时跳过；目录内文件缺失、大小或修改时间改变会重算，不能仅凭目录存在命中缓存。
- `--force`：忽略 manifest 强制执行；与 `--resume` 互斥。
- `--dry_run`：只打印执行计划。

manifest 位于输出根的 `.colite/stages/<stage>.json`。
当前 manifest schema 为 2，旧清单自动失效重算。MVS stage 在计划或缓存跳过前共用
`recon_mvs` 的参数校验，支持 `--name=value`，重复选项报错。输出清单指纹记录路径、
文件大小和修改时间，不是逐字节内容校验。

SfM的文件工作区缓存另使用私有schema 2。清单包含完成标记和自身FNV校验，原子发布；
产物及依赖目录以大小/修改时间指纹检查。`recon_sfm --resume`会修复缺失/截断的
特征、匹配、根模型或子模型；特征失效会重算frontend，只有模型产物损坏时继续复用
有效特征与frontend。独立extractor/matcher/mapper也检查各自声明的依赖与输出。
`recon_auto --resume`同时覆盖内部工作区，不会只因顶层模型存在而跳过修复。
旧私有清单会失效并重算，COLMAP BIN/TXT、feature v3和verified v2不改变。

## 全局日志参数

全局参数使用COLMAP 4.0名称，可出现在命令前后，支持 `--name value` 和 `--name=value`：

| 参数 | 含义及默认值 |
|---|---|
| `log_target` | `stderr` / `stdout` / `file` / `stderr_and_file`，默认后者 |
| `log_path` | 日志目录；默认空，由glog选择默认目录 |
| `log_level` | 有符号int32详细级别，默认0；1或更高启用逐图/BA诊断 |
| `log_severity` | 最低严重级别，0 INFO / 1 WARNING / 2 ERROR / 3 FATAL，默认0 |
| `log_color` | 布尔值，默认1 |

旧 `log_dir` / `log_to_stderr` / `v` 和字符串 `log_level=INFO` 已移除。
无效值在执行前报错；无效target直接拒绝，不采用原生的静默回退。
glog低于0.6时stdout按原生行为回退stderr并提示，文件/颜色能力由glog提供。
INFO记录阶段、缓存、数量与后端，`--log_level 1`另启用细化观测追踪。

## SfM 参数契约（COLMAP 4.0）

SfM 同义旧参数和别名已经移除，不提供兼容层。标准参数直接使用 COLMAP 4.0 名称；只有存储方式、实现后端或其他特有功能使用 `Colite.*`。`--name value` 与 `--name=value` 都可用，重复参数、无效数值、冲突相机分组和不适用的特征/匹配器族参数会在执行前报错。

相同名称应有相同含义、单位、布尔语义和枚举。当前只接受实际接入的选项；下文列出的算法/默认值差距仍需处理，不能把现阶段命令当成完整 COLMAP CLI 的替代品。目标是继续消除差距，而不是为旧设计永久保留差异。

### 特征提取

```bash
$COLITE feature_extractor --image_path data/building-18 \
  --Colite.workspace_path "$OUT/workspace" \
  --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \
  --FeatureExtraction.type SIFT --FeatureExtraction.num_threads 8 \
  --SiftExtraction.max_num_features 8192 --Colite.feature_extractor cuda_vlfeat
```

| 参数 | 当前行为 / 默认值 |
|---|---|
| `ImageReader.camera_model`、`camera_params` | SIMPLE_RADIAL；显式 CSV 参数按 COLMAP 顺序。初始焦距由完整 camera_params 表达，旧 focal_length 已移除 |
| `ImageReader.default_focal_length_factor` | 1.2；缺少 EXIF 焦距时使用 |
| `ImageReader.single_camera`、`single_camera_per_folder`、`single_camera_per_image` | 最多一个 true；全 false 时自动按 EXIF 相机身份分组。旧 camera_mode 已移除 |
| `FeatureExtraction.type` | SIFT / ALIKED_N16ROT / ALIKED_N32 |
| `FeatureExtraction.num_threads` | -1；提取 CPU 线程预算 |
| `SiftExtraction.max_num_features` | 8192；只接受 SIFT。CPU VLFeat 按基础点整层保留；GPU 与 COLMAP 的 -tc2 一样在方向展开后再次按整层限额，边界层可使数量超过预算 |
| `AlikedExtraction.max_num_features`、`min_score` | 2048 / 0.2；只接受 ALIKED |
| `AlikedExtraction.n16rot_model_path`、`n32_model_path` | 本地 ONNX 路径，必须对应选定模型；没有自动下载默认 URI 的行为 |
| `Colite.feature_extractor` | auto / vlsift / cvsift / cuda_opencv / cuda_vlfeat；auto 统一选择 CUDA VLFeat 或 CPU VLFeat |
| `FeatureExtraction.max_image_size` | 非正数按类型自动选择：SIFT 3200、ALIKED 1600；缩放提取后关键点回映原图，Camera 尺寸保持原始值 |
| `FeatureExtraction.use_gpu`、`gpu_index` | CPU/GPU 选择；单设备整数，-1 自动选设备 0。多 GPU CSV 暂不支持 |
| `SiftExtraction.first_octave`、`num_octaves`、`octave_resolution` | -1 / 4 / 3；first_octave 当前支持 -1 或 0 |
| `SiftExtraction.peak_threshold`、`edge_threshold`、`max_num_orientations` | 0.02/3 / 10 / 2；方向上限 1..4；四种 SIFT 后端均使用这些标准控制 |
| `descriptor_normalization` | `l1_root`；可选 `l2`，接受大小写，与 COLMAP 4.0 一致 |
| `Colite.learned_cuda_memory_limit_mb` | 8192；ONNX CUDA arena 预算 |

所有 SIFT 后端导出 COLMAP 像素中心坐标与 Gaussian sigma 尺度。`peak_threshold` 是归一化 DoG 阈值；OpenCV 实现内部将其乘 `octave_resolution` 转换为 contrast threshold，不再暴露另一组冗余阈值。CPU/GPU 的检测器与预算选择仍有差异，不承诺相同特征或重建结果。

非正 max_image_size 不表示关闭缩放；要保留原分辨率，指定不小于图像最长边的正尺寸。CLI 的 max_num_features 必须为正数，低层提取器测试 API 的 0=不限额不作为公开参数行为。

### 匹配与几何验证

```bash
$COLITE exhaustive_matcher --Colite.workspace_path "$OUT/workspace" \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 1 \
  --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 --FeatureMatching.guided_matching 1 \
  --TwoViewGeometry.max_error 4
$COLITE sequential_matcher --Colite.workspace_path "$OUT/workspace" \
  --SequentialMatching.overlap 10 --SequentialMatching.quadratic_overlap 1
```

匹配策略由命令选定；冗余 matching_type 已移除。给 exhaustive_matcher 传 SequentialMatching 参数等无效组合会报错。

| 参数 | 当前行为 / 默认值 |
|---|---|
| `FeatureMatching.type` | SIFT_BRUTEFORCE / SIFT_LIGHTGLUE / ALIKED_BRUTEFORCE / ALIKED_LIGHTGLUE；旧 matcher 已移除 |
| `SiftMatching.max_ratio`、`max_distance`、`cross_check` | 0.8 / 0.7 / true，只用于 SIFT BF 族 |
| `SiftMatching.lightglue_model_path`、`lightglue_min_score` | SIFT LightGlue 模型路径 / 0.1 |
| `AlikedMatching.brute_force_min_cossim`、`brute_force_max_ratio`、`brute_force_cross_check` | 0.85 / 1.0 / true，ALIKED BF；本地实现不需要 ONNX bruteforce_model_path |
| `AlikedMatching.lightglue_model_path`、`lightglue_min_score` | ALIKED LightGlue 模型路径 / 0.1 |
| `FeatureMatching.guided_matching` | matcher 与 recon_sfm 统一默认 false；SIFT BF重搜索按E/F/H分派，E使用去畸变归一化坐标，H使用前向像素传递门限 |
| `TwoViewGeometry.max_error`、`min_num_inliers`、`confidence`、`max_num_trials` | 4 px / 15 / 0.999 / 10000；min_num_inliers≥0，confidence∈[0,1]，max_num_trials≥100 |
| `TwoViewGeometry.min_inlier_ratio`、`random_seed` | 0.25 / -1；前者是 RANSAC 试验预算先验，不是最终内点比例门槛；-1 随机，非负值固定种子 |
| `TwoViewGeometry.compute_relative_pose` | false；默认匹配不保存位姿/视差，由 mapper 按需恢复；true 时匹配阶段恢复 |
| `TwoViewGeometry.detect_watermark`、`watermark_detection_max_error` | true / 4 px；检测边缘区域的平移模型，标为 WATERMARK 并排除建图 |
| `TwoViewGeometry.filter_stationary_matches`、`stationary_matches_max_error` | false / 4 px；启用时在几何验证前移除位移≤门限的对应关系 |
| `TwoViewGeometry.multiple_models`、`multiple_ignore_watermark` | false / true；迭代剥离内点，多个模型仅保存拼接内点，矩阵和位姿缺失 |
| `SequentialMatching.overlap`、`quadratic_overlap` | 10 / true；仅 sequential_matcher |
| `SpatialMatching.max_num_neighbors`、`max_distance` | 50 / 100；距离必须 >0，未实现 min_num_neighbors 补足规则 |
| `VocabTreeMatching.num_images` | 100；仅 vocab_tree_matcher |
| `FeatureMatching.use_gpu`、`gpu_index`、`num_threads` | SIFT 默认用可用 GPU；单设备索引 -1 自动选 0；线程预算默认 -1 自动。ALIKED BF 当前仅 CPU，显式 use_gpu=1 会拒绝；LightGlue 接到 ONNX provider |
| `Colite.guided_match_ratio`、`guided_min_inlier_gain` | 默认继承 SiftMatching.max_ratio，并沿用其 cross_check；显式 ratio 可覆盖。独立 matcher 内点增量门槛 10 |
| `Colite.vocab_tree_path` | COLITE 格式词树；没有使用 VocabTreeMatching.vocab_tree_path 假装可读取 COLMAP 词树 |

ONNX 设备与匹配并行预算使用标准 FeatureExtraction.* / FeatureMatching.*。特有 arena / I/O binding 仍使用 `Colite.learned_cuda_memory_limit_mb` / `Colite.learned_use_io_binding`。旧 Colite.matcher_backend、Colite.aliked_max_image_size、Colite.learned_device/gpu_index/matcher_num_threads 已移除，没有别名。

两视图命令行暴露的 13 个标准参数与 COLMAP 4.0 对齐；独立 matcher 与 recon_sfm 共用解析、
严格校验和缓存身份。RANSAC 使用无放回采样的成功概率、动态倍率3和 min_inlier_ratio 先验预算。
仅当两个相机都有焦距先验时估计 E；否则使用 F/H。平面对默认保留 PLANAR_OR_PANORAMIC，
在恢复 H 位姿后区分 PLANAR/PANORAMIC。相同输入的可复现 A/B 应显式设置 random_seed=0。
这没有复刻全部原生 LO/refinement。MULTIPLE 本身不含位姿，但 mapper 现会从其
verified对应重估初始标定几何，不再要求存在带位姿的单模型边；能否初始化仍取决于
重估内点、视差和三角化质量门限。

Colite 自建词树、词树保护顺序边和 transitive matcher 的控制也使用 `Colite.*`，以各命令 help 为准。它们不属于 COLMAP 同名文件/算法兼容承诺。

### Mapper 与组合流程

```bash
$COLITE mapper --Colite.workspace_path "$OUT/workspace" --output_path "$OUT/sparse" \
  --Mapper.multiple_models 1 --Mapper.max_num_models 50 \
  --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 --Colite.output_format both

$COLITE recon_sfm --image_path data/building-18 --output_path "$OUT/sfm" \
  --ImageReader.single_camera 1 --FeatureExtraction.type SIFT \
  --SiftExtraction.max_num_features 8192 --Colite.feature_extractor cuda_vlfeat \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 1 \
  --FeatureMatching.guided_matching 1
```

独立 mapper 的 `--output_path DIR` 在新建时将各模型写为 `DIR/0`、`DIR/1` 等；
提供 `--input_path` 续建时，只将续建模型直接写入 `DIR` 根目录，与 COLMAP 一致。支持 BIN/TXT/both；
不提供时只保留原来的 workspace sparse 缓存。`--image_path DIR` 可指定着色使用的图像根目录，
不提供时沿用 workspace 中的图像路径。导出与重建缓存独立：即使 mapper 命中缓存，也会
将完整模型导出到指定位置，补回被删除或损坏的导出文件。续建缓存区分输入路径、
格式及选定模型的文件内容（含可选rigs/frames）；auto优先BIN，输入修改后重新续建。

Mapper / recon_sfm 共用已实现的标准参数：

- 线程：`Mapper.num_threads=-1`；与 COLMAP 一致，-1/0 使用可用硬件，正整数指定 core/Ceres 映射阶段预算。独立与组合入口均恢复调用方原线程设置，不再继承匹配阶段预算。
- 多模型：`Mapper.multiple_models`、`max_num_models`、`min_model_size`。增量 mapper 始终保留首个含有效稀疏点的成功模型；后续模型的尺寸阈值为 `min(min_model_size, floor(有有效匹配的输入图像数 / 2))`，不随剩余分量缩小。`min_model_size` 默认 10，允许 0/1；0 关闭尺寸过滤，但不允许空点云或单张图像作为有效初始化。
- 初始化：`Mapper.init_min_num_inliers`、`init_max_error`、`init_max_forward_motion`、`init_min_tri_angle`、`init_num_trials`、`init_image_id1`、`init_image_id2`、`init_max_reg_trials`。前向运动门限默认0.95、范围[0,1]；相对平移归一化后的z绝对值达到门限则拒绝该初始对，平移尺度不影响筛选。0拒绝全部初始对，1仍拒绝严格轴向前后运动；放宽初始化内点/视差时不放宽此门限。
- 注册：`Mapper.max_reg_trials`、`abs_pose_max_error`、`abs_pose_min_num_inliers`、`abs_pose_min_inlier_ratio`、`structure_less_registration_fallback`。
- 过滤：`Mapper.filter_max_reproj_error`、`filter_min_tri_angle`。
- BA：`Mapper.ba_refine_focal_length`、`ba_refine_principal_point`、`ba_refine_extra_params`、`ba_local_num_images`、`ba_local_max_num_iterations`、`ba_local_max_refinements`、`ba_local_max_refinement_change`、`ba_global_max_refinements`、`ba_global_max_refinement_change`、`ba_global_max_num_iterations`。旧 ba_refine_params 逐名称选择入口已删除。每次全局细化默认最多 5 轮（包含首轮），观测变化率低于 0.0005 时停止；停滞恢复和收尾可分别触发细化，诊断轮数为累计值。最大轮数 0 关闭这些 BA 轮次，变化率阈值 0 表示不提前停止。局部精化默认最多2轮、观测变化率阈值0.001；首轮Soft L1尺度1，后续轮次普通最小二乘，阈值0不提前停止，最大轮数须为正int32。新点先由三角化角误差筛选，像素过滤在局部BA之后。全局BA默认普通最小二乘。

独立、局部和全局 BA 共用 Ceres 停止与 CPU 求解控制，迭代预算分别为 100/25/50。
Mapper 另支持 `ba_local_function_tolerance`、`ba_global_function_tolerance`（默认 0），以及
`ba_min_num_residuals_for_cpu_multi_threading`（默认 50000），后者同时作用于局部/全局 BA，
仍受 `Mapper.num_threads` 预算约束。局部求解器按局部窗口规模选择，外部固定观测不抬高该规模。
注册成功及局部细化后，按 COLMAP 4.0 的 `Mapper.ba_global_frames_ratio` /
`ba_global_points_ratio`（均为1.1、须大于1）和 `ba_global_frames_freq` /
`ba_global_points_freq`（500/250000、正int32）检查模型增长；任一门限达到时执行全局细化，
基准计数更新为细化后的模型大小。单相机中一张图像就是一个trivial frame，使用4.0的
`frames`参数名，不接受旧`images`名称。摘要的`global_ba_growth_refinements`统计增长触发次数，
`global_ba_refinements`统计全部触发来源累计的BA轮次。停滞恢复与最终细化继续使用同一轮数/变化率控制。
全局BA摘要使用 `global_ba_first_*` / `global_ba_last_*` 区分首末求解；不同轮次可能使用不同观测，不能直接把跨轮cost变化当作收敛率。`num_observations` / `mean_reprojection_error` / `max_reprojection_error`从最终保存模型计算，独立mapper与组合入口一致。旧含混的 `global_ba_initial_cost` / `global_ba_final_cost`等字段已移除。
`Mapper.tri_max_transitivity`默认1，`tri_complete_max_transitivity`默认5；均为非负int32，0关闭相应遍历，独立track合并不受补全深度零值影响。

`--log_level 1` 启用逐观测细化诊断，生成 `sparse/ba_refinement.tsv` 和 `ba_observation_changes.tsv`；其中事件变化率包含合并影响的已有观测，与观测集合净变化量不同。普通运行不构建详细快照。
- 三角化：`Mapper.tri_max_transitivity`、`tri_create_max_angle_error`、`tri_continue_max_angle_error`、`tri_merge_max_reproj_error`、`tri_complete_max_reproj_error`、`tri_complete_max_transitivity`、`tri_min_angle`、`tri_ignore_two_view_tracks`。

`input_path` 用于接续已有模型。`Colite.mapper` 选择 incremental / global / hierarchical；global 当前接入 Mapper.num_threads、Mapper.tri_*、Mapper.ba_global_max_num_iterations、Mapper.ba_global_function_tolerance 和 Mapper.ba_min_num_residuals_for_cpu_multi_threading，其他未实现 Mapper 参数明确拒绝。

`Colite.merge_models` 默认 false；增量建图保留独立子模型，与 COLMAP 默认流程一致。只有显式设为 1 才尝试 Sim(3) 合并，合并后的覆盖增加不保证跨组件位姿正确。

仍有意义的独立功能包括 `Colite.merge_models`、`Colite.registration_retry_error_multiplier` / `registration_retry_max_error`、`Colite.hierarchical_*`、`Colite.nearby_match_*`、`Colite.use_position_priors` / `position_prior_*`。标准 max_reg_trials、init_num_trials、structure_less_registration_fallback 的旧扁平名称不再接受。

### 图像遮罩

`ImageReader.mask_path`（目录）和`ImageReader.camera_mask_path`（单个全局图像）默认均为空。
独立`feature_extractor`、`recon_sfm`及Auto的`--sfm.ImageReader.*`支持相同参数。
例如图像`images/sub/view.jpg`优先读取`masks/sub/view.jpg.png`，不存在时尝试
`masks/sub/view.png`。两种遮罩同时指定时取交集；灰度0排除，其余非零值保留。

按编码原图的整数像素截断采样，超出遮罩范围的点排除；遮罩不随提取分辨率缩放，
也不在过滤后补足特征数。Colite会将EXIF朝向后的关键点映回编码原图，遮罩本身不随EXIF自动旋转。
文件名保持图像逻辑相对路径，包括嵌套符号链接；按目录共享相机和稀疏模型图名也保持该路径。

遮罩的内容变化会使特征及下游缓存失效，即使文件大小和mtime不变；Auto dry-run只计划，不改写产物。
指定的逐图遮罩必须全部可读。缺失/损坏遮罩在提取及缓存跳过前明确报错；这一失败策略
严于COLMAP4.0（原生忽略无效全局遮罩，缺失逐图遮罩跳过图像后仍可能返回0）。

### 增量重建默认行为与边界

Mapper / recon_sfm 已共用 COLMAP 默认配置：初始化 100 内点 / 16° / 每轮200次重建试验预算，abs pose 12 px / 30 内点 / 0.25 比率，过滤 1.5°，后续最小模型默认 10 张并按上述规则截断，structure-less 回退默认 true。几何候选检查不消耗重建试验预算，同轮多连通分量共享预算；按两轮“内点减半 → 角度减半”放宽时重置预算及已尝试对。默认 PnP 重试不隐式放宽用户误差阈值。

`Mapper.init_image_id1/2` 默认 `-1` 表示自动；非负 int32 是真实图像 ID（0 合法，须存在于输入）。
只指定任意一项时，该图像作为首图；两项均指定则固定该方向的初始对，失败不回退到其他对。
两项相同、越界或 `init_max_reg_trials<=0` 在读写工作区之前报错。
`Mapper.init_max_reg_trials` 默认2，只限制自动选择的首图；邻居及显式首图不受此筛选。
接受几何并注册种子时，两张图像的每图计数均增加，几何检查失败不增加；同轮跨分量共享，放宽时清零。
已有输入模型不使用初始图像选项重新选种子。并行原生搜索可能选择同分候选中的不同图像，
不能用单次初始对顺序判定质量等价。

初始对接受后，先用相同的稳健三角化器创建点，再执行两图全局BA、相机中心归一化和点过滤；
剩余点不足`abs_pose_min_num_inliers`时丢弃该候选并重选，相机/位姿/轨迹不泄漏到下一次。
`tri_create_max_angle_error`、`tri_max_transitivity`、`tri_ignore_two_view_tracks`也作用于初始点。
后者默认1：仅有纯两视图轨迹的连通分量不能产生初始点，需显式0允许，与COLMAP一致。
Mapper全局BA配置的基础预算为50轮，少于10张已注册图像时按原生规则加倍，三项停止容差除10；
独立BA仍为100轮、局部BA仍为25轮，不受此小模型规则影响。

初始化会从verified对应重新估计E/F/H并恢复位姿，`Mapper.init_max_error`同时控制
该RANSAC及初始点的像素误差，不再只依赖缓存位姿；标定检验使用提供的内参，即使
没有焦距先验也会尝试E模型。重估的min_num_iterations配置为30，初始中位视差须严格大于门限。
`Mapper.random_seed`默认-1（随机），非负int用于可复现的初始化/PnP/广义重定位；
独立`image_registrator`也支持同名种子，PnP默认同为12px/30内点/0.25。
重做前端与Mapper对照时分别设置`TwoViewGeometry.random_seed`和`Mapper.random_seed`；
`recon_auto`对应`sfm.TwoViewGeometry.random_seed`和`sfm.Mapper.random_seed`。

初始候选排序优先考虑焦距先验和首图的全图对应支撑，再按邻居的焦距先验及边内点数
排序；短基线等不能初始化的邻居仍可提供首图支撑，当前分量外的边不参与。相同分数按
图像 ID 稳定排序，初始对保留所选首图的方向，缓存匹配边仍使用规范 ID 顺序。

增量注册停滞时，全局细化一次后重试；连续无进展不重复细化，同一图像仍受 `Mapper.max_reg_trials` 约束。相对位姿射线无法初始化回退位姿时，有 PoseLib 的构建会使用 verified 2D–2D 匹配做广义相机重定位，不再要求匹配边先有可用 E 相对位姿。该路径要求至少两个不同相机中心、至少 `2 * abs_pose_min_num_inliers` 条内点对应、足够独立目标特征和邻居支撑，再通过正深度、重投影和视差验证。无 PoseLib 构建保留原有射线路径。已有一致相对位姿假设若未通过像素验证，仍拒绝注册。

这并不表示完整增量调度等价：并行候选次序、structure-less 求解器、局部细化和轨迹构造仍有实现差异。收尾全局细化已支持上述轮数与变化率控制。SIFT 仿射/DSP/upright 选项、顶层统一 random_seed（TwoViewGeometry.random_seed 已实现）、多 GPU CSV 等未接入项继续明确报错。ALIKED BF 尚无 GPU 实现，不能以接受 use_gpu=1 代替实际加速。

运行时继续使用文件 workspace。

### 全局与分层重建

`mapper --Colite.mapper incremental|global` 可从已匹配的工作区启动，`mapper --help` 列出全局模式的 `GlobalMapper.*` 参数。组合命令 `recon_sfm` 的 `--Colite.mapper` 额外接受 `hierarchical`，`recon_auto` 使用前缀 `--sfm.Colite.mapper`。默认均为增量式；全局和分层方式不支持以 `--input_path` 接续已有模型。用 `colite --capabilities` 确认当前构建能力。

### TXT / BIN 与模型维护

Colite 文件工作区使用 `Colite.workspace_path`；它不是 COLMAP SQLite database_path，不能通过改名宣称等价。模型文件保持 COLMAP world-to-camera、WXYZ、little-endian 与完整 POINT2D_IDX/track。

特征与 raw matches 支持离线双向交换，算法运行仍使用文件 workspace：

```bash
python3 scripts/convert_colmap_database_to_colite_workspace.py \
  --database_path colmap.db --image_path images --output_path new_workspace
python3 scripts/convert_colite_workspace_to_colmap_database.py \
  --input_path new_workspace --database_path exported.db --colmap_path colmap
# 新版 verified 数据可直接接续建图，无需再次匹配/几何验证。
colmap mapper --database_path exported.db --image_path images --output_path sparse
```

输出必须是新路径；失败不发布半成品，也不覆盖已有数据。导出器调用实际 COLMAP 4
创建 schema，并建立单相机 trivial rig/frame 关联。导入器旧 `--database/--workspace`
参数已移除，使用 `--database_path/--output_path`。
导入始终完整保存关键点、带类型描述子、raw matches及verified几何；旧`--include_descriptors`
开关及默认只导入几何的路径已移除。描述子缺失、行数或payload不合法时拒绝发布工作区，
零特征图像仍保留带类型的空描述子，便于再次匹配和导出。

feature v3 保存原始 2/4/6 列关键点形状与行顺序，含 payload 校验；读取仍支持历史 v1/v2。
xy 不平移、不重排；原始弧度或完整仿射系数可往返保留，内部计算仍用 scale/度制方向。
SIFT 按 COLMAP 的 `round(512*x)` 后裁剪到 [0,255] 保存 uint8，不另做归一化；
原生 uint8/512 可精确恢复，独立生成 float 的首次量化会有损，导出 JSON 报告误差和裁剪数。
ALIKED N16Rot/N32 保持 float32 字节及类型，数据库 cols 为512字节。
使用 ALIKED 时，两端 matcher 指定 `--FeatureMatching.type ALIKED_BRUTEFORCE`。
数据库不记录 SIFT 的 L1_ROOT/L2 选项；源为 L2 时，导入需传
`--descriptor_normalization l2`，默认 `l1_root`，与 COLMAP 4.0 提取命令一致；
提取器、`recon_sfm` 和离线导入工具共用这一名称（`recon_auto` 使用 `sfm.` 前缀）。
不再接受 `SiftExtraction.normalization`。response 在 COLMAP 中没有对应字段。

raw matches 保留零基索引、canonical pair_id 和空对。新版 `verified_matches.tsv`
使用 `COLITE_VERIFIED_GEOMETRIES 2` 头和每对元数据，保存 COLMAP 0–9 configuration、
row-major F/E/H、WXYZ 相对位姿及各字段是否存在；零矩阵和零平移不等于缺失。
内点仍为原来的 13 列诊断行，空对使用 -1/-1，reader 检查数量与元数据一致性。
离线反向默认同时导出 verified，可由 COLMAP mapper 直接读取。
COLMAP 不持久化 pair median angle，导回后由 mapper 按需要计算；默认未保存位姿的
CALIBRATED/UNCALIBRATED 对从 E/F 恢复位姿，平面/全景对从 H 分解，并在内存中计算视差，
不改写交换源文件。恢复后的非零基线 CALIBRATED、UNCALIBRATED、PLANAR 对可参与初始化。

旧 verified TSV 无法确定 F/E/H 与缺失状态，默认反向导出会明确报错。应重新运行
Colite matcher，或显式 `--Colite.export_verified 0` 仅导出特征/raw，再由 COLMAP matcher
完成几何验证。该开关不会补造 verified 数据。纯旋转即使有位姿，也不会用于平移基线初始化。
平面分解仍有多解歧义，不能把参数/格式对齐描述为与 COLMAP 数值等价。

COLMAP 4 从 EXIF 方向写入的仅重力 pose prior 可随特征交换，使用
`colmap_pose_priors.tsv` 保存原始 ID、单相机 sensor 关联、BLOB 与 NULL；
导出恢复原值。它是交换元数据，当前增量 mapper 不使用重力约束，也不旋转图像。
有效定位、协方差或已定义坐标系的 pose prior 交换尚不支持，明确拒绝，不能静默丢弃。
多传感器 rig、多图像 frame 不支持并明确拒绝；RIG_CALIBRATED
可以离线保留，但 mapper 明确拒绝执行，WATERMARK 对不进入建图对应关系。
此工具交换支持的数据字段，不迁移完整 COLMAP 数据库工程。

mapper / recon_sfm 用 `Colite.output_format txt|bin|both`，默认 both。模型维护命令用 `Colite.input_format auto|txt|bin` 选择输入（auto 优先完整 BIN），以及 `Colite.output_format`；model_converter 使用标准 `output_type TXT|BIN`，另支持扩展 BOTH。模型转换不优化，并保留 ID、参数、位姿、观测、RGB、ERROR 和 track。

建图或接续外部模型时，有限且有效的相机参数仍可能使部分像素无法反投影。位姿恢复、
三角化、PnP及视差检查会在需要射线时局部忽略这些观测；F/H继续使用有限像素对应，
特征/匹配文件的行顺序与索引不因此改变。非法相机参数和越界匹配仍明确报错。
```bash
$COLITE model_converter --input_path colmap/sparse/0 --output_path converted --output_type BOTH
colmap model_converter --input_path converted --output_path colmap_text --output_type TXT
$COLITE bundle_adjuster --input_path converted --output_path ba \
  --BundleAdjustment.refine_focal_length 1 --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 1 --BundleAdjustmentCeres.max_num_iterations 100
$COLITE point_triangulator --Colite.workspace_path workspace --input_path converted --output_path triangulated \
  --Mapper.tri_min_angle 1.5
$COLITE image_registrator --Colite.workspace_path workspace --input_path converted --output_path registered \
  --Mapper.abs_pose_max_error 12 --Mapper.abs_pose_min_num_inliers 30
```

旧 BundleAdjustment.max_num_iterations 与 max_reprojection_error 已删除。point_triangulator 只接受实际使用的 Mapper.tri_*，image_registrator 只接受实际使用的 Mapper.abs_pose_*。

`bundle_adjuster` 的常用单相机控制与 COLMAP 4.0 同名：

| 参数 | 默认及行为 |
|---|---|
| `BundleAdjustment.refine_points3D` | 1；设为 0 保持点坐标，只按其他开关调整位姿/内参 |
| `BundleAdjustment.refine_rig_from_world` | 1；单相机流程对应图像位姿，0 固定旋转和平移 |
| `BundleAdjustment.constant_rig_from_world_rotation` | 0；位姿开放时可只固定旋转 |
| `BundleAdjustment.refine_sensor_from_rig` | 1；当前仅支持 reference sensor 的 trivial rig，其 sensor-from-rig 恒为固定 identity，开关均不改变它 |
| `BundleAdjustment.min_track_length` | 0；非负 int，短 track 不进入 BA 目标，不删除其点或观测；全部被排除时输出未优化模型 |
| `BundleAdjustmentCeres.max_num_iterations`、`max_linear_solver_iterations` | 100 / 200；前者允许 0，后者必须为正 |
| `BundleAdjustmentCeres.function_tolerance`、`gradient_tolerance`、`parameter_tolerance` | 0 / 1e-4 / 0，有限非负数；不再隐式使用 Ceres 原默认 |
| `BundleAdjustmentCeres.min_num_residuals_for_cpu_multi_threading` | 50000 个标量残差，小问题单线程；其他情况用当前 CPU 预算 |
| `BundleAdjustmentCeres.max_num_images_direct_dense_cpu_solver`、`max_num_images_direct_sparse_cpu_solver` | 50 / 1000，要求 0≤dense<sparse；依次选择 dense/sparse/iterative Schur，缺少稀疏库时使用迭代路径 |
| `BundleAdjustmentCeres.use_gpu` | 只接受 0；GPU BA、GPU index/求解器门限未实现并拒绝，不伪装执行 |

输出分别给出请求/实际使用的 Ceres 求解器、线程、观测数和标量残差数；Ceres 可在固定
全部点等情况下调整线性求解器。基本重投影对应每观测两个标量残差，位置先验另加三维残差。
Colite 的位置先验扩展统一为 `Colite.ref_images_path`、`Colite.position_prior_stddev`、
`Colite.position_prior_loss_scale`、`Colite.position_prior_alignment_max_error`，后三者要求前者；
不再接受 BA 原先未加命名空间的 `ref_images_path` / `alignment_max_error`。
这不改变 `model_aligner` 中同名原生参数。

独立 `bundle_adjuster` 在构建优化目标前，按 COLMAP 清理已注册图像中相机深度
小于 `double epsilon` 的观测；若其 track 删除后不足两条，则删除整点并解除全部关联。
这不等于重投影/视差过滤，也不由 `min_track_length` 控制。清理后没有观测时成功
保存相机/图像和空点模型。日志单独报告 `num_filtered_negative_depth_observations`
与 `num_removed_negative_depth_points`；前者计直接触发的坏观测，不含整点删除时
连带解除的其他关联。原模型文件不被修改，后续求解失败不发布输出模型。
TXT/BIN 写出仍为三文件模型，不保留 COLMAP 4 rig/frame 元数据；写出时清理另一格式或旧 rig/frame，避免自动读取陈旧数据。SfM 改变几何后重算 ERROR；纯模型转换保留 ERROR。工作区特征/匹配缓存与 SQLite 保持各自格式。

## MVS 单步骤

MVS 使用共用的类型化参数表生成解析、help、默认值和阶段适用性检查。标准项使用
`PatchMatchStereo.*` / `StereoFusion.*`，扩展使用 `Colite.*`；旧扁平名称不再接受。
所有布尔参数显式传入 `0|1` 或 `false|true`，支持 `--name=value`，重复参数、数值后缀、
溢出、NaN/Inf 和不适用于当前命令的选项在读取输入前失败。

**0.1 收敛状态：参数接口、工作区、标准 dense 保存及质量默认配置已迁移。**
03/05/08 真值对照和端到端发布验收分别记录；不能将标准参数相同理解为两端数值结果相同。

### 图像去畸变

```bash
ROOT=output/mvs_single_building18_YYYYMMDD
$COLITE image_undistorter \
  --image_path data/building-18 \
  --input_path output/recon_sfm_building18_YYYYMMDD \
  --output_path "$ROOT" --output_type COLMAP \
  --max_image_size 1600 --num_threads 8
```

`output_type` 目前仅支持 COLMAP；`blank_pixels=0`、`min_scale=0.2`、`max_scale=2`。
直接写入 `$ROOT/{images,sparse,stereo}`，不再增加 `undistorted` 层。
同时生成 `stereo/patch-match.cfg`、`stereo/fusion.cfg`；
`--num_patch_match_src_images` 默认 20，控制写入的 `__auto__, N` 源视图数量。

### PatchMatch Stereo

```bash
$COLITE patch_match_stereo \
  --workspace_path "$ROOT" --workspace_format COLMAP \
  --Colite.patch_match_backend cuda \
  --PatchMatchStereo.num_iterations 5 --PatchMatchStereo.window_radius 5 \
  --PatchMatchStereo.num_samples 15 --PatchMatchStereo.num_threads 8 \
  --PatchMatchStereo.geom_consistency 1 --PatchMatchStereo.filter 1 \
  --Colite.patch_match_bilateral_ncc 1 --Colite.patch_match_directional_hmm 1 \
  --Colite.patch_match_checkerboard_rounds 4
```

`workspace_path` 默认解析其 `sparse/` 和 `images/`；也可显式提供 `input_path`、`image_path`
和 `output_path`。depth/normal 固定写 COLMAP binary，文件名为
`IMAGE.photometric.bin` / `IMAGE.geometric.bin`；法向按 x/y/z 通道分平面保存。
启用几何精化时保留未过滤的 photometric 快照，最终结果另写 geometric，不用同一文件覆盖两阶段。
`Colite.dense_map_format` 已移除；稀疏模型的 BIN/TXT 保存选项不受此影响。

`PatchMatchStereo.write_consistency_graph=1` 写标准像素记录和完整模型图像索引，默认 0。
内部 CSR/精化侧图仅为临时数据，成功后清理，不以公开 `.bin` 文件冒充 COLMAP graph。
`Colite.write_cost_maps=1` 额外保存 Colite binary 代价图；`Colite.write_debug_maps=1`
额外保存自定义调试 TXT depth/normal/cost，两个开关默认均为 0，互相独立。

使用 `workspace_path` 时读取 `stereo/patch-match.cfg`，也可用 `--config_path FILE` 指定配置。
每个参考图后跟一行源视图定义，支持 `__all__`、`__auto__, N` 和逗号分隔的显式图像名称。
仅计算配置中的参考图，源视图可来自完整注册图像集合；显式列表不再被
`Colite.max_num_source_images` 截断。`__auto__` 按 COLMAP 共享点/三角角规则选择。
无稀疏点的显式图像对需要同时提供有效的 `depth_min` 和 `depth_max`。
未知图像、重复参考图、自匹配、重复源视图、未配对行和非法数量均报错。

几何精化可直接消费两端已有的标准 photometric depth/normal：参考图和外部源图都可以复用。
有效交换输入保持字节不变，初始诊断标为 `persisted` / `reused_photometric=1`，未保存的
photometric 代价标为 unknown，不伪造零误差。每个图像的 map 必须匹配当前工作尺寸；
过大的 map 可降采样，小于工作尺寸时需使用相同的 `max_image_size` 或显式 `--force` 重算。
`PatchMatchStereo.allow_missing_files` 默认 0，源依赖缺失报错；设为 1 可跳过缺失源，
但完全没有可精化参考图时仍失败，不把未精化的 photometric 输出标成 geometric。
独立阶段的 `--force` 会重算当前参考图的 photometric 输入；`--resume` 检查全部已记录产物，
缺失/截断的参考图产物触发修复。已有 geometric 结果在重跑时重新精化，阶段级有效 resume 才整体跳过。

独立入口与 `recon_mvs` 共用 `.colite/photometric_cache.txt`：按实际工作尺寸、种子、
光度参数、解析后的参考/源列表、稀疏深度样本、相机位姿及图像文件身份判断自有光度缓存。
种子、输入或实现版本变化会重算；仅修改 Fusion 参数不使光度缓存失效。相同配置切换入口
可复用。记录在执行前标记未完成，成功后原子发布，并记录光度 depth/normal 的尺寸/时间指纹；
中断、损坏或缺失记录的旧 Colite 产物不会被当成外部交换输入继续借用。
新工作区中没有 Colite 所有权记录的标准外部 maps 仍可直接使用，不需要转换格式。

| 参数 | 当前默认与行为 |
|---|---|
| `PatchMatchStereo.num_iterations`、`window_radius`、`window_step` | 5 / 5 / 1；控制实际搜索与 NCC 取样；radius 1–32、step 1–2 |
| `PatchMatchStereo.num_samples` | 15，源 CDF 抽样；与 COLMAP 一样要求正数 |
| `PatchMatchStereo.gpu_index`、`num_threads` | -1 / -1；GPU 为单设备，CPU 线程 -1 自动，正数限制预算；多 GPU CSV 明确拒绝 |
| `PatchMatchStereo.max_image_size` | -1；最长边限制，独立于 Fusion |
| `PatchMatchStereo.depth_min`、`depth_max` | -1 / -1；自动稀疏范围，正值覆盖各视图的相机 Z 深度界限 |
| `PatchMatchStereo.sigma_spatial`、`sigma_color`、`ncc_sigma` | -1 / 0.2 / 0.6 |
| `PatchMatchStereo.min_triangulation_angle`、`incident_angle_sigma` | 1° / 0.9；源视图评分与抽样先验 |
| `PatchMatchStereo.geom_consistency` | 1；几何精化默认沿用 num_iterations；关闭后只写 photometric |
| `PatchMatchStereo.geom_consistency_regularizer`、`geom_consistency_max_cost` | 0.3 / 3 px；实际几何代价权重与截断上限 |
| `PatchMatchStereo.filter`、`filter_min_ncc`、`filter_min_triangulation_angle` | 1 / 0.1 / 3°；最终同源 NCC/几何过滤 |
| `PatchMatchStereo.filter_min_num_consistent`、`filter_geom_consistency_max_cost` | 2 / 1 px；源数量门限按当前可用源数裁剪，允许显式 0 |

有用扩展包括 `Colite.patch_match_backend`、`patch_match_pyramid_levels`、
`patch_match_directional_hmm`、`patch_match_bilateral_ncc`、`patch_match_checkerboard_rounds`、
`source_geometric_priors`、`robust_geometric_aggregation`、`randomize_initial_normals`、
`resolution_level`、`write_cost_maps`、`write_debug_maps`、`max_num_source_images` 和 `source_selection_mode`。
`source_geometric_priors`、`robust_geometric_aggregation` 和 `randomize_initial_normals`
均使用单个布尔选项，不再提供正反两个同义 flag；完整参数以本命令 help 为准。
源图HMM历史权重由搜索进度自动退火，与COLMAP的渐增历史思路对齐；不再提供
`Colite.patch_match_previous_selection_probability_weight`固定权重参数，传入会报错。
诊断表记录实际`selection_history_weights`，不是把固定0.5作为整个搜索过程的实际值。
CPU与CUDA搜索安排仍不同，参数语义对齐不代表逐像素数值完全相同。

`default_random_seed` 控制固定种子的初始化和扰动。`Colite.resolution_level` 默认 0，
由标准 max_image_size 限制工作尺寸。双边 NCC、动态选择、directional HMM 默认开启，
checkerboard rounds 默认 4；直接输入模型的源选择默认 colmap、最多 20 张。
只运行去畸变或 Fusion 时不会因几何默认开启而隐式执行 PatchMatch。
旧 combined-cost 过滤研究配置须显式设置 `PatchMatchStereo.filter=0`，避免两套门限同时生效。

### Stereo Fusion

```bash
$COLITE stereo_fusion \
  --workspace_path "$ROOT" --workspace_format COLMAP \
  --input_type geometric --output_path "$ROOT/fused.ply" \
  --StereoFusion.min_num_pixels 5 --StereoFusion.max_num_pixels 10000 \
  --StereoFusion.num_threads 8
```

`output_path` 是点云文件；同时写 `output_path.vis`，可见性与点云一起纳入 resume 完整性检查。
`input_type=photometric|geometric` 选择对应的标准 map 后缀，独立 Fusion 默认 geometric，
不再自动降级读取无阶段后缀的原生文件。`recon_mvs` 同时运行 PatchMatch/Fusion 时，
未显式指定 `input_type` 则跟随本次生成的 map 阶段；显式值始终优先。不会重跑 PatchMatch。
错误参数在执行或缓存判断前失败。
`stereo/fusion.cfg` 控制参与融合的图像；图像子集和缺失 map 不会改变 `.vis` 中完整模型的
图像索引（旧三文件模型按 images 记录顺序，有 trivial frames 时按 frame 记录顺序，
独立于图像 ID）。cfg 修改会使 resume 失效。
Fusion 支持 RGB、depth/normal 分辨率不同的标准工作区：按 map 实际尺寸缩放内参，
应用 `max_image_size` 时平滑降采样 map 并重归一化法向，保持相机 Z 深度含义。

COLMAP 4.0 的单参考相机 rig、单图像 frame 可读入，frame 位姿优先于 images 中的
冗余位姿；导出为保存等价独立相机位姿的三文件模型。多传感器 rig、多图像 frame、
不完整或非法关联暂不支持并报错，不静默丢弃约束。

| 参数 | 当前默认 |
|---|---|
| `StereoFusion.num_threads`、`max_image_size` | -1 / -1，独立于 PatchMatch |
| `StereoFusion.min_num_pixels`、`max_num_pixels`、`max_traversal_depth` | 5 / 10000 / 100，与 COLMAP 一致 |
| `StereoFusion.max_reproj_error`、`max_depth_error`、`max_normal_error` | 2 px / 0.01 / 10° |
| `StereoFusion.check_num_images` | 50；按共享稀疏点排序，不受 PatchMatch 源数量与角度门限限制 |
| `StereoFusion.use_cache`、`cache_size` | 0 / 32 GiB；显式开启后按完整视图字节限制 LRU |

扩展 `Colite.fusion_diagnostics=1` 写出轨迹、成对拒绝和误差分布；默认关闭。
`Colite.image_cache_size` 仅控制图像 workspace 的缓存数量，不替代完整视图的字节预算。
借用中的数据仍由 worker 持有，缓存预算不等于进程 RSS 的硬上限。

## 表面单步骤

### Poisson

```bash
$COLITE poisson_mesher \
  --input_path "$ROOT/fused.ply" \
  --output_path "$ROOT/surface/meshed-poisson.ply" \
  --PoissonMeshing.depth 10 \
  --PoissonMeshing.point_weight 1 \
  --PoissonMeshing.color 1 \
  --PoissonMeshing.trim 10 \
  --PoissonMeshing.num_threads -1
```

默认 `depth=13`、`point_weight=1`、`color=1`、`trim=10`。稀疏点云上 trim 过高可能得到空网格，A/B 必须记录 trim。

四个独立表面命令共用严格参数解析：支持 `--name=value`，布尔值接受
`0/1/false/true`，拒绝未知/重复选项、数值尾缀、非有限数及无符号负数。
参数范围在 dry-run 或 resume 跳过之前校验；`recon_auto` 也会在 SfM 开始前
校验所有表面参数，包括本次未运行的阶段。Delaunay 的 visibility 索引按模型
记录顺序解释，与 Fusion 的完整模型索引一致，不按数值 image ID 重新排序。

### Delaunay

```bash
$COLITE delaunay_mesher \
  --input_path "$ROOT" --input_type dense \
  --output_path "$ROOT/surface/meshed-delaunay.ply"
```

`input_type=dense|sparse` 默认 dense，不区分大小写。dense 输入目录需包含标准
`sparse/`、`fused.ply`、`fused.ply.vis`，无需读取 RGB 或 dense maps。
sparse 输入直接使用 COLMAP BIN/TXT 模型和点轨迹，无需 Fusion：

```bash
$COLITE delaunay_mesher --input_path "$ROOT/sparse" --input_type sparse \
  --output_path "$ROOT/surface/meshed-sparse.ply"
```

旧 `model_path` 选项已移除。`--DelaunayMeshing.` 参数包括 `max_proj_dist=20`、`max_depth_dist=0.05`、`visibility_sigma=3`、`distance_sigma_factor=1`、`quality_regularization=1`、`max_side_length_factor=25`、`max_side_length_percentile=95`、`num_threads=-1`。
sparse 点按稳定 ID 顺序处理，轨迹转换为注册视图索引；拒绝重复图像观测和不一致轨迹。
射线后方位于无限四面体时，只对有限 hull 面求交，避免读取 CGAL 无限顶点的未定义坐标。
两端点插入/射线累加顺序不同，不能保证与 COLMAP 面集合完全一致；退化或过强正则可得到空网格。

### 简化

```bash
$COLITE mesh_simplifier \
  --input_path "$ROOT/surface/meshed-poisson.ply" \
  --output_path "$ROOT/surface/mesh-simplified.ply" \
  --MeshSimplification.target_face_ratio 0.1
```

参数与 COLMAP 4.0 一致：`target_face_ratio=0.1`、`max_error=0`、`boundary_weight=1000`、
`interpolate_colors=1`、`num_threads=-1`。目标按输入面数比例计算，至少保留一个面的预算；
翻面、退化与拓扑检查可能使实际简化提前停止。`max_error` 是折叠的 QEM 代价上限，
不是欧氏距离，0 表示关闭；边界权重是软约束。颜色按折叠位置插值，关闭时选择
端点代价较低的颜色。线程参数控制初始化，归约顺序固定，运行后恢复调用方线程预算。
旧 `target_ratio`、`target_num_triangles`、`preserve_boundary`、`max_normal_change` 已移除，不提供别名。

### 纹理

```bash
$COLITE mesh_texturer \
  --workspace_path "$ROOT" \
  --input_path "$ROOT/surface/mesh-simplified.ply" \
  --output_path "$ROOT/surface/textured" \
  --output_type BIN
```

`workspace_path` 包含 `images/` 与 `sparse/`；模型中相机尺寸与实际图像不同时，RGB
按相机尺寸重采样。`output_path` 是目录，默认输出 `mesh.ply` 和 `texture.png`。
`output_type=BIN|TXT`（默认 BIN，不区分大小写）选择 PLY 编码，含标准
`comment TextureFile texture.png`、逐面 `texcoord` 和 float XYZ，与 COLMAP 4.0 双向交换。
UV 原点在左下。PLY 不保存额外顶点色或法线，与 COLMAP 纹理输出一致；
`--Colite.texture_write_obj 1` 可额外生成 OBJ/MTL。旧 `model_path/image_path` 入口已移除。

标准 `--MeshTextureMapping.` 参数与默认值：`min_cos_normal_angle=0.1`、
`min_visible_vertices=3`、`view_selection_smoothing_iterations=3`、`atlas_patch_padding=2`、
`inpaint_radius=5`、`texture_scale_factor=1`、`apply_color_correction=1`、
`color_correction_regularization=0.1`、`num_threads=-1`。线程参数用于视图评分，运行后恢复调用方预算。

Colite 扩展使用 `--Colite.texture_` 前缀：`atlas_size=4096`、`max_atlas_size=32768`、
`view_selection_min_component_size=64`、`visibility_resolution=1024`、
`occlusion_relative_tolerance=0.01`、`use_vertex_color_fallback=0`。旧同名
`MeshTextureMapping.*` 扩展已移除，不提供别名。遮挡容差只在受拓扑支持的孔洞恢复中完整使用；
主视图选择最多使用 0.005 的端点噪声带。

`occlusion_relative_tolerance=0.01` 只允许受近共面双邻面支持的孔洞恢复在
目标表面 1% 范围内忽略端点噪声；不建议在没有 A/B 的情况下继续放宽。没有
可靠相机视图的面默认映射到中性 texel，网格仍保留全部三角形；如需逐面
RGB 顶点色 patch，可显式设置 `--Colite.texture_use_vertex_color_fallback 1`。

大网格先用较小的 `MeshSimplification.target_face_ratio` 和 `Colite.texture_visibility_resolution=512` 验证，再逐级提高。

## 组合 MVS

`recon_mvs` 与独立 stage 使用相同的标准选项及交叉参数校验；阶段开关属于 `Colite.*`。
Fusion 默认在工作区根保存标准 `fused.ply` 和 `fused.ply.vis`，与 `sparse/` 同级，
可直接传给原生 COLMAP Delaunay。诊断及附加表面产物仍保存在 `dense/`。网格生成在 Fusion 完成、
其 map 缓存释放后运行；`Colite.mesh_simplifier`、`Colite.mesh_texturer` 对每个启用的
生成器依次执行，输出 `meshed-{poisson,delaunay}[-simplified].ply` 和
`textured-{poisson,delaunay}/`；Colite 的点云网格对应 `mesh[-simplified].ply`、`textured-mesh/`。
独立与组合流程都从实际保存的 PLY 接续，避免中间精度差异。
Poisson、Delaunay、纹理分别接受自身的 `*.num_threads`，不继承 Fusion 的线程参数。
纹理读取器按模型相机尺寸对齐 RGB，两个入口使用相同的图像处理。
组合命令的纹理格式由 `--Colite.texture_output_type BIN|TXT` 选择，默认 BIN；
`--output_type COLMAP` 仍属于去畸变工作区类型。`recon_auto` 将该格式参数传到独立纹理阶段，
也接受 `Colite.texture_write_obj` 及其余纹理扩展。

```bash
$COLITE recon_mvs \
  --input_path output/recon_sfm_building18_YYYYMMDD \
  --image_path data/building-18 --output_path output/recon_mvs_building18_YYYYMMDD \
  --Colite.image_undistorter 1 --Colite.run_patch_match 1 --Colite.fuse_stereo 1 \
  --Colite.patch_match_backend cuda --PatchMatchStereo.geom_consistency 1 \
  --PatchMatchStereo.num_threads 8 --StereoFusion.num_threads 8 \
  --Colite.poisson_mesher 1 --PoissonMeshing.depth 10
```

组合流程在工作区根保存标准 `fused.ply` / `.vis`，`dense/` 保留诊断与附加表面产物。
表面参数与独立命令同名，例如 `PoissonMeshing.*`、`DelaunayMeshing.*`、
`MeshSimplification.*`、`MeshTextureMapping.*`，不再接受对应的小写扁平别名。

## 完整自动流程

```bash
$COLITE recon_auto \
  --image_path data/building-18 \
  --output_path output/recon_auto_building18_YYYYMMDD \
  --sfm.SiftExtraction.max_num_features 8192 \
  --mvs.Colite.patch_match_backend cuda \
  --mvs.PatchMatchStereo.num_iterations 3 \
  --mesher poisson --simplify 1 --texture 1
```

- `start_from/stop_after`：`sfm|undistort|stereo|fusion|mesh|simplify|texture`。
- `mesher`：`poisson|delaunay|none`，默认 poisson。
- `simplify`、`texture` 默认 1。
- `--sfm.<canonical-option>` 转发到 `recon_sfm`，包括 ImageReader.*、FeatureExtraction.*、Mapper.* 和 Colite.*；dry_run 也先校验。旧 --sfm.camera_mode 等名称不再接受。
- `--mvs.<canonical-option>` 按适用阶段分发到去畸变、PatchMatch 和 Fusion；全部 MVS 参数会在 SfM 开始前预校验，包括本次未执行的阶段。
- 表面参数保持 `--PoissonMeshing.*`、`--DelaunayMeshing.*`、`--MeshSimplification.*`、`--MeshTextureMapping.*`。

断点示例：

```bash
$COLITE recon_auto --image_path data/building-18 --output_path "$OUT" --resume
$COLITE recon_auto --image_path data/building-18 --output_path "$OUT" \
  --start_from stereo --stop_after fusion
```

从中间阶段开始前，上游约定产物必须已经存在于相同输出结构中。

## COLMAP 模型比较

比较器支持 TXT/BIN 和混合格式；默认自动识别。也可以在同一模型目录中直接
比较两种保存格式：

```bash
$COLITE compare_colmap --reference_path converted --candidate_path converted \
  --reference_format txt --candidate_format bin --output_path format_comparison
```

```bash
$COLITE compare_colmap \
  --reference_path output/reference_colmap/sparse/0 \
  --candidate_path output/recon_sfm_building18_YYYYMMDD \
  --output_path output/compare_sfm_building18_YYYYMMDD
```

输出 `summary.txt`、`image_diff.tsv`、`pose_diff.tsv`、`point_diff.tsv` 和 `pair_diff.tsv`。位姿先按共同相机中心估计 Sim(3)；独立模型点 ID 不可视为对应关系。

可用标准库 Python 脚本执行真实双向互操作验证，覆盖全部 12 种相机模型，并可
额外传入已有重建结果：

```bash
python3 scripts/check_colmap_model_interop.py \
  --colite "$COLITE" --colmap /path/to/colmap \
  --output_path output/model_interop --input_path sparse/0
```

脚本独立解码 TXT/BIN，逐字段检查 `COLITE → COLMAP → COLITE`，保存模型、
进程日志和 `report.json`。本地 COLMAP 4.0 的 TXT 读取器会截断含空格的图像名，
并将点 ID 按有符号 64 位解析；这些特殊数据跨工具交换时应选择 BIN。


### 最终逐源 NCC/几何过滤

`recon_mvs` 和 `patch_match_stereo` 支持以下 COLMAP 同名参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--PatchMatchStereo.filter` | `1` | 启用独立最终过滤；接受 0/1/false/true |
| `--PatchMatchStereo.ncc_sigma` | `0.6` | NCC 似然分布的标准差，必须有限且为正；移除旧名称 `--patch_match_ncc_sigma` |
| `--PatchMatchStereo.filter_min_ncc` | `0.1` | 范围 [-1,1]；按 COLMAP 将 `1-NCC` 转为似然阈值，与逐源 HMM 后验概率比较，不是逐像素原始 NCC 硬截断 |
| `--PatchMatchStereo.filter_min_triangulation_angle` | `3` | 最小三角化角度，单位度 |
| `--PatchMatchStereo.filter_min_num_consistent` | `2` | 同时通过所有启用条件的源数下限 |
| `--PatchMatchStereo.filter_geom_consistency_max_cost` | `1` | 几何往返重投影像素误差上限 |

新路径只在最细尺度、法线细化后执行。几何细化使用的不可变光度源深度快照也用于最终
过滤；没有源深度的纯光度流程只检查 NCC 和视角。结果保存逐源一致性图，复用它而不再
针对已过滤的新深度重建图。`patch_match.tsv` 中 `final_filter_applied=1` 表示该图已生效。

不能与 `--Colite.filter_photometric_max_cost` 同时启用。该扩展选项在几何细化后过滤综合代价，
并非 NCC。新路径启用后，旧最终几何图重建的重投影/相对深度选项不再参与图构造；
应使用上述 `PatchMatchStereo.filter_*`，并将旧相对深度门槛保留为 0。
深度和法向固定保存为 COLMAP photometric/geometric binary；
`--Colite.write_debug_maps 1` 可额外写调试 TXT，`--Colite.write_cost_maps 1` 可写
Colite 代价图。`--PatchMatchStereo.write_consistency_graph 1` 额外保存标准 COLMAP
一致性图；内部 CSR 文件不属于交换格式。

Colite 和 COLMAP 的最终过滤默认开关均为 1。已对齐 COLMAP 的 NCC likelihood、
HMM 消息和后验阈值公式；HMM 使用纯光度代价，后验不再截断到 source pruning 的
`[0.05,0.999]`。历史后验随迭代更新，法线细化后重新更新；没有历史时从 0.5 推断。
`filter_min_ncc` 经转换得到的概率密度可能大于 1，保持 COLMAP 的原始行为。
PatchMatch 几何优化和过滤改用最近邻源深度；Fusion 深度采样维持现有双线性规则。
Colite 的迭代后行列 HMM 与 COLMAP 扫描内耦合更新仍有调度差异，CPU 假设搜索仍使用
best-K 聚合，不能据此认定整个算法等价。
