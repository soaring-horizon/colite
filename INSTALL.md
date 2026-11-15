# 构建与安装

Colite 0.2.0 使用 C++17，需要 CMake 3.20 或更新版本。CLI 可单独构建；桌面端另外需要 **Qt 5.15.2**（精确版本）、Widgets、OpenGL 和 libpng。下面的 CPU 配置适合首次构建，不需要 NVIDIA 显卡。

现有安装包的启动和已验证范围见[桌面使用说明](docs/desktop_user_guide.md)和[版本说明](docs/release_notes.md)。

## Windows x64

准备 Visual Studio 的 C++ 桌面工作负载、CMake、vcpkg、Python、NumPy 和 7-Zip。默认脚本支持 VS 2019，VS 2022 需选择对应生成器和 triplet。

```powershell
python scripts/fetch_qt.py --platform windows --destination output/sdk/qt --cache output/sdk/qt-downloads
python -m pip install numpy
./scripts/build_windows.ps1 -QtRoot output/sdk/qt/5.15.2/msvc2019_64 -VcpkgRoot C:/path/to/vcpkg
```

VS 2022 追加 `-Generator 'Visual Studio 17 2022' -Triplet x64-windows-colite-v143`。脚本构建并测试 GUI/CLI，安装和部署 DLL；默认入口是 `output/windows-release/install/bin/colite_gui.exe`，同目录有 `colite.exe`。不要单独移动两个 EXE，运行库和 Qt 插件在安装目录中。

需要 CUDA 时，准备驱动和与 MSVC/目标显卡兼容的 CUDA Toolkit，再给脚本传入 `-Cuda -CudaToolkitRoot <toolkit路径> -CudaArchitectures <计算能力>`。用独立的 `-BuildRoot` 和 `-InstallRoot` 可同时保留 CPU 版。显卡架构由构建者指定，不能从一次开发机验证推断所有硬件适用。

## Ubuntu 22.04 / 24.04 x64

Ubuntu 22.04 是 Linux 桌面包的 ABI 基线。先装基础依赖：

```bash
bash scripts/setup_ubuntu_22_04.sh
sudo apt install python3-numpy
```

只构建 CPU CLI：

```bash
cmake --preset wsl-release
cmake --build --preset wsl-release --parallel 2
ctest --preset wsl-release
cmake --install build-wsl-release --prefix "$PWD/output/cli-install"
./output/cli-install/bin/colite --version
```

这些 preset 也可在原生 Linux 使用。CUDA CLI 用 `wsl-colite-cuda` preset，可通过 `-DCMAKE_CUDA_COMPILER=...` 和 `-DCMAKE_CUDA_ARCHITECTURES=...` 选择工具链和架构。无 AVX2 的 CPU 构建时请加 `-DCOLITE_ENABLE_AVX2=OFF`。

构建原生 Linux 桌面端：

```bash
sudo apt install p7zip-full libgl1-mesa-dev libpng-dev libxcb-xinerama0 \
  libxkbcommon-x11-0 fonts-noto-cjk
python3 scripts/fetch_qt.py --platform linux --destination output/sdk/qt --cache output/sdk/qt-downloads
cmake -S . -B output/linux-desktop/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/output/sdk/qt/5.15.2/gcc_64" \
  -DCOLITE_BUILD_GUI=ON -DCOLITE_BUILD_TESTS=ON -DCOLITE_ENABLE_AVX2=OFF \
  -DCOLITE_ENABLE_CUDA_SIFT=OFF -DCOLITE_ENABLE_CUDA_MATCHER=OFF -DCOLITE_ENABLE_CUDA_MVS=OFF
cmake --build output/linux-desktop/build --parallel 2
ctest --test-dir output/linux-desktop/build --output-on-failure
cmake --install output/linux-desktop/build --prefix "$PWD/output/linux-desktop/install"
```

用 `output/linux-desktop/install/bin/colite_gui` 启动。如果从 Qt SDK 直接构建且系统无法找到 Qt 动态库，启动时设置 `LD_LIBRARY_PATH` 指向该 SDK 的 `lib/`。桌面启动需要 X11 或 XWayland。仅想编译而不跑测试时可配置 `-DCOLITE_BUILD_TESTS=OFF`。

## 可选算法与部署

- CUDA SIFT、匹配、PatchMatch 用三个独立的 `COLITE_ENABLE_CUDA_*` 开关配置。运行 `colite --capabilities` 确认该二进制的实际能力。
- Delaunay 需要 CGAL。若未找到 CGAL，仍可用随源码构建的 Poisson。没有 PoseLib 时常规 SfM 可用内置解算，部分广义注册功能不可用。
- ALIKED 和 LightGlue 需要 `-DCOLITE_ENABLE_ONNX=ON`。传入 `-DONNXRUNTIME_ROOT=<SDK>` 或使用 SDK 的 `onnxruntime_DIR`；重建后需要相应运行库与另行取得的模型。安装根目录的 `models/` 会自动查找。官方模型名称、下载源和 SHA-256 定义于 `src/colite/io/model_resource.cpp`。分发权重前应独立确认许可。
- CLI 和桌面程序来自同一源码版本。部署前运行 `colite --build-info`、`colite_gui --build-info` 检查版本和源码提交。无 Git 元数据的源码归档可用 `-DCOLITE_SOURCE_REVISION=<commit>` 写入冻结提交。

安装包含许可证、文档和离线 COLMAP 交换脚本（位于 `share/colite`），桥接需 Python 3 和 NumPy。从干净的输出目录进行安装检查：

```bash
python3 scripts/smoke_installed.py --prefix "$PWD/output/cli-install" \
  --output_path "$PWD/output/installed-smoke"
```

## 打包和核验

源码安装用于开发测试。可分发目录用 `scripts/package_windows.ps1` 或 `scripts/package_linux.py --help` 组合依赖；Linux 可用 `scripts/package_linux_deb.py` 生成带桌面入口的 DEB，其 `--work` 需要支持 Unix 权限的文件系统。Windows 可选 CPU ONNX SDK 用 `-OnnxRoot`、`-VcRuntimeRoot`，Linux 用 `--onnx-root`。CUDA execution provider 的依赖需要单独验证，不能套用 CPU ONNX 打包流程。

用 `scripts/collect_desktop_sources.py` 收集对应的第三方源码和许可。发行包应从确定的版本提交重新构建，确认工作区干净，在各自原生平台运行 `seal_desktop_candidate.py probe`，再运行 `seal`、`verify` 和 `archive`（参数见各脚本帮助）。封存记录源码、原生探测、依赖与逐文件校验和。发布前还应验证从独立包启动、导入和处理照片，以及在目标系统上的运行；开发机启动不能替代全新系统验收。
