# 参与开发

先阅读[架构与重建流程](docs/architecture_and_pipeline.md)和[构建与安装](INSTALL.md)。算法应留在相应核心模块，CLI 和桌面层负责参数、流程和产物管理。修改行为时更新相关说明，并运行涉及的 CPU 测试；改动 CUDA 路径时也验证对应后端。

请勿提交原始照片、模型权重、重建结果、构建目录和本机 SDK 配置。测试日志等生成文件放入本地 `output/`。提交前运行 `git diff --check`。

贡献遵循项目 [BSD 3-Clause](LICENSE) 许可。保留已有第三方署名；引入代码前确认许可证兼容。OpenMVS 仅作外部行为参考，不能把其 AGPL 代码复制或改写到 Colite。
