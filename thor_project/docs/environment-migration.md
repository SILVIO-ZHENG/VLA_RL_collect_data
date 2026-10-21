# 环境迁移与 PI0.5 依赖审计

本记录对应 Python 3.12 的控制端、视觉端和 PI0.5 服务端。环境目录、权重与密钥均不入库。

## 运行边界

| 进程 | 项目内解释器 | 重建入口 | 源码依赖 |
|---|---|---|---|
| 控制 API 与视觉 Sidecar | `.venv/bin/python` | `scripts/build_runtime_env.sh`、`uv.lock` | 本仓库；视觉 Ultralytics 8.4.146 |
| PI0.5 WebSocket 服务 | `.pi05-venv/bin/python` | `scripts/build_pi05_env.sh` | 本仓库与固定 LeRobot 上游提交加补丁 |
| 机器人客户端 | 机器人端现有 GDK 环境 | `src/nexus_deploy/robot/g2a` 按 `robot_manifest.txt` 打包 | 不导入 LeRobot |

`configs/5090.toml`、`configs/thor.toml` 和测试配置使用配置目录相对路径；解释器路径在代码中只做词法归一化，不解析符号链接。`.venv` 的 `pyvenv.cfg` 必须声明 `include-system-site-packages = false`。视觉权重另行提供，旧权重 SHA256 记录于 README；配置路径不会由构建脚本自动下载。

5090 本次安装为节省下载时间，先从已有 Python 3.12 环境复制与锁文件版本一致的已安装包，再由 `uv sync --locked` 补齐并核验。此操作仅用于本机加速；重建入口仍是锁文件和构建脚本，Thor 不复制 5090 的包或环境目录。

PI0.5 在 5090 上使用 `third_party/lerobot/requirements-5090.lock` 固定 78 个包版本；Thor 可设置 `NEXUS_PI05_CONSTRAINTS` 使用目标平台约束文件，必须在 Thor 本机验证 CUDA wheel 后生成。本次没有构建 Thor 环境。

## 仓库外 LeRobot 审计

原 5090 LeRobot 工作区位于 `/home/luoxiaolong/workspace/lerobot`，其 Git HEAD 为 `a9879e69edefb3c308f7025046d3d12b3cb2fe50`，存在未提交修改。原 PI0.5 服务通过 `PYTHONPATH` 指向该可变源码。现改为从该上游提交重建 `third_party/lerobot/source`，应用并校验 `third_party/lerobot/pi05-deployment.patch`（SHA256 `8ddd844c1c2bbf1271dcd2b5ed831d0107116236085683253be34485ce567c95`）。`source/` 是本地生成并忽略提交的固定源码检出；补丁、依赖锁和说明文件保留在同一个 `third_party/lerobot/` 目录。

| 外部变更 | 推理是否依赖 | 迁移方式 |
|---|---|---|
| `policies/factory.py` 预处理覆盖与 tokenizer 选择 | 是，离线 tokenizer | 固定补丁 |
| `policies/pi05/configuration_pi05.py` tokenizer 与训练元数据字段 | 是，读取检查点配置 | 固定补丁；默认 tokenizer 恢复上游值，现场通过 `--tokenizer-path` 指定本地资源 |
| `policies/pi05/processor_pi05.py` tokenizer 配置 | 是 | 固定补丁 |
| `policies/pi05/modeling_pi05.py` 严格失败、快速加载与缓冲区重建 | 是，现有服务的快速加载路径 | 固定补丁 |
| `robots/g2a/`、`robots/__init__.py` G2A 扩展 | 服务仅使用 24 维关节名，机器人客户端不依赖它 | 从本仓库 `nexus_deploy.robot.g2a.constants` 读取同序常量；不移植 LeRobot G2A 类 |
| `rollout/context.py`、训练脚本与日志工具改动 | 当前服务不依赖 | 不进入服务补丁；以后训练环境需另行审计 |
| LeRobot 仓库中其他未追踪数据、文档与实验脚本 | 当前服务不依赖 | 不复制、不纳入环境 |

上游原有的 `lerobot.policies.rtc.reanchor_relative_rtc_prefix`、`RTCConfig`、processor API 和 PI0.5 policy 仍是版本固定的库依赖，不能单独用本仓库代码替代。补丁校验脚本检查提交、补丁哈希、实际差异及额外源码改动；失败时停止构建。

## 离线核验

```bash
.venv/bin/python scripts/prepare_pi05_lerobot.py --verify
.venv/bin/python -m nexus_deploy.site --site 5090 --check
.venv/bin/python -m pytest -q
.pi05-venv/bin/python -m nexus_deploy.vla.pi05.server --help
```

5090 的 `--check` 在 x86_64 上运行；Thor 的站点启动检查在 Thor 上运行，两份 TOML 可在开发机解析。以上检查不连接机器人或启动推理监听。模型权重加载、视觉准确率、GPU 并行时延、Thor CUDA wheel 与真机执行仍需目标平台验证。Thor 不能复用 5090 的虚拟环境目录；需在 Thor 本机重建并确认 Torch/CUDA wheel 兼容。
