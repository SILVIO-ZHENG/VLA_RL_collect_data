# Nexus Thor 部署与维护 / Deployment and Maintenance — DF-01-02

## 中文

### 1. 来源与交接范围

本分支保存 **2026-10-01 从 Thor 采集的部署源码**。仓库：`http://61.157.75.168:8010/luoxl/nexus-deploy-platform`；分支：`DF-01-02`；对照 main 基准：`d480a839`。Thor 工作目录没有 `.git`，以 [源码清单](docs/deployment/thor-20261001/source-manifest.json) 的来源、时间与 SHA-256 记录实机版本；系统、服务和环境见 [部署证据目录](docs/deployment/thor-20261001/)。

5090 到 Thor 的地址变化、需同步检查的文件和后续更新步骤，单独列在 [迁移适配清单](docs/5090_TO_THOR_MIGRATION.md)。

从模型生成动作到机器人真正执行的完整调用链，见 [机器人执行代码详解](docs/THOR_ROBOT_EXECUTION_CODE_GUIDE.md)：逐个列出 **67 个文件**的作用及本机/Thor/机器人地址，解释频率、插值、平滑、限速、碰撞安全、夹爪、暂停与复位。另有 [硬件与安全逐函数说明](docs/execution/ROBOT_HARDWARE_FUNCTION_REFERENCE.md)，覆盖九个模块的245个类/函数。

2026-10-08 前次注释整理为 **63 个 Python 文件**新增 **3,061 行中英文注释**，中文在前、英文在后。该次原可执行语句、docstring和各行顺序保持；AST及非注释token逐文件一致，见 [验证记录](docs/execution/annotation-verification.json)。四个视觉规则核心受源码SHA守卫固定，保留原字节并在文档及wrapper解释。该次仅更新仓库注释/文档，没有部署至Thor或机器人、没有启动动作；下述调度交还修复是后续单独的功能变更。

下文路径缩写均为 Linux 绝对路径：

```text
ROOT = /home/anu/deploy/nexus_deploy_5090_copy_20260928
ROBOT_ROOT = /home/agi/deploy/nexus_deploy
```

`ROOT` 是已核实的控制进程工作目录；`ROBOT_ROOT` 是配置目标。G2 `.101`、`.102` 的 SSH 探测超时，未取得机器人当前快照，不能证明 Thor 客户端源码与机器人运行文件一致。采集包含控制、视觉、VLA、客户端源、资产和 LeRobot；权重、环境、运行数据与凭据不入 Git。本次发布没有启动机器人动作。

Thor 运行控制网页、推理、视觉和只读三维；G2 负责相机、动作队列、安全和 GDK。人工与调度共用后端，同时只接受一个任务，忙时拒绝、不建立业务等待队列。控制台 `/` 或 `/monitor`，模型页 `/index.html`，运行配置 `/session.html?id=<部署ID>`，记录 `/history.html`。关闭浏览器不等于结束任务。

### 2. 设备 IP、内外网、有线无线

“实测”指本次 2026-10-01 采集；“配置”不等于连通；“历史”不能作为当前 DHCP 地址使用。

| 设备 / 网络 | 地址、接口与状态 | 来源 / 用途 |
|---|---|---|
| Thor 有线内网 | `10.42.1.77/24`，`enP2p1s0`，UP | 实测；机器人链路、推理广告地址、遥测 |
| Thor 无线内网 | `192.168.110.14/24`，`wlP1p1s0`，UP、DHCP | 实测；SSH 用户 `anu`、当前控制网页 |
| 无线网关 | `192.168.110.1` | 实测默认路由，`wlP1p1s0`、metric 600 |
| Thor USB bridge | `192.168.55.1/24`，`l4tbr0`，DOWN | 实测；经 `192.168.55.100` 的另一默认路由为 linkdown，不是有效出口证据 |
| Thor Docker bridge | `172.17.0.1/16`，`docker0`，DOWN | 实测本地虚拟网段 |
| G2A 有线管理 | `10.42.1.102:22`；历史 `enp1s0`、`/24` | 配置目标，SSH 超时；用户 `agi`，预期 AID `G2A0004C201260` |
| 同一 G2A 厂商别名 | 历史 `10.42.1.101/24` | 本次 SSH 也超时；不是另一台机器人 |
| G2A 无线 | 历史 `192.168.110.40`；更早 `.237`、`.30` | `.40` 记录于 09-15，本次从 Thor 探测 SSH 超时；`.237/wlan0` 于 09-10 记录；当前接口/身份未核实 |
| 5090 跳板 `anu-server` 有线 | `10.42.1.200/24`、`192.168.110.140/24`，同属 `eno1`，UP | 本次实测两个有线地址；旧文档称 `.140` 为无线的描述不适用于当前配置 |
| 5090 跳板无线 | `192.168.110.60/24`，`wlp68s0`，UP | 本次实测无线地址；不是 `.140` |
| 5090 跳板默认路由 | 均经 `192.168.110.1`：`eno1` metric 100，`wlp68s0` metric 600 | 本次实测；有线默认路由优先 |
| Git 公网入口 | `61.157.75.168:8010` | 用户指定并已确认可达；不说明 Thor 拥有该公网 IP |
| 5090 公网 SSH 跳板 | `61.157.75.168:2281` | 本次已登录并核实主机 `anu-server`；Thor 运行无需依赖跳板 |
| Windows 维护笔记本无线 | `192.168.27.34/24`，WLAN；网关 `192.168.27.250` | 本次 ipconfig 实测；当前为异地维护网络，物理以太网断开 |
| 维护笔记本虚拟网卡 | `26.26.26.1/29`，LetsTAP，无默认网关 | 本次实测虚拟适配器，不是机器人现场网络 |
| 维护笔记本现场有线规划 | 历史建议 `10.42.1.78/24` | 仅到现场接线时的规划建议，当前未使用；公网出口未核实 |
| 调度主机 / 各设备独立公网 IP | 未核实 | 不根据 Git 地址推断 NAT 或公网接口 |

Thor `alternate_endpoints=[]`，没有机器人无线回退。历史机器人网络服务保留厂商 `.101` 并添加 `.102`；迁移时核实 `scripts/robot-network/`，不因发布代码顺便改网络。

| 服务 | 地址与采集状态 |
|---|---|
| 控制 UI/API | **`http://192.168.110.14:21865/`，实测监听** |
| 遥测 | `tcp://10.42.1.77:8768`，实测监听 |
| Viewer / SSH | `0.0.0.0:8766/8767`、`:22` 实测监听；8767 另有 IPv6 |
| PI0.5 | 配置 `ws://10.42.1.77:5556`，绑定 `0.0.0.0`；采集时未见监听 |
| Presence ingress | 配置 `tcp://10.42.1.77:8776`；采集时未见监听，按任务建立 |
| Presence callback | 配置 `http://192.168.110.14:21865`，与实际控制入口一致 |
| 对齐 | 历史 Thor `:8775/alignment`，旧服务 inactive、未见监听；G2 模板 `.102:8080` 未核实 |

**配置差异：** `configs/thor.toml [host]` 仍是 `.77:8765`，实际 service/进程覆盖为 `.14:21865`。直接运行 `nexus_deploy.site --site thor` 不是当前服务的等价重启。以后统一端口，须同时修改 service、`[host]`、presence callback、浏览器与调度地址并验证。

同 Wi-Fi 直接打开当前地址。Windows 隧道示例，保持终端运行后访问 `http://127.0.0.1:18765/`：

```powershell
ssh -NT -o ExitOnForwardFailure=yes -L 18765:192.168.110.14:21865 -L 18766:10.42.1.77:8766 -L 18767:10.42.1.77:8767 anu@192.168.110.14
```

当前在外网时，经已核实的公网跳板访问 Thor；将 `JUMP_USER` 替换为自己的已授权跳板账号，SSH 会分别验证跳板与 Thor：

```powershell
ssh -NT -o ExitOnForwardFailure=yes -J JUMP_USER@61.157.75.168:2281 -L 18765:192.168.110.14:21865 -L 18766:10.42.1.77:8766 -L 18767:10.42.1.77:8767 anu@192.168.110.14
```

DHCP 改址后先用 `ip -br -4 address` 核实，再同步绑定、回调和访问地址。

### 3. 模块文件与部署位置

下表相对路径以 `ROOT` 为根，展开即 Thor 完整地址。配置 `../` 从 `ROOT/configs/` 解析，模型参数 `models/...` 从模型工作目录 `ROOT` 解析。

| 模块 | 仓库相对路径 / 绝对部署路径 |
|---|---|
| 配置 / 启动 | `configs/thor.toml`；`src/nexus_deploy/site.py` |
| 控制 / 部署 / 准入 | `src/nexus_deploy/control_api.py`、`src/nexus_deploy/deployment.py`；`src/nexus_deploy/control/` 下 `tasks.py`、`vision.py`、`robot.py`、`defaults.py` |
| 调度 / 门禁 | `src/nexus_deploy/sorting_api.py`、`src/nexus_deploy/presence_gate.py` |
| 网页 / 三维 | `src/nexus_deploy/static/`、`src/nexus_deploy/viewer/` |
| 对齐 | `src/nexus_deploy/alignment/`；现场 `alignment.toml`，模板 `alignment.example.toml` |
| PI0.5 | `src/nexus_deploy/vla/pi05/`；入口 `nexus_deploy.vla.pi05.server` |
| 实机 LeRobot / RTC | `ROOT/third_party/lerobot/source/src/lerobot/policies/` 下 `pi05/`、`rtc/` |
| 入库 LeRobot / 重建 | `snapshots/lerobot/`；`scripts/prepare_pi05_lerobot.py`、`third_party/lerobot/pi05-deployment.patch`；快照不是自动 import 入口 |
| PI 权重 / tokenizer | `models/pi05/dahuan/<实验>/checkpoint-<步数>/`；`models/tokenizers/paligemma-3b-pt-224/` |
| 视觉入口 / 运行时 | `src/nexus_deploy/vision/parcel/run_nexus_presence_sidecar.py`；同目录 `runtime/` 的 `nexus_frame_stream.py`、`estimator.py`、`config.py`、`service.py` |
| 视觉核心 / ROI / 权重 | `src/nexus_deploy/vision/parcel/rules_core/`、`src/nexus_deploy/vision/parcel/configs/g2_head.yaml`；`models/vision/parcel/yolo26m-seg-final.pt` |
| 机器人源码 / manifest | `src/nexus_deploy/robot/g2a/`；`robot_manifest.txt` |
| 客户端关键文件 | 上行目录内 `client.py`、`action_plan.py`、`gdk_adapter.py`、`async_safety.py`、`safety_layer.py`、`constants.py`、`assets/` 与安全 JSON |
| 图像 / 遥测 | 同目录 `presence_stream.py`、`telemetry.py`、`client.py` |
| bundle / 远端目标 | `dist/thor/g2a_robot_client/`；`/home/agi/deploy/nexus_deploy/robot_client/`，复位 `reset_all_joints.sh`；实际远端未核实 |
| 状态 / 归档 / 调度配置 | `runtime/thor/`、`runtime/thor/robot-sessions/`、`runtime/thor/sorting.json`；远端归档默认 `/home/agi/.cache/nexus-platform-sessions` |
| 控制/视觉 Python / PI Python | `ROOT/.venv/bin/python` / `ROOT/.pi05-venv/bin/python` |
| service / 凭据来源 | `/home/anu/.config/systemd/user/nexus-thor-control.service`；`/home/anu/.config/nexus-thor-control.env`（只记录位置，内容不入库） |

当前安全文件 `src/nexus_deploy/robot/g2a/G2_dahuan_standard_table_20260922_contact_refined_safety_execute_v2.json`，SHA-256 `1596634e164161fa0aebe583efc6898fec622154953d18415a8b8203a9eccb94`。最终任务 `--safety-config`、AID、URDF 和标定来源仍须核对，不能直接用于另一现场。

该文件明确记录 `measurement_mode=contact-refined-temporary`，`final_camera_extrinsic_calibration_pending=true`、`replacement_apriltag_pending=true`，碰撞几何仍为 `provisional`。迁移清单必须包含最终相机外参、AprilTag、更换工位后的桌面边界与真实几何验收；文件名中的 execute 不表示这些项目已完成。

旧 `/home/anu/test_20260916_senbao/...`、独立视觉目录、`/home/agi/deploy/g2a_robot_client*` 是历史路径。G2 对齐 service 模板仍指旧 `/home/agi/deploy/g2a_robot_client`，与新 remote_directory 不一致；恢复连接后单独核实迁移。

### 4. 服务、环境与当前配置

`nexus-thor-control.service` 实测 active/running，工作目录 `ROOT`，使用 `.venv/bin/python -m nexus_deploy.control_api`，加载 `configs/thor.toml`、`runtime/thor/sorting.json`，开放 actions/execute/reset。旧 `g2a-control`、`g2a-alignment`、`g2a-pi05` 均 inactive。

**已查明有效启动配置：** 基础 unit 的参数被 drop-in `10-unauthenticated.conf` 覆盖，实际使用 `--password-env NEXUS_UI_PASSWORD_DISABLED_20260929 --allow-unauthenticated-private-lan`。另一个 `20-sorting-token.conf` 只加载 `EnvironmentFile=/home/anu/.config/nexus-thor-sorting-5090.env`。两份 drop-in 保存于 `ops/systemd/nexus-thor-control.service.d/`，组合配置见证据 `effective-control-unit.txt`；这已解释基础 unit 与运行 argv 的差异。内网网页准入与调度 token 是不同通道，凭据内容不入库。维护时查看 `systemctl --user cat nexus-thor-control` / `show ... -p ExecStart`，不要只读取基础 unit。

| 项目 | 2026-10-01 实测 |
|---|---|
| OS / 架构 / kernel / L4T | Ubuntu 24.04.4 LTS / aarch64 / `6.8.12-1021-tegra` / R39.2.0；未推断 JetPack 商品版本 |
| 两套 Python | 3.12.3；`include-system-site-packages=false` |
| 两套 imported Torch / CUDA / GPU | `2.11.0+cu130` / `13.0` / `NVIDIA Thor`；CUDA available |
| `.venv` 包元数据 | Torch `2.11.0`、Torchvision `0.26.0`、Ultralytics `8.4.146`、NumPy `2.2.6`、OpenCV `4.11.0.86`、SciPy `1.18.0` |
| `.pi05-venv` 包元数据 | Torch `2.11.0+cu130`、Torchvision `0.26.0+cu130`、Transformers `5.5.4`、PEFT `0.20.0`、LeRobot `0.6.1`、NumPy `2.2.6` |

`.venv` distribution 元数据与 imported Torch 标记不同；重建须核对实际导入位置、wheel 来源和 ABI。详见证据目录 `*-environment.json`、`*-pyvenv.cfg`。

站点 TOML 默认 **read-only、RTC、执行 30 FPS**；模型 30 FPS、horizon 30、CUDA 0、port 5556、启动超时 600 秒。手臂速度 3.76、加速度 8、`urdf-velocity-uniform-acceleration`、savgol；双低通关闭，workspace safety 开启，stationary hold feedback 上限 **100 ms**。

2026-10-01 采集时，service 加载的 `runtime/thor/sorting.json` 中，`parcel-sort` profile 是 **execute、RTC、执行 15 FPS、servo 100 Hz**；presence **enabled=true、automatic、3 秒暂停、6 秒复位、auto_resume=true**，复用默认 PI0.5 版本，callback_url 未设置。历史脱敏记录见 [sorting.config.redacted.json](docs/deployment/thor-20261001/sorting.config.redacted.json)，保留当时数据。它与 TOML/源码默认值是不同层，交接必须同时记录；配置存在不等于启动了物理任务，也不构成新的安全认证。

**2026-10-08 调度交还修复：** 当前运行方案与 `parcel-sort` 统一为 `server_mode=reuse` 和 `G2_dahuan_standard_table_20260922_contact_refined_safety_execute_v2.json`。按操作员要求，连续无包裹 **3 秒暂停、5 秒触发自动复位**；5 秒指无包裹等待时间，不是机械复位运动时长。`control/defaults.py`、`PresenceTaskGateConfig` 默认值及 Thor 的调度配置一致。更新后仍保持人工模式、暂停接单，不启动机器人动作。

`/api/workstation` 和 `/api/profiles` 的模板状态现在包含 `reason` 与 `config_mismatches`，后者列出字段、当前部署值和模板值；页面及交还按钮提示显示这些原因。页面的“配置匹配模板”基于模板与已保存的部署状态，不代表模型已通过本次交接的实时健康检查；交还调度时仍检查模型实时健康，`NOT_READY` 会具体说明模型未确认运行或实时健康检查失败。完整配置匹配、机器人空闲、成功复位、健康检查及 30 秒内机器人检测要求保持。控制服务沿用 systemd 默认的 `KillMode=control-group`：重启控制服务会停止同一 cgroup 中的 PI0.5 模型进程，本次没有改变该清理策略。模型停止后，必须通过正常模型准备、保存并应用流程重新加载真实模型并确认健康，再使用 `reuse` 配置；`reuse` 不会自行加载缺失的模型服务。模型服务策略是模型准备阶段参数；要从 `real` 改为 `reuse`，应准备/选择复用服务的候选方案，再保存并应用，不能直接修改当前方案的加载前参数。仅切换“允许新任务”不会同步固定调度模板。

只读检查：`systemctl --user status nexus-thor-control`、`ip -br -4 address`、`ss -ltn`。重启影响受管模型、任务和 CSRF，仅在任务结束且客户端退出后操作。发布代码无需启动或重启机器人。

### 5. 下次更换模型

**PI0.5 使用新增版本。** 默认 ID `large_qr_clean_v2_seed1000_007500` 对应 `ROOT/models/pi05/dahuan/large_qr_clean_v2_seed1000/checkpoint-007500`。实机资产清单确认 8 个配置版本中只有此 checkpoint 可用；另 7 个（`large_qr_clean_v2_seed1000_005000`、`per_package_001500/004500`、`per_sample_20260820_001500/007500`、`per_sample_20260824_005000/007500`）未部署，不能因 UI 可选就使用。

默认目录中 `model.safetensors` 为 **9,354,050,752 字节**，SHA-256 **`a15a67f4a45f6b7c5359c8b48b16fb4bf8894cb6e7ff5ea1777a3ffb9d2ddd74`**；本地 tokenizer 已存在。完整文件记录见 [model-assets.json](docs/deployment/thor-20261001/model-assets.json)。本次检查了资产，未重新加载模型。

1. 记录旧版本、资产哈希、源码/LeRobot、tokenizer、归一化和安全配置。结束活动任务、关闭模型。
2. 新完整 checkpoint 放 `models/pi05/dahuan/<新实验>/checkpoint-<步数>/`；保留旧目录，不用可变 `current` 软链接。
3. 在 `configs/thor.toml` 新增版本表；下例替换 ID/路径。在已有 variant 表修改 `default_version`，或先网页选择新版本验收。ID 格式 `[a-z0-9][a-z0-9_.-]{0,79}`。

```toml
[models.pi05.effector_variants.dahuan.versions.experiment_20261001_010000]
display_name = "Experiment 2026-10-01 · Step 10000"
normalization_source = "checkpoint_embedded"
normalization_label = "Experiment 2026-10-01 · embedded Quantiles"
real_args = ["--checkpoint=models/pi05/dahuan/experiment_20261001/checkpoint-010000"]
```

4. `common_real_args` 前置；版本 `real_args` 替换继承项，不自动追加。保留 `--tokenizer-path=models/tokenizers/paligemma-3b-pt-224`。外部统计 `external_training_asset` 必须配 `--robot-norm-path=...`；base_args 不硬编码 `--effector-profile`。
5. 验证相机 key/顺序/尺寸/RGB、16D 状态动作映射、绝对/相对语义、Dahuan 米制 0–0.025 与反向动作映射、FPS/chunk/RTC。新 ID 是 `dahuan`，不是旧 `dh`。
6. 检查命令后，按现有服务方式重载配置、刷新网页、加载与 warmup；先离线/只读再现场动作验收。`--check` 只检查配置/路径，不证明模型可用，也不复制 service 端口覆盖。

```bash
.venv/bin/python -m nexus_deploy.site --site thor --check
.venv/bin/python -m nexus_deploy.deployment --config configs/thor.toml server command pi05
.pi05-venv/bin/python -m nexus_deploy.vla.pi05.server --help
```

回滚：结束任务/模型，恢复旧 ID、配置与配套源码/环境，再加载验证。兼容权重通常无需重部署客户端。新策略家族新增 `[models.<id>]` 的 module、Python、repo、端口/GPU、限制、模式和机器人兼容性；协议变化同步 server/client/contracts/manifest/tests。`ui_models` 只是展示项。

**YOLO 不能只换路径。** `presence_runtime.weights` 选文件，parcel 下 `runtime/config.py` 固定哈希；`rules_core/model_adapter.py` 要求类别 `{0: package, 1: left_gripper, 2: right_gripper}` 和 segmentation masks；`runtime/core_bridge.py` 校验核心代码。本次实测权重 SHA-256 与预期一致：`6ca0492ba45211e14e8f5d4f2eeac6bc16c29476f5f0646b4503b4b5c0d40c2c`。新权重经标注视频验收后更新版本/哈希/契约；改核心同时更新经审核的核心哈希。不得跳过校验或以框代 mask。测试三路、遮挡、断流/缺图到 UNKNOWN、并发 GPU 延迟；支架/头姿/裁剪/FOV 改变后重标 ROI。

### 6. 更新 Nexus 时的 Thor 适配清单

本节未写完整前缀的 Python 模块路径均相对 `ROOT/src/nexus_deploy/`；脚本与 TOML 路径仍相对 `ROOT`。

| 项目 | 文件与复核点 |
|---|---|
| 站点 / 网络 | `configs/thor.toml`、service：aarch64、ROOT、`.14:21865` 覆盖、`.77` 推理/遥测、callback，处理 `[host]` 差异 |
| Python / ARM64 | `deployment.py::_interpreter`、`site.py` 保留 venv 词法路径，不 `.resolve()` 成基础 Python；两个环境在 Thor 重建 |
| 模型 / 夹爪 | 版本表、`dahuan`、tokenizer/Quantiles、16D/RTC，核实完整资产 |
| 三路传输 | `robot/g2a/presence_stream.py`、`vision/parcel/runtime/nexus_frame_stream.py`、`deployment.py`：认证/代次/序列校验，head-only 兼容；presence 会话用 `NEXUS_PRESENCE_FRAME_MODE=three` |
| 三路融合 | sidecar、control ingest：head/left wrist/right wrist 各 3 个真实样本；全部空才空，任一路确认有则有，否则 UNKNOWN；异常不判空 |
| 时序 / ROI | capture gap 超过 0.75 秒清证据；头部 ROI、手腕全图，头图比例 1.60±3%；运行时三帧覆盖 YAML 旧四帧 |
| 暂停 / 完成 | `control/defaults.py`、`tasks.py`、`presence_gate.py`、`control_api.py`：当前为 3 秒暂停/5 秒复位触发；核对默认、服务内存和固定模板；保留所有权、复位和新计划屏障 |
| 操作员结束 | 已执行调度任务的 `operator_end` 是业务完成；`conveyor_delivery_independently_verified=false`、reset pending；取消保持 cancelled，失败复位阻止下一物理任务 |
| 安全 / 客户端 | adapter、safety、queue、URDF/mesh、安全 JSON、manifest 一起核对，不放宽失效条件掩盖延迟；对齐旧 service 路径单独迁移 |
| 上游 / 发布 | 固定 LeRobot commit+实际 patch+快照；逐文件来源/哈希，分别记录离线、GPU、只读、物理测试 |

本次 canonical `robot/g2a/gdk_adapter.py`、`safety_layer.py` 同基准 main。09-30 安全 staging 的共享 FK 候选不能宣称已进入正式源码，不盲目覆盖。三路与 operator_end 同改 `control_api.py`，合并须保留两者。

构建入口 `scripts/build_runtime_env.sh`、`scripts/build_pi05_env.sh` 固定 Torch 2.11.0/Torchvision 0.26.0；runtime 用 `uv sync --locked`。PI 的 `NEXUS_PI05_CONSTRAINTS` 不能覆盖冲突精确版本。保留 CUDA13 ARM64 wheel 来源/哈希，版本变化须审核构建输入；不复制 5090 环境。`NEXUS_PYTHON312` 选择 Python，`NEXUS_LEROBOT_MIRROR` 可选择镜像。

实测 LeRobot HEAD `a9879e69edefb3c308f7025046d3d12b3cb2fe50`，有 factory、PI05 configuration/modeling/processor 四个修改；实际 patch 与源码保存于证据及快照。`snapshots/lerobot/` 仅保存源码、不含上游 `.git`；标准 `prepare_pi05_lerobot.py` 仍通过 clone/checkout/apply/verify 重建带 Git 元数据的 `third_party/lerobot/source/`。不要把快照直接覆盖该运行检出而跳过验证；snapshots 不自动参与导入。

```bash
./scripts/build_runtime_env.sh
./scripts/build_pi05_env.sh
.venv/bin/python scripts/prepare_pi05_lerobot.py --verify
.venv/bin/python -c 'import sys,torch,nexus_deploy; print(sys.executable,sys.prefix,sys.base_prefix); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(nexus_deploy.__file__)'
.pi05-venv/bin/python -c 'import sys,torch,lerobot; print(sys.executable,sys.prefix,sys.base_prefix); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(lerobot.__file__)'
```

以上为维护步骤，本次未重建线上环境。检查隔离的 `pyvenv.cfg`，对两个解释器运行 `uv pip check --python <解释器>`。`readlink -f` 仅诊断，配置保持 venv 路径。CUDA available 不代表全部算子或物理流程已验收。

### 7. 部署、采集与验证边界

机器人恢复连接、AID 正确、任务/GDK 空闲后才更新客户端；本次未执行：

```bash
.venv/bin/python -m nexus_deploy.deployment --config configs/thor.toml client bundle --robot g2a
.venv/bin/python -m nexus_deploy.deployment --config configs/thor.toml client deploy --robot g2a
```

核对 manifest、远端哈希、工作目录/import 来源、对齐 agent、安全 profile。`alignment.example.toml` 的 `target_configured=false`，示例位姿不是现场测量。

采集由实际进程/解释器/import、配置、editable 包、上游依赖、机器人源/资产确定；逐文件记录主机/来源/时间/目的路径/SHA-256，说明映射到 `snapshots/lerobot` 等位置。嵌套 `vision/parcel/runtime/` 是源码，不可按名字排除所有 runtime。Git 排除权重/tokenizer、venv、日志、数据、视频、帧、缓存、`.env`、私钥、token；资产另交接版本/路径/哈希/大小/获取方式。clone 后仍需模型、环境与受限现场配置。

发布源码清单 `published-source-manifest.json` 记录 **1,570 个文件**，其中上游 LeRobot **955 个**。27 个文档符号链接已展开为文件，保留 `source_symlink` 来源字段。`vision/parcel/runtime/` 源码已完整补齐。原始采集源码保存于注释前基线提交 `2409e97cb2745974128989eb9311113eca5fd44d`；该清单SHA继续记录原始版本，不改成注释版hash。2026-10-08新增注释改变文件字节/行号，注释版hash及行为文本一致性另见验证记录；Windows checkout还需考虑Git换行转换。

本次在 Thor 的实际 `.venv` 上完成：presence ingest **4 passed**；其余 10 个离线测试模块 **364 passed**；站点 `--check` 与 LeRobot 固定源码 `--verify` **通过**。测试分两个进程运行，避免测试中的持久 stub 相互污染，不代表启动模型或机器人。结果见 `pytest-presence-ingest.txt`、`pytest-offline.txt`、`site-check.txt`、`lerobot-verify.txt`。这不是全仓库测试或物理验收声明。

机器人仍不可达，未取得其运行快照，未启动动作、未重载 PI/YOLO。源码快照已发布到 `DF-01-02`；每次普通 push 后检查远端提交与 README，不使用 force push。后续另做模型加载、真实只读链路、现场物理和最终标定验收。

参考：[代码结构](docs/code-structure.md)、[环境](docs/environment-migration.md)、[模型](models/README.md)、[调度](docs/scheduler-api.md)、[视觉](docs/parcel-vision.md)。旧文档 5090 地址、单头相机和历史验收不能替代本次证据。

---

## English

### 1. Provenance and handover scope

This branch preserves **deployment source collected from Thor on 2026-10-01**. Repository: `http://61.157.75.168:8010/luoxl/nexus-deploy-platform`; branch: `DF-01-02`; main comparison baseline: `d480a839`. The live project has no `.git`. Its version is established through provenance, timestamp, and SHA-256 in the [source manifest](docs/deployment/thor-20261001/source-manifest.json); system, service, and environment records are in [deployment evidence](docs/deployment/thor-20261001/).

The separate [migration checklist](docs/5090_TO_THOR_MIGRATION.md) records 5090-to-Thor address changes, files to inspect together, and steps for later updates.

The [robot execution code guide](docs/THOR_ROBOT_EXECUTION_CODE_GUIDE.md) follows predictions through physical dispatch and lists **67 files**, with their roles and Windows/Thor/robot locations. It explains rates, interpolation, smoothing, limits, collision safety, effectors, pause and reset. A [hardware/safety function reference](docs/execution/ROBOT_HARDWARE_FUNCTION_REFERENCE.md) covers245symbols across nine modules.

The earlier annotation pass on 2026-10-08 added **3,061 bilingual comment lines** to **63 Python files**, Chinese before English. That pass preserved executable statements, docstrings and original line order; per-file AST and non-comment tokens were identical, as recorded in [verification evidence](docs/execution/annotation-verification.json). Four source-hash-pinned vision core files retain their bytes and are explained through wrappers/documentation. That pass updated repository commentary only, with no Thor/robot deployment or motion; the scheduler handback fix below is a separate subsequent behavior change.

Path abbreviations denote Linux absolute paths:

```text
ROOT = /home/anu/deploy/nexus_deploy_5090_copy_20260928
ROBOT_ROOT = /home/agi/deploy/nexus_deploy
```

`ROOT` is the verified control-process working directory; `ROBOT_ROOT` is configured. SSH to G2 `.101` and `.102` timed out, so no current robot snapshot was collected and equality with Thor's client source is not established. Collection covers control, vision, VLA, client source/assets, and LeRobot. Weights, environments, runtime data, and credentials are excluded from Git. Publication started no robot actions.

Thor hosts UI, inference, vision, and display-only 3-D; G2 owns cameras, action queues, safety, and GDK. Operator and scheduler requests share the backend. One task is admitted at a time; busy requests are rejected, not queued. Console: `/` or `/monitor`; models: `/index.html`; runtime settings: `/session.html?id=<deployment-id>`; history: `/history.html`. Closing a browser does not end a task.

### 2. Device IPs, public/private, wired/wireless

“Observed” means this 2026-10-01 collection. “Configured” does not prove connectivity. Historical addresses are not current DHCP evidence.

| Device / network | Address, interface, state | Evidence / purpose |
|---|---|---|
| Thor private wired | `10.42.1.77/24`, `enP2p1s0`, UP | Observed; robot link, inference advertisement, telemetry |
| Thor private Wi-Fi | `192.168.110.14/24`, `wlP1p1s0`, UP, DHCP | Observed; SSH user `anu`, current UI |
| Wi-Fi gateway | `192.168.110.1` | Observed default route, `wlP1p1s0`, metric 600 |
| Thor USB bridge | `192.168.55.1/24`, `l4tbr0`, DOWN | Observed; other default via `192.168.55.100` is linkdown, not evidence of working egress |
| Thor Docker bridge | `172.17.0.1/16`, `docker0`, DOWN | Observed local virtual network |
| G2A wired management | `10.42.1.102:22`; historical `enp1s0`, `/24` | Configured, SSH timeout; user `agi`, expected AID `G2A0004C201260` |
| Same G2A vendor alias | Historical `10.42.1.101/24` | SSH also timed out; not a second robot |
| G2A Wi-Fi | Historical `192.168.110.40`; earlier `.237`, `.30` | `.40` recorded 09-15 and SSH timed out when probed from Thor now; `.237/wlan0` recorded 09-10; current interface/identity unverified |
| 5090 jump host `anu-server`, wired | `10.42.1.200/24` and `192.168.110.140/24`, both on `eno1`, UP | Both wired addresses observed now; older descriptions of `.140` as Wi-Fi do not describe the current configuration |
| 5090 jump host, Wi-Fi | `192.168.110.60/24`, `wlp68s0`, UP | Observed wireless address; not `.140` |
| 5090 jump host default routes | Both via `192.168.110.1`: `eno1` metric 100, `wlp68s0` metric 600 | Observed; wired default has priority |
| Public Git endpoint | `61.157.75.168:8010` | User-specified and reachable; does not establish a public IP on Thor |
| 5090 public SSH jump | `61.157.75.168:2281` | Login and hostname `anu-server` verified now; not required for Thor runtime |
| Windows maintenance laptop Wi-Fi | `192.168.27.34/24`, WLAN; gateway `192.168.27.250` | Observed with ipconfig; current remote-maintenance network, physical Ethernet disconnected |
| Maintenance laptop virtual adapter | `26.26.26.1/29`, LetsTAP, no default gateway | Observed virtual adapter, not the robot's site network |
| Maintenance laptop onsite wired plan | Historical suggested `10.42.1.78/24` | Planning suggestion for onsite cabling, not currently used; public egress unverified |
| Scheduler / dedicated public device IPs | Unverified | No NAT/public interface inferred from the Git address |

Thor has `alternate_endpoints=[]`, with no robot Wi-Fi fallback. Historical robot networking preserves vendor `.101` and adds `.102`. Verify `scripts/robot-network/` during migration; publication should not incidentally alter networking.

| Service | Endpoint and collection state |
|---|---|
| Control UI/API | **`http://192.168.110.14:21865/`, observed listening** |
| Telemetry | `tcp://10.42.1.77:8768`, observed listening |
| Viewer / SSH | `0.0.0.0:8766/8767`, `:22` observed; 8767 also on IPv6 |
| PI0.5 | Configured `ws://10.42.1.77:5556`, binds `0.0.0.0`; absent at collection |
| Presence ingress | Configured `tcp://10.42.1.77:8776`; absent at collection, created for tasks |
| Presence callback | Configured `http://192.168.110.14:21865`, matches control endpoint |
| Alignment | Historical Thor `:8775/alignment`, old service inactive/no listener; G2 template `.102:8080` unverified |

**Configuration discrepancy:** `configs/thor.toml [host]` still says `.77:8765`; actual service/process overrides it with `.14:21865`. Running `nexus_deploy.site --site thor` directly is not an equivalent restart. Any later consolidation must update service, `[host]`, presence callback, browser, and scheduler together, then validate.

Open the current URL directly on the same Wi-Fi. Windows tunnel example; keep it open and browse `http://127.0.0.1:18765/`:

```powershell
ssh -NT -o ExitOnForwardFailure=yes -L 18765:192.168.110.14:21865 -L 18766:10.42.1.77:8766 -L 18767:10.42.1.77:8767 anu@192.168.110.14
```

From outside the LAN, reach Thor through the verified public jump host. Replace `JUMP_USER` with your authorized jump-host account; SSH authenticates the jump host and Thor separately:

```powershell
ssh -NT -o ExitOnForwardFailure=yes -J JUMP_USER@61.157.75.168:2281 -L 18765:192.168.110.14:21865 -L 18766:10.42.1.77:8766 -L 18767:10.42.1.77:8767 anu@192.168.110.14
```

After DHCP changes, confirm with `ip -br -4 address`, then update bindings, callback, and access addresses.

### 3. Module files and deployed locations

Relative paths below are rooted at `ROOT`; expansion gives the complete Thor path. Configuration `../` resolves from `ROOT/configs/`, while model arguments `models/...` resolve from model working directory `ROOT`.

| Module | Repository path / absolute deployment path |
|---|---|
| Configuration / launcher | `configs/thor.toml`; `src/nexus_deploy/site.py` |
| Control / deployment / admission | `src/nexus_deploy/control_api.py`, `src/nexus_deploy/deployment.py`; `tasks.py`, `vision.py`, `robot.py`, `defaults.py` under `src/nexus_deploy/control/` |
| Scheduler / presence policy | `src/nexus_deploy/sorting_api.py`, `src/nexus_deploy/presence_gate.py` |
| Browser / 3-D | `src/nexus_deploy/static/`, `src/nexus_deploy/viewer/` |
| Alignment | `src/nexus_deploy/alignment/`; site `alignment.toml`, template `alignment.example.toml` |
| PI0.5 | `src/nexus_deploy/vla/pi05/`; entry `nexus_deploy.vla.pi05.server` |
| Live LeRobot / RTC | `pi05/` and `rtc/` under `ROOT/third_party/lerobot/source/src/lerobot/policies/` |
| Archived LeRobot / rebuild | `snapshots/lerobot/`; `scripts/prepare_pi05_lerobot.py`, `third_party/lerobot/pi05-deployment.patch`; snapshots are not automatic imports |
| PI weights / tokenizer | `models/pi05/dahuan/<experiment>/checkpoint-<step>/`; `models/tokenizers/paligemma-3b-pt-224/` |
| Vision entry / runtime | `src/nexus_deploy/vision/parcel/run_nexus_presence_sidecar.py`; sibling `runtime/nexus_frame_stream.py`, `estimator.py`, `config.py`, `service.py` |
| Vision core / ROI / weights | `src/nexus_deploy/vision/parcel/rules_core/`, `src/nexus_deploy/vision/parcel/configs/g2_head.yaml`; `models/vision/parcel/yolo26m-seg-final.pt` |
| Robot source / manifest | `src/nexus_deploy/robot/g2a/`; `robot_manifest.txt` |
| Key client files | That directory's `client.py`, `action_plan.py`, `gdk_adapter.py`, `async_safety.py`, `safety_layer.py`, `constants.py`, `assets/`, safety JSON |
| Images / telemetry | Same directory: `presence_stream.py`, `telemetry.py`, `client.py` |
| Bundle / remote target | `dist/thor/g2a_robot_client/`; `/home/agi/deploy/nexus_deploy/robot_client/`, reset `reset_all_joints.sh`; actual remote unverified |
| State / archives / scheduler config | `runtime/thor/`, `runtime/thor/robot-sessions/`, `runtime/thor/sorting.json`; remote archive default `/home/agi/.cache/nexus-platform-sessions` |
| Control/vision Python / PI Python | `ROOT/.venv/bin/python` / `ROOT/.pi05-venv/bin/python` |
| Service / credential source | `/home/anu/.config/systemd/user/nexus-thor-control.service`; `/home/anu/.config/nexus-thor-control.env` (path only, contents excluded) |

Selected safety file: `src/nexus_deploy/robot/g2a/G2_dahuan_standard_table_20260922_contact_refined_safety_execute_v2.json`, SHA-256 `1596634e164161fa0aebe583efc6898fec622154953d18415a8b8203a9eccb94`. Recheck final task `--safety-config`, AID, URDF, and calibration provenance; do not reuse it blindly at another workcell.

The file explicitly records `measurement_mode=contact-refined-temporary`, `final_camera_extrinsic_calibration_pending=true`, `replacement_apriltag_pending=true`, and `provisional` collision geometry. Migration must include final camera extrinsics, AprilTag, table boundaries after relocation, and physical geometry acceptance. An execute filename does not establish their completion.

Old `/home/anu/test_20260916_senbao/...`, standalone vision, and `/home/agi/deploy/g2a_robot_client*` are historical paths. The G2 alignment service template still references `/home/agi/deploy/g2a_robot_client`, differing from the new remote_directory; inspect/migrate it separately after connectivity returns.

### 4. Services, environments, and current configuration

`nexus-thor-control.service` was active/running, working directory `ROOT`, using `.venv/bin/python -m nexus_deploy.control_api`, `configs/thor.toml`, and `runtime/thor/sorting.json`, enabling actions/execute/reset. Old `g2a-control`, `g2a-alignment`, and `g2a-pi05` were inactive.

**Effective launch configuration is now explained:** drop-in `10-unauthenticated.conf` overrides the base unit with `--password-env NEXUS_UI_PASSWORD_DISABLED_20260929 --allow-unauthenticated-private-lan`. `20-sorting-token.conf` only loads `EnvironmentFile=/home/anu/.config/nexus-thor-sorting-5090.env`. Both are preserved in `ops/systemd/nexus-thor-control.service.d/`; combined configuration is in evidence `effective-control-unit.txt`. These overrides explain the base-unit/running-argv difference. Private-LAN UI admission and scheduler-token authentication are separate channels; credentials are excluded. Inspect `systemctl --user cat nexus-thor-control` / `show ... -p ExecStart`, not just the base unit.

| Item | Observed 2026-10-01 |
|---|---|
| OS / architecture / kernel / L4T | Ubuntu 24.04.4 LTS / aarch64 / `6.8.12-1021-tegra` / R39.2.0; no JetPack product version inferred |
| Both Python environments | 3.12.3; `include-system-site-packages=false` |
| Both imported Torch / CUDA / GPU | `2.11.0+cu130` / `13.0` / `NVIDIA Thor`; CUDA available |
| `.venv` package metadata | Torch `2.11.0`, Torchvision `0.26.0`, Ultralytics `8.4.146`, NumPy `2.2.6`, OpenCV `4.11.0.86`, SciPy `1.18.0` |
| `.pi05-venv` metadata | Torch `2.11.0+cu130`, Torchvision `0.26.0+cu130`, Transformers `5.5.4`, PEFT `0.20.0`, LeRobot `0.6.1`, NumPy `2.2.6` |

The `.venv` distribution metadata differs from imported Torch. Check actual import locations, wheel provenance, and ABI when rebuilding. Full details are in evidence `*-environment.json` and `*-pyvenv.cfg`.

Site TOML defaults are **read-only, RTC, execution 30 FPS**; model 30 FPS, horizon 30, CUDA 0, port 5556, 600-second startup timeout. Arm velocity 3.76, acceleration 8, `urdf-velocity-uniform-acceleration`, savgol; both low-pass filters disabled, workspace safety enabled, stationary-hold feedback limit **100 ms**.

At capture on 2026-10-01, the service-loaded `runtime/thor/sorting.json` profile `parcel-sort` was **execute, RTC, execution 15 FPS, servo 100 Hz**, with presence **enabled=true, automatic, pause 3 seconds, reset 6 seconds, auto_resume=true**. It reused the default PI0.5 version and had no callback_url. The historical [sorting.config.redacted.json](docs/deployment/thor-20261001/sorting.config.redacted.json) retains those values. This is a separate layer from TOML/source defaults; record both. Configuration existence does not establish robot motion or new safety certification.

**Scheduler handback fix on 2026-10-08:** the active scheme and `parcel-sort` use `server_mode=reuse` and `G2_dahuan_standard_table_20260922_contact_refined_safety_execute_v2.json`. At the operator's request, continuous no-parcel detection pauses after **3 seconds** and requests automatic reset after **5 seconds**. Five seconds is the no-parcel delay, not the duration of physical reset motion. Defaults in `control/defaults.py` and `PresenceTaskGateConfig` match Thor's scheduler configuration. The workstation remains in manual mode with admission paused; this update starts no robot motion.

Profile views from `/api/workstation` and `/api/profiles` now expose `reason` and `config_mismatches`, including field names and deployment/profile values. The page and handback-button tooltip display these diagnostics. The “configuration-matched templates” indicator uses template and saved deployment state; it does not confirm that the model has passed the live health check for this handback. Handback still checks live model health, and `NOT_READY` now distinguishes an unconfirmed running model from a failed live health check. Exact configuration matching, robot idle/reset barriers, health checks, and the 30-second robot-probe requirement remain. The control service retains systemd's default `KillMode=control-group`: restarting it stops PI0.5 model processes in the same cgroup. This update does not change that cleanup policy. After the model stops, reload the real model through the normal model-preparation, save, and apply flow and confirm its health before using a `reuse` configuration; `reuse` does not load a missing model service. Model service strategy is a preparation-stage parameter: moving from `real` to `reuse` requires preparing/selecting a reuse candidate, then saving and applying it; it cannot be changed as a runtime parameter of the current scheme. Toggling admission alone does not synchronize fixed scheduler profiles.

Read-only checks: `systemctl --user status nexus-thor-control`, `ip -br -4 address`, `ss -ltn`. Restart affects managed models, tasks, and CSRF; do it only after task termination and client exit. Publication requires no robot start/restart.

### 5. Replacing models next time

**Add a PI0.5 version.** Default ID `large_qr_clean_v2_seed1000_007500` maps to `ROOT/models/pi05/dahuan/large_qr_clean_v2_seed1000/checkpoint-007500`. Live asset inventory confirms only this checkpoint is available among eight configured versions. The other seven (`large_qr_clean_v2_seed1000_005000`, `per_package_001500/004500`, `per_sample_20260820_001500/007500`, `per_sample_20260824_005000/007500`) are not deployed; a UI choice does not make them usable.

Its `model.safetensors` is **9,354,050,752 bytes**, SHA-256 **`a15a67f4a45f6b7c5359c8b48b16fb4bf8894cb6e7ff5ea1777a3ffb9d2ddd74`**. The local tokenizer is present. Full files are recorded in [model-assets.json](docs/deployment/thor-20261001/model-assets.json). Assets were inspected without reloading the model.

1. Record old version, asset hashes, source/LeRobot, tokenizer, normalization, and safety configuration. End the active task and close the model.
2. Place complete new assets under `models/pi05/dahuan/<new-experiment>/checkpoint-<step>/`, retaining old directories; avoid mutable `current` symlinks.
3. Add a version table to `configs/thor.toml`, replacing sample ID/path below. Change `default_version` in the existing variant table, or select the new UI version for acceptance first. IDs match `[a-z0-9][a-z0-9_.-]{0,79}`.

```toml
[models.pi05.effector_variants.dahuan.versions.experiment_20261001_010000]
display_name = "Experiment 2026-10-01 · Step 10000"
normalization_source = "checkpoint_embedded"
normalization_label = "Experiment 2026-10-01 · embedded Quantiles"
real_args = ["--checkpoint=models/pi05/dahuan/experiment_20261001/checkpoint-010000"]
```

4. `common_real_args` is prepended; version `real_args` replaces inherited arguments rather than appending. Retain `--tokenizer-path=models/tokenizers/paligemma-3b-pt-224`. `external_training_asset` requires `--robot-norm-path=...`; do not hard-code `--effector-profile` in base_args.
5. Validate camera keys/order/size/RGB, 16D state/action mapping, absolute/relative semantics, Dahuan metres and 0–0.025 range with inverted action mapping, FPS/chunk/RTC. The profile ID is `dahuan`, not historical `dh`.
6. Inspect commands below, then reload configuration through installed service management, refresh the UI, load/warm up, and perform offline/read-only before physical acceptance. `--check` validates configuration/paths, not model health, and does not reproduce service port overrides.

```bash
.venv/bin/python -m nexus_deploy.site --site thor --check
.venv/bin/python -m nexus_deploy.deployment --config configs/thor.toml server command pi05
.pi05-venv/bin/python -m nexus_deploy.vla.pi05.server --help
```

Rollback: end task/model, restore old ID, configuration, and matching source/environment, then reload/validate. Compatible weight-only changes normally need no client redeploy. Another policy family needs `[models.<id>]` module, Python, repo, port/GPU, bounds, modes, and robot compatibility; protocol changes update server/client/contracts/manifest/tests together. `ui_models` is display inventory only.

**YOLO needs more than a new path.** `presence_runtime.weights` chooses the file; parcel's `runtime/config.py` pins its hash. `rules_core/model_adapter.py` requires classes `{0: package, 1: left_gripper, 2: right_gripper}` and segmentation masks; `runtime/core_bridge.py` pins core code. Measured weight SHA-256 matches the expected value: `6ca0492ba45211e14e8f5d4f2eeac6bc16c29476f5f0646b4503b4b5c0d40c2c`. Accept labeled videos before updating version/hash/contracts; core changes require reviewed core hashes. Do not bypass verification or replace masks with boxes. Test all cameras, occlusion, missing/gapped frames becoming UNKNOWN, and concurrent GPU latency. Recalibrate ROI after mount/head-pose/crop/FOV changes.

### 6. Thor checklist for future Nexus updates

Python module paths without a full prefix in this section are relative to `ROOT/src/nexus_deploy/`; script and TOML paths remain relative to `ROOT`.

| Item | Files and checks |
|---|---|
| Site / network | `configs/thor.toml`, service: aarch64, ROOT, `.14:21865` override, `.77` inference/telemetry, callback; reconcile `[host]` discrepancy |
| Python / ARM64 | Preserve lexical venv paths in `deployment.py::_interpreter` and `site.py`, not `.resolve()` to base Python; rebuild both environments on Thor |
| Model / gripper | Version tables, `dahuan`, tokenizer/Quantiles, 16D/RTC, complete assets |
| Three-camera transport | `robot/g2a/presence_stream.py`, `vision/parcel/runtime/nexus_frame_stream.py`, `deployment.py`: authentication/generation/sequence, head-only compatibility; presence sessions use `NEXUS_PRESENCE_FRAME_MODE=three` |
| Fusion | Sidecar/control ingest: three actual samples per head/left wrist/right wrist; all empty means empty, any confirmed present means present, otherwise UNKNOWN; errors never become empty |
| Timing / ROI | Gaps above 0.75 seconds clear evidence; head ROIs, whole-image wrists, head aspect 1.60±3%; runtime three samples override old YAML four-sample default |
| Pause / completion | `control/defaults.py`, `tasks.py`, `presence_gate.py`, `control_api.py`: current pause/reset-request delays are 3/5 seconds; verify defaults, service memory, and fixed profile; retain ownership, reset, and fresh-plan barriers |
| Operator completion | Executed scheduler task `operator_end` is business completion; `conveyor_delivery_independently_verified=false`, reset pending; cancellation stays cancelled, failed reset blocks the next physical task |
| Safety / client | Reconcile adapter, safety, queue, URDF/meshes, safety JSON, manifest together; do not weaken invalidation to hide latency; migrate old alignment service paths separately |
| Upstream / publication | Fixed LeRobot commit+actual patch+snapshot; per-file provenance/hashes; separate offline, GPU, read-only, physical results |

Collected canonical `robot/g2a/gdk_adapter.py` and `safety_layer.py` match baseline main. Shared-FK candidates in 09-30 safety staging are not thereby deployed canonical changes; do not overlay them blindly. Three-camera and operator_end changes both touch `control_api.py`; preserve both.

Build entry points `scripts/build_runtime_env.sh` and `scripts/build_pi05_env.sh` pin Torch 2.11.0/Torchvision 0.26.0; runtime uses `uv sync --locked`. PI `NEXUS_PI05_CONSTRAINTS` cannot override conflicting exact versions. Retain CUDA13 ARM64 wheel provenance/hashes; review build inputs when changing versions, and do not copy 5090 environments. `NEXUS_PYTHON312` selects Python; `NEXUS_LEROBOT_MIRROR` can select a mirror.

Observed LeRobot HEAD is `a9879e69edefb3c308f7025046d3d12b3cb2fe50`, with four modified files: factory and PI05 configuration/modeling/processor. Actual patch/source are in evidence and snapshots. `snapshots/lerobot/` contains source without upstream `.git`; standard `prepare_pi05_lerobot.py` still clones/checks out/applies/verifies a Git-backed `third_party/lerobot/source/`. Do not overwrite that runtime checkout with an unverified snapshot; snapshots are not automatically imported.

```bash
./scripts/build_runtime_env.sh
./scripts/build_pi05_env.sh
.venv/bin/python scripts/prepare_pi05_lerobot.py --verify
.venv/bin/python -c 'import sys,torch,nexus_deploy; print(sys.executable,sys.prefix,sys.base_prefix); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(nexus_deploy.__file__)'
.pi05-venv/bin/python -c 'import sys,torch,lerobot; print(sys.executable,sys.prefix,sys.base_prefix); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(lerobot.__file__)'
```

These are maintenance instructions; live environments were not rebuilt. Verify isolated `pyvenv.cfg` and run `uv pip check --python <interpreter>` for both. `readlink -f` is diagnostic; retain venv paths in configuration. CUDA availability does not establish all operators or physical acceptance.

### 7. Deployment, collection, and validation boundaries

Update the client only after connectivity returns, AID matches, and task/GDK are idle. These commands were not performed here:

```bash
.venv/bin/python -m nexus_deploy.deployment --config configs/thor.toml client bundle --robot g2a
.venv/bin/python -m nexus_deploy.deployment --config configs/thor.toml client deploy --robot g2a
```

Check manifest, remote hashes, working/import origins, alignment agent, and safety profile. `alignment.example.toml` has `target_configured=false`; example poses are not field measurements.

Collection scope comes from actual process/interpreter/import paths, configuration, editable packages, upstream dependencies, and robot source/assets. Record host/source/time/destination/SHA-256 per file and mappings into `snapshots/lerobot`. Nested `vision/parcel/runtime/` is source; do not blanket-exclude every runtime directory. Exclude weights/tokenizers, venvs, logs, data, videos, frames, caches, `.env`, keys, and tokens. Hand assets over separately with version/path/hash/size/acquisition procedure. A clone still requires models, environments, and restricted site configuration.

The publication manifest `published-source-manifest.json` records **1,570 files**, including **955 upstream LeRobot files**. Twenty-seven documentation symlinks were materialized with `source_symlink` provenance retained. The complete `vision/parcel/runtime/` source is included. Original collected sources remain in pre-annotation baseline commit `2409e97cb2745974128989eb9311113eca5fd44d`; manifest hashes remain historical evidence. Comments added on 2026-10-08 change source bytes/line numbers; annotated hashes and text-level execution equivalence are recorded separately. Windows Git line-ending conversion also affects byte comparison.

Completed on Thor's actual `.venv`: presence ingest **4 passed**; ten other offline test modules **364 passed**; site `--check` and pinned LeRobot `--verify` **passed**. Tests ran in separate processes to avoid persistent test-stub interference; they did not start inference or robot execution. Results are in `pytest-presence-ingest.txt`, `pytest-offline.txt`, `site-check.txt`, and `lerobot-verify.txt`. This is not a full-repository or physical acceptance claim.

The robot remains unreachable; no running robot snapshot, motion, or PI/YOLO reload was performed. The source snapshot has been published to `DF-01-02`; after each ordinary push, check the remote commit and README without force-pushing. Model loading, real read-only integration, physical acceptance, and final calibration remain separate field checks.

References: [Code](docs/code-structure.md), [Environment](docs/environment-migration.md), [Models](models/README.md), [Scheduler](docs/scheduler-api.md), [Vision](docs/parcel-vision.md). Historical 5090 addresses, head-only descriptions, and earlier acceptance do not replace this evidence.
