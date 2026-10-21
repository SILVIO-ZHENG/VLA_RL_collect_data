# G2 头部相机包裹阶段实时封装 / G2 Head-Camera Parcel-Stage Runtime

[中文说明](#中文说明) | [English guide](#english-guide)

## 中文说明

### 1. 这套程序做什么

它把 G2 智元机器人头部相机画面转换成一个稳定的“包裹阶段值”：

```text
cam_head RGB 图像 → 10 Hz 取最新帧 → RGB 转 BGR
→ YOLO26m-seg 实例分割（batch=1）→ 包裹/夹爪/ROI 几何判断
→ 历史帧确认 → StageResult 或单行 JSON
```

本目录只负责部署：不训练、不修改模型，也不发送机器人动作。独立入口只会访问 GDK Camera。

默认且唯一认可的权重是：

```text
models/vision/parcel/yolo26m-seg-final.pt
SHA256: 6ca0492ba45211e14e8f5d4f2eeac6bc16c29476f5f0646b4503b4b5c0d40c2c
```

`--weights` 可以指定权重的新位置，但程序仍会校验上述哈希和类别映射。

### 2. 仓库目录

```text
项目根目录/
├── src/nexus_deploy/vision/parcel/          # 你从这里运行程序
├── src/nexus_deploy/vision/parcel/rules_core/                 # 已验证并锁定哈希的阶段判断核心
│   ├── dataset_io.py
│   ├── model_adapter.py
│   ├── state_engine.py
│   ├── temporal_filter.py
│   └── configs/test_rois.yaml
└── models/vision/parcel/
    └── yolo26m-seg-final.pt   # 指定的 best52/final 权重
```

运行时和规则核心随 `nexus_deploy` 包一起安装。确需替换规则核心位置时，可设置
`PARCEL_STATE_CORE_DIR`；权重位置通过配置或 `PARCEL_STATE_WEIGHTS` 指定。

纯小白建议按顺序看：

1. `README.md`：安装、运行、输出含义。
2. `run_g2_head.py`：直接连接 G2 相机的命令行入口。
3. `example_existing_g2_loop.py`：接入已有机器人循环的最小例子。
4. `src/nexus_deploy/vision/parcel/runtime/estimator.py`：一帧图像怎样变成阶段结果。
5. `src/nexus_deploy/vision/parcel/runtime/service.py`：怎样限制到 10 Hz、只取最新帧。
6. `CORE_ALGORITHM_GUIDE.md`：核心算法的中英文逐层说明。

### 3. 环境与自检

- Python 至少 3.10；已在 Python 3.10.21 验证。
- GPU 推理需要可用的 PyTorch + CUDA；服务器已有正常环境时优先复用。
- `agibot_gdk` 由 G2 机器人环境提供，不通过本 requirements 安装。
- 不要为了部署盲目重装 PyTorch、NumPy 或 OpenCV，以免破坏 CUDA/ABI 兼容。

先检查已有环境：

```bash
python -m pip check
python -c 'import torch,cv2,numpy,yaml,ultralytics; print(torch.__version__, torch.cuda.is_available())'
```

缺少普通依赖时再执行：

```bash
cd /path/to/nexus_deploy
./scripts/build_runtime_env.sh
```

完整部署自检：

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.verify_install \
  --weights models/vision/parcel/yolo26m-seg-final.pt \
  --device 0 --require-cuda --require-gdk --load-model
```

看到 `"status": "ok"` 和 `"model_loaded_and_warmed": true` 才表示指定 GPU、GDK、
核心代码、权重及模型预热全部通过。

### 4. 方式一：程序独占 GDK，直接读取头部相机

```bash
cd /path/to/nexus_deploy
.venv/bin/python -m nexus_deploy.vision.parcel.run_g2_head \
  --weights models/vision/parcel/yolo26m-seg-final.pt \
  --device 0 --detect-hz 10 --camera-fps 30
```

程序以 30 FPS 请求 `cam_head/kHeadColor`，按 10 Hz 处理最新帧，每次输出一行 JSON。
`--camera-fps` 只是生成帧编号时间尺度的名义值，不负责修改 GDK 相机硬件帧率。

该入口拥有全局 `gdk_init()`/`gdk_release()` 生命周期。已有另一个 GDK/G2A 程序运行时，
不要同时启动它；请改用下一节的嵌入方式。Ctrl+C 会正常释放资源。连续取帧超时或解码失败时
程序会退出，应由 systemd、Docker 或你的上层进程负责重启。

30 秒冒烟测试：

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.run_g2_head --device 0 --max-seconds 30
```

没有机器人相机时可检查软件链路（仍会加载真实模型）：

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.run_g2_head --device 0 --dry-run --max-results 5
```

### 5. 方式二：接入已有 G2A 控制循环（推荐）

现有程序已经持有 `G2A` 时，绝对不要再次初始化 GDK。首先保证头部相机输出
`640×400 RGB uint8`。G2A 常见默认值 `320×240` 与本模型 ROI 不兼容：

```python
from g2a import G2AConfig

cfg = G2AConfig(
    camera_width=640,
    camera_height=400,
    camera_fps=30,
    # read_only/configure_on_connect 按你的控制程序安全策略设置。
)
```

然后在程序启动时只创建一次 estimator 和 service，并在整个循环中复用：

```python
from nexus_deploy.vision.parcel.runtime import ParcelStateEstimator, RateLimitedParcelStateService

estimator = ParcelStateEstimator(device=0)
estimator.warmup()
service = RateLimitedParcelStateService(estimator, detect_hz=10.0)

# 放进已有 while 循环中：
capture = robot.capture_camera_frames()
result = service.process_if_due(
    capture.frames["cam_head"],
    color_space="RGB",
    source_id="cam_head",
    capture_timestamp_ns=capture.frame_wall_time_ns["cam_head"],
)
if result is not None:               # 未到下一个 100 ms 时返回 None，属于正常情况。
    print(result.state)               # 例如 "IN_LEFT_GRIPPER"
    print(result.state_id)            # 例如 2
    print(result.to_json())           # 完整结构化 JSON
```

相机重连、新任务开始或视角/FOV 改变后调用 `service.reset()`，清除旧历史帧状态。

### 6. 直接传入单帧图像

```python
from nexus_deploy.vision.parcel.runtime import ParcelStateEstimator

estimator = ParcelStateEstimator(device=0)  # 只创建一次
estimator.warmup()
result = estimator.process_frame(
    image_rgb,                         # numpy.ndarray, uint8, H×W×3
    color_space="RGB",
    source_id="cam_head",
    frame_index=123,
    capture_timestamp_ns=time.time_ns(),
)
print(result.state, result.state_id)
```

`capture_timestamp_ns` 应与 `time.time_ns()` 属于同一墙钟时间域，否则
`end_to_end_ms` 没有意义。下游使用历史确认后的 `state`，不要用诊断字段 `raw_state`。
不要每帧重新创建 estimator，否则会丢失历史并反复加载模型。

### 7. 用视频测试 API

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.run_video \
  --video /path/to/file-001.mp4 \
  --weights models/vision/parcel/yolo26m-seg-final.pt \
  --device 0 --detect-hz 10 \
  --output-jsonl runs/file-001-states.jsonl
```

视频入口离线尽快处理，不按真实时间等待；OpenCV 帧按 BGR 输入。它会保存每个采样帧的
JSONL；若要人工复核，可额外传入 `--output-video runs/file-001-review.mp4`，生成带有旧版
实例分割掩膜、包裹/左右夹爪检测框、`bin`/`table_left` ROI 轮廓、阶段和四帧 Presence
信息的 MP4，且不会额外执行 YOLO 推理。复核视频只写入实际 10 Hz 推理样本，因此框和掩膜
永远对应当前输出帧，而不是复用到未推理的相邻帧。若输出文件已存在会拒绝覆盖；确认后加
`--overwrite`。每行额外包含 `video_timestamp_seconds`，方便定位原视频时间。

### 8. 输出阶段值

| state_id | state | 中文含义 | English meaning |
|---:|---|---|---|
| 0 | `NO_PACKAGE` | 未检测到活动包裹 | No active package detected |
| 1 | `ON_TABLE_LEFT` | 包裹在配置的左侧台面 ROI | Package in configured left-table ROI |
| 2 | `IN_LEFT_GRIPPER` | 包裹在左夹爪 | Package held by left gripper |
| 3 | `IN_RIGHT_GRIPPER` | 包裹在右夹爪 | Package held by right gripper |
| 4 | `IN_BIN` | 已跟踪包裹进入料箱 ROI | Tracked package entered bin ROI |
| 5 | `UNKNOWN` | 暖机、证据不足或歧义 | Warm-up, insufficient evidence, or ambiguity |

当前协议只选一个全局活动包裹；双臂同时操作不同包裹时可能输出 `UNKNOWN`。

主要 JSON 字段：

| 字段 | 含义 |
|---|---|
| `state`, `state_id` | 历史帧确认后的阶段名称和数字值 |
| `confidence` | 几何规则证据分数，不是校准后的概率 |
| `changed` | 稳定阶段是否在本次结果中改变 |
| `actionable` | 仅表示 `state != UNKNOWN`，**不表示允许机器人执行动作** |
| `status` | 当前处理状态，正常为 `ok` |
| `raw_state`, `raw_confidence` | 当前单帧原始判断，仅供诊断 |
| `candidate_state`, `candidate_frames` | 尚在等待连续帧确认的候选阶段及帧数 |
| `reason`, `raw_reason` | 稳定结果和单帧结果的解释 |
| `active_package_index`, `active_centroid` | 当前活动包裹检测索引和中心点 |
| `inference_ms` | 单次模型与规则处理耗时 |
| `end_to_end_ms` | 从同一墙钟域采集时间到输出的耗时 |
| `model_sha256` | 实际加载权重的哈希 |

`actionable=true` 不能替代碰撞检测、安全互锁或机器人动作许可。

### 9. 频率、延迟和图像约束

- 相机目标输入：30 FPS；检测和阶段输出目标：10 Hz；实时 batch=1。
- 夹爪状态连续 3 个采样帧确认，名义约 0.3 秒。
- 台面/料箱状态连续 4 帧确认，名义约 0.4 秒。
- 无包裹连续 5 帧确认，名义约 0.5 秒。
- 这些是采样计数，不是硬实时保证。单次推理超过 100 ms 时，服务跳过错过的时隙而不排队。
- 输入必须是 `numpy.ndarray`、`uint8`、H×W×3；G2 用 `RGB`，OpenCV 用 `BGR`。
- 画幅必须接近 640:400（1.60，容差约 3%）。程序不做隐式裁剪；比例正确时归一到
  640×400，随后由 Ultralytics 内部执行 `imgsz=640` letterbox。
- 相机支架、头部姿态、裁剪或 FOV 改变后，必须重新标定 `configs/g2_head.yaml` 中的 ROI。

### 10. 常见问题

- `python: command not found`：激活原虚拟环境，或使用环境内 Python 的绝对路径。
- `weight hash mismatch`：不是指定权重，重新复制 `yolo26m-seg-final.pt`。
- `core incomplete/unverified`：四个核心 `.py` 文件缺失或被修改；重新解压完整压缩包。
- `agibot_gdk import failed`：需要在 G2 官方运行环境执行。
- `selected CUDA device unavailable`：检查 `nvidia-smi`、PyTorch CUDA 版本和 `--device`。
- CUDA OOM：确认没有其他任务占卡；本服务实时 batch 固定为 1。
- `no image`/超时：检查 `cam_head` 与 `kHeadColor`、GDK 权限和设备连接。
- `aspect ratio mismatch`：把 G2A 相机设为 640×400，不要使用默认 320×240。
- 长时间 `UNKNOWN`：检查目标检测、ROI/FOV、双包裹/双夹爪歧义和光照。
- 输出低于 10 Hz：查看 `inference_ms`；模型速度不足时不会积压旧帧。

### 11. 测试与命令帮助

```bash
python -m unittest discover -s tests -v
.venv/bin/python -m nexus_deploy.vision.parcel.run_g2_head --help
.venv/bin/python -m nexus_deploy.vision.parcel.run_video --help
.venv/bin/python -m nexus_deploy.vision.parcel.verify_install --help
```

## English guide

### Purpose and fixed model

This deployment-only package turns G2 head-camera frames into a stable parcel-stage value:

```text
cam_head RGB → latest frame at 10 Hz → RGB-to-BGR → YOLO26m-seg (batch=1)
→ package/gripper/ROI geometry → temporal confirmation → StageResult / JSON line
```

It does not train or alter the model and never sends robot motion commands. The standalone
entry point accesses only the GDK Camera. The accepted default weight is
`models/vision/parcel/yolo26m-seg-final.pt`, pinned to SHA256
`6ca0492ba45211e14e8f5d4f2eeac6bc16c29476f5f0646b4503b4b5c0d40c2c`.

Keep the runtime and rules core under `src/nexus_deploy/vision/parcel`, and place the weight
under `models/vision/parcel`. Read the source documentation for an explanation of the
hash-pinned core files.

### Requirements and preflight

Use Python 3.10 or newer (tested with 3.10.21), a CUDA-enabled PyTorch environment for GPU
inference, and the vendor-provided `agibot_gdk` on the robot. Prefer the existing working
environment and install `requirements.txt` only for missing dependencies.

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.verify_install \
  --weights models/vision/parcel/yolo26m-seg-final.pt \
  --device 0 --require-cuda --require-gdk --load-model
```

Proceed when the report contains `"status": "ok"` and
`"model_loaded_and_warmed": true`.

### Standalone GDK mode

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.run_g2_head --device 0 --detect-hz 10 --camera-fps 30
```

This process owns the global GDK lifecycle and reads `cam_head/kHeadColor`. Do not run it
beside another GDK/G2A owner. `--camera-fps` is a nominal frame-index timescale, not a camera
hardware configuration command. Ctrl+C releases resources. Persistent capture/decode errors
terminate the process so a supervisor can restart it.

### Existing G2A loop

Configure G2A for 640×400 RGB; the common 320×240 default is incompatible with the calibrated
ROIs. Create one long-lived `ParcelStateEstimator`, call `warmup()`, wrap it in one
`RateLimitedParcelStateService`, and pass each `cam_head` frame to `process_if_due()` as shown
in the Chinese example and `example_existing_g2_loop.py`. A `None` result simply means the
next 100 ms slot is not due. Call `reset()` after reconnects, new tasks, or FOV changes.

### Video mode

```bash
.venv/bin/python -m nexus_deploy.vision.parcel.run_video --video /path/to/input.mp4 --device 0 \
  --output-jsonl runs/input-states.jsonl
```

Video mode processes sampled frames as fast as possible and writes JSONL. Pass
`--output-video runs/file-001-review.mp4` to create a review MP4 with the original masks,
boxes and ROI outlines plus the four-sample presence line. It writes only actual detector
samples (normally 10 FPS), so the geometry never becomes stale, and it does not run YOLO a
second time. Existing output is protected; use `--overwrite` intentionally. Each row adds
`video_timestamp_seconds`.

### Output, timing, and safety

The stable values are `NO_PACKAGE=0`, `ON_TABLE_LEFT=1`, `IN_LEFT_GRIPPER=2`,
`IN_RIGHT_GRIPPER=3`, `IN_BIN=4`, and `UNKNOWN=5`. Use `state`, not the diagnostic
`raw_state`, downstream. `actionable` only means the state is not `UNKNOWN`; it is never a
robot-motion authorization or safety interlock.

The target is a 30 FPS camera feed with 10 Hz detection/output. Gripper, table/bin, and
no-package transitions require 3, 4, and 5 sampled frames respectively (nominally 0.3, 0.4,
and 0.5 seconds). Slow inference skips missed slots instead of queueing stale frames.

Input must be an H×W×3 `uint8` NumPy array. Use `RGB` for G2 and `BGR` for OpenCV. The aspect
ratio must remain near 640:400; changes to crop, mount, head pose, or FOV require ROI
recalibration. Use capture timestamps from the `time.time_ns()` wall-clock domain if
`end_to_end_ms` must be meaningful.

### Tests and help

```bash
python -m unittest discover -s tests -v
.venv/bin/python -m nexus_deploy.vision.parcel.run_g2_head --help
.venv/bin/python -m nexus_deploy.vision.parcel.run_video --help
.venv/bin/python -m nexus_deploy.vision.parcel.verify_install --help
```
