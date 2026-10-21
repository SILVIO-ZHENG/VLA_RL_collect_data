# G2A PI0.5 真机运行说明

本文档只说明如何启动真实模型 Server 和真机 Client，以及真机执行时需要调整的参数。

## 可用的 model-profile

启动 Server 前先选择模型：

| Model profile | 实验 | Action 表示 |
| --- | --- | --- |
| `24d_absolute` | 24D LoRA | 24D absolute |
| `16d_absolute` | 16D LoRA | 双臂和夹爪 absolute |
| `16d_absolute_full` | 16D full fine-tuning | 双臂和夹爪 absolute |
| `16d_absolute_full_aug` | 16D full fine-tuning + OpenPI 图像增强 | 双臂和夹爪 absolute |
| `16d_delta` | 16D LoRA | 双臂 delta、夹爪 absolute |
| `16d_delta_full` | 16D full fine-tuning | 双臂 delta、夹爪 absolute |

无论选择 absolute 还是 delta，Server 发给 Client 的结果都是 canonical 24D absolute action。
Client 不会再次累加 delta。

下面只保留一条通用 Server 命令和一条通用 Client 命令。示例默认使用：

- `16d_absolute_full` 模型；
- 不开启 RTC；
- Client 使用 async 推理；
- 首次只执行 10 秒；
- 100 Hz 下发前使用 2 Hz 低通。

切换模型或 RTC 时只修改后面的参数，不需要换另一套程序。

## 1. 运行前准备

开始前确认：

- GPU Server 使用 conda 的 `lerobot` 环境。
- 机器人端部署目录为 `/home/agi/deploy/g2a_pi05_robot_client`。
- GPU Server 和机器人处于同一局域网，机器人能够访问 Server 的 5556 端口。
- 三路相机、左右夹爪和 24D state 读取正常。
- 已执行完整 reset，使双臂、双夹爪、头部和腰部回到训练数据初始姿态。
- 急停可用，机器人工作范围内没有人员和障碍物。

16D 模型虽然只输出双臂和夹爪，开始前也必须 reset 头部和腰部。推理过程中头部和腰部保持
reset 后的实测位置。

## 2. 启动 Server

在 GPU Server 上运行以下通用命令：

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_absolute_full \
  --device=cuda:0 \
  --execution-horizon=30 \
  --host=0.0.0.0 \
  --port=5557 \
  --log-file=outputs/g2a_pi05_deploy/logs/server.log
```

Server 默认先使用Nexus标准起始state、三路 `640×400` 图像和任务prompt完成一次本地
warmup，再打开端口。看到下面的日志后，表示 warmup 已完成并开始监听：

```text
G2A PI0.5 server listening on ws://0.0.0.0:5556
```

Server 会自动选择该 profile 下 step 最大且文件完整的 checkpoint。当前两个 full fine-tuning
profile 都会选择 30K checkpoint。

### Server 可变参数

| 参数 | 示例/默认 | 说明 |
| --- | --- | --- |
| `--model-profile` | `16d_absolute_full` | 选择模型，具体列表见下表 |
| `--device` | `cuda:0` | 模型使用的 GPU |
| `--host` | `0.0.0.0` | 允许机器人从局域网连接 |
| `--port` | `5556` | 必须和 Client 的端口一致 |
| `--model-fps` | `30` | Server仅公布训练频率元数据，通常无需显式设置 |
| `--checkpoint` | 不设置 | 可选；显式指定某个 `pretrained_model` 目录 |
| `--log-file` | `outputs/g2a_pi05_deploy/logs/server.log` | Server 日志路径 |
| `--command-head-waist` | 不设置 | 仅适用于 24D；默认不让模型控制头腰，真机首次测试不要开启 |
| `--warmup-requests` | `1` | 监听端口前执行的本地预热次数；诊断时可设为 `0` 关闭 |
| `--warmup-task` | 数据集 prompt | 预热使用的任务文本；实际任务不同时可覆盖 |

### 选择 async 或 RTC

Server 启动后始终具备 RTC 能力，不需要添加 RTC 开关，也不需要为了切换模式重启 Server。
是否使用 RTC 只由 Client 决定：

```text
普通异步推理：--inference-mode=async
RTC 推理：     --inference-mode=rtc
```

下面三个 Server 参数只定义 RTC 算法配置；Client 选择 `async` 时不会使用旧 chunk：

| 参数 | 建议起点 | 说明 |
| --- | ---: | --- |
| `--execution-horizon` | `30` | RTC prefix 长度；Server 默认值 |
| `--max-guidance-weight` | `10.0` | RTC prefix guidance 最大权重 |
| `--prefix-attention-schedule` | `EXP` | 可选 `ZEROS`、`ONES`、`LINEAR`、`EXP` |

## 3. 启动真机 Client

Server 完成模型加载、机器人完成全身 reset 后，在机器人电脑运行：

```bash
cd /home/agi/deploy/g2a_pi05_robot_client

./robot_client/run.sh \
  --host=<GPU_SERVER_IP> \
  --port=5556 \
  --inference-mode=rtc \
  --execute \
  --duration-s=10 \
  --execution-fps=30 \
  --servo-fps=100 \
  --prediction-horizon=50 \
  --quality-horizon=50 \
  --min-replan-after-steps=1 \
  --request-lead-margin-steps=8 \
  --watchdog-timeout-s=3.0 \
  --max-relative-target=0.05 \
  --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --action-low-pass-cutoff-hz=2 \
  --log-file=./logs/robot_client.log \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

必须把 `<GPU_SERVER_IP>` 替换成 GPU Server 的实际局域网 IP。`run.sh` 会自动加载机器人上的
`/home/agi/app/env.sh`。

首次执行保留 `--duration-s=10`。确认动作、夹爪、相机、网络和急停均正常后，才可以把它改成
`--duration-s=0` 持续运行；使用 `Ctrl-C` 停止。

### Client 可变参数

| 参数 | 当前建议 | 说明 |
| --- | ---: | --- |
| `--host` | Server IP | GPU Server 的局域网 IP |
| `--port` | `5556` | 与 Server 保持一致 |
| `--inference-mode` | `rtc` | `async` 为普通异步推理；`rtc` 为 RTC，由 Client 请求决定 |
| `--duration-s` | 首次 `10` | `0` 表示持续执行 |
| `--fps` | `30` | 模型 action 频率，不要改成 100 |
| `--servo-fps` | `100` | GDK 固定下发频率 |
| `--prediction-horizon` | `50` | 必须与 Server 返回的 chunk 长度一致 |
| `--quality-horizon` | `50` | 当前整包 50 步均为可信范围；延迟降到 0.5 秒以下再比较 35 |
| `--min-replan-after-steps` | `1` | 安装新包后至少推进 1 步才允许重规划 |
| `--request-lead-margin-steps` | `8` | rolling P95 延迟之外的提前请求余量；覆盖偶发延迟峰值 |
| `--watchdog-timeout-s` | `3.0` | 超时后清空 action queue 并保持最后成功目标 |
| `--request-timeout-s` | `30` | 单次推理请求等待上限 |
| `--camera-timeout-ms` | `1000` | 每路 GDK 相机读取超时 |
| `--action-low-pass-cutoff-hz` | `2` | 最终 100 Hz 下发前的因果低通；删除该参数表示关闭 |
| `--relative-target-mode` | `warn` | 超过相邻目标阈值时只打印 warning，不裁剪 |
| `--max-relative-target` | `0.05` | 机械臂相邻 100 Hz 目标变化 warning 阈值 |
| `--max-gripper-relative-target` | `0.05` | 夹爪相邻 100 Hz 目标变化 warning 阈值 |
| `--max-message-mb` | `64` | 单次 WebSocket 消息大小上限 |
| `--log-file` | `./logs/robot_client.log` | Client 日志路径 |
| `--left-effector-type` | 自动识别 | 自动识别失败时指定左夹爪 GDK 名称 |
| `--right-effector-type` | 自动识别 | 自动识别失败时指定右夹爪 GDK 名称 |
| `--hold-effectors` | 不设置 | 设置后夹爪保持启动位置，不执行模型夹爪 action |

`--relative-target-mode=warn` 只告警、不裁剪机械臂动作。左右夹爪的物理绝对范围
`[-0.785, 0]` 仍会强制裁剪。只有显式改成 `--relative-target-mode=clip` 时，相邻目标阈值才会
用于裁剪；真机模型对比时建议保持 `warn`，避免引入额外变量。

## 4. 推荐运行顺序

1. 将机器人完整 reset 到训练数据初始姿态，包括头部和腰部。
2. 启动 Server，确认日志打印实际 profile 和自动选择的 checkpoint。
3. 等待 Server 出现 `listening` 日志。
4. 确认急停可用、工作区无人后启动 Client。
5. 首次只执行 10 秒，观察左右臂、夹爪、桌面间隙和日志 warning。
6. 确认正常后再延长执行时间。

## 5. 模型对比要求

对比不同模型时，除 `--model-profile` 外，尽量保持以下条件完全一致：

- 机器人完整初始姿态；
- 包裹位置和朝向；
- 三路相机和图像尺寸；
- task prompt；
- `execution-fps=15`、`servo-fps=100`；
- inference mode 和 RTC 参数；
- `prediction/quality horizon`、rolling P95窗口和请求余量；
- 2 Hz 低通设置；
- 每个模型的测试次数。

建议记录成功率、完成时间、掉包/碰撞/人工停止次数、左右臂抖动、夹爪表现、推理延迟、RTT、
queue underrun、watchdog 和相邻 action warning 次数。

## 6. 日志位置

按照通用命令运行时：

```text
Server: outputs/g2a_pi05_deploy/logs/server.log
Client: /home/agi/deploy/g2a_pi05_robot_client/logs/robot_client.log
```

切换模型做对比时，建议为每次测试使用不同的日志文件名，避免多个实验内容写入同一个文件。
