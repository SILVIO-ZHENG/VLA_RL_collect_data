# G2A PI0.5 Client/Server 启动命令

本文档只提供常用启动命令。完整 pipeline、协议和参数说明请参考
[README.md](README.md)。

## 1. 推荐组合：16D Delta + RTC

### GPU 服务器

在 GPU 服务器上运行：

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_delta \
  --device=cuda:0 \
  --execution-horizon=30 \
  --max-guidance-weight=10.0 \
  --prefix-attention-schedule=EXP \
  --host=0.0.0.0 \
  --port=5556
```

Server 默认会在打开端口前完成一次本地模型 warmup。等待日志中出现：

```text
G2A PI0.5 server listening on ws://0.0.0.0:5556
```

检查 server 是否启动完成：

```bash
curl http://127.0.0.1:5556/healthz
```

预期返回：

```text
OK
```

### Robot Client：打包并复制到机器人

在GPU服务器的LeRobot仓库中生成独立部署包：

```bash
cd /path/to/nexus_deploy
./scripts/build_g2a_robot_client_bundle.sh
```

把命令输出的`g2a_pi05_robot_client_*.tar.gz`复制到机器人。机器人端解压并进入
原生GDK环境：

```bash
tar -xzf g2a_pi05_robot_client_*.tar.gz
cd g2a_pi05_robot_client
source ~/app/env.sh
./robot_client/install_offline.sh
```

### Robot Client：只读取真机 observation

先使用 `--read-only` 检查 state、图像、网络和模型输出。该模式不会向机器人发送
action。

在机器人电脑上运行：

```bash
./robot_client/run.sh \
  --host=<GPU_SERVER_IP> \
  --port=5556 \
  --inference-mode=rtc \
  --read-only \
  --duration-s=60 \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

需要替换：

```text
<GPU_SERVER_IP>           GPU 服务器的局域网 IP
"Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward." 训练数据中实际使用的任务 prompt
```

### Robot Client：固定OBS只读诊断模型输出

这个命令复用第一次采集的state和三路图像，连续请求模型，但不会向GDK发送
action：

```bash
./robot_client/run.sh \
  --host=<GPU_SERVER_IP> \
  --port=5556 \
  --inference-mode=async \
  --read-only \
  --duration-s=20 \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 \
  --action-trace-dir=./logs/action_traces \
  --freeze-trace-observation \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

采集目录会打印在终端。复制到GPU服务器后运行：

```bash
conda run --no-capture-output -n lerobot \
  .venv/bin/python tools/pi05/analyze_action_trace.py <TRACE_DIR>
```

结果保存在`<TRACE_DIR>/analysis/`。`--action-trace-dir`被安全锁定，只能和
`--read-only`一起使用。

为了判断重复推理差异是否来自PI0.5随机noise，可重启server并加入
`--fixed-noise-seed=123`，再重复上面的固定OBS只读测试。默认不固定noise。

需要继续只读验证最终下发平滑时，在client命令追加
`--action-low-pass-cutoff-hz=4 --hold-effectors`。两个开关默认关闭。

### Robot Client：执行真机 action

确认 `--read-only` 测试正常、急停可用并完成安全检查后，才能运行：

```bash
cd /path/to/nexus_deploy

./robot_client/run.sh \
  --host=<GPU_SERVER_IP> \
  --port=5556 \
  --inference-mode=rtc \
  --execute \
  --duration-s=10 \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 \
  --watchdog-timeout-s=3.0 \
  --max-relative-target=0.05 \
  --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

当前两个 `0.05` 参数仅作为 WARNING 阈值；`--relative-target-mode=warn`
表示超出时不做相对裁剪。夹爪物理绝对范围仍会裁剪。首次真机测试建议保留
`--duration-s=10`。确认运行安全后，可以改成：

```text
--duration-s=0
```

`0` 表示持续运行，使用 `Ctrl-C` 停止。

Server只通过metadata公布训练频率`model_fps`，不决定真机执行速度。
`--execution-fps=30`由Client用于action播放、RTC stale steps和buffer时间换算；
它与`model_fps`不一致时仅打印时间缩放warning。`--servo-fps=100`控制GDK固定下发频率。

## 2. 使用指定 checkpoint

如果不希望 server 自动选择 profile 中 step 最大的完整 checkpoint，可以显式
指定：

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_delta \
  --checkpoint=/absolute/path/to/checkpoints/030000/pretrained_model \
  --device=cuda:0 \
  --execution-horizon=30 \
  --host=0.0.0.0 \
  --port=5556
```

`--checkpoint` 必须指向 `pretrained_model` 目录，而不是实验输出根目录。

## 3. 16D Absolute 模型

### Server：LoRA

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_absolute \
  --device=cuda:0 \
  --host=0.0.0.0 \
  --port=5556
```

### Server：full fine-tuning

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_absolute_full \
  --device=cuda:0 \
  --host=0.0.0.0 \
  --port=5556
```

### Client

```bash
./robot_client/run.sh \
  --host=<GPU_SERVER_IP> \
  --port=5556 \
  --inference-mode=async \
  --read-only \
  --duration-s=60 \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

如果要测试 16D absolute + RTC，只需把 client 改为 `--inference-mode=rtc`；
server 始终具备 RTC 能力。

## 4. 16D Delta full fine-tuning

### Server

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_delta_full \
  --device=cuda:0 \
  --host=0.0.0.0 \
  --port=5556
```

Client 与第 1 节相同：普通异步推理使用 `--inference-mode=async`，RTC 使用
`--inference-mode=rtc`；server 命令不需要改变。

## 5. 24D Absolute 模型

### Server

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile=24d_absolute \
  --device=cuda:0 \
  --host=0.0.0.0 \
  --port=5556
```

该命令默认锁定头部和腰部。首轮真机实验不要添加
`--command-head-waist`。

### Client

```bash
./robot_client/run.sh \
  --host=<GPU_SERVER_IP> \
  --port=5556 \
  --inference-mode=async \
  --read-only \
  --duration-s=60 \
  --execution-fps=30 \
  --min-replan-after-steps=1 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

## 6. Mock 联调

Mock 模式不加载模型、不使用 GPU，也不连接真机。

### Mock Server

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --mock \
  --mock-inference-delay-s=0.4 \
  --host=127.0.0.1 \
  --port=18082
```

### Dry-run Client

在同一台电脑的另一个终端运行（从仓库测试时）：

```bash
cd /path/to/nexus_deploy

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.robot.g2a \
  --host=127.0.0.1 \
  --port=18082 \
  --inference-mode=rtc \
  --dry-run \
  --duration-s=10 \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

在 0.4 秒模拟延迟和 30 Hz 下，后续日志通常会出现：

```text
drop=12
rtc_prefix=True
```

## 7. 指定日志文件

Server：

```bash
--log-file=outputs/g2a_pi05_deploy/logs/server_16d_delta_rtc.log
```

Client：

```bash
--log-file=outputs/g2a_pi05_deploy/logs/client_g2a_rtc.log
```

server不指定时会自动生成带时间戳的日志；独立client默认写入解压目录的
`logs/robot_client.log`：

```text
outputs/g2a_pi05_deploy/logs/server_YYYYMMDD_HHMMSS.log
g2a_pi05_robot_client/logs/robot_client.log
```

## 8. 最短可复制版本

Server：

```bash
cd /path/to/nexus_deploy
conda run --no-capture-output -n lerobot python -m nexus_deploy.vla.pi05.server \
  --model-profile=16d_delta --device=cuda:0 \
  --execution-horizon=30 --host=0.0.0.0 --port=5556
```

Read-only Client：

```bash
./robot_client/run.sh \
  --host=<GPU_SERVER_IP> --port=5556 --inference-mode=rtc --read-only \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

真机 Client：

```bash
./robot_client/run.sh \
  --host=<GPU_SERVER_IP> --port=5556 --inference-mode=rtc --execute \
  --execution-fps=30 \
  --servo-fps=100 \
  --min-replan-after-steps=1 --watchdog-timeout-s=3.0 \
  --max-relative-target=0.05 --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```
