# G2A PI0.5 16D 真机 RTC 对比实验

本文档用于直接在真机上比较现有的五个 16D 模型。所有实验固定使用 30000 step，
只改变模型，其他部署参数、初始状态、任务和场景保持一致。

## 1. 对比模型

| 实验 | Model profile | Step | 训练方式 | Action | 图像增强 |
| --- | --- | ---: | --- | --- | --- |
| 16D Absolute LoRA | `16d_absolute` | `030000` | LoRA | 双臂和夹爪 absolute | 无 |
| 16D Absolute Full | `16d_absolute_full` | `030000` | 全参 | 双臂和夹爪 absolute | 无 |
| 16D Absolute Full Aug | `16d_absolute_full_aug` | `030000` | 全参 | 双臂和夹爪 absolute | OpenPI augmentation |
| 16D Delta LoRA | `16d_delta` | `030000` | LoRA | 双臂 delta、夹爪 absolute | 无 |
| 16D Delta Full | `16d_delta_full` | `030000` | 全参 | 双臂 delta、夹爪 absolute | 无 |

对应 checkpoint：

| Model profile | Checkpoint |
| --- | --- |
| `16d_absolute` | `outputs/g2a_pi05_16d_absolute_lora_r16_30k/checkpoints/030000/pretrained_model` |
| `16d_absolute_full` | `outputs/g2a_pi05_16d_absolute_full_30k/checkpoints/030000/pretrained_model` |
| `16d_absolute_full_aug` | `outputs/g2a_pi05_16d_absolute_full_openpi_aug_30k/checkpoints/030000/pretrained_model` |
| `16d_delta` | `outputs/g2a_pi05_16d_armdelta_lora_r16_30k/checkpoints/030000/pretrained_model` |
| `16d_delta_full` | `outputs/g2a_pi05_16d_armdelta_full_30k/checkpoints/030000/pretrained_model` |

不要使用 `checkpoints/last`。即使 profile 能自动选择最新 checkpoint，正式实验也必须通过
`--checkpoint` 显式指定上述 `030000/pretrained_model` 路径。

## 2. 固定实验条件

所有模型使用相同条件：

- 相同机器人、相机位置、光照、物体和传送带状态；
- 相同 task prompt；
- 每次实验前执行相同的全身 reset，先打开夹爪，再复位头、腰和双臂；
- Server 使用 `model_fps=30`，Client 使用 `execution_fps=15`；
- RTC、100 Hz GDK 下发、低通、速度/加速度限制和 watchdog 参数保持一致；
- 一次只运行一个模型 Server，避免模型之间产生 GPU/CPU 竞争；
- 每次实验保存独立的 Server 日志、Client 日志、timing JSONL 和现场录像。

`model_fps` 是模型训练时间轴的元数据，`execution_fps` 是真机消费 action 的频率，两者不要求相同。

## 3. 启动 Server

在 GPU Server 上设置当前实验的 profile、checkpoint 和日志名，然后启动：

```bash
cd /path/to/nexus_deploy

PROFILE=16d_absolute
CHECKPOINT=outputs/g2a_pi05_16d_absolute_lora_r16_30k/checkpoints/030000/pretrained_model
RUN_NAME=16d_absolute_step030000

conda run --no-capture-output -n lerobot \
  python -m nexus_deploy.vla.pi05.server \
  --model-profile="$PROFILE" \
  --checkpoint="$CHECKPOINT" \
  --device=cuda:0 \
  --model-fps=30 \
  --execution-horizon=30 \
  --host=0.0.0.0 \
  --port=5556 \
  --warmup-requests=1 \
  --log-file="outputs/g2a_pi05_deploy/comparison_16d_rtc/$RUN_NAME/server.log"
```

切换模型时只替换表格中的 `PROFILE`、`CHECKPOINT` 和对应的 `RUN_NAME`。Server 出现监听日志后，
再启动机器人 Client。

## 4. 启动 RTC Client

在 G2 机器人上运行。每次切换模型时同步修改 `RUN_NAME`，其他参数保持不变：

```bash
cd /home/agi/deploy/g2a_pi05_robot_client

RUN_NAME=16d_absolute_step030000

./robot_client/run.sh \
  --host=192.168.110.140 \
  --port=5556 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward." \
  --inference-mode=rtc \
  --execute \
  --duration-s=10 \
  --execution-fps=15 \
  --servo-fps=100 \
  --prediction-horizon=50 \
  --quality-horizon=50 \
  --min-replan-after-steps=1 \
  --request-lead-margin-steps=8 \
  --min-rtc-blend-steps=8 \
  --warmup-requests=1 \
  --execution-preflight-requests=5 \
  --wrist-image-width=640 \
  --wrist-image-height=400 \
  --wrist-jpeg-quality=90 \
  --watchdog-timeout-s=3.0 \
  --max-relative-target=0.05 \
  --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --gdk-position-limit-margin=0.01 \
  --max-consecutive-gdk-rejections=3 \
  --action-low-pass-cutoff-hz=2 \
  --effector-action-low-pass-cutoff-hz=2 \
  --log-file="./logs/comparison_16d_rtc/$RUN_NAME/client.log"
```

## 5. 实验流程

每个模型依次执行：

1. 确认急停可用，清空机器人周围人员和障碍物。
2. 完成统一的全身 reset，并按固定基准摆放物体。
3. 首先执行一次 3～5 秒的安全短测；出现快速跳变、撞桌趋势或持续抖动时立即急停，并将该模型记为安全检查失败。
4. 安全短测通过后，每个模型正式执行至少 10 次，每次仍从相同初始条件开始。
5. 每次保存日志和录像，并立即记录结果，不根据中途结果修改 RTC、滤波或限速参数。

为了减少电量、温度、机械状态和人工摆放造成的顺序偏差，正式实验建议按轮次交错执行：

```text
第1轮：A → B → C → D → E
第2轮：E → D → C → B → A
后续轮次继续正序、倒序交替
```

## 6. 结果记录表

| Run ID | Profile | Step | 场景 | 是否成功 | 完成时间 | 急停/碰撞 | 抖动/停顿 | RTC fallback | 备注 |
| --- | --- | ---: | --- | --- | ---: | --- | --- | ---: | --- |
| 001 | `16d_absolute` | 30000 | 标准 |  |  |  |  |  |  |

最终汇总：

| Profile | Step | 有效次数 | 成功率 | 急停次数 | 碰桌次数 | 平均完成时间 | RTC fallback 总数 | 主观平滑度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `16d_absolute` | 30000 |  |  |  |  |  |  |  |
| `16d_absolute_full` | 30000 |  |  |  |  |  |  |  |
| `16d_absolute_full_aug` | 30000 |  |  |  |  |  |  |  |
| `16d_delta` | 30000 |  |  |  |  |  |  |  |
| `16d_delta_full` | 30000 |  |  |  |  |  |  |  |

模型选择优先级固定为：安全性、任务成功率、动作连续性、完成时间。任何模型未通过安全短测时，
不要为了补齐实验次数继续执行。
