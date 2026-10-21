# G2A 真机 Client 启动说明

当前 Nexus 正式部署目录是 `/home/agi/deploy/g2a_robot_client`。PI0.5 Server
的有线地址为 `10.42.1.200:5557`，无线回退地址为
`192.168.110.140:5557`。运行前确认 Server 已完成模型加载，并确认
没有其他 GDK Client 占用机器人。

模型通过 Server 的 `--model-profile` 选择，机器人 Client 不需要设置 model profile。

## 安全启动命令

推荐由 Nexus 任务页的“开始观测”启动。手工联调时，先用 read-only
读取 joints/图像并执行推理，不下发 action：

```bash
cd /home/agi/deploy/g2a_robot_client
pgrep -af 'python3? -m [r]obot_client'

./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5557 \
  --inference-mode=async \
  --read-only \
  --duration-s=60 \
  --execution-fps=30 \
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
  --action-low-pass-cutoff-hz=2 \
  --effector-action-low-pass-cutoff-hz=2 \
  --log-file=./logs/robot_client.log \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

使用无线回退网络时，把 `--host` 改为 `192.168.110.140`。若 `pgrep`
显示已有 Client，不要重复启动。结束手工 Client 时按 `Ctrl+C`，等待
GDK 独占锁释放。

`--execute` 会真实下发动作，不是普通启动方式。当前 Nexus 配置仍锁定
execute；未完成已标定安全 profile、急停、低速验收和人工授权时，不得
把上述 `--read-only` 改成 `--execute`。

## 参数说明

| 参数 | 建议值 | 说明 |
| --- | ---: | --- |
| `--host` | Server IP | GPU Server 的局域网 IP |
| `--port` | `5557` | PI0.5 端口；LingBot 使用 `5558` |
| `--inference-mode` | `async` | 当前真机默认使用普通异步推理；`rtc` 仅用于 `--read-only` 验收 |
| `--read-only` | 安全默认 | 读取真机并推理，但不发送 action |
| `--duration-s` | 首次 `10` | 执行时长；`0` 表示持续执行 |
| `--execution-fps`（兼容别名 `--fps`） | `15` | Client消费action的频率；30 FPS训练轨迹按2倍时长执行 |
| `--servo-fps` | `100` | GDK 固定 action 下发频率 |
| `--prediction-horizon` | `50` | 必须与 Server 返回的 chunk 长度一致 |
| `--quality-horizon` | `50` | 每包中允许执行的可信范围；当前不要改为 35 |
| `--min-replan-after-steps` | `1` | 安装新包后至少推进1个执行频率step才允许重规划 |
| `--request-lead-margin-steps` | `8` | 自适应提前请求的安全余量；用于覆盖偶发网络/调度延迟峰值 |
| `--min-rtc-blend-steps` | `8` | RTC 响应至少保留的预计有效融合步数，不足时本次退化为 async |
| `--latency-window-size` | `20` | rolling P95 延迟统计窗口；默认 20 次请求 |
| `--warmup-requests` | `1` | 启动控制线程前预热一次模型；结果不进入 action queue 和延迟 P95 |
| `--execution-preflight-requests` | `5` | 真机解锁前测量5次warm-model请求并建立rolling P95 |
| `--wrist-image-width/height` | `640/400` | 左右腕图像在机器人端缩放后的传输尺寸 |
| `--wrist-jpeg-quality` | `90` | 腕部缩放图像的 JPEG 质量；原图对照可用 `--no-resize-wrist-images` |
| `--watchdog-timeout-s` | `3.0` | 超时后保持最后一次成功 action |
| `--action-low-pass-cutoff-hz` | `2` | 手臂和头腰的100Hz低通 |
| `--effector-action-low-pass-cutoff-hz` | `2` | 夹爪独立100Hz低通，用于抑制模型在开/合目标间的短时反复切换；之后仍执行速度/加速度限制 |
| `--max-arm-tracking-error` | `0.4` | command与实测手臂误差上限；连续3次超限会停机 |
| `--max-effector-tracking-error` | `0.5` | command与实测夹爪误差上限 |
| `--max-head-waist-tracking-error` | `0.2` | command与实测头腰误差上限 |
| `--relative-target-mode` | `warn` | 超过相邻目标阈值时只打印 warning，不裁剪 |
| `--max-relative-target` | `0.05` | 机械臂相邻 100 Hz 目标变化 warning 阈值 |
| `--max-gripper-relative-target` | `0.05` | 夹爪相邻 100 Hz 目标变化 warning 阈值 |
| `--gdk-position-limit-margin` | `0.01` | 所有身体关节目标与厂商硬限位保留的安全余量（rad），避免边界浮点误差被 GDK 拒绝 |
| `--max-consecutive-gdk-rejections` | `3` | 单次拒绝保持最后成功 action；连续达到该次数才安全停机 |
| `--log-file` | `./logs/robot_client.log` | Client 日志文件 |
| `--timing-log-file` | 自动生成 | 逐请求 JSONL；默认与普通日志同目录并以 `_timing.jsonl` 结尾 |
| `--task` | 固定训练 prompt | 保持与训练数据完全一致 |

Client 使用 `observation_monotonic_time` 计算真实 stale steps。async 在队列剩余步数小于等于
`rolling P95 延迟步数 + request-lead-margin-steps` 时发起下一次请求；RTC 的提前量还会保证
不小于 Server 的 execution horizon。同一时间只会有一个
请求在途。若实测延迟无法满足连续执行预算，`--execute` 会在安装模型 action 前直接退出；
真机 action 线程只有在5次warm-model延迟预检和首个可执行chunk都通过后才会解锁，因此
预检失败不会下发hold或模型action。15 FPS下50步约3.33秒，30步RTC horizon约2秒。

RTC 的完整行为、prefix 不足处理和只读验收见上级目录的 `ASYNC_RTC.md`。

如果左右夹爪无法自动识别，可以额外设置：

```text
--left-effector-type=<LEFT_TYPE> \
--right-effector-type=<RIGHT_TYPE>
```
