# G2A 三种推理模式真机命令

以下命令使用 `16d_absolute_full` Server（端口 `5557`），Client 以 `15 FPS`
消费 action，并以固定 `100 Hz` 向 GDK 下发平滑后的目标。

每次测试前必须完整 reset 双臂、夹爪、头部和腰部，并保持任务物体、相机视角和机器人初始状态一致。
首次测试均使用 `--duration-s=5`，确认正常后再延长时间。

## 1. Sync

当前 chunk 执行完成后再请求下一包。等待期间保持最后一次成功 action，因此 chunk 之间存在预期停顿。

```bash
cd /home/agi/deploy/g2a_pi05_robot_client

./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5557 \
  --inference-mode=sync \
  --execute \
  --duration-s=10 \
  --execution-fps=30 \
  --servo-fps=100 \
  --prediction-horizon=50 \
  --quality-horizon=50 \
  --warmup-requests=1 \
  --wrist-image-width=640 \
  --wrist-image-height=400 \
  --wrist-jpeg-quality=90 \
  --watchdog-timeout-s=3 \
  --max-relative-target=0.05 \
  --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --action-low-pass-cutoff-hz=2 \
  --effector-action-low-pass-cutoff-hz=2 \
  --log-file=./logs/execute_full_sync_15fps_5s.log \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

## 2. Async

执行当前 chunk 时提前请求下一包，不使用旧 chunk 作为模型 prefix。提前余量使用 8 steps，
用于覆盖偶发网络或线程调度延迟峰值。

```bash
cd /home/agi/deploy/g2a_pi05_robot_client

./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5557 \
  --inference-mode=async \
  --execute \
  --duration-s=5 \
  --execution-fps=30 \
  --servo-fps=100 \
  --prediction-horizon=50 \
  --quality-horizon=50 \
  --min-replan-after-steps=1 \
  --request-lead-margin-steps=8 \
  --warmup-requests=1 \
  --execution-preflight-requests=5 \
  --wrist-image-width=640 \
  --wrist-image-height=400 \
  --wrist-jpeg-quality=90 \
  --watchdog-timeout-s=3 \
  --max-relative-target=0.05 \
  --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --action-low-pass-cutoff-hz=2 \
  --effector-action-low-pass-cutoff-hz=2 \
  --log-file=./logs/execute_full_async_15fps_5s.log \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

## 3. RTC

RTC 同样异步请求，但 Server 使用旧 chunk 的剩余 action 作为 prefix，引导新旧 chunk 平滑衔接。
Client 要求响应安装后至少保留 8 个有效融合 steps，否则真机执行会被拒绝。

```bash
cd /home/agi/deploy/g2a_pi05_robot_client

./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5557 \
  --inference-mode=rtc \
  --execute \
  --duration-s=5 \
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
  --watchdog-timeout-s=3 \
  --max-relative-target=0.05 \
  --max-gripper-relative-target=0.05 \
  --relative-target-mode=warn \
  --action-low-pass-cutoff-hz=2 \
  --effector-action-low-pass-cutoff-hz=2 \
  --log-file=./logs/execute_full_rtc_15fps_5s.log \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward."
```

## 运行时重点日志

- sync 的 `SYNC CHUNK COMPLETE` 表示正常等待下一包，不是故障。
- async/RTC 出现 `ACTION QUEUE UNDERRUN` 表示提前请求量没有覆盖实际延迟。
- `WATCHDOG ENTER HOLD` 表示推理响应长时间中断。
- `Joint tracking error exceeds` 连续达到限制会停止物理执行。
- RTC 应看到 `rtc_prefix=True`、`rtc_fallback=None`，并且 `actual_blend` 不低于 8。
- 三种模式都必须保持 `deadline_misses=0`。

日志保存在机器人端：

```text
/home/agi/deploy/g2a_pi05_robot_client/logs/
```
