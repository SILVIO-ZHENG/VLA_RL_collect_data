# G2A 当前 RTC 配置

本文记录当前已经通过有线网络只读验收的 RTC 配置，作为后续真机测试和参数对比的基线。

## 1. 网络与模型

| 项目 | 当前配置 |
| --- | --- |
| Server 有线接口 | `eno1` |
| Server 有线 IP | `10.42.1.200` |
| G2 有线接口 | `enp1s0` |
| G2 有线 IP | `10.42.1.101` |
| 有线协商速率 | `1000 Mbps` |
| Server 端口 | `5557` |
| Model profile | `16d_absolute_full` |
| Checkpoint | `030000` |
| 模型训练 FPS | `30` |
| Client action 执行 FPS | `15` |
| GDK servo FPS | `100` |

Client 必须使用有线 Server IP：

```text
--host=10.42.1.200
```

## 2. 当前 RTC 参数

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `prediction_horizon` | `50` | 模型每次返回50步action |
| `quality_horizon` | `50` | 当前允许使用全部50步预测 |
| `execution_fps` | `30` | Client每秒消费30个policy action |
| `servo_fps` | `100` | 最终固定100 Hz向GDK下发 |
| `rtc_execution_horizon` | `30` | Server最多使用旧chunk的30步作为RTC prefix |
| `min_rtc_blend_steps` | `8` | 响应安装时至少保留8步有效融合，否则拒绝真机执行 |
| `min_replan_after_steps` | `1` | 安装新chunk后至少推进1步才允许重新规划 |
| `request_lead_margin_steps` | `8` | rolling P95之外的请求提前余量 |
| `latency_window_size` | `20` | rolling P95使用最近20次成功请求 |
| `warmup_requests` | `1` | Client控制线程启动前的模型预热请求数 |
| `execution_preflight_requests` | `5` | 真机action解锁前的warm-model延迟测试数 |
| `watchdog_timeout_s` | `3.0` | async/RTC长时间无成功响应时进入HOLD |
| `prefix_attention_schedule` | `EXP` | RTC prefix attention调度 |
| `max_guidance_weight` | `10.0` | RTC prefix guidance最大权重 |

RTC Server参数由Server启动时定义：

```text
--execution-horizon=30
--prefix-attention-schedule=EXP
--max-guidance-weight=10.0
```

Client决定是否启用RTC：

```text
--inference-mode=rtc
```

## 3. 最终action安全参数

| 参数 | 当前值 | 说明 |
| --- | ---: | --- |
| `action_low_pass_cutoff_hz` | `默认关闭` | 生产链路不使用 C2 轨迹重建；低通仅作为显式诊断开关 |
| `effector_action_low_pass_cutoff_hz` | `默认关闭` | 夹爪保留事件时序，仅使用独立运动学限速 |
| `max_arm_velocity` | `0.6 rad/s` | 双臂最终速度限制 |
| `max_arm_acceleration` | `2.0 rad/s²` | 双臂最终加速度限制 |
| `max_effector_velocity` | `1.0 units/s` | 夹爪最终速度限制 |
| `max_effector_acceleration` | `4.0 units/s²` | 夹爪最终加速度限制 |
| `max_relative_target` | `0.05` | 双臂相邻目标warning阈值，不裁剪 |
| `max_gripper_relative_target` | `0.05` | 夹爪相邻目标warning阈值，不裁剪 |
| `relative_target_mode` | `warn` | 超阈值打印 warning，最终由连续轨迹和运动学限制器处理 |

模型 30 Hz chunk 始终保持原策略时间轴，由 100 Hz 线性采样和必要的关节限速执行。
C2 轨迹重定时已从客户端移除，不能用于真机或只读链路。
运行时不再默认执行零相位 chunk SG，也不对夹爪做普通滤波。

16D模型只输出双臂和夹爪。头部和腰部在Client启动时锁定为实测初始位置，执行前仍必须完成
全身reset。

## 4. 当前有线RTC实测结果

测试条件：

```text
模式：RTC read-only
时长：20秒
action执行频率：15 FPS
请求大小：约164–174 KiB
```

| 指标 | 结果 |
| --- | ---: |
| 请求数 | `16` |
| observation-to-response平均 | `343.0 ms` |
| observation-to-response P95 | `356.5 ms` |
| observation-to-response最大 | `388.2 ms` |
| WebSocket RTT平均 | `295.3 ms` |
| WebSocket RTT P95 | `307.9 ms` |
| 模型推理平均 | `271.3 ms` |
| 模型推理 P95 | `286.0 ms` |
| Server外部残差平均 | `7.6 ms` |
| Server外部残差 P95 | `11.4 ms` |
| `stale_steps` | `5–6` |
| `actual_blend_steps` | `24–25` |
| RTC fallback | `0` |
| action queue underrun | `0` |
| watchdog | `0` |
| 100 Hz deadline miss | `0` |

有线大包上行测试约为 `919 Mbps`。此前Wi-Fi测试的Server外部残差最大达到约
`1521 ms`，会令RTC只剩1个有效融合step并触发安全退出；当前有线结果已经消除该问题。

测试日志：

```text
/home/agi/deploy/g2a_pi05_robot_client/logs/wired_rtc_readonly_15fps_20s.log
/home/agi/deploy/g2a_pi05_robot_client/logs/wired_rtc_readonly_15fps_20s_timing.jsonl
```

## 5. 当前RTC运行逻辑

```text
模型返回50步
  → 当前队列剩余约30步时触发下一次请求
  → 有线端到端延迟消耗约5–6步
  → 根据observation_monotonic_time丢弃新chunk前5–6个stale steps
  → RTC prefix仍保留约24–25个有效融合steps
  → 原子安装剩余约44–45步
  → 100 Hz插值、低通、速度和加速度限制后发送给GDK
```

同一时间只允许一个推理请求in-flight。若实际融合低于8步，Client在安装响应前停止物理执行，
不会用缩短后的不连续prefix强行接管机器人。

## 6. 后续RTC参数对比

当前基线保持：

```text
prediction_horizon=50
quality_horizon=50
execution_fps=30
rtc_execution_horizon=30
min_rtc_blend_steps=8
request_lead_margin_steps=8
```

建议下一组只改变 `rtc_execution_horizon`：

| 对比值 | 当前有线延迟下预计有效融合 | 预期特点 |
| ---: | ---: | --- |
| `30` | `24–25 steps` | 当前基线，连续性余量最大 |
| `20` | `14–15 steps` | 仍满足8步要求，对新观测可能更灵活 |

对比时保持模型、初始姿态、物体位置、任务prompt、执行FPS、滤波和安全限制完全一致。重点记录：

- `actual_blend_steps`是否始终不低于8；
- 是否出现RTC fallback或queue underrun；
- chunk边界的运动连续性；
- 右臂方向变化是否更及时；
- 抓取和放置阶段是否存在prefix过强导致的反应滞后。
