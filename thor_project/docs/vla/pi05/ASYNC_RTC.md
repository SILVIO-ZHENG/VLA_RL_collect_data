# G2A PI0.5 异步 RTC 设计与使用说明

## 1. 当前结论

当前真机执行优先使用普通异步推理：

```text
--inference-mode=async
```

RTC 在完成只读验收前不要用于真机 `--execute`。RTC 的目标是让连续两包 action chunk
在时间上平滑衔接，但它不能替代最底层的关节速度和加速度保护。

当前推荐参数：

```text
prediction_horizon       = 50
quality_horizon          = 50
rtc_execution_horizon    = 30
min_replan_after_steps   = 1
request_lead_margin_steps = 8
min_rtc_blend_steps      = 8
latency_estimator        = rolling P95
```

`rtc_execution_horizon=30` 是最大 RTC prefix 长度，不是固定长度，也不表示机器人必须执行
30 步后才能重新推理。

## 2. Pipeline

```text
Robot Client                         GPU Server
────────────                         ──────────
采集 observation/state/images
记录 observation_monotonic_time
快照当前 action 队列
        │
        ├── observation + 当前旧 chunk 的位置 ──>
        │                                      取旧 chunk 尚未执行的真实动作
旧 action 继续以 30 Hz 执行                  最多截取30步作为 prefix
GDK 插值层继续以 100 Hz 下发                  PI0.5 + 可选 RTC 生成50步
        │
        <──────────── 新 action chunk ──────────┤
按墙钟时间计算 stale_steps
丢弃 new_actions[:stale_steps]
安装 new_actions[stale_steps:50]
```

同一时刻最多有一个推理请求 in-flight。相机只在需要推理时采集，action 发送线程和相机
采集不共用 Robot 控制锁。

## 3. 三种长度

需要区分以下三个量：

```text
source_prefix_steps     = 发请求时，旧 chunk 中尚未执行的真实动作数量
effective_horizon       = min(source_prefix_steps, rtc_execution_horizon)
estimated_blend_steps   = effective_horizon - estimated_delay_steps
```

例如旧 chunk 剩24步：

```text
source_prefix_steps = 24
effective_horizon   = min(24, 30) = 24
```

Server 直接使用这24步，不会补6个零。RTC 模型为了张量形状可能在内部补零，但会先把
有效 horizon 限制为24，补零区域的权重为零，不会被当作旧动作约束。

## 4. 请求触发

Client 用最近20次 observation-to-response 延迟的 rolling P95 估计推理延迟：

```text
latency_steps = ceil(P95_latency_seconds × fps)
```

普通 async 的请求提前量：

```text
async_request_lead = latency_steps + request_lead_margin_steps
```

RTC 的请求提前量：

```text
rtc_request_lead = max(
    latency_steps + request_lead_margin_steps,
    rtc_execution_horizon,
)
```

当 action buffer 的剩余步数小于或等于该提前量，并且从上一包至少执行了
`min_replan_after_steps`，立即采集 observation 并请求下一包。

使用30步 RTC horizon 的原因是当前推理延迟通常约18～20步。若要求至少8步有效融合：

```text
required_prefix ≈ 20 + 8 = 28 steps
```

因此当前使用30步较合理。延迟稳定低于约0.5秒后，可以再比较24步与30步。

## 5. Prefix 不足时的处理

RTC prefix 永远只包含真实旧动作：

```text
prefix = remaining_previous_actions[:rtc_execution_horizon]
```

不允许把短 prefix 用零补齐。请求时预计的融合区间为：

```text
estimated_blend_steps = effective_horizon - estimated_delay_steps
```

只有满足以下条件才启用本次 RTC：

```text
estimated_blend_steps >= min_rtc_blend_steps
```

否则 Client 本次不发送 RTC prefix，自动按普通 async 推理，并在本地日志记录
`RTC request falls back to async`。Server 响应会标记：

```text
rtc_fallback=request_disabled
```

Server 还保留独立的防御检查；若请求方错误地要求 RTC、但 prefix 的预计融合仍不足，
Server 会返回 `rtc_fallback=insufficient_blend_steps`。

响应到达后还会使用真实 `stale_steps` 再检查一次：

```text
actual_blend_steps = effective_horizon - stale_steps
```

如果预计值满足要求、但某次实际延迟尖峰使 `actual_blend_steps < 8`，read-only/dry-run会
打印告警并保留该包用于诊断；真机 `--execute` 会拒绝安装该包并停止，不能用预计P95掩盖
单次实际延迟超限。

第一次请求没有上一包 action，也会按预期使用普通推理：

```text
rtc_prefix=False
rtc_fallback=request_disabled
```

## 6. 新 action chunk 的安装

模型返回的50步以采集 observation 的时刻为时间原点。推理期间已经过去的动作必须丢弃：

```text
stale_steps = ceil(
    (response_monotonic_time - observation_monotonic_time) × fps
)

installed_actions = response_actions[stale_steps:quality_horizon]
```

新动作替换未来队列，不追加到旧 chunk 尾部。如果：

```text
stale_steps >= quality_horizon
```

整包已经过期，Client 丢弃整包，并保留旧队列中仍然有效的动作；不会执行过期动作。

RTC 的平滑来自模型在 prefix 有效区间内对新旧轨迹进行约束。它不是把两段动作在 Client
端直接拼接，也不是重复执行旧 prefix。

## 7. 已修复的问题

### 7.1 短 prefix 被补零

旧实现会把24步真实 prefix 补成30步：

```text
[24步真实动作] + [6步零]
```

模型把这些零误认为有效约束，在真实 prefix 与零 padding 的边界产生大幅关节跳变。
当前 Server 只截断、不补齐，并把实际长度传给 RTC 模型。

### 7.2 RTC 请求过晚

旧实现仅在 buffer 剩余约 `P95延迟+3` 步时请求，通常只剩21～24步，无法给30步 RTC
窗口留下足够融合空间。当前 RTC 请求提前量至少为 Server 公布的 execution horizon。

### 7.3 融合空间过短仍强制使用 RTC

当前 Client 和 Server 都检查最小融合预算，默认至少8步。预算不足时明确退化为 async，
并在 response metadata 和日志中记录原因。

### 7.4 使用推理耗时估算过期动作

当前使用 `observation_monotonic_time` 到收到响应的真实墙钟时间计算 `stale_steps`，其中包含
相机、编码、网络、排队、Server 前后处理和模型推理，不再只使用模型推理时间。

## 8. 日志检查

每次成功安装 action chunk 会打印：

```text
stale=<实际丢弃步数>
queue=<安装后队列长度>
lead=<当前请求提前量>
latency_p95=<滚动P95延迟>
rtc_prefix=<是否真正使用RTC>
rtc_source/effective/blend=<原始prefix/有效prefix/预计融合步数>
actual_blend=<按真实stale计算的融合步数>
rtc_fallback=<降级原因>
```

RTC 只读验收时应满足：

- 不出现从真实 prefix 跳到零 padding 的边界；
- `effective <= source` 且 `effective <= 30`；
- `rtc_prefix=True` 时 `blend >= 8`；
- `stale >= 50` 的响应被整包丢弃；
- action chunk 安装期间 action buffer 不被意外清空；
- LoRA/Full 的 RTC 结果没有明显大于对应 async 的关节相邻步跳变。

## 9. 只读测试命令

在机器人上测试时必须使用 `--read-only`，不要添加 `--execute`：

```bash
cd /home/agi/deploy/g2a_pi05_robot_client

./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5556 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward." \
  --inference-mode=rtc \
  --read-only \
  --duration-s=10 \
  --fps=30 \
  --servo-fps=100 \
  --prediction-horizon=50 \
  --quality-horizon=50 \
  --min-replan-after-steps=1 \
  --request-lead-margin-steps=8 \
  --min-rtc-blend-steps=8 \
  --action-trace-dir=./logs/action_traces/rtc_validation \
  --no-trace-images \
  --log-file=./logs/rtc_validation.log
```

普通异步基线只需改为：

```text
--inference-mode=async
```

## 10. 暂未处理的问题

最终 GDK 100 Hz 下发层尚未加入真正的关节速度和加速度限制。必须先完成 RTC/async 的
只读对比并确认模型与队列接缝行为，再设计该限制，避免同时改变多个变量。加入底层限制前，
即使 RTC 通过只读测试，也应先低速、短时、急停可达地逐步恢复真机执行。
