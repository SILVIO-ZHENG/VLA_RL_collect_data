# G2A 训练部署延迟诊断

这套诊断保持以下控制参数不变：

- `prediction_horizon=50`
- `quality_horizon=50`
- `rtc_execution_horizon=30`
- `min_replan_after_steps=1`
- `request_lead_margin_steps=3`

真机执行仍优先使用 `async`。RTC 在只读模式下验收，并且应先把
observation-to-response P95 稳定降低到 `500–600 ms`（30 Hz 下约 15–18 steps）。

## 自动分段计时

Server 和机器人 Client 各自写普通日志和逐请求 JSONL。若不指定
`--timing-log-file`，文件会自动生成在普通日志旁边，名称以 `_timing.jsonl` 结尾。

Client 每个请求记录：

- 三路相机采集时间；
- 左右腕图像 JPEG 解码、resize、重新编码时间；
- MessagePack 请求序列化时间和字节数；
- WebSocket send 调用时间；
- 等待响应时间；
- 响应反序列化时间；
- observation-to-response 和 rolling P95。

Server 每个请求记录：

- 请求完整到达时刻和字节数；
- MessagePack 反序列化时间；
- 单 worker executor 排队时间；
- worker 墙钟时间和线程 CPU 时间；
- policy 预处理、模型推理、后处理时间；
- 响应序列化、WebSocket send 和 handler 总时间；
- asyncio event-loop lag 超过 `50 ms` 时的 warning。

Server 默认限制数值库和 PyTorch 的 CPU 线程，避免共享 64 核机器上的线程过量：

```text
OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4
--torch-num-threads=4
--torch-num-interop-threads=1
```

环境变量在外部显式设置时优先；CLI 参数可以单独修改 PyTorch 线程数。修改后需要重启
Server 才会生效。

Client 与 Server 使用同一个 `request_id`，可以关联两端记录。跨机器的 wall clock
只用于辅助定位；各阶段耗时使用本机 monotonic clock，不要求两台机器时钟完全同步。

## Warmup

Client 默认在启动 action/inference 线程之前执行一次非 RTC warmup：

```text
--warmup-requests=1
```

Warmup 返回的 action 会被丢弃，不会进入 action queue，也不会进入 rolling P95。
只有 warmup 完成后的成功推理响应才用于 RTC 延迟估计。设置为 `0` 可以关闭，但不建议。

## 腕部图像传输

默认只把左右腕相机从 GDK 原始分辨率缩放到训练数据分辨率后重新 JPEG 编码：

```text
--wrist-image-width=640
--wrist-image-height=400
--wrist-jpeg-quality=90
```

头部相机保持 GDK 原始 JPEG。机器人当前使用 Pillow 编解码，不使用 ABI 不兼容的
OpenCV。可用 `--no-resize-wrist-images` 做原始图像对照实验。

## 汇总请求耗时

把机器人 Client 的 timing JSONL 复制到 GPU Server，然后执行：

```bash
conda run --no-capture-output -n lerobot \
  python src/nexus_deploy/vla/pi05/analyze_request_timing.py \
  /path/to/robot_client_timing.jsonl \
  --server-timing-log=/path/to/server_timing.jsonl \
  --output=outputs/g2a_pi05_deploy/latency_summary.md
```

默认排除 warmup；需要查看冷启动时增加 `--include-warmup`。

## 网络链路诊断

在 GPU Server 上运行：

```bash
./tools/pi05/diagnose_robot_link.sh
```

脚本检查：

- 双向 ping、抖动和丢包；
- Server 和机器人到对端的真实路由/出口网卡；
- `ip -s link` 丢包与错误；
- 机器人 Wi-Fi 信号、协商速率和 power-save；
- 两端都有 `iperf3` 时自动测量上下行吞吐。

机器人当前通过 `wlan0` 连接 Server，并且 power-save 为 on。机器人尚未安装
`iperf3` 时脚本会明确跳过吞吐测试，不会修改网络配置。不要把 G2 内部 EtherCAT
接口当作外部部署网口。

## Server CPU/GPU 竞争诊断

启动推理请求的同时，在 GPU Server 另一个终端执行：

```bash
DURATION_S=60 ./tools/pi05/diagnose_server_load.sh
```

结果默认保存在 `outputs/g2a_pi05_deploy/system/<timestamp>/`，包括：

- `nvidia-smi dmon` 的 GPU 利用率、显存、功耗和 PCIe 活动；
- `vmstat` 的 CPU、运行队列、内存和 IO；
- `pidstat`（若已安装）的逐进程 CPU/IO；
- 测试前后的进程列表和网卡统计。

判断重点：

- `queue_ms` 高：GPU worker 被前一个任务或请求占用；
- `worker_ms` 明显高于 `worker_cpu_ms`，且 GPU 推理没变慢：线程被调度挂起；
- event-loop lag warning 多：Server CPU/事件循环竞争；
- `response_wait_ms - server worker` 很高：优先检查网络、WebSocket 和调度；
- 图像 resize 后请求字节数、Server preprocess 应同时下降。
