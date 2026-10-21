# 推理工作站 Agent / 上层调度 API 对接说明

本说明按 2026-10-13 当前 Thor 控制服务的代码、服务配置及实际只读响应核对。对外调度接口与网页共享任务状态和执行逻辑。同事的后端服务调用下面的 HTTP 接口即可，无需机器人 SDK、SSH 或模型环境。

本次核对只读取健康、模型、机器人和历史任务状态，没有提交生产任务、停止任务或执行复位。旧 5090 地址 `192.168.110.140:8765` 是历史部署地址；当前使用下表中的 Thor 地址。

## 1. 地址、认证与当前状态

| 项目 | 当前值 |
| --- | --- |
| Base URL | `http://192.168.110.14:21865` |
| 同服务有线地址 | `http://10.42.1.77:21865`，仅供具备该网段路由的调用方使用 |
| 固定任务方案 | `parcel-sort` |
| 认证 | `Authorization: Bearer <调度Token>` |
| POST 数据格式 | `Content-Type: application/json` |
| 通知方式 | 当前主动轮询，`callback_url` 未配置 |
| 并发 | 每台机器人同一时刻接受一个任务；忙时立即拒绝，没有业务等待队列 |

所有 `/api/sorting/*` 接口都要求调度 Token。标定页面的管理员密码 `admin` 不能用于此接口。由管理员私下交付实际 Token，调用方存入自己的服务器密钥配置；当前没有登录换取 Token 的接口。当前 Thor 服务从 `/home/anu/.config/nexus-thor-sorting-5090.env` 加载 `G2A_SORTING_TOKEN`；文件名保留了历史站点名称，文件仅在工作站本机存在，不要放进前端或转发到公开文档。

请求应由同事的后端服务发出，不带浏览器 `Origin`，使用约定 IP 与端口。Token 不需要每次任务更换。调度机须能访问对应 TCP 21865；本机核对不代表同事的主机网络已连通。

核对时模型 `pi05 / training_rtc_009500` 已就绪、机器人空闲且复位成功；工作站仍处于 `debug`（手动）模式，所以 `ready_for_next_task=false`。现场操作员须在运行设置切换到 `automatic`（自动/API），并确认允许新任务，调度请求才能启动。该状态会随现场操作变化，以实时查询及启动接口准入结果为准。

## 2. 开始任务

```http
POST /api/sorting/tasks
Authorization: Bearer <调度Token>
Content-Type: application/json

{"request_id":"order-20261013-001","profile":"parcel-sort"}
```

请求体必须且只能包含 `request_id` 和 `profile`：

- `request_id`：同事为每个新业务生成并保存唯一编号，1–128 字符，只允许英文字母、数字、下划线、点、冒号和连字符。
- `profile`：固定方案名，当前为 `parcel-sort`。模型版本、机器人地址、动作指令、回调地址都由现场方案配置，不在请求中传入。

首次接受返回 **HTTP 202** 和任务 JSON，表示已接受本次执行，需要继续查询结果。相同 `request_id`、相同请求体重试返回 **HTTP 200** 和原任务，不会再启动；相同编号但参数不同返回 **409 / IDEMPOTENCY_CONFLICT**。

下例为响应核心字段示意，非实时响应；完整响应还包含第 4 节的公共任务字段：

```json
{
  "task_id": "0123456789abcdef0123456789abcdef",
  "source": "scheduler",
  "request_id": "order-20261013-001",
  "profile": "parcel-sort",
  "status": "starting",
  "lifecycle_state": "executing",
  "error": null,
  "callback": {
    "enabled": false,
    "event_id": null,
    "status": "not_pending",
    "attempts": 0
  }
}
```

调用方立即保存 `request_id -> task_id` 对应关系。启动 POST 超时、500 或断线时，执行结果可能不确定，应使用原编号和原请求体重试；已得到 `task_id` 则优先查询。不能为同一业务换新编号重试。被拒绝的请求不会在机器人空闲后自动执行。

Bash 示例（在同事服务器设置实际 Token；启动会执行当前真机方案）：

```bash
export NEXUS_URL='http://192.168.110.14:21865'
export NEXUS_TOKEN='<私下交付的实际Token>'

curl --noproxy '*' -sS -i --connect-timeout 5 --max-time 30 \
  -X POST "$NEXUS_URL/api/sorting/tasks" \
  -H "Authorization: Bearer $NEXUS_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"request_id":"order-20261013-001","profile":"parcel-sort"}'
```

## 3. 接收进度和最终结果：当前使用轮询

```http
GET /api/sorting/tasks/{task_id}
Authorization: Bearer <调度Token>
```

返回 **HTTP 200** 和同一任务的最新详情。建议每 1–2 秒查询一次，读取顶层 `status`；网页关闭不会使任务停止。不要靠 HTTP 200、`deployment_status` 或模型健康判断业务完成。

```bash
TASK_ID='<启动返回的task_id>'
curl --noproxy '*' -sS --connect-timeout 5 --max-time 15 \
  "$NEXUS_URL/api/sorting/tasks/$TASK_ID" \
  -H "Authorization: Bearer $NEXUS_TOKEN"
```

| `status` | 同事应如何处理 |
| --- | --- |
| `starting`、`running`、`pausing`、`paused`、`resetting`、`stopping`、`cancelling` | 过程状态，继续查询；暂停不等于结束 |
| `completed` | 业务完成，继续看 `completion_rule`、`completion_evidence`、`reset_state` 区分完成方式及是否复位 |
| `simulated` | 模拟/只读流程结束，不能当作真机完成 |
| `cancelled` | 已取消，不是业务成功；可能仍需复位 |
| `failed` | 执行失败，记录 `error` 并交现场处理 |
| `recovery_required` | 执行或状态结果不确定，停止自动派发并要求现场核实 |
| 未识别的状态 | 保存原任务并继续查询/核实，不猜测完成 |

### 自动空场完成

下面为当前接口中已保存的历史真机任务响应的核心字段，ID、业务编号做了示意替换，并省略命令日志及其他公共字段：

```json
{
  "task_id": "0123456789abcdef0123456789abcdef",
  "source": "scheduler",
  "request_id": "order-20261013-001",
  "profile": "parcel-sort",
  "status": "completed",
  "lifecycle_state": "ended_reset",
  "reset_state": "ready",
  "completion_rule": "continuous_work_area_empty_then_reset_success",
  "completion_evidence": {
    "rule": "continuous_work_area_empty_then_reset_success",
    "empty_seconds": 6.021429564,
    "required_empty_seconds": 6.0,
    "observed_state": "NO_PACKAGE",
    "reset_status": "succeeded",
    "physical_execution": true,
    "conveyor_delivery_independently_verified": false
  },
  "error": null,
  "callback": {
    "enabled": false,
    "event_id": null,
    "status": "not_pending",
    "attempts": 0
  }
}
```

`continuous_work_area_empty_then_reset_success` 表示工作区连续空场证据成立，并且机器人复位成功。所需空场时间来自任务配置，读取 `required_empty_seconds`，不要把历史示例的 6 秒写死在调度代码。`conveyor_delivery_independently_verified=false` 表示未通过独立传送带/扫码信号验证送达；接口不返回包裹扫码清单。

### 网页操作员结束调度任务

当前版本将操作员在网页结束已启动的真机调度任务作为业务完成，会出现：

```json
{
  "status": "completed",
  "lifecycle_state": "ended_unreset",
  "reset_state": "not_reset",
  "completion_rule": "operator_end",
  "completion_evidence": {
    "rule": "operator_end",
    "physical_execution": true,
    "reset_status": "pending",
    "conveyor_delivery_independently_verified": false
  }
}
```

这表示操作员确认结束，机器人仍待复位。后续成功复位会更新为 `reset_state=ready`、`lifecycle_state=ended_reset` 和 `completion_evidence.reset_status=succeeded`。复位失败可转为 `recovery_required`。同事若只认可视觉自动完成，应检查 `completion_rule`；不要假设所有 `completed` 都已经复位。

新业务开始前查询 `GET /api/sorting/robot`，仅当 `ready_for_next_task=true` 才尝试派发；POST 仍会原子重新检查准入。

## 4. 任务公共字段

| 字段 | 含义 |
| --- | --- |
| `task_id` / `request_id` | Nexus 任务 ID / 调用方业务编号 |
| `source` | 调度接口创建时为 `scheduler` |
| `profile` / `profile_revision` | 固定方案名称 / 本次任务配置摘要 |
| `status` | 任务进度或结果，调度方的主要判断字段 |
| `lifecycle_state` | `executing`、`ended_unreset`、`ended_reset`、`abnormal`、`simulated` |
| `reset_state` | `ready`、`not_reset`、`resetting`、`unknown` |
| `completion_rule` / `completion_evidence` | 完成依据和证据；未完成时证据可能为空 |
| `error` | 错误说明，正常时通常为 `null` |
| `created_at` / `updated_at` | Unix 秒级时间戳 |
| `revision` | 任务版本，可用于界面状态控制 |
| `deployment_status` | 内部部署状态，不用作业务完成判据 |
| `services` | 模型、视觉服务状态，与业务结果分开读取 |
| `callback` | 是否配置通知、通知状态、事件 ID、发送次数 |
| `allowed_actions` / `unavailable_reasons` | 主要供网页操作员使用；不代表有同名调度写接口 |

## 5. 其他接口及拒绝响应

所有路径都带相同的 Bearer Token。

| 方法与路径 | 用途 |
| --- | --- |
| `GET /api/sorting/health` | 调度服务健康和方案列表；`ok=true` 不代表机器人可接单 |
| `GET /api/sorting/model` | 模型状态；`state=ready`、`profile=parcel-sort` 表示当前部署匹配 |
| `GET /api/sorting/robot` | 机器人、复位状态和 `ready_for_next_task` |
| `POST /api/sorting/tasks/{task_id}/stop` | 取消调度任务，请求体 `{"command_id":"stop-order-20261013-001"}` |
| `POST /api/sorting/robot/reset` | 对当前部署请求待完成的受控复位，请求体 `{"command_id":"reset-order-20261013-001"}`；会执行真机动作 |

停止首次接受返回 202，相同 `command_id` 重试返回 200。停止后通常返回 `cancelled`，仍需查询确认；停止结果未知会要求人工核实。调度停止与上面的网页操作员完成是不同的结果约定。停止不自动复位。

复位接口要求 automatic 模式、execute 方案及当前存在复位要求，返回 202 和机器人状态；相同 command_id 可用于核对原复位。该接口会关闭接单，复位成功后仍需由现场核实并恢复接单；复位成功不等于允许下一业务立即启动。结果用 `/api/sorting/robot` 查询，不要靠 202 判断复位完成。

常规拒绝响应示例：

```json
{"error":"机器人已有任务占用或需要人工核实；请求不会排队","reason_code":"ROBOT_BUSY"}
```

| HTTP / `reason_code` | 处理 |
| --- | --- |
| 400 / `INVALID_REQUEST` | 修正字段或编号格式 |
| 401 | Token 缺失/不一致，或请求错误地携带 Origin |
| 404 | 路径、任务不存在或调度接口未启用 |
| 409 / `MODE_CONFLICT` | 现场尚未切换到自动模式 |
| 409 / `ROBOT_BUSY`、`RESET_REQUIRED` | 占用未解除或复位未完成，请求不会排队 |
| 409 / `IDEMPOTENCY_CONFLICT` | 同一 request_id 或 command_id 的参数冲突 |
| 409 / `STATE_CONFLICT`、`SCHEME_NOT_ACTIVE`、`ROBOT_STATE_UNKNOWN` | 状态/方案不满足或机器人状态未知，按错误说明核实 |
| 503 / `NOT_ACCEPTING`、`NOT_READY`、`SCHEME_SWITCHING`、`PERSISTENCE_FAILED` | 接单关闭、部署未就绪、配置切换或持久化异常 |
| 500 / 网络超时、断线 | 结果可能不确定；保留原业务编号及请求体核实，不换新编号重复启动 |

服务重启后默认关闭接单；未完成的任务可能要求人工恢复。调度方也须持久化自己的业务编号和任务 ID，重启后继续核对原任务。

## 6. 如需 Nexus 主动通知同事

当前 `callback_url` 为空，只有轮询会得到结果。要启用主动通知，同事需提供一个 Nexus 能访问的固定 HTTP(S) POST 接收地址，以及双方约定的回调认证 Token（可选）。管理员在 Nexus 端配置 `callback_url` 和可选 `callback_token_env`；任务 POST 不接受 callback_url。

终态事件请求示例：

```http
POST /同事提供的接收路径
Content-Type: application/json
Idempotency-Key: <event_id>
Authorization: Bearer <回调Token，仅在配置后发送>

{"event_id":"事件ID","event_type":"sorting.completed","task":{"task_id":"任务ID","request_id":"业务编号","status":"completed"}}
```

完整 `task` 为事件生成时的公共任务详情。事件类型为 `sorting.completed`、`sorting.simulated`、`sorting.cancelled`、`sorting.failed` 或 `sorting.recovery_required`，不是每个进度状态都主动通知。

接收方按 `event_id` 去重，处理并返回 2xx；失败或 5 秒内没有成功响应，Nexus 会重试，退避从 2 秒增长至最多 300 秒，event_id 和事件载荷保持一致。重复通知不会重新执行机器人任务。任务结果与通知送达状态分开，通知成功可查到 `callback.status=delivered`。

当前实现为每个任务创建一次终态事件，事件载荷是当时的快照。尤其 `operator_end` 的完成通知可能发生在复位前；后续复位状态变化不会保证产生新的通知。因此即使启用回调，也应按 task_id 查询最新任务，并用 `/api/sorting/robot` 判断下一任务准入。
