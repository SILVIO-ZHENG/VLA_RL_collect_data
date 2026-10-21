# Nexus Simulator 与平台的关系和边界

## 1. 当前状态

Nexus 生产平台已经拥有 `src/nexus_deploy/viewer` 实时只读渲染器。仓库内的
`tools/offline_simulator` 不替换、不启动也不被生产控制面导入；它只承担 Episode 回放、
离线安全对照和实验场景开发。

## 2. 运行形态

Nexus Simulator 保持独立进程和独立端口。当前不建立仿真计划 API，也不把离线回放
混入真机会话状态机。

开发阶段：

```text
Nexus Platform   http://192.168.110.140:8765
Nexus Simulator  http://192.168.110.140:8766
```

未来如需统一入口，可由反向代理只读暴露：

```text
/simulation/*  → Nexus Simulator
```

不建议把 MuJoCo 循环放入当前控制 API 进程；仿真负载、崩溃或内存泄漏不能影响部署控制台。

## 3. 历史接口草案（未启用）

以下内容只保留为历史设计草案，不代表当前平台接口。曾计划在部署配置页增加：

```text
在仿真中验证
```

它创建仿真计划并跳到独立页面。该按钮不能与“开始执行任务”共用颜色、权限或后端接口。

## 4. 计划创建接口草案

```http
POST /api/v1/simulations
Content-Type: application/json
```

```json
{
  "robot_model": "g2_omnipicker",
  "scenario": "table_sorting_basic",
  "work_surface": {
    "height": 0.75,
    "hard_margin": 0.01,
    "xy_bounds": null
  },
  "guard": {
    "path_samples": 2,
    "deadline_ms": 8.0,
    "collision_response": "hold_previous_safe_action"
  },
  "action_source": {
    "type": "fixture",
    "id": "approach_surface_v1"
  }
}
```

响应：

```json
{
  "id": "sim_...",
  "status": "ready",
  "simulation_only": true,
  "viewer_url": "/simulations/sim_..."
}
```

接口不接受 robot instance、IP、SSH、server mode 或 hardware mode。

## 5. 会话状态

```text
created → loading_assets → ready → playing ↔ paused → completed
                      └──────────────→ failed
```

状态只描述仿真。不要复用 Nexus Platform 的真机任务 `running/paused` 对象，以免误操作。

## 6. 报告回传

仿真器返回摘要：

- 配置哈希
- URDF/mesh 版本
- 安全核心版本
- 总动作数、通过数、拒绝数
- 保持上一步次数、保持姿态次数
- guard mean/P95/P99/max
- deadline miss 数
- 最小工作面距离
- 可复现场景 ID

Nexus Platform 只保存摘要和报告路径，不把“仿真通过”转换成“允许真机执行”。

## 7. 认证与部署

- 开发阶段可仅监听 loopback，通过 SSH tunnel 使用。
- 局域网开放时复用 Nexus Platform 登录或由反向代理统一认证。
- WebSocket 校验 Origin 和会话 token。
- 不把 `admin/admin` 写进新工程源码；认证策略由部署环境注入。
- 服务应使用独立 systemd unit、日志目录和资源限制。

## 8. 实时关节与安全遥测

真机接入时，不把三维渲染塞进 `nexus-policy-websocket` 推理请求。G2 Client
继续在 100 Hz servo 链路内同步完成：

```text
实测 joints → 候选 action → 工作面安全守门 → 采用/保持动作 → GDK
```

遥测使用独立 latest-only 通道，以 10–20 Hz 发送小数据摘要：

- 24D `measured_joints`；
- 可选 `candidate_action` 与 `adopted_action`；
- `collision_detected`、拦截原因、最小净空和 guard 耗时；
- session ID、单调时钟和递增 sequence。

浏览器默认跟随视频解码帧以 30 FPS 更新。慢消费者丢弃旧显示帧，不阻塞
Robot Client；遥测超过 500 ms 显示延迟警告，超过 2 s 明确标记“最后一帧”。

## 9. 页面与参数边界

- 机器人型号、URDF SHA、安全 profile、工作面高度和硬安全余量在模型加载前
  确定，并冻结到作业快照；
- 模型加载后的任务页只读展示快照和实时结果；
- 仿真页的“安全限制”开关只用于对比拦截效果，不能直接复用为生产真机开关；
- execute 中修改安全边界必须先软暂停、保持实测姿态、重新预检并创建新 session；
- 三维面板不新增第二套暂停/继续/结束按钮，继续复用 Nexus Platform 任务状态机。

## 10. 视角与渲染器迁移

当前只保留已验收的“双臂工作区”视角。原全景、左夹爪和右夹爪特写会因
MeshCat 重建 camera 后 OrbitControls target 回到世界原点而错位，已从页面和公开
API 移除。

其他视角只能在下列验收后恢复：

1. 同时原子更新 camera pose 和 OrbitControls target；
2. 夹爪 target 每次从当前 FK 世界变换计算；
3. 在 Episode 第 0 帧、中间帧和最大伸展帧检查出画、桌面遮挡和近裁剪；
4. 预设后手动旋转、平移和缩放仍正常。

生产化阶段建议使用 Three.js/WebGL + 带 SHA 的 GLB 资产取代跨端口 MeshCat
iframe，并对高面数夹爪生成近/中/远 LOD。

## 11. 迁移顺序

1. 冻结 URDF/mesh、数据集、安全核心、正确视角和性能基线；
2. 增加仿真计划 API、报告和 Platform 反向代理；
3. 先用记录数据完成端到端集成；
4. 再接入 G2 read-only 的 measured joints；
5. 最后才增加 execute 的 candidate/adopted 安全决策可视化。

任何阶段都必须保证 Simulator 停止、页面关闭或遥测断开不影响模型服务、
Robot Client 和 100 Hz 安全守门。

完整的跨工程方案位于 Nexus Platform：

```text
/data/company/data_common/lxl/workspace/g2a_deploy_project/docs/Nexus仿真可视化集成迁移方案.md
```
