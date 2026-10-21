# 协同平滑轨迹实验室（仅离线）

这是 Nexus 部署项目内的隔离原型，不是新的真机控制器。没有被平台、机器人客户端、启动脚本或部署包导入；不连接 GDK，不发送动作，不修改原来的 100 Hz 控制、50 Hz 碰撞配置或 RTC 行为。

## 文件

- `trajectory.py`：带共享时间轴的分段五次轨迹、前瞻拟合、导数/路径偏差边界、夹爪事件映射。
- `executor.py`：用于单元测试的确定性队列模型，模拟已检查前缀、采用进度、响应过期、暂停、拦截保持和新段接续。
- `replay.py`：读取已验证的 30 Hz DH 16D 数据集，在本地生成对照 JSON/CSV，可选择输出大体积 NPZ。
- `dataset_limits.py`：分别统计训练集动作/反馈的逐关节导数，只生成离线实验参数。
- `profiles/dh_30hz_20260914.offline.json`：本轮训练数据参考值，**不是平台或机器人配置**。
- `tests/test_coordinated_trajectory_lab.py`：项目根目录下的专项测试。

## 运行

在项目根目录执行（该环境已有 NumPy、PyArrow）：

```bash
/home/luoxiaolong/miniforge3/envs/lerobot/bin/python -m tools.trajectory_lab.replay \
  --dataset /home/luoxiaolong/workspace/datasets/logistics_g2a_dh_20260818_20260824_merged_v30_16d \
  --episodes 0,50,150,250,350,400 \
  --fps 30 15 \
  --jerk 10 \
  --smoothing-window 9 \
  --knot-tolerance 0.003 \
  --legacy-client src/nexus_deploy/robot/g2a/action_filter.py \
  --output outputs/diagnostics/my-coordinated-replay
```

输出目录必须不存在，防止覆盖已有结果。需要逐点离线回放数据时增加 `--save-samples`；这些 NPZ **不是可以直接下发给机器人的动作包**。

```bash
.venv/bin/python -m pytest -q tests/test_coordinated_trajectory_lab.py
```

## 明确边界

- 实验 jerk=10 rad/s³、身体拟合容差和实验位置范围不是批准的真机参数。
- `Trajectory.bounds()` 是数学导数边界，不是碰撞检查结果。
- `OfflineExecutor` 的 checker 是本地同步回调，仅模拟“先检查最终输出再采用”。它没有真实异步进程、几何证书有效期、GDK 回执或机器人反馈；不能把 `tick()` 接到 SDK。
- `RequestAnchor` 区分已采用进度与冻结前缀边界，**不是当前模型服务器支持的 RTC 协议**。冻结边界变化时明确拒绝旧响应，真实系统仍需实现响应重定位和前缀转换。
- 碰撞/暂停/结束保持会中断运动，平滑的导数保证不覆盖这种紧急保持跳变或物理刹车过程。
- 全行程夹爪碰撞包络、真实身体限位及 G2 上的端到端 100 Hz 验证，尚未接入这个原型。

完整结果及接入清单见 `docs/G2协同平滑轨迹离线原型与验证-20260914.md`。

## 数据参考参数与新版回放

新版对照增加：

```bash
--limits-profile tools/trajectory_lab/profiles/dh_30hz_20260914.offline.json --tangent-method auto
```

该参数文件覆盖统一速度/加速度/jerk。`auto` 按简化三次引导、完整节点三次引导、谐和方法尝试，所有候选保留相同导数和路径限制，失败不会强行放行。仅回放默认使用 `auto`，`plan()` 默认保留谐和方法；时间扩展使用阻尼更新，因此旧命令不再逐数值复现上一轮历史报告。

逐关节参数与上一轮未通过案例见 [数据参考优化说明](../../docs/G2基于训练数据的关节限速与平滑优化-20260914.md)。后续已用局部路径加密修复这 3 个 episode，全部 56 个案例通过数学约束，仍不能作为真机发布版本。

回放默认 `--path-refinements 6`，仅在偏差超限区间增加内部插值节点；不删除源动作、不新增夹爪事件、不放宽路径或导数限制。加密后重新计时，边界计算同时覆盖原始和新增节点。使用 `--path-refinements 0` 可关闭该功能。底层 `plan()` 默认保持关闭，`plan_recorded()` 默认启用。

最新证据与接入边界见 [起步与快速变化段路径修复](../../docs/G2起步与快速变化段路径修复-20260914.md)。

## 滚动接续实验

`rolling.py` 提供未来参考交接点；它不是碰撞已检查队列，也没有接入真实 RTC 协议。`rolling_replay.py` 使用数据集 50 帧窗口和历史响应延迟，在虚拟 100 Hz 时钟中验证接续；可选加载现有安全模块做本机影子几何检查，不导入 GDK。

运动中非零导数边界会尝试 3/6 源帧跨度的身体短过渡，保留完整原始路径约束和夹爪事件。目前长片段仍有规划拒绝，生产控制不变。详见 [滚动轨迹交接与碰撞影子验证](../../docs/G2滚动轨迹交接与碰撞影子验证-20260914.md)。

### 可选的运动边界凸优化（2026-09-15）

滚动回放可增加 `--constrained-fallback`，在原有快速方法全部失败后，使用 SciPy/HiGHS 联合求解短窗口身体 q/v/a。保持原始路径偏差预算和夹爪事件，两个候选共用 350 ms 后备预算；这不是 100 Hz 回调，默认关闭且没有接入生产。

目前降低了规划拒绝，但仍有结果晚到；`--adaptive-lead` 的试验未改善晚到，不能作为默认建议。详细约束、失败结果和复现命令见 [运动中交接约束求解优化](../../docs/G2运动中交接约束求解优化-20260915.md)。

滚动实验同时允许在旧参考耗尽后从其已验证的静止端点重新预约，避免连续晚到后永远无法接续。此功能不解除碰撞/暂停保持、不外推运动，实际采用样本仍经过 checker。扩大到 20 秒的模板几何影子检查出现危险点，不能将规划可行率当作碰撞通过率。
