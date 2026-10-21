# Nexus Simulator

Nexus Simulator 是 Nexus 仓库内的独立离线安全仿真与可视化工具，位于
`tools/offline_simulator`。它在不连接机器人、不导入 GDK、不发送机器人指令的前提下，
验证 G2 的 URDF、关节映射、工作面安全约束、动作轨迹和安全层行为。

当前状态：**阶段 2 数据集回放原型已可运行**。已经实现 G2 模型资产校验、DAE 内嵌材质配色、URDF/FK、LeRobot v3 Episode 0 回放、头部相机同步、单一工作面调节，以及固定哈希导入的真机同源安全参考判定；尚未引入 MuJoCo 动力学。

## 与生产仿真器的边界

- `src/nexus_deploy/viewer` 是生产会话使用的实时、只读渲染器；
- `tools/offline_simulator` 负责数据集 Episode 回放、离线安全对照和实验场景；
- 工具与 Nexus 平台共用根目录 `.venv`，但作为独立进程运行，不进入控制面、模型 Server 或机器人 Client 进程；
- 工具不具备机器人 IP、SSH、GDK 或动作下发接口。

## 访问方式

仿真计算运行在服务器，本地浏览器通过 HTTP/WebSocket 查看三维画面：

```text
服务器：MuJoCo + Nexus 安全计算 + MeshCat
                         │
                    HTTP/WebSocket
                         │
本地浏览器：http://<服务器IP>:8766
```

阶段 1 的控制页和 MeshCat 画面分别使用 8766、8767。无法直接访问服务器端口时同时转发：

```bash
ssh -L 8766:127.0.0.1:8766 -L 8767:127.0.0.1:8767 <服务器SSH地址>
```

然后访问 `http://127.0.0.1:8766`。

## 环境与安装说明

从 Nexus 仓库根目录执行：

```bash
uv sync --extra server --extra dev --extra offline-simulator
./scripts/nexus-sim sync-assets
```

本工具是根目录 `uv` workspace 的成员：统一使用 `<Nexus仓库>/.venv` 和根目录
`uv.lock`，不再创建 `tools/offline_simulator/.venv` 或维护第二份锁文件。工具自己的
`pyproject.toml` 仍然保留，用于说明包边界、viewer 依赖以及未来可选的 MuJoCo 依赖。

`sync-assets` 从当前仓库的 `dist/g2a_robot_client` 复制权威 URDF/mesh 到工具的
Git 忽略目录，并生成 SHA-256 清单；不会反向修改生产客户端或内置仿真器。

## 启动回放界面

只允许本机/SSH 转发访问：

```bash
./scripts/nexus-sim serve \
  --dataset-root /path/to/lerobot-dataset \
  --host 127.0.0.1 --port 8766 --meshcat-port 8767
```

局域网内访问：

```bash
./scripts/nexus-sim serve \
  --dataset-root /path/to/lerobot-dataset \
  --host 192.168.110.140 --port 8766 --meshcat-port 8767
```

打开 `http://192.168.110.140:8766`。控制页和 MeshCat 画面当前使用两个端口。

默认仍加载已验证的 `omnipicker-legacy` 可视模型。DH 夹爪模型直接读取
`src/nexus_deploy/viewer/assets/g2/G2_dahuan` 中的 Nexus 权威版本，不再维护工具专用副本：

```bash
./scripts/nexus-sim serve \
  --dataset-root /path/to/lerobot-dataset \
  --host 192.168.110.140 --port 8770 --meshcat-port 8771 \
  --robot-model G2_dahuan
```

打开 `http://192.168.110.140:8770`。该变体已支持 Episode 旧夹爪角度到 DH 夹爪直线位移的显示映射；outer joint 由权威 URDF 的 mimic 关系驱动。DH URDF 尚未与生产安全碰撞几何重新标定，预览期间碰撞判定仍明确使用已验证的旧版安全参考，不能据此宣称新模型的几何安全边界已完成验证。

新 DH 数据集已经使用 `0~0.025 m` 原生夹钳值，而且 observation 与 action 极性相反。
仿真会从转换清单自动识别该协议。可用以下命令启动 Episode 0：

```bash
./scripts/nexus-sim serve \
  --host 0.0.0.0 --port 8772 --meshcat-port 8773 \
  --robot-model G2_dahuan \
  --dataset-root /path/to/logistics-g2a-dh-dataset \
  --episode 0
```

数据集包含 Episode `0~427`，修改 `--episode` 后重启即可选择其他 Episode。

数据集路径必须通过 `--dataset-root` 显式提供，避免工具绑定开发机目录。使用
`--episode` 切换 Episode。原始轨迹为 30 FPS；浏览器使用视频解码帧作为时钟，
同步发送三维与安全判定请求。仿真工作面默认高度为 `0.75 m`。

SSH 转发需要同时转发两个端口：

```bash
ssh -L 8766:127.0.0.1:8766 -L 8767:127.0.0.1:8767 <服务器SSH地址>
```

然后打开 `http://127.0.0.1:8766`。

## 检查与测试

```bash
./scripts/nexus-sim doctor \
  --config configs/g2_work_surface.simulation.json
./scripts/nexus-sim verify-assets
uv run --extra offline-simulator ruff check tools/offline_simulator
uv run --extra offline-simulator pytest -q tools/offline_simulator/tests
```

这些命令不导入 GDK、不连接机器人。

## 文档导航

1. [总体方案](docs/01-总体方案.md)
2. [系统架构与通信流程](docs/02-系统架构与通信流程.md)
3. [分阶段实施与验收](docs/03-分阶段实施与验收.md)
4. [安全隔离与真机边界](docs/04-安全隔离与真机边界.md)
5. [Nexus 平台关系与边界](docs/05-Nexus平台关系与边界.md)
6. [配置与数据协议](docs/06-配置与数据协议.md)
7. [决策记录](docs/07-决策记录.md)
8. [现有安全基线与仿真对照](docs/08-现有安全基线与仿真对照.md)
9. [阶段 1 实现记录](docs/09-阶段1实现记录.md)
10. [阶段 2 安全轨迹可视化方案](docs/10-阶段2安全轨迹可视化方案.md)
11. [数据集 Episode 回放实现记录](docs/11-数据集Episode回放实现记录.md)
12. [新 G2 完整夹钳 URDF 检查与切换](docs/12-新G2完整夹钳URDF检查与切换.md)

## 已确定的原则

- 生产界面只配置一个通用“工作面高度”，不区分桌面和传送带。
- 仿真配置必须保持 `simulation_only=true`、`calibrated=false`、`deployable=false`。
- 仿真代码不得导入 GDK，不得提供 SSH、机器人 IP 或动作下发入口。
- 数据集保持原始 30 FPS 时间语义，浏览器三维与参考判定跟随实际解码帧以 30 FPS 更新；完整离线扫描和 100 Hz 性能基准独立运行。
- 碰撞风险默认拒绝候选动作，优先保持上一步经过重新校验的安全动作，否则保持当前测量姿态。
- 碰撞检测始终开启；运行时可关闭“安全限制执行”。关闭后危险候选动作不再被拦截，但页面和后台必须继续给出醒目的红色 WARNING。

## 当前明确未实现

- MuJoCo 动力学、接触和箱子场景。
- 100 Hz 安全循环与 30 FPS 最新帧广播。
- 离线报告导出与场景资产规范化。
- 任何真机或 GDK 能力。
