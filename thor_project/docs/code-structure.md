# 代码结构

仓库使用标准 `src` 布局，所有产品代码都属于唯一顶层包 `nexus_deploy`。配置、测试、
运维脚本和离线工具与产品包分开，运行时不再从旧工程目录导入代码。

```text
nexus_deploy/
├── configs/                         # 5090、Thor 与测试站点配置
├── docs/                            # 架构、环境、视觉、VLA 与机器人说明
├── models/                          # 本地模型资产；仅 README 纳入 Git
├── scripts/                         # 环境构建和明确的运维入口
├── src/nexus_deploy/
│   ├── control/                     # 任务准入、HTTP、遥测和适配器边界
│   ├── robot/g2a/                   # 可独立打包到 G2 的机器人客户端
│   ├── vision/parcel/               # Sidecar、录像工具、ROI 和规则核心
│   ├── vla/pi05/                    # PI0.5 WebSocket 服务
│   ├── alignment/                   # 工作面标定服务与机器人只读代理
│   ├── viewer/                      # 生产会话的只读三维查看器
│   ├── deployment.py                # 模型与机器人部署适配
│   ├── control_api.py               # ControlPlane 和兼容 HTTP 入口
│   ├── presence_gate.py             # 唯一视觉门控策略
│   └── sorting_api.py               # 调度认证、查询与结果通知
├── tests/                            # 全部产品回归，包括 tests/vision
├── third_party/lerobot/              # 固定补丁、依赖锁和忽略提交的 source
└── tools/
    ├── offline_simulator/           # 与机器人隔离的独立回放应用
    ├── trajectory_lab/              # 轨迹算法实验
    └── pi05/                         # PI0.5 日志与延迟分析工具
```

## 模块边界

- `control/tasks.py` 是模式、接单、原子准入和 `TaskView` 的公共用例层。界面与调度 API
  都调用它；仓库没有业务等待队列。
- `control/vision.py` 定义真实 Sidecar、Mock 和录像回放适配器。视觉只产生观察，不能
  调用机器人命令。
- `control/robot.py` 是窄机器人命令端口，实际 SSH/GDK 安全流程仍由 `deployment.py`
  和 `robot/g2a` 实现。
- `presence_gate.py` 是连续空场、断流和代次判断的唯一权威策略。视觉包中的
  `presence_control.py` 只保留兼容导出。
- `viewer` 只读取遥测；`tools/offline_simulator` 是独立进程，两者都不拥有机器人控制权。

`scripts/` 只保留环境构建、网络和明确的运维入口。可复用分析程序放 `tools/`，说明材料
放 `docs/`。模型权重、虚拟环境、运行状态和生成缓存均被 Git 忽略。
