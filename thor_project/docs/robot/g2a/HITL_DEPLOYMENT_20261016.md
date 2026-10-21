# 2026-10-16 人在环路平板采集部署记录

Thor 的 21865 服务目前运行 `/home/anu/deploy/nexus_deploy_hil_live_20261016_r1`，由 `/home/anu/.config/systemd/user/nexus-thor-control.service.d/50-hitl-tablet-start.conf` 指向。此目录以原在线工程为底本，合入本工程的人在环路功能，并保留 2026-10-16 新增的两份 RTC checkpoint 选项和复位流程。本开发目录 `/home/anu/deploy/nexus_deploy_hil` **不是**当前在线服务的代码目录；后续修改或再部署应先同步到在线候选目录并测试。

G2 客户端部署到 `/home/agi/deploy/g2a_robot_client_hil_tabletstart_20261016_r4`。旧目录未覆盖；本次更新前的 r4 包保存在同级 `.backup_20261016_140853`。Thor 状态仍在原工程的 `runtime/thor`，切换前副本位于候选目录的 `deployment_backups/`。

自动验证：候选版 HITL 与控制相关测试 279 项通过。添加最后 3 项参数化测试前，全量测试（未包含另需 Pi0.5 依赖环境的 6 项）有 1032 项通过、17 项失败，其中 14 项可在原在线工程复现，1 项因候选目录没有 Git 元数据，2 项因模型文件采用指向原在线目录的软链接而与“位于工程内”的测试断言不符。另用 Pi0.5 依赖环境运行的 6 项测试通过。G2 的只读录制监听返回 `ready`；复位姿态检测返回 `not_home`，没有进行机器人动作试验。

现场验收前，确认外部 GDK 任务已停止、机器人复位门禁完成，工作站设为人工模式并允许接单。保存并应用 `human_in_loop=true` 的真机配置；页面显示平板可开始采集后，在平板录制一包。核对页面自动进入任务启动/运行状态、G2 先进入 HOLD，之后松开 X/Y 再按 Y 开启 Policy。保存第一包并核对封存，再在同一 Nexus 部署内录制第二包，确认独立封存。操作说明见 [HITL_TABLET_COLLECTION.md](HITL_TABLET_COLLECTION.md)。

回退 Thor：删除上述 `50-hitl-tablet-start.conf`，执行 `systemctl --user daemon-reload` 和 `systemctl --user restart nexus-thor-control.service`。旧 G2 包与原在线工程均保留。
