# G2A 机器人客户端

本目录是可直接发布到 G2 的独立 Client 工程。完整参数、安全语义和
故障排查见 [G2机器人客户端使用说明.md](G2机器人客户端使用说明.md)。

## 最先看：如何启动 Client

Nexus 当前约定：

| 模型 Server | 有线地址 | 无线回退地址 | 端口 |
| --- | --- | --- | ---: |
| PI0.5 | `10.42.1.200` | `192.168.110.140` | `5557` |
| LingBot-VLA-v2 | `10.42.1.200` | `192.168.110.140` | `5558` |

推荐由 Nexus 任务页的“开始观测”启动 Client。手工启动前先确认 Server
健康，并确认没有其他 GDK Client 占用机器人：

```bash
ssh g2
cd /home/agi/deploy/g2a_robot_client
pgrep -af 'python3? -m [r]obot_client'
```

完全不访问 GDK，只验证网络、协议和模型链路：

```bash
./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5557 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward." \
  --inference-mode=async \
  --effector-profile=dahuan \
  --dry-run \
  --duration-s=30
```

读取真机 joints 和三路图像、执行真实推理，但禁止发送 action：

```bash
./robot_client/run.sh \
  --host=10.42.1.200 \
  --port=5557 \
  --task="Pick a package from the box with the left gripper and place it on the table. Then pick it up with the right gripper and place it on the conveyor belt, keeping the QR code facing upward." \
  --inference-mode=async \
  --effector-profile=dahuan \
  --read-only \
  --duration-s=60
```

使用无线回退网络时，把 `--host` 改为 `192.168.110.140`。使用 LingBot 时把
`--port` 改为 `5558`。终止手工 Client 时在它的终端中按 `Ctrl+C`，
并等待 GDK 独占锁释放。

当前正式配置为大寰双夹爪。`--effector-profile=dahuan` 将模型的
`0=张开、0.025=闭合` action 转换为 GDK 的 `0=闭合、0.025=张开` 物理目标；
转换在 action chunk 安装前完成，100 Hz 滤波、安全守门和动作发送只处理物理坐标。
Nexus 界面启动时会自动注入该参数并与 Server metadata 校验，手工联调时应显式填写。
`omnipicker_legacy` 只用于旧 checkpoint/旧夹爪的明确兼容测试。

## 独立测试大寰夹爪

脚本 `robot_client/test_dahuan_gripper.py` 不包含双臂、头部、腰部、底盘或整机伺服调用。
默认模式只读取左右夹爪状态：

```bash
cd /home/agi/deploy/g2a_robot_client
./robot_client/debug_python.sh -m robot_client.test_dahuan_gripper \
  --action=observe --observe-s=3
```

现场确认夹爪周围无人员和物体后，可执行双夹爪“松开—夹紧—最终松开”测试：

```bash
./robot_client/debug_python.sh -m robot_client.test_dahuan_gripper \
  --action=cycle --side=both --execute \
  --confirmation="TEST DAHUAN GRIPPER CYCLE BOTH"
```

脚本使用 `move_ee_pos/dual_tool/dahuan`，每阶段最多发送 3 次，并以状态反馈连续
3 次到位作为成功条件；循环测试异常时会额外尝试一次最终松开。

`--execute` 会真实下发机器人动作，不是普通启动模式。没有已标定安全
profile、现场急停、低速验收和人工授权时不得使用；正式环境优先从 Nexus
受控界面启动。

## 平台实时仿真隔离

Nexus 注入 `--platform-telemetry-endpoint` 后，Client 使用后台 latest-only
单向 TCP 发布器上传实测状态（旧平台仍可使用 HTTP）。该连接不接收 joint、action
或控制命令。`read-only` 会按遥测频率独立刷新 joints，使仿真不再
依赖推理请求频率；该分支仍然不会进入 `_send_combined_action()`。

`execute` 不会为了仿真增加 GDK 读取、网络等待或动作调用：开启安全限制时
直接复用 100 Hz 安全循环已经取得的实测反馈；网络慢时只覆盖或丢弃旧显示
帧，绝不排队追赶，也不改变 servo、碰撞守门或动作发送频率。平台和仿真器
都只消费遥测，不能通过该通道向机器人回写 joint 或 action。

保留本文件仅用于 Git 托管平台、部署工具和既有链接的兼容入口。
