"""Nexus 内置的 G2 实时三维可视化服务。

该包不包含机器人通信或控制能力；它只接收平台转发的最新关节状态，
并把 URDF 正向运动学结果发送给浏览器中的 MeshCat。渲染子进程与平台
共享同一 Python 环境，但保持独立进程故障边界。
"""

from .service import INTEGRATED_SIMULATOR_COMPONENT, SimulatorProcess

__all__ = ["INTEGRATED_SIMULATOR_COMPONENT", "SimulatorProcess"]
