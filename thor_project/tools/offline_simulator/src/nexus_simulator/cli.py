"""Nexus Simulator 管理命令。当前只提供无硬件副作用的 doctor。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nexus_simulator.assets import AssetError, sync_assets, verify_assets
from nexus_simulator.config import SimulationConfigError, load_simulation_config
from nexus_simulator.model_variants import LEGACY_MODEL_ID, model_variant_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="校验仿真配置和安全隔离标记")
    doctor.add_argument(
        "--config",
        default="configs/g2_work_surface.simulation.json",
        help="仿真配置文件",
    )
    doctor.add_argument("--project-root", default=".", help="Nexus Simulator 工程根目录")
    sync = commands.add_parser("sync-assets", help="从权威源码同步 G2 模型资产")
    sync.add_argument("--project-root", default=".", help="Nexus Simulator 工程根目录")
    sync.add_argument("--source-root", default=None, help="覆盖清单中的 G2 资产源目录")
    verify = commands.add_parser("verify-assets", help="校验本地 G2 模型资产哈希")
    verify.add_argument("--project-root", default=".", help="Nexus Simulator 工程根目录")
    serve = commands.add_parser("serve", help="启动阶段 1 浏览器可视化服务")
    serve.add_argument("--project-root", default=".", help="Nexus Simulator 工程根目录")
    serve.add_argument("--config", default="configs/g2_work_surface.simulation.json")
    serve.add_argument("--host", default="127.0.0.1", help="控制页面监听地址")
    serve.add_argument("--port", type=int, default=8766, help="控制页面端口")
    serve.add_argument("--meshcat-port", type=int, default=8767, help="三维画面端口")
    serve.add_argument(
        "--dataset-root",
        required=True,
        help="LeRobot v3 数据集根目录（只读）",
    )
    serve.add_argument("--episode", type=int, default=0, help="启动时加载的 Episode")
    serve.add_argument(
        "--robot-model",
        choices=model_variant_ids(),
        default=LEGACY_MODEL_ID,
        help="受控的 G2 可视模型版本；默认保留已验证旧模型",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sync-assets":
        try:
            result = sync_assets(
                Path(args.project_root),
                Path(args.source_root) if args.source_root is not None else None,
            )
        except AssetError as exc:
            print(f"Nexus Simulator asset sync: FAILED: {exc}")
            return 2
        print(f"Nexus Simulator asset sync: OK: {result['file_count']} files")
        return 0
    if args.command == "verify-assets":
        try:
            result = verify_assets(Path(args.project_root))
        except AssetError as exc:
            print(f"Nexus Simulator asset verify: FAILED: {exc}")
            return 2
        print(f"Nexus Simulator asset verify: OK: {result['file_count']} files")
        return 0
    if args.command == "serve":
        from nexus_simulator.server import serve

        return serve(
            project_root=Path(args.project_root),
            config_path=Path(args.config),
            host=args.host,
            port=args.port,
            meshcat_port=args.meshcat_port,
            dataset_root=Path(args.dataset_root),
            episode_index=args.episode,
            robot_model=args.robot_model,
        )
    if args.command != "doctor":
        raise AssertionError("未知命令")
    try:
        config = load_simulation_config(args.config)
    except SimulationConfigError as exc:
        print(f"Nexus Simulator doctor: FAILED: {exc}")
        return 2
    print("Nexus Simulator doctor: OK")
    print(f"profile={config.profile}")
    print("simulation_only=true calibrated=false deployable=false")
    print("hardware_access=disabled gdk_import=forbidden robot_commands=disabled")
    try:
        assets = verify_assets(Path(args.project_root))
    except AssetError as exc:
        print(f"assets=not_ready detail={exc}")
    else:
        print(f"assets=verified files={assets['file_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
