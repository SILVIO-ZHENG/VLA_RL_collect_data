"""Offline handoff comparison: fake clock/command sink, real FK and limits.

No SSH, model service, GDK initialization, or physical robot command is used.
Run with the project's pytest/numpy-capable development Python environment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Frozen robot_client/gdk_adapter.py")
    parser.add_argument("--output", type=Path, help="New JSON report; refuses overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    root = Path(__file__).resolve().parents[1]
    suite = load_module("nexus_handoff_suite", root / "tests/test_safety_layer.py")
    versions = {"candidate": suite.gdk_adapter.G2AAdapter}
    if args.baseline is not None:
        baseline = load_module("nexus_deploy.robot.g2a._handoff_baseline", args.baseline)
        versions = {"baseline": baseline.G2AAdapter, **versions}
    logging.disable(logging.CRITICAL)
    rows = []
    for version, adapter_type in versions.items():
        for fps in (15, 30):
            with tempfile.TemporaryDirectory(prefix="nexus-handoff-") as directory:
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(suite.gdk_adapter, "G2AAdapter", adapter_type)
                    result = suite.run_async_jitter_schedule(Path(directory), patch, policy_fps=fps)
                    rows.append(dict(version=version, policy_fps=fps,
                                     **{k: v for k, v in result.items() if k != "records"}))
    report = dict(simulation_only=True, physical_commands=False, actual_log_replay=False,
                  description="Same synthetic latency/jitter/30ms feedback-lag schedule; real FK and pacing checks",
                  baseline=None if args.baseline is None else str(args.baseline.resolve()), results=rows)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
