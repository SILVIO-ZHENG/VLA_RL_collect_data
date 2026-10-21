"""Conservative generic robot client; physical execution is opt-in."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from .contracts import ContractError, PolicyCapabilities
from .plugins import RobotAdapter, load_plugin
from .wire import DEFAULT_MAX_BYTES, pack_message, unpack_message


def ensure_compatible(robot: RobotAdapter, policy: PolicyCapabilities) -> None:
    if robot.state_spec.names != policy.state.names:
        raise ContractError(
            "Robot/model state schema mismatch: "
            f"robot={robot.state_spec.names}, model={policy.state.names}"
        )
    if robot.action_spec.names != policy.action.names:
        raise ContractError(
            "Robot/model action schema mismatch: "
            f"robot={robot.action_spec.names}, model={policy.action.names}"
        )
    missing = [item.name for item in policy.images if item.required and item.name not in robot.camera_names]
    if missing:
        raise ContractError(f"Robot does not provide required cameras: {missing}")


def _load_json(value: str) -> dict[str, Any]:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def run(
    robot: RobotAdapter,
    *,
    uri: str,
    task: str,
    duration_s: float,
    execute: bool,
    mode: str,
) -> None:
    try:
        from websockets.sync.client import connect
    except ImportError as error:
        raise RuntimeError("Install the 'client' dependency group: pip install -e '.[client]'") from error
    session_id = uuid.uuid4().hex
    request_number = 0
    robot.connect()
    try:
        with connect(uri, compression=None, max_size=DEFAULT_MAX_BYTES) as socket:
            capabilities = PolicyCapabilities.from_metadata(unpack_message(socket.recv()))
            ensure_compatible(robot, capabilities)
            if mode not in capabilities.modes:
                raise ContractError(f"Mode {mode!r} is unsupported by {capabilities.model_id}")
            deadline = time.monotonic() + duration_s if duration_s > 0 else float("inf")
            while time.monotonic() < deadline:
                request_number += 1
                request_id = f"{session_id}-{request_number:08d}"
                observation = robot.observe(task)
                capabilities.state.validate(observation.state)
                message = {
                    "type": "inference_request",
                    "request_id": request_id,
                    "session_id": session_id,
                    "mode": mode,
                    "reset": request_number == 1,
                    "observation": observation.to_wire(),
                }
                socket.send(pack_message(message))
                response = unpack_message(socket.recv())
                if response.get("status") != "ok":
                    raise RuntimeError(str(response.get("error", "policy request failed")))
                actions = capabilities.action.validate(response.get("actions"), rows=True)
                if execute:
                    for action in actions:
                        robot.execute(action)
                        time.sleep(1.0 / capabilities.model_fps)
                else:
                    print(f"dry-run request={request_id} actions={actions.shape}")
                    time.sleep(1.0 / capabilities.model_fps)
    finally:
        robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-adapter", required=True, help="plugin as package.module:Class")
    parser.add_argument("--robot-config", default="{}", help="JSON or @path")
    parser.add_argument("--uri", required=True, help="ws://server:port")
    parser.add_argument("--task", required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--mode", choices=("sync", "async", "rtc"), default="async")
    parser.add_argument("--execute", action="store_true", help="send actions to hardware")
    args = parser.parse_args()
    robot = load_plugin(args.robot_adapter, _load_json(args.robot_config))
    if not isinstance(robot, RobotAdapter):
        raise TypeError(f"{args.robot_adapter} must produce a RobotAdapter")
    run(
        robot,
        uri=args.uri,
        task=args.task,
        duration_s=args.duration_s,
        execute=args.execute,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
