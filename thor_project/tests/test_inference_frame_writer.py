from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

CLIENT_SOURCE = Path(__file__).resolve().parents[1] / "src/nexus_deploy/robot/g2a"
SPEC = importlib.util.spec_from_file_location(
    "nexus_robot_telemetry_test", CLIENT_SOURCE / "telemetry.py"
)
assert SPEC is not None and SPEC.loader is not None
telemetry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = telemetry
SPEC.loader.exec_module(telemetry)


def test_latest_inference_frame_writer_publishes_complete_three_camera_frame(
    tmp_path: Path,
) -> None:
    writer = telemetry.LatestInferenceFrameWriter(tmp_path)
    jpeg = b"\xff\xd8test-jpeg\xff\xd9"
    images = {
        camera: {"encoding": "jpeg", "data": jpeg}
        for camera in telemetry.INFERENCE_CAMERA_NAMES
    }
    writer.start()
    writer.submit("request-12", images, captured_at=123.5)
    deadline = time.monotonic() + 1.0
    while not (tmp_path / "latest.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    writer.close()

    manifest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert manifest["request_id"] == "request-12"
    assert manifest["captured_at"] == 123.5
    assert set(manifest["images"]) == set(telemetry.INFERENCE_CAMERA_NAMES)
    for metadata in manifest["images"].values():
        assert (tmp_path / metadata["path"]).read_bytes() == jpeg
