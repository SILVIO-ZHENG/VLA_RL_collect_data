"""Tests for the no-GDK Nexus presence sidecar boundary.

The sidecar must only post observations to the control plane.  These tests use
an in-process HTTP recorder; they do not open a model, camera, GDK, SSH session,
or robot-control channel.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARCEL_ROOT = PROJECT_ROOT / "src" / "nexus_deploy" / "vision" / "parcel"

from nexus_deploy.vision.parcel.runtime.nexus_frame_stream import HeadFramePacket  # noqa: E402


def load_sidecar():
    path = PARCEL_ROOT / "run_nexus_presence_sidecar.py"
    spec = importlib.util.spec_from_file_location("staged_nexus_presence_sidecar", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SIDECAR = load_sidecar()
JOB_ID = "0123456789abcdef0123456789abcdef"
TOKEN = "sidecar-control-token-0123456789"


class _Recorder(BaseHTTPRequestHandler):
    received: list[tuple[str, dict[str, str], dict]] = []

    def log_message(self, *_args):  # noqa: D401 - suppress test server noise
        """Suppress normal HTTP server logs during tests."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.received.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")


class NexusPresenceSidecarTests(unittest.TestCase):
    def setUp(self):
        _Recorder.received = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1.0)

    def test_publisher_posts_only_bounded_observation_with_separate_token(self):
        self.enterContext(mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:9",
                                                      "http_proxy": "http://127.0.0.1:9",
                                                      "NO_PROXY": "", "no_proxy": ""}))
        host, port = self.server.server_address
        publisher = SIDECAR.ControlPlanePresencePublisher(
            f"http://{host}:{port}/api/deployments/{JOB_ID}/presence-gate",
            TOKEN,
        )
        packet = HeadFramePacket(
            job_id=JOB_ID,
            generation=3,
            sequence=12,
            capture_timestamp_ns=100,
            received_timestamp_ns=200,
            jpeg=b"\xff\xd8jpeg",
        )
        presence = SIDECAR._unknown_presence("controlled test failure")

        self.assertTrue(publisher.publish(packet=packet, presence=presence, stage=None))
        self.assertEqual(len(_Recorder.received), 1)
        path, headers, payload = _Recorder.received[0]
        self.assertEqual(path, f"/api/deployments/{JOB_ID}/presence-gate")
        self.assertEqual(headers["X-Nexus-Presence-Token"], TOKEN)
        self.assertEqual(payload["generation"], 3)
        self.assertEqual(payload["sequence"], 12)
        self.assertEqual(payload["presence"]["state"], "UNKNOWN")
        self.assertEqual(payload["presence"]["observed_at_ns"], 200)
        self.assertNotIn("decision", payload)
        self.assertNotIn("robot_command", payload)

    def test_unknown_result_is_not_a_control_permission(self):
        payload = SIDECAR._unknown_presence("camera unavailable")
        self.assertEqual(payload["state"], "UNKNOWN")
        self.assertEqual(payload["window_states"], [])
        self.assertNotIn("action", payload)

    def test_control_url_rejects_credentials_and_queries(self):
        with self.assertRaises(ValueError):
            SIDECAR._validate_control_url("http://user:secret@127.0.0.1:8765")
        with self.assertRaises(ValueError):
            SIDECAR._validate_control_url("http://127.0.0.1:8765/path?token=secret")


if __name__ == "__main__":
    unittest.main()
