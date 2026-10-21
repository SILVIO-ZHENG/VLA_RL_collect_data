"""Contract tests for the isolated robot-to-Thor head-frame transport.

These tests deliberately use no GDK, CUDA, YOLO model, or robot command.  They
prove only that the normal robot client can offer its existing ``cam_head`` JPEG
to the authenticated latest-frame ingress without queueing old frames.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARCEL_ROOT = PROJECT_ROOT / "src" / "nexus_deploy" / "vision" / "parcel"
CLIENT_STREAM = PROJECT_ROOT / "src" / "nexus_deploy" / "robot" / "g2a" / "presence_stream.py"
INGRESS_STREAM = PARCEL_ROOT / "runtime" / "nexus_frame_stream.py"
JOB_ID = "0123456789abcdef0123456789abcdef"
TOKEN = "test-presence-ingress-token-0123456789"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CLIENT = load_module("test_client_presence_stream", CLIENT_STREAM)
INGRESS = load_module("staged_nexus_frame_stream", INGRESS_STREAM)


class NexusFrameStreamTests(unittest.TestCase):
    def make_server(self):
        return INGRESS.PresenceFrameIngressServer(
            ("127.0.0.1", 0),
            job_id=JOB_ID,
            generation=1,
            ingress_token=TOKEN,
        )

    def test_client_packet_is_accepted_by_thor_ingress(self):
        server = self.make_server()
        try:
            encoded = CLIENT.encode_head_frame_packet(
                job_id=JOB_ID,
                generation=1,
                sequence=7,
                capture_timestamp_ns=123_456_789,
                jpeg=b"\xff\xd8tiny-test-jpeg",
                ingress_token=TOKEN,
            )
            self.assertTrue(server.ingest_payload(encoded[4:]))
            packet = server.mailbox.take(timeout_s=0.01)
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.job_id, JOB_ID)
            self.assertEqual(packet.generation, 1)
            self.assertEqual(packet.sequence, 7)
            self.assertEqual(packet.capture_timestamp_ns, 123_456_789)
            self.assertEqual(packet.jpeg, b"\xff\xd8tiny-test-jpeg")
            self.assertIsNotNone(packet.received_timestamp_ns)
        finally:
            server.server_close()

    def test_wrong_token_and_old_generation_are_rejected(self):
        server = self.make_server()
        try:
            wrong_token = CLIENT.encode_head_frame_packet(
                job_id=JOB_ID,
                generation=1,
                sequence=1,
                capture_timestamp_ns=1,
                jpeg=b"\xff\xd8bad-token",
                ingress_token="not-the-right-token-0123456789",
            )
            old_generation = CLIENT.encode_head_frame_packet(
                job_id=JOB_ID,
                generation=2,
                sequence=2,
                capture_timestamp_ns=2,
                jpeg=b"\xff\xd8old-generation",
                ingress_token=TOKEN,
            )
            self.assertFalse(server.ingest_payload(wrong_token[4:]))
            self.assertFalse(server.ingest_payload(old_generation[4:]))
            self.assertIsNone(server.mailbox.take(timeout_s=0.01))
            self.assertEqual(server.stats()["rejected_packets"], 2)
        finally:
            server.server_close()

    def test_ingress_keeps_only_the_latest_and_rejects_stale_sequences(self):
        server = self.make_server()
        try:
            for sequence in (1, 2):
                encoded = CLIENT.encode_head_frame_packet(
                    job_id=JOB_ID,
                    generation=1,
                    sequence=sequence,
                    capture_timestamp_ns=sequence,
                    jpeg=b"\xff\xd8packet-" + str(sequence).encode("ascii"),
                    ingress_token=TOKEN,
                )
                self.assertTrue(server.ingest_payload(encoded[4:]))
            duplicate = CLIENT.encode_head_frame_packet(
                job_id=JOB_ID,
                generation=1,
                sequence=2,
                capture_timestamp_ns=3,
                jpeg=b"\xff\xd8duplicate",
                ingress_token=TOKEN,
            )
            self.assertFalse(server.ingest_payload(duplicate[4:]))
            packet = server.mailbox.take(timeout_s=0.01)
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.sequence, 2)
            self.assertEqual(server.stats()["replaced_pending"], 1)
            self.assertEqual(server.stats()["rejected_stale"], 1)
        finally:
            server.server_close()

    def test_background_publisher_sends_without_blocking_the_submitter(self):
        server = self.make_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        publisher = None
        try:
            host, port = server.server_address
            publisher = CLIENT.LatestHeadFramePublisher(
                f"tcp://{host}:{port}",
                TOKEN,
                job_id=JOB_ID,
                generation=1,
                fps=15.0,
                timeout_s=0.5,
            )
            publisher.start()
            self.assertTrue(
                publisher.submit(
                    {"cam_head": {"encoding": "jpeg", "data": b"\xff\xd8head-frame"}},
                    capture_timestamp_ns=999,
                )
            )
            packet = server.mailbox.take(timeout_s=2.0)
            self.assertIsNotNone(packet)
            assert packet is not None
            self.assertEqual(packet.capture_timestamp_ns, 999)
            self.assertEqual(packet.sequence, 0)
            deadline = time.monotonic() + 1.0
            while publisher.stats()["sent"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(publisher.stats()["sent"], 1)
        finally:
            if publisher is not None:
                publisher.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
