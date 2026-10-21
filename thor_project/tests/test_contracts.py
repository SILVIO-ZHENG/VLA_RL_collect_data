from __future__ import annotations

import asyncio

import numpy as np
import pytest

from nexus_deploy.adapters.mock_model import MockModelAdapter
from nexus_deploy.adapters.mock_robot import MockRobotAdapter
from nexus_deploy.client import ensure_compatible
from nexus_deploy.contracts import (
    ActionChunk,
    ContractError,
    ImageSpec,
    PolicyCapabilities,
    TensorSpec,
    parse_inference_request,
    validate_action_chunk,
)
from nexus_deploy.plugins import load_plugin
from nexus_deploy.server import PolicyServer
from nexus_deploy.wire import pack_message, unpack_message


def capabilities() -> PolicyCapabilities:
    names = ("left_joint", "right_joint")
    return PolicyCapabilities(
        model_id="test-vla",
        state=TensorSpec(names),
        action=TensorSpec(names),
        images=(ImageSpec("front"),),
        chunk_size=3,
        model_fps=20,
    )


def test_metadata_round_trip_preserves_dynamic_schema() -> None:
    original = capabilities()
    decoded = PolicyCapabilities.from_metadata(original.to_metadata())
    assert decoded == original


def test_wire_round_trip_numpy_without_losing_dtype_or_shape() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    decoded = unpack_message(pack_message({"array": array}))
    np.testing.assert_array_equal(decoded["array"], array)
    assert decoded["array"].dtype == np.float32


def test_request_validation_rejects_missing_required_camera() -> None:
    request = {
        "type": "inference_request",
        "request_id": "r1",
        "session_id": "s1",
        "observation": {
            "state": np.zeros(2, dtype=np.float32),
            "images": {},
            "task": "move",
            "timestamp_ns": 1,
        },
    }
    with pytest.raises(ContractError, match="Missing required images"):
        parse_inference_request(request, capabilities())


def test_request_validation_rejects_malformed_image() -> None:
    request = {
        "request_id": "r1",
        "session_id": "s1",
        "observation": {
            "state": np.zeros(2, dtype=np.float32),
            "images": {"front": {"encoding": "jpeg", "width": 10, "height": 10, "data": "not-bytes"}},
            "task": "move",
        },
    }
    with pytest.raises(ContractError, match="data must be bytes"):
        parse_inference_request(request, capabilities())


def test_action_chunk_validation_rejects_wrong_dimension() -> None:
    chunk = ActionChunk(np.zeros((3, 3), dtype=np.float32), "chunk")
    with pytest.raises(ContractError, match="shape"):
        validate_action_chunk(chunk, capabilities())


def test_schema_names_prevent_same_dimension_wrong_order() -> None:
    robot = MockRobotAdapter(
        {"joint_names": ["right_joint", "left_joint"], "camera_names": ["front"]}
    )
    with pytest.raises(ContractError, match="schema mismatch"):
        ensure_compatible(robot, capabilities())


def test_reference_plugins_form_a_compatible_pair() -> None:
    config = {"joint_names": ["a", "b", "c"], "camera_names": ["head"]}
    model = MockModelAdapter(config)
    robot = MockRobotAdapter(config)
    ensure_compatible(robot, model.capabilities)


def test_plugin_loader_uses_module_object_contract() -> None:
    plugin = load_plugin(
        "nexus_deploy.adapters.mock_model:MockModelAdapter",
        {"joint_names": ["joint"], "camera_names": ["front"]},
    )
    assert isinstance(plugin, MockModelAdapter)


def test_policy_server_handler_runs_adapter_and_returns_valid_chunk() -> None:
    config = {
        "joint_names": ["left_joint", "right_joint"],
        "camera_names": ["front"],
        "chunk_size": 3,
    }
    request = {
        "type": "inference_request",
        "request_id": "r1",
        "session_id": "s1",
        "reset": True,
        "observation": {
            "state": np.asarray([1.0, 2.0], dtype=np.float32),
            "images": {
                "front": {
                    "encoding": "raw",
                    "width": 1,
                    "height": 1,
                    "data": b"\x00\x00\x00",
                }
            },
            "task": "hold",
            "timestamp_ns": 1,
        },
    }

    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages = [pack_message(request)]
            self.sent: list[bytes] = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.messages:
                raise StopAsyncIteration
            return self.messages.pop(0)

        async def send(self, payload: bytes) -> None:
            self.sent.append(payload)

    socket = FakeWebSocket()
    server = PolicyServer(MockModelAdapter(config))
    asyncio.run(server.handle(socket))
    assert len(socket.sent) == 2
    metadata, response = (unpack_message(item) for item in socket.sent)
    assert metadata["protocol"] == {"name": "nexus-policy-websocket", "version": 3}
    assert response["status"] == "ok"
    np.testing.assert_array_equal(response["actions"], np.asarray([[1.0, 2.0]] * 3))


def test_v3_metadata_accepts_previous_protocol_name_during_migration() -> None:
    metadata = capabilities().to_metadata()
    metadata["protocol"] = {"name": "g2a-policy-websocket", "version": 3}

    assert PolicyCapabilities.from_metadata(metadata) == capabilities()


def test_v3_metadata_rejects_unknown_protocol_name() -> None:
    metadata = capabilities().to_metadata()
    metadata["protocol"] = {"name": "unknown-policy-websocket", "version": 3}

    with pytest.raises(ContractError, match="Unsupported protocol"):
        PolicyCapabilities.from_metadata(metadata)
