from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/gripper_step_benchmark.py"
SPEC = spec_from_file_location("gripper_step_benchmark", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_open_step_metrics_have_expected_milestones():
    samples = [
        {"t_s": 0.00, "position_m": 0.0000},
        {"t_s": 0.05, "position_m": 0.0002},
        {"t_s": 0.10, "position_m": 0.0030},
        {"t_s": 0.20, "position_m": 0.0130},
        {"t_s": 0.30, "position_m": 0.0230},
        {"t_s": 0.40, "position_m": 0.0245},
        {"t_s": 0.45, "position_m": 0.0247},
        {"t_s": 0.50, "position_m": 0.0248},
    ]
    metrics = MODULE.compute_metrics(samples, "open")
    assert metrics["onset_s"] == pytest.approx(0.10)
    assert metrics["t10_s"] == pytest.approx(0.10)
    assert metrics["t50_s"] == pytest.approx(0.20)
    assert metrics["t90_s"] == pytest.approx(0.30)
    assert metrics["settled_s"] == pytest.approx(0.40)


def test_close_step_metrics_use_reverse_direction():
    samples = [
        {"t_s": 0.0, "position_m": 0.025},
        {"t_s": 0.1, "position_m": 0.020},
        {"t_s": 0.2, "position_m": 0.010},
        {"t_s": 0.3, "position_m": 0.001},
        {"t_s": 0.4, "position_m": 0.0008},
        {"t_s": 0.5, "position_m": 0.0007},
    ]
    metrics = MODULE.compute_metrics(samples, "close")
    assert metrics["onset_s"] == pytest.approx(0.1)
    assert metrics["t50_s"] == pytest.approx(0.2)
    assert metrics["t90_s"] == pytest.approx(0.3)
    assert metrics["settled_s"] == pytest.approx(0.3)


def test_rejects_start_near_target():
    with pytest.raises(ValueError, match="too close"):
        MODULE.compute_metrics(
            [{"t_s": 0.0, "position_m": 0.024}, {"t_s": 0.1, "position_m": 0.025}],
            "open",
        )


class _FakeGdk:
    class JointStates:
        pass

    class JointState:
        pass

    class JointServoControlReq:
        def __init__(self):
            self.joint_velocities = []
            self.joint_accelerations = []


def test_move_ee_request_contains_one_gripper_and_no_body_joint():
    request = MODULE._make_move_ee_request(_FakeGdk, "left", 0.0)
    assert request.group == "left_tool"
    assert request.target_type == "dahuan"
    assert request.nums == 1
    assert [state.position for state in request.states] == [0.0]


def test_servo_request_contains_exactly_one_gripper_joint():
    request = MODULE._make_servo_request(_FakeGdk, "right", 0.025, 0.01)
    assert request.joint_names == ["idx71_gripper_r_inner_joint1"]
    assert request.joint_positions == [0.025]
    assert request.joint_velocities == [0.0]
    assert request.joint_accelerations == [0.0]
    assert request.control_period == pytest.approx(0.01)
