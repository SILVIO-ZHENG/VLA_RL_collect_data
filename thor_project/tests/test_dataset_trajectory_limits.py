"""Dataset-derived values remain offline; no SDK or network involved."""

import json

import numpy as np
import pytest

from tools.trajectory_lab.dataset_limits import ARMS, derivatives, experimental_limits, load_profile
from tools.trajectory_lab.replay import plan_recorded
from tools.trajectory_lab.trajectory import Limits, PlanningError, path_error_bound, plan


def test_cubic_fit_derivatives_have_correct_units_and_no_edge_padding():
    t = np.arange(101) / 30
    q = np.stack((2 + 3 * t + 4 * t**2 + 5 * t**3, -2 * t), axis=1)
    result = derivatives(q, 30)
    center = t[4:-4]
    np.testing.assert_allclose(result["velocity"][:, 0], 3 + 8 * center + 15 * center**2, atol=1e-9)
    np.testing.assert_allclose(result["acceleration"][:, 0], 8 + 30 * center, atol=1e-9)
    np.testing.assert_allclose(result["jerk"][:, 0], 30, atol=1e-8)
    np.testing.assert_allclose(result["velocity"][:, 1], -2, atol=1e-10)
    np.testing.assert_allclose(result["jerk"][:, 1], 0, atol=1e-9)


@pytest.mark.parametrize("window", [0, 2, 3, 6, 9.5])
def test_fit_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        derivatives(np.zeros((30, 2)), 30, window)


def test_experimental_profile_uses_both_channels_per_joint_not_dataset_max():
    stats = {
        channel: {
            "fitted": {name: {"p99": np.full(14, value).tolist()} for name in ("velocity", "acceleration", "jerk")}
        }
        for channel, value in (("action", 1), ("observation.state", 3))
    }
    stats["action"]["fitted"]["velocity"]["p99"][0] = 4
    result = experimental_limits(stats)
    assert result["velocity"][0] == pytest.approx(4.4)
    assert result["velocity"][1] == pytest.approx(3.3)
    assert result["acceleration"][1] == pytest.approx(3.3)
    assert result["jerk"][1] == 10


def profile_file(tmp_path):
    names = [f"j{i}" for i in range(16)]
    profile = {
        "schema": "nexus.offline-dataset-limits.v1",
        "hardware_ready": False,
        "joint_indices": ARMS,
        "joint_names": [names[i] for i in ARMS],
        "units": {"velocity": "rad/s", "acceleration": "rad/s^2", "jerk": "rad/s^3"},
        "limits": {name: [1.0] * 14 for name in ("velocity", "acceleration", "jerk")},
    }
    path = tmp_path / "limits.offline.json"
    path.write_text(json.dumps(profile))
    return path, profile, names


def test_load_profile_preserves_joint_mapping(tmp_path):
    path, profile, names = profile_file(tmp_path)
    profile["limits"]["velocity"] = list(range(1, 15))
    path.write_text(json.dumps(profile))
    _, vectors = load_profile(path, names)
    np.testing.assert_array_equal(vectors["velocity"][ARMS], np.arange(1, 15))
    assert vectors["velocity"][7] == vectors["velocity"][15] == 1  # excluded placeholders


@pytest.mark.parametrize("problem", ["hardware", "mapping", "units", "nan", "negative", "dimension"])
def test_bad_profile_rejected(tmp_path, problem):
    path, profile, names = profile_file(tmp_path)
    if problem == "hardware":
        profile["hardware_ready"] = True
    elif problem == "mapping":
        profile["joint_indices"] = list(range(14))
    elif problem == "units":
        profile["units"]["velocity"] = "degree/s"
    elif problem == "dimension":
        profile["limits"]["velocity"].pop()
    else:
        profile["limits"]["velocity"][0] = float("nan") if problem == "nan" else -1
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_profile(path, names)


def test_cubic_guide_is_c2_bounded_and_does_not_smooth_gripper():
    t = np.arange(61) / 30
    points = np.stack((0.05 * np.sin(t), 0.1 * np.sin(2 * t), (t > 1) * 0.025), axis=1)
    limits = Limits(np.full(3, 0.6), np.full(3, 2), np.full(3, 10), np.full(3, -3), np.full(3, 3))
    trajectory = plan(points, policy_fps=30, limits=limits, device_indices=(2,), tangent_method="cubic")
    assert all(value <= 1 + 1e-8 for value in trajectory.bounds().values())
    assert path_error_bound(trajectory) <= 0.02
    for t in trajectory.times[1:-1]:
        left, right = trajectory.sample(t - 1e-9), trajectory.sample(t + 1e-9)
        for field in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(left, field)[:2], getattr(right, field)[:2], atol=1e-7)
    event = trajectory.times[31]
    assert trajectory.sample(event - 1e-9).position[2] == 0
    assert trajectory.sample(event + 1e-9).position[2] == 0.025


@pytest.mark.parametrize("damping", [0, -1, 1.1, float("nan")])
def test_invalid_damping_rejected(damping):
    limits = Limits(*[np.ones(2)] * 3, -np.ones(2), np.ones(2))
    with pytest.raises(PlanningError):
        plan(np.zeros((3, 2)), policy_fps=30, limits=limits, retiming_damping=damping)


def test_recorded_search_retries_without_changing_any_limits(monkeypatch):
    calls = []
    sentinel = object()
    limit = object()

    def fake_plan(points, **kwargs):
        calls.append(kwargs)
        assert kwargs["limits"] is limit
        assert kwargs["max_path_deviation"] == 0.02
        if len(calls) < 3:
            raise PlanningError("path deviation")
        return sentinel

    monkeypatch.setattr("tools.trajectory_lab.replay.plan", fake_plan)
    result, attempts = plan_recorded([[0], [1]], limits=limit, policy_fps=30, max_path_deviation=0.02)
    assert result is sentinel
    assert [(r["method"], r["knot_tolerance_rad"]) for r in attempts] == [
        ("cubic", 0.003),
        ("cubic", 0),
        ("harmonic", 0.003),
    ]
    assert [r["status"] for r in attempts] == ["rejected", "rejected", "bounded"]


def test_recorded_search_does_not_release_if_all_attempts_fail():
    limits = Limits(*[np.ones(2)] * 3, -np.ones(2), np.ones(2))
    with pytest.raises(PlanningError, match="All offline candidates rejected"):
        plan_recorded([[0, 0], [2, 2]], policy_fps=30, limits=limits)


def test_cubic_suffix_preserves_nonzero_boundary_when_feasible():
    limits = Limits(np.ones(2), np.full(2, 3), np.full(2, 30), -np.ones(2), np.ones(2))
    v, a = np.array([0.05, -0.02]), np.array([0.01, 0.005])
    p = plan(
        np.array([[0, 0], [0.01, -0.005], [0.02, -0.01], [0.03, -0.015]]),
        policy_fps=30,
        limits=limits,
        start_velocity=v,
        start_acceleration=a,
        tangent_method="cubic",
    )
    np.testing.assert_allclose(p.sample(0).velocity, v, atol=1e-12)
    np.testing.assert_allclose(p.sample(0).acceleration, a, atol=1e-12)
    assert all(value <= 1 + 1e-8 for value in p.bounds().values())
