"""No robot, inference server, or real-time claims: test future handoff contracts."""

from dataclasses import replace

import numpy as np
import pytest

from tools.trajectory_lab.rolling import RollingTimeline
from tools.trajectory_lab.trajectory import Limits, PlanningError, plan


def initial():
    limits = Limits(np.ones(3), np.full(3, 5), np.full(3, 80), -np.ones(3), np.ones(3))
    return plan(np.tile([0, 0, 0.025], (51, 1)), policy_fps=30, limits=limits, device_indices=(2,))


def suffix(request, active):
    q = np.tile(request.sample.position, (51, 1))
    q[:, 0] += np.linspace(0, 0.05, 51)
    q[20:, 2] = 0.004
    return plan(
        q,
        policy_fps=30,
        limits=active.limits,
        device_indices=(2,),
        phases=request.sample.phase + np.arange(51),
        start_velocity=request.sample.velocity,
        start_acceleration=request.sample.acceleration,
    )


def test_future_boundary_survives_consumption_and_preserves_prefix():
    old = initial()
    timeline = RollingTimeline(old)
    request = timeline.reserve(0, lead_s=0.6)
    new = suffix(request, old)
    assert timeline.last.phase == 0  # reservation/planning is not adoption
    for t in np.arange(0.01, 0.31, 0.01):
        np.testing.assert_array_equal(timeline.adopt(t, lambda a, b: True).position, old.sample(t).position)
    timeline.submit(new, request, now=0.31)
    before = timeline.adopt(request.at - 1e-9, lambda a, b: True)
    after = timeline.adopt(request.at + 1e-9, lambda a, b: True)
    for name in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(before, name), getattr(after, name), atol=1e-7)
    assert timeline.switches == 1
    assert timeline.active is new


def test_handoff_never_uses_old_gripper_value_at_source_boundary():
    old = initial()
    q = old.original_points.copy()
    q[15:, 2] = 0.0
    p = plan(q, policy_fps=30, limits=old.limits, device_indices=(2,))
    timeline = RollingTimeline(p, start=10000.0)
    request = timeline.reserve(10000.0, lead_s=0.49)
    assert request.sample.position[2] == 0


def test_late_response_leaves_old_reference_intact():
    old = initial()
    timeline = RollingTimeline(old)
    request = timeline.reserve(0, lead_s=0.6)
    with pytest.raises(PlanningError, match="Late handoff"):
        timeline.submit(suffix(request, old), request, now=request.at)
    assert timeline.reservation is None
    assert timeline.active is old
    assert timeline.scheduled is None
    timeline.adopt(request.at + 0.01, lambda a, b: True)
    assert timeline.switches == 0


def test_exhausted_reference_can_handoff_from_verified_rest_without_motion_extrapolation():
    old = initial()
    timeline = RollingTimeline(old)
    now = old.duration + 0.2
    timeline.adopt(now, lambda a, b: True)
    with pytest.raises(PlanningError, match="Insufficient planned horizon"):
        timeline.reserve(now, lead_s=0.6)
    request = timeline.reserve(now, lead_s=0.6, allow_terminal_hold=True)
    assert request.sample.phase == old.original_phases[-1]
    np.testing.assert_array_equal(request.sample.position, old.sample(old.duration).position)
    np.testing.assert_array_equal(request.sample.velocity, np.zeros(3))
    new = suffix(request, old)
    timeline.submit(new, request, now=now + 0.3)
    np.testing.assert_array_equal(timeline.adopt(request.at - 0.01, lambda a, b: True).position, old.points[-1])
    seen = []
    timeline.adopt(request.at + 0.02, lambda a, b: seen.append(b.copy()) or True)
    assert len(seen) == 1  # terminal resumption does not bypass checking
    assert timeline.switches == 1
    assert timeline.last.phase > old.original_phases[-1]


def test_late_terminal_handoff_can_retry_but_collision_hold_cannot_auto_resume():
    old = initial()
    timeline = RollingTimeline(old)
    now = old.duration + 1
    timeline.adopt(now, lambda a, b: True)
    request = timeline.reserve(now, lead_s=0.6, allow_terminal_hold=True)
    with pytest.raises(PlanningError, match="Late"):
        timeline.submit(suffix(request, old), request, now=request.at)
    retry = timeline.reserve(request.at, lead_s=0.6, allow_terminal_hold=True)
    assert retry.token != request.token
    timeline.adopt(request.at + 0.01, lambda a, b: False)
    with pytest.raises(PlanningError, match="held"):
        timeline.reserve(request.at + 0.02, lead_s=0.6, allow_terminal_hold=True)


def test_terminal_handoff_never_skips_original_gripper_events():
    old = initial()
    q = old.original_points.copy()
    q[-2:, 2] = 0.004
    old = plan(q, policy_fps=30, limits=old.limits, device_indices=(2,))
    timeline = RollingTimeline(old)
    now = old.duration - 0.2
    request = timeline.reserve(now, lead_s=0.6, allow_terminal_hold=True)
    assert request.sample.position[2] == 0.004
    timeline.submit(suffix(request, old), request, now=now + 0.1)
    assert timeline.adopt(old.times[-3], lambda a, b: True).position[2] == 0.025
    assert timeline.adopt(old.times[-2] + 1e-9, lambda a, b: True).position[2] == 0.004


def test_replay_recovers_after_two_forced_late_replies(monkeypatch):
    from tools.trajectory_lab import rolling_replay

    original = rolling_replay.plan_recorded
    elapsed, calls = [0.0], []

    def measured_plan(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        # Initial plan is prepared offline; then force two late responses,
        # followed by a fast valid plan. No sleeps or timing-dependent test.
        elapsed[0] += 0.8 if len(calls) in (2, 3) else 0.001
        return result

    monkeypatch.setattr(rolling_replay, "plan_recorded", measured_plan)
    monkeypatch.setattr(rolling_replay.time, "perf_counter", lambda: elapsed[0])
    points = np.zeros((250, 16))
    points[:, [7, 15]] = 0.025
    limits = Limits(np.ones(16), np.full(16, 5), np.full(16, 80), -np.ones(16), np.ones(16))
    report = rolling_replay.run_case(
        points,
        fps=30,
        limits=limits,
        delays=[0.27],
        seconds=4,
        lead_s=0.6,
        checker=lambda a, b: True,
    )
    assert [e["status"] for e in report["events"][:2]] == ["response_rejected"] * 2
    assert any(e["terminal_handoff"] and e["status"] == "scheduled" for e in report["events"])
    assert report["terminal_hold_ticks"] > 0  # no pretending that a pause vanished
    assert not report["ticks"][-1]["terminal_hold"]
    assert report["ticks"][-1]["phase"] > 50
    assert report["switches"] >= 1


def test_replay_does_not_report_future_reserved_end_as_already_executed():
    from tools.trajectory_lab.rolling_replay import run_case

    points = np.zeros((52, 16))
    points[-1, [7, 15]] = 0.025
    limits = Limits(np.ones(16), np.full(16, 5), np.full(16, 80), -np.ones(16), np.ones(16))
    report = run_case(points, fps=30, limits=limits, delays=[0.05], seconds=4, lead_s=0.6, checker=lambda a, b: True)
    assert report["dataset_end_reached"]
    assert report["ticks"][-1]["phase"] == 51
    assert report["ticks"][-1]["gripper"] == [0.025, 0.025]


def test_collision_invalidates_reserved_and_scheduled_work_and_can_recover(caplog):
    old = initial()
    timeline = RollingTimeline(old)
    request = timeline.reserve(0, lead_s=0.6)
    new = suffix(request, old)
    timeline.submit(new, request, now=0.2)
    previous = timeline.last
    held = timeline.adopt(0.3, lambda a, b: False)
    np.testing.assert_array_equal(held.position, previous.position)
    assert held.phase == previous.phase
    assert timeline.scheduled is None and timeline.held
    assert "rejected" in caplog.text
    with pytest.raises(PlanningError, match="Stale"):
        timeline.submit(new, request, now=0.31)
    # Replan from the actually held state, never restore the old scheduled plan.
    recovery = plan(
        np.tile(held.position, (51, 1)),
        policy_fps=30,
        limits=old.limits,
        device_indices=(2,),
        phases=held.phase + np.arange(51),
    )
    timeline.recover(recovery, now=0.32)
    assert not timeline.held
    assert timeline.adopt(0.4, lambda a, b: True).phase > held.phase


@pytest.mark.parametrize("result", [None, 1, "yes"])
def test_nonboolean_checker_holds_and_invalidates(result):
    timeline = RollingTimeline(initial())
    timeline.reserve(0, lead_s=0.6)
    with pytest.raises(PlanningError, match="explicit boolean"):
        timeline.adopt(0.01, lambda a, b: result)
    assert timeline.held and timeline.reservation is None


def test_checker_exception_holds():
    timeline = RollingTimeline(initial())

    def checker(a, b):
        raise RuntimeError("geometry failed")

    with pytest.raises(RuntimeError, match="geometry failed"):
        timeline.adopt(0.01, checker)
    assert timeline.held


def test_copied_or_superseded_reservation_rejected():
    timeline = RollingTimeline(initial())
    request = timeline.reserve(0, lead_s=0.6)
    proposal = suffix(request, timeline.active)
    with pytest.raises(PlanningError, match="Stale"):
        timeline.submit(proposal, replace(request), now=0.1)
    timeline.cancel(request)
    timeline.reserve(0.1, lead_s=0.6)
    with pytest.raises(PlanningError, match="Stale"):
        timeline.submit(proposal, request, now=0.2)


def test_handoff_rejects_limit_changes():
    timeline = RollingTimeline(initial())
    request = timeline.reserve(0, lead_s=0.6)
    new = suffix(request, timeline.active)
    new.limits = Limits(np.full(3, 2), np.full(3, 5), np.full(3, 80), -np.ones(3), np.ones(3))
    with pytest.raises(PlanningError, match="changed trajectory limits"):
        timeline.submit(new, request, now=0.2)


@pytest.mark.parametrize("now", [-1, float("nan"), float("inf")])
def test_invalid_time_rejected(now):
    timeline = RollingTimeline(initial())
    with pytest.raises(PlanningError):
        timeline.adopt(now, lambda a, b: True)


def test_insufficient_horizon_does_not_truncate_reservation():
    timeline = RollingTimeline(initial())
    with pytest.raises(PlanningError, match="Insufficient planned horizon"):
        timeline.reserve(0, lead_s=10)
    assert timeline.reservation is None


def test_checker_cannot_modify_adopted_target():
    timeline = RollingTimeline(initial())

    def checker(a, b):
        b[:] = 99
        return True

    np.testing.assert_array_equal(timeline.adopt(0.01, checker).position, [0, 0, 0.025])


def test_seam_blend_preserves_original_events_and_path():
    from tools.trajectory_lab.trajectory import path_error_bound

    q = np.tile([0.0, 0, 0.025], (51, 1))
    q[:, 0] = np.arange(51) * 0.001
    q[2:4, 2] = 0.004
    p = plan(
        q,
        policy_fps=30,
        limits=initial().limits,
        device_indices=(2,),
        start_velocity=[0.03, 0, 0],
        start_blend_phase=6,
        tangent_method="cubic",
    )
    assert p.phases[1] == 6
    np.testing.assert_array_equal(p.original_points, q)
    assert path_error_bound(p) <= 0.02
    for phase, goal in [(2.5, 0.004), (4.5, 0.025)]:
        t = np.interp(phase, p.phases, p.times)
        assert p.sample(t).position[2] == goal
    np.testing.assert_allclose(p.sample(0).velocity, [0.03, 0, 0], atol=1e-12)
    assert all(v <= 1 + 1e-8 for v in p.bounds().values())


def test_seam_blend_cannot_hide_original_large_excursion():
    q = np.zeros((51, 3))
    q[2, 0] = 0.3
    with pytest.raises(PlanningError, match="path deviation"):
        plan(q, policy_fps=30, limits=initial().limits, device_indices=(2,), start_blend_phase=6)


@pytest.mark.parametrize("span", [-1, 13, float("nan")])
def test_invalid_seam_span_rejected(span):
    with pytest.raises(PlanningError):
        plan(np.zeros((5, 3)), policy_fps=30, limits=initial().limits, start_blend_phase=span)
