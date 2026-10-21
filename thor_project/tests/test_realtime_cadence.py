"""Failure boundaries for the opt-in robot cadence scheduler."""

import threading
import time
from types import SimpleNamespace

import numpy as np

import pytest

from nexus_deploy.robot.g2a import realtime_cadence as cadence
from nexus_deploy.robot.g2a import client


def test_execution_preflight_matches_rtc_request_cadence_without_arming(monkeypatch):
    runner = object.__new__(client.RobotClientRunner)
    captures = []
    records = []
    runner.hardware_mode = "execute"
    runner.inference_mode = "rtc"
    runner.execution_preflight_requests = 3
    runner.quality_horizon = 50
    runner.fps = 30.0
    runner.session_id = "preflight-test"
    runner._request_number = 0
    runner.min_rtc_blend_steps = 8
    runner.rtc_state_aligned_anchor = False
    runner._observation_latencies_s = []
    runner.shutdown_event = threading.Event()
    runner.policy_client = SimpleNamespace(infer=lambda _request: {
        "chunk_id": "synthetic", "rtc": {"prefix_used": True},
    })
    monkeypatch.setattr(runner, "_request_lead_steps", lambda: 30)
    monkeypatch.setattr(runner, "_estimated_delay_steps", lambda: 0)
    monkeypatch.setattr(runner, "_enforce_execution_continuity_budget", lambda: True)
    monkeypatch.setattr(runner, "_latency_p95_s", lambda: 0.01)
    monkeypatch.setattr(runner, "_latency_steps", lambda: 1)
    monkeypatch.setattr(runner, "_record_request_timing", lambda **kw: records.append(kw))

    def capture():
        now = time.perf_counter()
        captures.append(now)
        return {"observation_monotonic_time": now}

    monkeypatch.setattr(runner, "_capture_observation", capture)
    runner._perform_execution_latency_preflight()

    assert len(records) == 3
    assert all(item["phase"] == "execution_preflight" for item in records)
    assert all(0.62 <= later - earlier < 0.85
               for earlier, later in zip(captures, captures[1:]))


def worker(*, alive=True):
    return SimpleNamespace(feedback_thread=SimpleNamespace(native_id=123, is_alive=lambda: alive))


def fake_scheduler(monkeypatch, *, fail_tid=None):
    calls = []
    monkeypatch.setattr(cadence.threading, "get_native_id", lambda: 456)
    monkeypatch.setattr(cadence.os, "sched_getscheduler", lambda tid: cadence.os.SCHED_OTHER)

    def schedule(tid, policy, param):
        calls.append((tid, policy, param.sched_priority))
        if tid == fail_tid and policy == cadence.os.SCHED_RR:
            raise PermissionError("RT promotion denied")

    monkeypatch.setattr(cadence.os, "sched_setscheduler", schedule)
    return calls


def test_promotes_feedback_and_action_then_restores(monkeypatch):
    calls = fake_scheduler(monkeypatch)
    pair = cadence.RealtimeCadence(worker())

    pair.start()
    pair.stop()

    assert calls == [
        (123, cadence.os.SCHED_RR, 1),
        (456, cadence.os.SCHED_RR, 1),
        (456, cadence.os.SCHED_OTHER, 0),
        (123, cadence.os.SCHED_OTHER, 0),
    ]


def test_failed_action_promotion_rolls_back_feedback(monkeypatch):
    calls = fake_scheduler(monkeypatch, fail_tid=456)
    pair = cadence.RealtimeCadence(worker())

    with pytest.raises(PermissionError, match="denied"):
        pair.start()
    pair.stop()

    assert calls == [
        (123, cadence.os.SCHED_RR, 1),
        (456, cadence.os.SCHED_RR, 1),
        (123, cadence.os.SCHED_OTHER, 0),
    ]


def test_missing_feedback_fails_without_scheduling(monkeypatch):
    calls = fake_scheduler(monkeypatch)
    with pytest.raises(RuntimeError, match="running 50 Hz feedback"):
        cadence.RealtimeCadence(worker(alive=False)).start()
    assert calls == []


def test_preflight_rejects_missing_privilege_before_changing_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(cadence.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cadence.resource, "getrlimit", lambda limit: (0, 0))
    monkeypatch.setattr(cadence.resource, "setrlimit", lambda *args: calls.append(args))

    with pytest.raises(RuntimeError, match="priority limit"):
        cadence.prepare_realtime_cadence(worker())
    assert calls == []


def test_preflight_sets_bounded_rt_cpu_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(cadence.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cadence.os, "sched_getscheduler", lambda tid: cadence.os.SCHED_OTHER)
    monkeypatch.setattr(cadence.resource, "getrlimit", lambda limit: (1, 1) if limit == cadence.resource.RLIMIT_RTPRIO else (cadence.resource.RLIM_INFINITY,) * 2)
    monkeypatch.setattr(cadence.resource, "setrlimit", lambda *args: calls.append(args))

    cadence.prepare_realtime_cadence(worker())

    assert calls == [(cadence.resource.RLIMIT_RTTIME, (50_000, 100_000))]


def test_missing_rt_permission_never_prepares_physical_execution(monkeypatch):
    runner = object.__new__(client.RobotClientRunner)
    prepared = []
    runner.adapter = SimpleNamespace(
        config=SimpleNamespace(
            safety_enforcement_enabled=True,
            async_safety_enabled=True,
            rolling_safety_enabled=True,
        ),
        _async_worker=worker(),
        prepare_execution=lambda: prepared.append(True),
    )
    runner.policy_client = SimpleNamespace(metadata={})
    runner.plan = SimpleNamespace(set_hold_action=lambda state: None)
    runner.hardware_mode = "execute"
    runner.rt_cadence = True
    runner.servo_fps = 100.0
    runner._hold_effectors = False
    runner._action_filter = None
    runner._effector_action_filter = None
    runner._start_paused = False
    runner._last_action = None
    monkeypatch.setattr(runner, "validate_server", lambda: None)
    monkeypatch.setattr(runner, "warn_initial_state", lambda: np.zeros(24))
    monkeypatch.setattr(runner, "_perform_warmup", lambda: None)
    monkeypatch.setattr(runner, "_perform_execution_latency_preflight", lambda: None)
    def reject_rt_permission(_worker):
        raise PermissionError("no RT limit")

    monkeypatch.setattr(client, "prepare_realtime_cadence", reject_rt_permission)

    with pytest.raises(PermissionError, match="no RT limit"):
        runner.run(0.1)
    assert prepared == []


def test_action_promotion_failure_stops_before_any_send(monkeypatch):
    runner = object.__new__(client.RobotClientRunner)
    runner.adapter = SimpleNamespace(_async_worker=worker())
    runner.rt_cadence = True
    runner.shutdown_event = threading.Event()
    runner._execution_armed_event = threading.Event()
    runner._execution_armed_event.set()
    runner._fatal_error = None
    sent = []
    stopped = []
    runner._send_action = lambda *args, **kwargs: sent.append((args, kwargs))

    class FailedCadence:
        def __init__(self, feedback_worker):
            assert feedback_worker is runner.adapter._async_worker

        def start(self):
            raise PermissionError("promotion denied")

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(client, "RealtimeCadence", FailedCadence)
    runner._action_loop()

    assert runner.shutdown_event.is_set()
    assert "promotion denied" in runner._fatal_error
    assert stopped == [True]
    assert sent == []


def test_action_thread_restores_scheduler_when_stopped(monkeypatch):
    runner = object.__new__(client.RobotClientRunner)
    runner.adapter = SimpleNamespace(_async_worker=worker())
    runner.rt_cadence = True
    runner.shutdown_event = threading.Event()
    runner._execution_armed_event = threading.Event()
    runner._execution_armed_event.set()
    runner._fatal_error = None
    runner.servo_fps = 100.0
    calls = []

    class OneTickCadence:
        def __init__(self, feedback_worker):
            pass

        def start(self):
            calls.append("start")
            runner.shutdown_event.set()

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(client, "RealtimeCadence", OneTickCadence)
    runner._action_loop()

    assert calls == ["start", "stop"]
    assert runner._fatal_error is None


def test_failed_scheduler_restore_is_reported_and_stops(monkeypatch):
    runner = object.__new__(client.RobotClientRunner)
    runner.adapter = SimpleNamespace(_async_worker=worker())
    runner.rt_cadence = True
    runner.shutdown_event = threading.Event()
    runner._execution_armed_event = threading.Event()
    runner._execution_armed_event.set()
    runner._fatal_error = None
    runner.servo_fps = 100.0

    class FailedRestore:
        def __init__(self, feedback_worker):
            pass

        def start(self):
            runner.shutdown_event.set()

        def stop(self):
            raise RuntimeError("restore denied")

    monkeypatch.setattr(client, "RealtimeCadence", FailedRestore)
    runner._action_loop()

    assert runner.shutdown_event.is_set()
    assert "restore denied" in runner._fatal_error
