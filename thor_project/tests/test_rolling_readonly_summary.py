"""Diagnostic summaries must distinguish evidence, missing evidence and pause."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "rolling_readonly_summary", Path(__file__).parents[1] / "tools/summarize_rolling_readonly.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
BASE = summary._log_timestamp("2026-09-11 16:00:00,000 INFO test")


def row(ms, kind="servo_dispatch", request=1, **values):
    result = dict(timestamp_ns=BASE + ms * 1_000_000, record_type=kind,
                  generation=1, safety_request_id=request, gdk_command_accepted=None)
    if kind == "servo_dispatch":
        result.update(collision_response=summary.MOTION, certificate_age_ms=5,
                      proof_renewal="sampled_path_feedback_bound", certified_buffer_steps=5)
    if kind == "worker_check":
        result.update(hard_limit_triggered=False, safety_guard_latency_ms=8,
                      check_within_budget=True, check_started=ms / 1000)
    result.update(values)
    return result


def write_run(path, rows, log="", code=124):
    (path / "client.log").write_text(log)
    (path / "client_safety.jsonl").write_text("\n".join(json.dumps(value) for value in rows))
    (path / "command.json").write_text(json.dumps(["--rolling-safety", "--rolling-geometry=legacy"]))
    (path / "result.json").write_text(json.dumps(dict(ssh_returncode=code)))


def test_startup_steady_and_pause_are_separate(tmp_path):
    hold = dict(collision_response="async_checked_hold", certified_buffer_steps=0)
    rows = [row(0, "worker_check"), row(5, **hold), row(15, **hold), row(25),
            row(35, hold_reason="awaiting_fresh_window", **hold),
            row(45, hold_reason="awaiting_fresh_window", **hold), row(55),
            row(100, **hold), row(110, **hold)]
    write_run(tmp_path, rows, "2026-09-11 16:00:00,100 INFO SOFT PAUSE ACTIVE\n")
    result = summary.summarize(tmp_path)
    assert result["completed"] is True
    assert result["dispatch_count"] == 6
    assert result["startup_hold_run_count"] == result["steady_hold_run_count"] == 1
    assert result["steady_hold_tick_count"] == 2
    assert [run["duration_ms"] for run in result["hold_runs"]] == [20, 20]
    assert result["hold_runs"][1]["hold_reasons"] == {"awaiting_fresh_window": 2}
    assert result["motion_audit_failures"] == []
    assert result["no_physical_acceptance"] is True
    assert result["steady_queue"]["min"] == 0


def test_worker_rate_uses_actual_start_intervals_not_configured_fps(tmp_path):
    rows = [row(0, "worker_check"), row(40, "worker_check", request=2),
            row(80, "worker_check", request=3)]
    write_run(tmp_path, rows)
    result = summary.summarize(tmp_path)
    assert result["worker_timing"]["actual_start_hz"] == pytest.approx(25)
    assert result["worker_timing"]["start_interval_ms"]["median"] == pytest.approx(40)
    assert result["dispatch_hz"] is None
    assert result["motion_fraction"] is None
    assert result["completed"] is False


def test_failed_tail_is_censored_without_invented_final_tick(tmp_path):
    rows = [row(0, "worker_check"), row(10),
            row(20, collision_response="async_checked_hold"),
            row(30, collision_response="async_checked_hold")]
    write_run(tmp_path, rows, "Traceback (most recent call last):\nRuntimeError: hold expired\n", code=1)
    result = summary.summarize(tmp_path)
    assert result["completed"] is False
    assert result["active_end_reason"] == "log_ended_without_pause"
    assert result["steady_hold_run_count"] == 1
    tail = result["hold_runs"][0]
    assert tail["right_censored"] is True
    assert tail["duration_ms"] is None
    assert tail["observed_hold_span_ms"] == 10
    assert result["failure_lines"] == ["RuntimeError: hold expired"]


def test_admission_discard_and_no_evidence_are_not_conflated(tmp_path):
    rows = [row(0, "worker_check"), row(5, "safety_event", event="result_admitted"), row(10),
            row(20, "worker_check", request=2),
            row(25, "safety_event", request=2, event="result_discarded", reason="anchor_identity"),
            row(40, "worker_check", request=3)]
    write_run(tmp_path, rows)
    result = summary.summarize(tmp_path)
    outcomes = {entry["safety_request_id"]: entry for entry in result["request_outcomes"]}
    assert outcomes[1]["outcome"] == "motion_dispatched"
    assert outcomes[1]["admission_observed"] is True
    assert outcomes[2]["outcome"] == "explicit_discard"
    assert outcomes[2]["events"][0]["reason"] == "anchor_identity"
    assert outcomes[3]["outcome"] == "no_motion_observed_reason_unknown"
    assert outcomes[3]["discard_observed"] is False


def test_no_empty_audit_success_or_jitter_statistics(tmp_path):
    write_run(tmp_path, [], "RuntimeError: GDK occupied\n", code=1)
    result = summary.summarize(tmp_path)
    assert result["completed"] is False
    assert result["no_physical_acceptance"] is None
    assert result["active_seconds"] is None
    assert result["check_p95_ms"] is None
    assert result["queue"]["min"] is None
    assert result["hold_runs"] == []
    json.dumps(result, allow_nan=False)


def test_physical_acceptance_after_pause_is_not_hidden(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10), row(100, gdk_command_accepted=True)],
              "2026-09-11 16:00:00,050 INFO SOFT PAUSE ACTIVE\n")
    result = summary.summarize(tmp_path)
    assert result["dispatch_count"] == 1
    assert result["no_physical_acceptance"] is False


def test_truncated_audit_reports_missing_evidence(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10)])
    with (tmp_path / "client_safety.jsonl").open("a") as stream:
        stream.write('\n{"timestamp_ns":')
    result = summary.summarize(tmp_path)
    assert len(result["malformed_audit_rows"]) == 1
    assert result["no_physical_acceptance"] is None
    assert result["completed"] is False


def test_active_log_counters_exclude_pause(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10)],
              "2026-09-11 16:00:00,020 INFO End-to-end timing phase=inference\n"
              "2026-09-11 16:00:00,050 INFO SOFT PAUSE ACTIVE\n"
              "2026-09-11 16:00:00,060 INFO End-to-end timing phase=inference\n"
              "2026-09-11 16:00:00,070 WARN ACTION QUEUE UNDERRUN\n")
    result = summary.summarize(tmp_path)
    assert result["inference_responses"] == 1
    assert result["queue_underruns"] == 0


def test_worker_turnaround_and_consumption_are_independent(tmp_path):
    rows = [row(0, "worker_check", submitted_at=1.0, compute_started=1.003,
                check_completed=1.014, checked_points=5),
            row(5, "safety_event", event="result_admitted", checked_points=5), row(10)]
    write_run(tmp_path, rows)
    result = summary.summarize(tmp_path)
    assert result["worker_timing"]["submit_to_complete_ms"]["median"] == pytest.approx(14)
    assert result["worker_timing"]["submit_to_compute_ms"]["median"] == pytest.approx(3)
    assert result["worker_timing"]["checked_points"]["median"] == 5
    request = result["request_outcomes"][0]
    assert request["admitted_points"] == 5
    assert request["motion_dispatches"] == 1


def test_log_with_pause_but_no_runner_exit_is_not_confirmed_complete(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10)],
              "2026-09-11 16:00:00,050 INFO SOFT PAUSE ACTIVE\n")
    (tmp_path / "result.json").unlink()
    result = summary.summarize(tmp_path)
    assert result["automatic_pause_confirmed"] is True
    assert result["completed"] is False


def test_requested_pause_hold_is_transition_not_steady_gap(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10), row(40),
                        row(60, collision_response="async_checked_hold", generation=3),
                        row(80, collision_response="async_checked_hold", generation=3)],
              "2026-09-11 16:00:00,050 WARNING Configured active duration 50.000s reached; requesting automatic soft pause\n"
              "2026-09-11 16:00:00,075 WARNING SOFT PAUSE ACTIVE\n")
    result = summary.summarize(tmp_path)
    assert result["completed"] is True
    assert result["active_end_reason"] == "automatic_pause_requested"
    assert result["dispatch_count"] == 2
    assert result["steady_hold_run_count"] == 0
    assert result["pause_transition"]["hold_tick_count"] == 1
    assert result["pause_transition"]["duration_ms"] == 25


def test_genuine_hold_before_pause_request_is_not_hidden(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10),
                        row(40, collision_response="async_checked_hold"),
                        row(60, collision_response="async_checked_hold", generation=3)],
              "2026-09-11 16:00:00,050 WARNING Configured active duration 50.000s reached; requesting automatic soft pause\n"
              "2026-09-11 16:00:00,075 WARNING SOFT PAUSE ACTIVE\n")
    result = summary.summarize(tmp_path)
    assert result["steady_hold_run_count"] == 1
    assert result["steady_hold_tick_count"] == 1
    assert result["hold_runs"][0]["duration_ms"] == 10
    assert result["hold_runs"][0]["right_censored"] is True
    assert result["pause_transition"]["hold_tick_count"] == 1


def test_pause_request_without_confirmation_is_not_completed(tmp_path):
    write_run(tmp_path, [row(0, "worker_check"), row(10),
                        row(60, collision_response="async_checked_hold")],
              "2026-09-11 16:00:00,050 WARNING Configured active duration 50.000s reached; requesting automatic soft pause\n")
    result = summary.summarize(tmp_path)
    assert result["automatic_pause_confirmed"] is False
    assert result["completed"] is False
    assert result["steady_hold_run_count"] == 0
    assert result["pause_transition"]["hold_tick_count"] == 1
    assert result["pause_transition"]["duration_ms"] is None
