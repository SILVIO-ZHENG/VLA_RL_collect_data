"""Recorded reply replay validates provenance/completion without a robot."""

import json
from pathlib import Path

import numpy as np
import pytest

from tools.trajectory_lab.response_windows import main, replay

PROFILE = Path(__file__).resolve().parents[1] / "tools/trajectory_lab/profiles/dh_30hz_20260914.offline.json"


def trace(tmp_path, *, replies=True, dropped=0):
    rows = [dict(record_type="trace_start", policy_fps=30, servo_fps=100, inference_mode="rtc")]
    rows.append(dict(record_type="control", monotonic=0, adopted_target=[0] * 24, planner_velocity=[0] * 24))
    if replies:
        targets = np.zeros((50, 24))
        targets[20:, [7, 15]] = 0.025
        rows.append(
            dict(
                record_type="inference_response",
                installed=True,
                installed_monotonic=0.3,
                control_targets=targets.tolist(),
                dropped_steps=10,
                chunk_id="r1",
                observation_state=[0] * 24,
                rtc={"prefix_used": True},
            )
        )
    rows.append(dict(record_type="trace_end", dropped=dropped))
    path = tmp_path / "control.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_actual_reply_coordinates_events_and_dropped_prefix(tmp_path):
    report = replay(trace(tmp_path), PROFILE, 20)
    assert report["physical_commands"] == 0 and not report["closed_loop_counterfactual"]
    assert report["evaluated"] == 1
    case = report["cases"][0]
    assert case["status"] == "bounded" and case["dropped_steps"] == 10
    assert case["duration_s"] == pytest.approx(40 / 30)


@pytest.mark.parametrize("kwargs", [{"replies": False}, {"dropped": 1}])
def test_missing_evidence_is_not_replayed_as_success(tmp_path, kwargs):
    with pytest.raises(ValueError):
        replay(trace(tmp_path, **kwargs), PROFILE, 20)


def test_cli_writes_and_prints_summary(tmp_path, monkeypatch, capsys):
    pytest.importorskip("scipy.optimize")
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "response_windows",
            "--trace",
            str(trace(tmp_path)),
            "--limits-profile",
            str(PROFILE),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out) == {"evaluated": 1, "bounded": 1}
    assert (output / "report.json").exists()
