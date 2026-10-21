from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nexus_deploy.safety_profiles import (
    AlignmentSafetyProfileRepository,
    SafetyProfileGenerationError,
)


def write_template(tmp_path: Path) -> Path:
    path = tmp_path / "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "calibrated": False,
                "read_only_ready": True,
                "execute_approved": False,
                "base_frame": "base_link",
                "robot_model": {
                    "type": "urdf",
                    "path": "assets/G2_dahuan_safety.urdf",
                    "sha256": "urdf-sha",
                },
                "surfaces": [
                    {
                        "name": "work_surface",
                        "frame": "base_link",
                        "height": 0.75,
                        "hard_margin": 0.02,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def measurement() -> dict[str, object]:
    return {
        "schema_version": 1,
        "measurement_id": "standard_table-20260910-103522",
        "profile_id": "standard_table",
        "created_at": 1789007722.0,
        "base_frame": "base_link",
        "surface": {
            "eligible": True,
            "height_m": 0.78342,
            "pose": {"x_m": 0.81, "y_m": -0.22, "z_m": 0.78342},
            "normal": [0.004, -0.003, 0.9999875],
        },
        "robot": {"id": "g2-102", "aid": "G2A-TEST", "host": "10.42.1.102"},
        "camera_calibration_version": "camera-v1",
        "robot_model": {"id": "G2_dahuan_safety", "sha256": "urdf-sha"},
    }


@pytest.mark.parametrize("aid", ["", "OTHER"])
def test_expected_robot_aid_cannot_be_missing_or_mismatched(tmp_path: Path, aid: str) -> None:
    data = measurement()
    data["robot"]["aid"] = aid
    repository = AlignmentSafetyProfileRepository(tmp_path / "safety-profiles")
    with pytest.raises(SafetyProfileGenerationError, match="AID"):
        repository.generate(
            measurement=data,
            measurement_path=tmp_path / "measurement.json",
            robot_id="g2a",
            template_path=write_template(tmp_path),
            expected_robot_aid="G2A-TEST",
        )


def test_alignment_measurement_generates_immutable_dated_safety_profile(
    tmp_path: Path,
) -> None:
    source = tmp_path / "alignment-surfaces/standard_table/versions/source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(measurement()), encoding="utf-8")
    repository = AlignmentSafetyProfileRepository(tmp_path / "safety-profiles")

    generated = repository.sync(
        measurements_root=tmp_path / "alignment-surfaces",
        robot_id="g2a",
        template_path=write_template(tmp_path),
        expected_robot_aid="G2A-TEST",
        allowed_robot_hosts={"10.42.1.102"},
    )

    assert len(generated) == 1
    assert generated[0].name.startswith("G2_dahuan_standard_table_20260910T")
    assert re.fullmatch(
        r"G2_dahuan_standard_table_20260910T103522\+0800_[0-9a-f]{8}\.json",
        generated[0].name,
    )
    profile = json.loads(generated[0].read_text(encoding="utf-8"))
    assert profile["surfaces"][0]["height"] == pytest.approx(0.78342)
    assert profile["surfaces"][0]["origin"] == pytest.approx([0.81, -0.22, 0.78342])
    assert profile["surfaces"][0]["normal"] == pytest.approx(
        [0.004, -0.003, 0.9999875]
    )
    assert profile["surface_calibration"]["measurement_id"] == measurement()["measurement_id"]
    assert profile["surface_calibration"]["camera_calibration_version"] == "camera-v1"
    assert profile["read_only_ready"] is True
    assert profile["execute_approved"] is True
    assert profile["calibrated"] is True

    # Reconciliation is deterministic and never creates a second mutable copy.
    assert repository.sync(
        measurements_root=tmp_path / "alignment-surfaces",
        robot_id="g2a",
        template_path=tmp_path / "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json",
        expected_robot_aid="G2A-TEST",
        allowed_robot_hosts={"10.42.1.102"},
    ) == generated


def test_unvalidated_inter_arm_geometry_cannot_inherit_execute_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "measurement.json"
    source.write_text(json.dumps(measurement()), encoding="utf-8")
    template = write_template(tmp_path)
    value = json.loads(template.read_text(encoding="utf-8"))
    value["inter_arm_collision"] = {
        "enabled": True,
        "validation_status": "read-only-candidate",
    }
    template.write_text(json.dumps(value), encoding="utf-8")

    output = AlignmentSafetyProfileRepository(tmp_path / "profiles").generate(
        measurement=measurement(),
        measurement_path=source,
        robot_id="g2a",
        template_path=template,
        expected_robot_aid="G2A-TEST",
    )
    profile = json.loads(output.read_text(encoding="utf-8"))

    assert profile["read_only_ready"] is True
    assert profile["execute_approved"] is False


def test_alignment_measurement_without_plane_geometry_is_rejected(tmp_path: Path) -> None:
    data = measurement()
    del data["surface"]["normal"]  # type: ignore[index]
    source = tmp_path / "measurement.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SafetyProfileGenerationError, match="origin or normal"):
        AlignmentSafetyProfileRepository(tmp_path / "profiles").generate(
            measurement=data,
            measurement_path=source,
            robot_id="g2a",
            template_path=write_template(tmp_path),
            expected_robot_aid="G2A-TEST",
        )

def test_sync_retires_existing_snapshot_with_missing_aid(tmp_path: Path) -> None:
    repository = AlignmentSafetyProfileRepository(tmp_path / "safety-profiles")
    data = measurement()
    data["robot"]["aid"] = ""
    template = write_template(tmp_path)
    (tmp_path / "measurement.json").write_text(json.dumps(data), encoding="utf-8")
    old = repository.generate(
        measurement=data, measurement_path=tmp_path / "measurement.json",
        robot_id="g2a", template_path=template,
    )
    assert repository.sync(
        measurements_root=tmp_path / "absent", robot_id="g2a",
        template_path=template, expected_robot_aid="G2A-TEST",
    ) == []
    assert old.exists()  # Preserve immutable audit evidence.


def test_sync_ignores_malformed_existing_calibration_metadata(tmp_path: Path) -> None:
    repository = AlignmentSafetyProfileRepository(tmp_path / "safety-profiles")
    robot_directory = repository.robot_directory("g2a")
    robot_directory.mkdir(parents=True)
    malformed = robot_directory / "malformed.json"
    malformed.write_text(
        json.dumps(
            {
                "generated_profile": {"source_template_sha256": "irrelevant"},
                "surface_calibration": [],
            }
        ),
        encoding="utf-8",
    )
    assert repository.sync(
        measurements_root=tmp_path / "absent",
        robot_id="g2a",
        template_path=write_template(tmp_path),
        expected_robot_aid="G2A-TEST",
    ) == []


def test_modified_generated_snapshot_is_preserved_but_not_selectable(tmp_path: Path) -> None:
    source = tmp_path / "alignment-surfaces/standard_table/versions/source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(measurement()), encoding="utf-8")
    repository = AlignmentSafetyProfileRepository(tmp_path / "safety-profiles")
    template = write_template(tmp_path)
    generated = repository.sync(
        measurements_root=tmp_path / "alignment-surfaces",
        robot_id="g2a",
        template_path=template,
        expected_robot_aid="G2A-TEST",
    )
    assert len(generated) == 1
    tampered_path = generated[0]
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["surfaces"][0]["height"] += 0.1
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")

    assert repository.sync(
        measurements_root=tmp_path / "alignment-surfaces",
        robot_id="g2a",
        template_path=template,
        expected_robot_aid="G2A-TEST",
    ) == []
    assert tampered_path.exists()


def test_template_change_retires_old_snapshot_without_deleting_audit_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "alignment-surfaces/standard_table/active.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(measurement()), encoding="utf-8")
    repository = AlignmentSafetyProfileRepository(tmp_path / "safety-profiles")
    template = write_template(tmp_path)
    old = repository.sync(
        measurements_root=tmp_path / "alignment-surfaces",
        robot_id="g2a",
        template_path=template,
        expected_robot_aid="G2A-TEST",
        allowed_robot_hosts={"10.42.1.102"},
    )
    payload = json.loads(template.read_text(encoding="utf-8"))
    payload["surfaces"][0]["hard_margin"] = 0.01
    template.write_text(json.dumps(payload), encoding="utf-8")

    current = repository.sync(
        measurements_root=tmp_path / "alignment-surfaces",
        robot_id="g2a",
        template_path=template,
        expected_robot_aid="G2A-TEST",
        allowed_robot_hosts={"10.42.1.102"},
    )

    assert len(old) == len(current) == 1
    assert old[0] != current[0]
    assert old[0].is_file()
    assert json.loads(current[0].read_text())["surfaces"][0]["hard_margin"] == 0.01


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_frame", "world", "base_link"),
        ("robot_host", "10.42.1.103", "robot address"),
        ("urdf_sha", "different", "URDF"),
    ],
)
def test_foreign_or_mismatched_measurement_is_rejected(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    data = measurement()
    if field == "robot_host":
        data["robot"] = {**data["robot"], "host": value}  # type: ignore[arg-type]
    elif field == "urdf_sha":
        data["robot_model"] = {**data["robot_model"], "sha256": value}  # type: ignore[arg-type]
    else:
        data[field] = value
    source = tmp_path / "source.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    repository = AlignmentSafetyProfileRepository(tmp_path / "profiles")

    with pytest.raises(SafetyProfileGenerationError, match=message):
        repository.generate(
            measurement=data,
            measurement_path=source,
            robot_id="g2a",
            template_path=write_template(tmp_path),
            expected_robot_aid="G2A-TEST",
            allowed_robot_hosts={"10.42.1.102"},
        )
