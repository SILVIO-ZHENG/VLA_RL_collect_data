#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path
from shutil import copytree
from uuid import uuid4

import numpy as np
import pytest

from nexus_deploy.robot.g2a.hitl.common import sidecar as sidecar_module
from nexus_deploy.robot.g2a.hitl.common.contract import EVENT_SCHEMA_VERSION
from nexus_deploy.robot.g2a.hitl.common.sidecar import (
    CAMERA_SOURCES,
    EVENTS_WAL,
    MANIFEST_FINAL,
    MANIFEST_PARTIAL,
    FlowPROSidecarWriter,
    inspect_vendor_episode,
    recover_open_sidecar,
    seal_sidecar,
    validate_sealed_sidecar,
)
from nexus_deploy.robot.g2a.hitl.offline.ingest import FilesystemBundlePuller, ImmutableRawStore

TIMESTAMPS = (1_788_253_877_000_000_000, 1_788_253_877_033_333_333, 1_788_253_877_066_666_666)
STATS_SHA256 = "1" * 64


@pytest.fixture(autouse=True)
def _read_synthetic_h5_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    def read(path: Path) -> list[int]:
        value = json.loads(path.read_text(encoding="utf-8"))
        return [int(timestamp) for timestamp in value["main_timestamps_ns"]]

    monkeypatch.setattr(sidecar_module, "_read_h5_timestamps", read)


def _make_vendor_episode(root: Path, name: str, *, offset_ns: int = 0, camera_skew_ns: int = 0) -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    (episode / "record").mkdir()
    (episode / "meta_info.json").write_text(
        json.dumps(
            {
                "clip_start_time": "2026-09-01T17:11:17+08:00",
                "clip_end_time": "2026-09-01T17:11:18+08:00",
                "data_validate": {"validate": True},
                "integrity": {"integrity": True, "reason": 0},
            }
        ),
        encoding="utf-8",
    )
    (episode / "record/aligned_joints.h5").write_text(
        json.dumps({"main_timestamps_ns": [timestamp + offset_ns for timestamp in TIMESTAMPS]}),
        encoding="utf-8",
    )
    for prefix in CAMERA_SOURCES.values():
        video = episode / f"{prefix}.h265"
        timestamp_file = episode / f"{prefix}.txt"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"synthetic-h265")
        timestamp_file.write_text(
            "".join(f"{value + offset_ns + camera_skew_ns} frame\n" for value in TIMESTAMPS),
            encoding="utf-8",
        )
    return episode


def _make_writer(root: Path, *, session_id: str | None = None) -> FlowPROSidecarWriter:
    return FlowPROSidecarWriter.create(
        root,
        flowpro_session_id=session_id or str(uuid4()),
        robot_id="g2-sorter-01",
        task_id="package_sorting",
        checkpoint_id="pi05-g2a-test",
        checkpoint_stats_sha256=STATS_SHA256,
        native_action_dim=16,
        chunk_size=50,
        effector_profile="dh",
        processor_version="pi05-g2a/v1",
    )


def _observation_event(writer: FlowPROSidecarWriter, observation_id: str, timestamp_ns: int) -> dict:
    state = writer.store_observation_state(observation_id, np.arange(24, dtype=np.float32))
    images = {
        camera_key: writer.store_inference_jpeg(
            b"\xff\xd8synthetic-jpeg\xff\xd9",
            camera_key=camera_key,
            vendor_timestamp_ns=timestamp_ns,
            width=640,
            height=400,
        )
        for camera_key in CAMERA_SOURCES
    }
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": str(uuid4()),
        "flowpro_session_id": writer.flowpro_session_id,
        "event_type": "ObservationCaptured",
        "producer": "g2_supervisor",
        "monotonic_ns": 100,
        "wall_time_utc": "2026-09-01T09:11:17Z",
        "vendor_timestamp_ns": timestamp_ns,
        "observation_id": observation_id,
        "control_owner": "hold",
        "payload": {
            "anchor_timestamp_ns": timestamp_ns,
            "state": state,
            "task": "sort the parcel",
            "images": images,
        },
    }


def _record_candidate(writer: FlowPROSidecarWriter, observation_id: str) -> dict:
    return writer.record_policy_candidate(
        np.arange(50 * 16, dtype=np.float32).reshape(50, 16),
        event_id=str(uuid4()),
        observation_id=observation_id,
        request_id="infer-000001",
        candidate_id=str(uuid4()),
        generation_id=str(uuid4()),
        monotonic_ns=200,
        wall_time_utc="2026-09-01T09:11:18Z",
        control_owner="hold",
    )


def test_sidecar_seals_and_revalidates_complete_vendor_join(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "w-9.1_20260901171117")
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    writer.append_event(_observation_event(writer, observation_id, TIMESTAMPS[0]))
    image = writer.store_inference_jpeg(
        b"\xff\xd8synthetic-jpeg\xff\xd9",
        camera_key="cam_head",
        vendor_timestamp_ns=TIMESTAMPS[0],
        width=640,
        height=400,
    )
    assert (writer.open_directory / image["relative_path"]).is_file()
    candidate_event = _record_candidate(writer, observation_id)
    writer.close()

    open_directory = writer.open_directory
    sealed = seal_sidecar(open_directory, episode, vendor_record_root=record_root)
    retried = seal_sidecar(open_directory, episode, vendor_record_root=record_root)
    report = validate_sealed_sidecar(
        sealed,
        vendor_episode_path=episode,
        vendor_record_root=record_root,
    )

    assert not open_directory.exists()
    assert retried == sealed
    assert sealed.name.endswith(".sealed")
    assert report.flowpro_session_id == writer.flowpro_session_id
    assert report.record_name == episode.name
    assert report.event_count == 3
    assert report.observation_count == 1
    assert report.candidate_count == 1
    assert candidate_event["payload"]["size_bytes"] == 50 * 16 * 4
    assert not (sealed / MANIFEST_PARTIAL).exists()
    manifest = json.loads((sealed / MANIFEST_FINAL).read_text(encoding="utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["vendor_episode"]["episode_path"] == str(episode.resolve())
    assert {item["relative_path"] for item in manifest["sidecar_files"]} >= {
        EVENTS_WAL,
        image["relative_path"],
        candidate_event["payload"]["relative_path"],
    }


def test_sealed_validation_allows_hash_checked_cross_host_staging_remap(tmp_path: Path) -> None:
    source_root = tmp_path / "g2-record"
    episode = _make_vendor_episode(source_root, "L-9.2_20260902160655")
    writer = _make_writer(tmp_path / "sidecars")
    writer.close()
    sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=source_root)

    staging_root = tmp_path / "5090-staging"
    staged_episode = copytree(episode, staging_root / episode.name)
    report = validate_sealed_sidecar(
        sealed,
        vendor_episode_path=staged_episode,
        vendor_record_root=staging_root,
    )

    assert report.record_name == episode.name
    with pytest.raises(ValueError, match="<vendor_record_root>/<record_name>"):
        validate_sealed_sidecar(
            sealed,
            vendor_episode_path=staged_episode,
            vendor_record_root=tmp_path,
        )


def test_thor_puller_archives_copied_episode_under_local_record_root(tmp_path: Path) -> None:
    g2_record_root = tmp_path / "g2-record"
    episode = _make_vendor_episode(g2_record_root, "L-9.2_20260902160655")
    writer = _make_writer(tmp_path / "g2-sidecars")
    writer.close()
    sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=g2_record_root)

    thor_sidecars = tmp_path / "thor" / "sidecars"
    thor_vendor = tmp_path / "thor" / "vendor"
    copytree(sealed, thor_sidecars / sealed.name)
    copytree(episode, thor_vendor / episode.name)
    store = ImmutableRawStore(tmp_path / "thor" / "store")
    report = FilesystemBundlePuller(
        sidecar_root=thor_sidecars,
        vendor_record_root=thor_vendor,
        raw_store=store,
    ).pull_once()

    assert report.failed == []
    assert report.ingested_bundle_ids == [writer.flowpro_session_id]
    assert store.verify_receipt(writer.flowpro_session_id).record_name == episode.name


def test_one_nexus_run_archives_two_separate_tablet_recordings(tmp_path: Path) -> None:
    g2_record_root = tmp_path / "g2-record"
    thor_sidecars = tmp_path / "thor" / "sidecars"
    thor_vendor = tmp_path / "thor" / "vendor"
    session_ids = []
    for name in ("parcel-001", "parcel-002"):
        episode = _make_vendor_episode(g2_record_root, name)
        writer = _make_writer(tmp_path / "g2-sidecars")
        writer.close()
        sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=g2_record_root)
        copytree(sealed, thor_sidecars / sealed.name)
        copytree(episode, thor_vendor / episode.name)
        session_ids.append(writer.flowpro_session_id)

    store = ImmutableRawStore(tmp_path / "thor" / "store")
    report = FilesystemBundlePuller(
        sidecar_root=thor_sidecars,
        vendor_record_root=thor_vendor,
        raw_store=store,
    ).pull_once()
    assert report.failed == []
    assert set(report.ingested_bundle_ids) == set(session_ids)
    assert {store.verify_receipt(session_id).record_name for session_id in session_ids} == {
        "parcel-001", "parcel-002",
    }


@pytest.mark.parametrize(
    ("data_validate", "integrity", "message"),
    [
        ({"validate": False}, {"integrity": True}, "data_validate.validate"),
        ({"validate": True}, {"integrity": False}, "integrity.integrity"),
        (True, {"integrity": True}, "data_validate must be an object"),
        ({"validate": True}, True, "integrity must be an object"),
        (None, None, "active-recording placeholder"),
    ],
)
def test_vendor_meta_requires_final_nested_validation(
    tmp_path: Path,
    data_validate: object,
    integrity: object,
    message: str,
) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "invalid-meta")
    meta_path = episode / "meta_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["data_validate"] = data_validate
    meta["integrity"] = integrity
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        inspect_vendor_episode(episode, vendor_record_root=record_root)


def test_join_rejects_policy_candidate_without_observation_event(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "missing-observation")
    writer = _make_writer(tmp_path / "sidecars")
    _record_candidate(writer, str(uuid4()))
    writer.close()

    with pytest.raises(ValueError, match="missing ObservationCaptured"):
        seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)

    assert writer.open_directory.exists()
    events = [json.loads(line) for line in (writer.open_directory / EVENTS_WAL).read_text().splitlines()]
    assert all(event["event_type"] != "DataSealed" for event in events)


def test_wrong_vendor_episode_is_rejected_without_poisoning_retry(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    correct = _make_vendor_episode(record_root, "correct")
    wrong = _make_vendor_episode(record_root, "wrong", offset_ns=10_000_000_000)
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    writer.append_event(_observation_event(writer, observation_id, TIMESTAMPS[1]))
    _record_candidate(writer, observation_id)
    writer.close()

    with pytest.raises(ValueError, match="outside the bound episode"):
        seal_sidecar(writer.open_directory, wrong, vendor_record_root=record_root)

    sealed = seal_sidecar(writer.open_directory, correct, vendor_record_root=record_root)
    assert sealed.is_dir()


def test_seal_allows_legacy_tablet_finished_event_after_clip_end(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "tablet-finished-after-end")
    writer = _make_writer(tmp_path / "sidecars")
    writer.append_event(
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "LogicalInput",
            "producer": "g2_supervisor",
            "monotonic_ns": 1,
            "wall_time_utc": "2026-09-08T00:00:00Z",
            "vendor_timestamp_ns": TIMESTAMPS[-1] + 1_000_000_000,
            "control_owner": "hold",
            "payload": {
                "name": "tablet_recording_finished",
                "outcome_authority": "external_tablet_recorder",
            },
        }
    )
    writer.close()

    sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)

    assert sealed.is_dir()


def test_seal_still_rejects_other_logical_input_after_clip_end(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "non-lifecycle-after-end")
    writer = _make_writer(tmp_path / "sidecars")
    writer.append_event(
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "LogicalInput",
            "producer": "g2_supervisor",
            "monotonic_ns": 1,
            "wall_time_utc": "2026-09-08T00:00:00Z",
            "vendor_timestamp_ns": TIMESTAMPS[-1] + 1_000_000_000,
            "control_owner": "hold",
            "payload": {"name": "y_press"},
        }
    )
    writer.close()

    with pytest.raises(ValueError, match="outside the bound episode"):
        seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)

    assert writer.open_directory.is_dir()


def test_seal_allows_recovery_lifecycle_event_after_monotonic_epoch_reset(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "reboot-recovery")
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    observation = _observation_event(writer, observation_id, TIMESTAMPS[0])
    observation["monotonic_ns"] = 10_000_000_000
    writer.append_event(observation)
    writer.append_event(
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "EpisodeOutcome",
            "producer": "g2_supervisor",
            "monotonic_ns": 100,
            "wall_time_utc": "2026-09-02T09:11:17Z",
            "vendor_timestamp_ns": None,
            "control_owner": "hold",
            "payload": {"outcome": "success", "detail": {"recovered": True}},
        }
    )
    writer.close()

    sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)
    assert sealed.is_dir()


def test_seal_still_rejects_control_event_after_monotonic_regression(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "bad-control-order")
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    observation = _observation_event(writer, observation_id, TIMESTAMPS[0])
    observation["monotonic_ns"] = 10_000
    writer.append_event(observation)
    writer.append_event(
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": str(uuid4()),
            "flowpro_session_id": writer.flowpro_session_id,
            "event_type": "LogicalInput",
            "producer": "g2_supervisor",
            "monotonic_ns": 1,
            "wall_time_utc": "2026-09-02T09:11:17Z",
            "vendor_timestamp_ns": None,
            "control_owner": "hold",
            "payload": {"name": "synthetic_regression"},
        }
    )
    writer.close()

    with pytest.raises(ValueError, match="monotonic_ns moved backwards"):
        seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)


def test_crash_partial_wal_tail_requires_explicit_recovery(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "wal-recovery")
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    writer.append_event(_observation_event(writer, observation_id, TIMESTAMPS[0]))
    with (writer.open_directory / EVENTS_WAL).open("ab") as stream:
        stream.write(b'{"partial_event":')
    writer.close()

    with pytest.raises(ValueError, match="crash-partial"):
        seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)

    recovery = recover_open_sidecar(writer.open_directory)
    assert recovery["recovered"] is True
    assert recovery["discarded_bytes"] > 0
    reopened = FlowPROSidecarWriter.open_existing(writer.open_directory)
    _record_candidate(reopened, observation_id)
    reopened.close()
    sealed = seal_sidecar(reopened.open_directory, episode, vendor_record_root=record_root)
    assert (sealed / "recovery.jsonl").is_file()


def test_sealed_validation_detects_sidecar_tampering(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "tamper-sidecar")
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    writer.append_event(_observation_event(writer, observation_id, TIMESTAMPS[0]))
    candidate = _record_candidate(writer, observation_id)
    writer.close()
    sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)

    candidate_path = sealed / candidate["payload"]["relative_path"]
    with candidate_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="Sidecar files|artifact mismatch"):
        validate_sealed_sidecar(
            sealed,
            vendor_episode_path=episode,
            vendor_record_root=record_root,
        )


def test_sealed_validation_detects_vendor_tampering(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    episode = _make_vendor_episode(record_root, "tamper-vendor")
    writer = _make_writer(tmp_path / "sidecars")
    observation_id = str(uuid4())
    writer.append_event(_observation_event(writer, observation_id, TIMESTAMPS[0]))
    _record_candidate(writer, observation_id)
    writer.close()
    sealed = seal_sidecar(writer.open_directory, episode, vendor_record_root=record_root)

    camera_timestamp = episode / f"{CAMERA_SOURCES['cam_head']}.txt"
    camera_timestamp.write_text(
        "".join(f"{value + 1_000_000} frame\n" for value in TIMESTAMPS),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no longer matches"):
        validate_sealed_sidecar(
            sealed,
            vendor_episode_path=episode,
            vendor_record_root=record_root,
        )


def test_vendor_inspection_rejects_missing_files_or_excessive_skew(tmp_path: Path) -> None:
    record_root = tmp_path / "record"
    incomplete = _make_vendor_episode(record_root, "incomplete")
    (incomplete / f"{CAMERA_SOURCES['cam_head']}.h265").unlink()
    with pytest.raises(FileNotFoundError, match="missing required files"):
        inspect_vendor_episode(incomplete, vendor_record_root=record_root)

    skewed = _make_vendor_episode(record_root, "skewed", camera_skew_ns=41_000_000)
    with pytest.raises(ValueError, match="exceeds|reuses"):
        inspect_vendor_episode(skewed, vendor_record_root=record_root)


def test_candidate_id_is_idempotent_only_for_identical_bytes(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path / "sidecars")
    candidate_id = str(uuid4())
    values = np.zeros((50, 16), dtype=np.float32)

    first = writer.store_candidate(candidate_id, values)
    second = writer.store_candidate(candidate_id, values.copy())
    assert first == second

    with pytest.raises(ValueError, match="collision"):
        writer.store_candidate(candidate_id, np.ones((50, 16), dtype=np.float32))


def test_rollback_plan_source_timestamps_are_persisted_immutably(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path / "sidecars")
    intervention_id = str(uuid4())
    (writer.open_directory / "executed_segments" / intervention_id).mkdir(parents=True)
    waypoints = np.zeros((3, 24), dtype=np.float32)
    timestamps = np.asarray([300, 200, 100], dtype=np.int64)

    plan = writer.store_rollback_plan(intervention_id, waypoints)
    source_times = writer.store_rollback_plan_source_timestamps(intervention_id, timestamps)

    assert plan["shape"] == [3, 24]
    assert source_times["shape"] == [3]
    stored = np.fromfile(writer.open_directory / source_times["relative_path"], dtype="<i8")
    np.testing.assert_array_equal(stored, timestamps)
