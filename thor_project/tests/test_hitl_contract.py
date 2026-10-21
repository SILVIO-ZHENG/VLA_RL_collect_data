"""The archived data contract follows the selected model, not an r14 preset."""

import json
from uuid import uuid4

from nexus_deploy.robot.g2a.hitl.common.contract import validate_manifest
from nexus_deploy.robot.g2a.hitl.common.sidecar import FlowPROSidecarWriter


def test_sidecar_keeps_selected_model_cadence_and_effector(tmp_path):
    writer = FlowPROSidecarWriter.create(
        tmp_path,
        flowpro_session_id=str(uuid4()),
        robot_id="g2a",
        task_id="sorting",
        checkpoint_id="current-thor-model",
        checkpoint_stats_sha256="a" * 64,
        native_action_dim=16,
        chunk_size=50,
        effector_profile="dahuan",
        processor_version="nexus-pi05-g2a/v1",
        model_fps=25.0,
    )
    manifest = json.loads((writer.open_directory / "manifest.partial.json").read_text())
    assert manifest["policy"]["model_fps"] == 25.0
    assert manifest["policy"]["effector_profile"] == "dahuan"
    validate_manifest(manifest)
