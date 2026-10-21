"""Conservative coverage and broad/narrow semantics for the G2 terminal model."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_deploy.robot.g2a.safety_layer import (
    CollisionCapsule,
    InterArmCollisionConstraint,
)


ASSET = ROOT / "src/nexus_deploy/viewer/assets/g2/G2_dahuan"
OVERLAY = ASSET / "collision_overlay.json"
URDF = ASSET / "G2_dahuan.urdf"
BUILDER = runpy.run_path(str(ROOT / "tools/build_g2_interarm_capsules.py"))
PROFILE_BUILDER = runpy.run_path(str(ROOT / "tools/build_g2_readonly_v2_profile.py"))


def _distance_to_capsule(points: np.ndarray, capsule: dict[str, object]) -> np.ndarray:
    start = np.asarray(capsule["start"], dtype=np.float64)
    end = np.asarray(capsule["end"], dtype=np.float64)
    segment = end - start
    parameter = np.clip(
        (points - start) @ segment / float(np.dot(segment, segment)), 0.0, 1.0
    )
    return np.linalg.norm(points - (start + parameter[:, None] * segment), axis=1)


def test_generated_v2_overlay_is_reproducible_and_triangle_conservative() -> None:
    overlay = json.loads(OVERLAY.read_text())
    assert BUILDER["build"](URDF, overlay) == overlay
    assert overlay["geometry"] == "cross-chain-capsules-v2"
    assert overlay["validation_status"] == "read-only-candidate"

    triangles = BUILDER["_terminal_triangles"](URDF)
    for side, prefix in (("left", "l"), ("right", "r")):
        capsules = overlay[f"{side}_capsules"]
        for part in ("base", "inner", "outer"):
            frame = f"gripper_{prefix}_{part}_link" + ("" if part == "base" else "1")
            if part == "base":
                frame = f"gripper_{prefix}_base_link"
            candidates = [capsule for capsule in capsules if capsule["frame"] == frame]
            assert candidates
            for triangle in triangles[frame]:
                # A capsule is convex. Enclosing all three vertices therefore
                # encloses the complete triangle, not only sampled points.
                assert any(
                    np.max(_distance_to_capsule(triangle, capsule))
                    <= float(capsule["radius"]) + 1e-9
                    for capsule in candidates
                )


def test_v2_execute_profile_is_separate_and_explicitly_approved() -> None:
    source = {
        "display_name": "v1",
        "calibrated": True,
        "read_only_ready": True,
        "execute_approved": True,
        "execution_approval": {
            "surface_hard_margin_m": 0.003,
            "surface_warning_margin_m": 0.005,
        },
    }
    overlay = json.loads(OVERLAY.read_text())
    read_only = PROFILE_BUILDER["build"](source, overlay, OVERLAY)
    execute = PROFILE_BUILDER["build"](
        source, overlay, OVERLAY, execute_approved=True
    )

    assert read_only["execute_approved"] is False
    assert read_only["inter_arm_collision"]["validation_status"] == "read-only-candidate"
    assert "execution_approval" not in read_only
    assert execute["execute_approved"] is True
    assert execute["inter_arm_collision"]["validation_status"] == "execute-approved"
    assert execute["execution_approval"]["inter_arm_geometry"] == "cross-chain-capsules-v2"
    assert execute["execution_approval"]["surface_hard_margin_m"] == 0.003


class _TranslatingFK:
    def frame_poses(self, q, frames):
        poses = []
        for frame in frames:
            pose = np.eye(4)
            if frame.startswith("right"):
                pose[0, 3] = float(q[0])
            poses.append(pose)
        return tuple(poses)


def test_broad_phase_skips_far_narrow_pass_but_true_overlap_uses_narrow() -> None:
    broad_left = (CollisionCapsule("broad_l", "left", (-1, 0, 0), (1, 0, 0), .2),)
    broad_right = (CollisionCapsule("broad_r", "right", (-1, 0, 0), (1, 0, 0), .2),)
    fine_left = (CollisionCapsule("fine_l", "left", (-.2, 0, 0), (.2, 0, 0), .05),)
    fine_right = (CollisionCapsule("fine_r", "right", (-.2, 0, 0), (.2, 0, 0), .05),)
    constraint = InterArmCollisionConstraint(
        fine_left,
        fine_right,
        hard_margin=.008,
        warning_margin=.020,
        broad_left=broad_left,
        broad_right=broad_right,
        broad_phase_activation_margin=.030,
    )
    far = constraint.evaluate(np.asarray([3.0]), _TranslatingFK())
    overlap = constraint.evaluate(np.asarray([0.0]), _TranslatingFK())
    assert far.phase == "broad" and far.clearance > 0
    assert overlap.phase == "narrow" and overlap.clearance < 0
