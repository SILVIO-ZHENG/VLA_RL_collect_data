from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAFETY_SOURCE = (
    PROJECT_ROOT
    / "src"
    / "nexus_deploy"
    / "robot"
    / "g2a"
    / "safety_layer.py"
)
ROBOT_CLIENT_SOURCE = SAFETY_SOURCE.parent
OFFICIAL_SAFETY_TEMPLATE = ROBOT_CLIENT_SOURCE / "SAFETY_CONFIG_TEMPLATE.json"
DAHUAN_SAFETY_TEMPLATE = ROBOT_CLIENT_SOURCE / "G2_DAHUAN_SAFETY_CONFIG_TEMPLATE.json"
SPEC = importlib.util.spec_from_file_location("g2a_robot_safety_layer", SAFETY_SOURCE)
assert SPEC is not None and SPEC.loader is not None
safety = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safety
SPEC.loader.exec_module(safety)
LEROBOT_ROOT = PROJECT_ROOT
sys.path.insert(0, str(LEROBOT_ROOT))
gdk_adapter = importlib.import_module("nexus_deploy.robot.g2a.gdk_adapter")


class TranslationFK:
    """Test FK: joint 0 is tool Z; joint 1 rotates the tool around X."""

    def frame_pose(self, joint_positions: np.ndarray, frame: str) -> np.ndarray:
        assert frame == "tool"
        z, angle = joint_positions[:2]
        cosine, sine = math.cos(float(angle)), math.sin(float(angle))
        pose = np.eye(4)
        pose[:3, :3] = (
            (1.0, 0.0, 0.0),
            (0.0, cosine, -sine),
            (0.0, sine, cosine),
        )
        pose[2, 3] = z
        return pose


def make_layer(*, points=(safety.CollisionPoint("tcp_bottom", "tool", (0.0, 0.0, 0.0)),)):
    return safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(points),
        surface=safety.TableConstraint(height=0.0, hard_margin=0.02),
        motion_indices=(0, 1),
        passthrough_indices=(2,),
        path_samples=8,
        search_iterations=32,
    )


def write_calibrated_official_config(tmp_path: Path, *, table_height: float) -> Path:
    data = json.loads(OFFICIAL_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["calibrated"] = True
    data["robot_model"]["path"] = str(
        ROBOT_CLIENT_SOURCE / data["robot_model"]["path"]
    )
    for mesh in data["collision_meshes"]:
        mesh["path"] = str(ROBOT_CLIENT_SOURCE / mesh["path"])
    data["surfaces"] = [
        {
            "name": "table",
            "enabled": True,
            "frame": "base_link",
            "height": table_height,
            "hard_margin": 0.02,
            "xy_bounds": None,
        }
    ]
    path = tmp_path / f"official_{table_height:.4f}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_calibrated_dahuan_config(tmp_path: Path, *, table_height: float = 0.0) -> Path:
    data = json.loads(DAHUAN_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["calibrated"] = True
    data["robot_model"]["path"] = str(
        ROBOT_CLIENT_SOURCE / data["robot_model"]["path"]
    )
    for mesh in data["collision_meshes"]:
        mesh["path"] = str(ROBOT_CLIENT_SOURCE / mesh["path"])
    data["surfaces"][0]["height"] = table_height
    data["surfaces"][0]["origin"][2] = table_height
    path = tmp_path / "dahuan.calibrated.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_twenty_centimetres_high_one_centimetre_down_is_unchanged() -> None:
    layer = make_layer()
    current = np.asarray([0.20, 0.0, -0.2])
    target = np.asarray([0.19, 0.0, -0.5])

    result = layer.project(current, target)

    np.testing.assert_allclose(result.safe_action, target)
    assert result.hard_limit_triggered is False
    assert result.action_scale == 1.0
    assert result.current_z_min == pytest.approx(0.20)


def test_warning_margin_reports_proximity_without_blocking() -> None:
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (safety.CollisionPoint("finger", "tool", (0.0, 0.0, 0.0)),)
        ),
        surface=safety.TableConstraint(
            height=0.0,
            hard_margin=0.001,
            warning_margin=0.003,
        ),
        motion_indices=(0, 1),
        passthrough_indices=(2,),
    )
    pose = np.asarray([0.002, 0.0, 0.0])

    result = layer.project(pose, pose)

    assert result.hard_limit_triggered is False
    assert result.soft_limit_triggered is True
    assert result.distance_to_surface == pytest.approx(0.002)


def test_calibrated_tilted_plane_blocks_recorded_right_finger_contact_pose() -> None:
    # Recorded 2026-09-15 pose: the old horizontal-height check reported
    # 4.194 mm clearance. The AprilTag plane is 2.883 mm higher at this XY,
    # leaving only 1.311 mm and therefore requiring a hard hold.
    point = safety.CollisionPoint(
        "right_inner_finger:vertex_420:opening_box_1",
        "tool",
        (0.552075618, -0.0777523275, 0.7317839635),
    )

    class FixedFK:
        def frame_pose(self, _joint_positions: np.ndarray, frame: str) -> np.ndarray:
            assert frame == "tool"
            return np.eye(4)

    layer = safety.ActionSafetyLayer(
        fk=FixedFK(),
        geometry=safety.GripperGeometry((point,)),
        surface=safety.TableConstraint(
            name="work_surface",
            height=0.72759,
            origin=(0.83664, -0.41725, 0.72759),
            normal=(0.0048785, -0.0044026, 0.9999784),
            hard_margin=0.004,
            warning_margin=0.007,
        ),
        motion_indices=(0,),
        passthrough_indices=(),
    )

    result = layer.guard(np.zeros(1), np.zeros(1))

    assert result.distance_to_surface == pytest.approx(0.00131095, abs=1e-8)
    assert result.hard_limit_triggered is True
    assert result.limiting_point == point.name


def test_surface_normal_is_normalized_and_must_point_upward() -> None:
    constraint = safety.TableConstraint(
        height=0.75,
        origin=(0.1, 0.2, 0.75),
        normal=(0.0, 0.2, 2.0),
    )
    assert np.linalg.norm(constraint.normal) == pytest.approx(1.0)
    with pytest.raises(safety.SafetyConfigurationError, match="point upward"):
        safety.TableConstraint(height=0.75, normal=(0.0, 0.0, -1.0))


def test_warning_margin_cannot_be_smaller_than_hard_margin() -> None:
    with pytest.raises(safety.SafetyConfigurationError, match="Warning margin"):
        safety.TableConstraint(
            height=0.0,
            hard_margin=0.003,
            warning_margin=0.001,
        )


def test_hard_margin_blocks_further_downward_motion_and_preserves_gripper_command() -> None:
    layer = make_layer()
    current = np.asarray([0.02, 0.0, -0.2])
    target = np.asarray([0.00, 0.0, -0.7])

    result = layer.project(current, target)

    assert result.hard_limit_triggered is True
    assert result.action_scale == pytest.approx(0.0, abs=1e-8)
    assert result.safe_action[0] == pytest.approx(current[0])
    # Gripper open/close is passthrough, not multiplied by arm motion alpha.
    assert result.safe_action[2] == pytest.approx(target[2])
    assert result.safe_predicted_z_min >= 0.02 - 1e-6


def test_candidate_gripper_target_does_not_resize_collision_geometry() -> None:
    class OpeningSensitiveFK:
        """Expose accidental passthrough use by mapping gripper opening to tool Z."""

        def frame_pose(self, joint_positions: np.ndarray, frame: str) -> np.ndarray:
            assert frame == "tool"
            pose = np.eye(4)
            pose[2, 3] = float(joint_positions[2])
            return pose

    layer = safety.ActionSafetyLayer(
        fk=OpeningSensitiveFK(),
        geometry=safety.GripperGeometry(
            (safety.CollisionPoint("finger", "tool", (0.0, 0.0, 0.0)),)
        ),
        surface=safety.TableConstraint(height=0.0, hard_margin=0.02),
        motion_indices=(0, 1),
        passthrough_indices=(2,),
        path_samples=4,
    )
    current = np.asarray([0.0, 0.0, 0.10])
    target = np.asarray([0.0, 0.0, -0.10])

    projected = layer.project(current, target)
    guarded = layer.guard(current, target)

    assert projected.hard_limit_triggered is False
    assert guarded.hard_limit_triggered is False
    assert projected.predicted_z_min == pytest.approx(0.10)
    assert guarded.predicted_z_min == pytest.approx(0.10)
    assert projected.safe_action[2] == pytest.approx(target[2])
    assert guarded.safe_action[2] == pytest.approx(target[2])


def test_binary_projection_finds_largest_safe_motion_fraction() -> None:
    layer = make_layer()
    current = np.asarray([0.03, 0.0, -0.2])
    target = np.asarray([0.00, 0.0, -0.6])

    result = layer.project(current, target)

    assert result.hard_limit_triggered is True
    assert result.action_scale == pytest.approx(1.0 / 3.0, abs=2e-6)
    assert result.safe_action[0] == pytest.approx(0.02, abs=1e-6)
    assert result.safe_action[2] == pytest.approx(-0.6)


def test_wrist_rotation_with_constant_tcp_height_detects_fingertip_dip() -> None:
    points = (
        safety.CollisionPoint("palm_bottom", "tool", (0.0, 0.0, 0.0)),
        safety.CollisionPoint("fingertip", "tool", (0.0, 0.10, 0.0)),
    )
    layer = make_layer(points=points)
    current = np.asarray([0.11, 0.0, -0.2])
    # TCP translation z stays 0.11 m; rotation sends the fingertip to 0.01 m.
    target = np.asarray([0.11, -math.pi / 2, -0.2])

    result = layer.project(current, target)

    assert result.predicted_z_min == pytest.approx(0.01, abs=1e-7)
    assert result.hard_limit_triggered is True
    assert result.action_scale < 1.0
    assert result.limiting_point == "fingertip"
    assert result.safe_predicted_z_min >= 0.02 - 1e-6


def test_minimum_z_uses_all_geometry_points_not_tcp_origin() -> None:
    points = (
        safety.CollisionPoint("tcp", "tool", (0.0, 0.0, 0.0)),
        safety.CollisionPoint("finger_bottom", "tool", (0.0, 0.0, -0.04)),
        safety.CollisionPoint("palm_bottom", "tool", (0.0, 0.0, -0.02)),
    )
    layer = make_layer(points=points)

    minimum = layer.minimum_z(np.asarray([0.10, 0.0, -0.2]))

    assert minimum.point_name == "finger_bottom"
    assert minimum.z == pytest.approx(0.06)


def test_uncalibrated_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "safety.json"
    path.write_text(json.dumps({"version": 1, "calibrated": False}), encoding="utf-8")

    with pytest.raises(safety.SafetyConfigurationError, match="calibrated=true"):
        safety.load_safety_layer(path)


def test_read_only_ready_config_requires_explicit_loader_opt_in(tmp_path: Path) -> None:
    data = json.loads(DAHUAN_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["robot_model"]["path"] = str(
        ROBOT_CLIENT_SOURCE / data["robot_model"]["path"]
    )
    for mesh in data["collision_meshes"]:
        mesh["path"] = str(ROBOT_CLIENT_SOURCE / mesh["path"])
    path = tmp_path / "dahuan.read-only.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(safety.SafetyConfigurationError, match="allow_read_only_ready=True"):
        safety.load_safety_layer(path)

    layer = safety.load_safety_layer(path, allow_read_only_ready=True)
    assert isinstance(layer.fk, safety.URDFForwardKinematics)
    assert len(layer.geometry.points) > 1000
    assert layer.surfaces.surfaces[0].xy_bounds == (0.3, 1.0, -0.8, 0.8)
    assert layer.surfaces.surfaces[0].hard_margin == pytest.approx(0.003)
    assert layer.surfaces.surfaces[0].warning_margin == pytest.approx(0.005)


def test_calibrated_serial_chain_config_drives_fk_projection(tmp_path: Path) -> None:
    config = {
        "version": 1,
        "calibrated": True,
        "base_frame": "base",
        "surface": {"frame": "base", "height": 0.0, "hard_margin": 0.02},
        "motion_indices": [0, 1],
        "passthrough_indices": [2],
        "frame_chains": {
            "tool": {
                "base": {"xyz": [0.0, 0.0, 0.11]},
                "joints": [
                    {
                        "action_index": 1,
                        "axis": [1.0, 0.0, 0.0],
                        "origin": {"xyz": [0.0, 0.0, 0.0]},
                    }
                ],
                "frame": {"xyz": [0.0, 0.0, 0.0]},
            }
        },
        "collision_points": [
            {"name": "fingertip", "frame": "tool", "xyz": [0.0, 0.10, 0.0]}
        ],
    }
    path = tmp_path / "calibrated.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    layer = safety.load_safety_layer(path)

    result = layer.project(
        np.asarray([0.0, 0.0, -0.2]),
        np.asarray([0.0, -math.pi / 2, -0.2]),
    )

    assert result.hard_limit_triggered is True
    assert result.predicted_z_min == pytest.approx(0.01, abs=1e-7)


def test_official_g2_urdf_meshes_and_joint_mapping_load(tmp_path: Path) -> None:
    layer = safety.load_safety_layer(
        write_calibrated_official_config(tmp_path, table_height=0.0)
    )
    configuration = np.zeros(24)
    configuration[7] = configuration[15] = 0.025

    minimum = layer.minimum_z(configuration)
    open_pose = layer.fk.frame_pose(configuration, "gripper_l_outer_link4")
    configuration[7] = 0.0
    closed_pose = layer.fk.frame_pose(configuration, "gripper_l_outer_link4")

    assert isinstance(layer.fk, safety.URDFForwardKinematics)
    assert len(layer.geometry.points) == 1128
    assert minimum.point_name.startswith(("left_", "right_"))
    assert minimum.frame.startswith("gripper_")
    # The passive outer chain follows the configured active GDK gripper joint.
    assert not np.allclose(closed_pose, open_pose)


def test_official_config_table_height_changes_hard_constraint(tmp_path: Path) -> None:
    low_layer = safety.load_safety_layer(
        write_calibrated_official_config(tmp_path, table_height=0.0)
    )
    configuration = np.zeros(24)
    configuration[7] = configuration[15] = 0.025
    current_minimum = low_layer.minimum_z(configuration).z
    high_layer = safety.load_safety_layer(
        write_calibrated_official_config(tmp_path, table_height=current_minimum - 0.01)
    )

    low_result = low_layer.project(configuration, configuration)
    high_result = high_layer.project(configuration, configuration)

    assert low_result.hard_limit_triggered is False
    assert high_result.hard_limit_triggered is True
    assert high_result.limiting_surface == "table"


def test_dahuan_hybrid_urdf_and_terminal_collision_meshes_load(tmp_path: Path) -> None:
    layer = safety.load_safety_layer(write_calibrated_dahuan_config(tmp_path))
    opened = np.zeros(24)
    opened[7] = opened[15] = 0.025
    closed = opened.copy()
    closed[7] = closed[15] = 0.0

    open_inner = layer.fk.frame_pose(opened, "gripper_l_inner_link1")
    open_outer = layer.fk.frame_pose(opened, "gripper_l_outer_link1")
    closed_inner = layer.fk.frame_pose(closed, "gripper_l_inner_link1")
    closed_outer = layer.fk.frame_pose(closed, "gripper_l_outer_link1")

    assert isinstance(layer.fk, safety.URDFForwardKinematics)
    assert len(layer.geometry.points) > 1000
    assert not np.allclose(open_inner, closed_inner)
    assert not np.allclose(open_outer, closed_outer)
    # Both fingers are driven from the same 24D left-gripper state via the
    # controlled URDF mimic relationship.
    assert np.linalg.norm(open_inner[:3, 3] - closed_inner[:3, 3]) == pytest.approx(0.025)
    assert np.linalg.norm(open_outer[:3, 3] - closed_outer[:3, 3]) == pytest.approx(0.025)


def test_generated_profile_can_resolve_assets_from_explicit_robot_client_root(
    tmp_path: Path,
) -> None:
    data = json.loads(DAHUAN_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["calibrated"] = True
    profile = tmp_path / "runtime/safety-profiles/g2a/generated.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps(data), encoding="utf-8")

    layer = safety.load_safety_layer(profile, asset_root=ROBOT_CLIENT_SOURCE)

    assert isinstance(layer.fk, safety.URDFForwardKinematics)
    assert len(layer.geometry.points) == 1190


def test_dahuan_convex_support_vertices_are_exact_and_used_by_realtime_guard(
    tmp_path: Path,
) -> None:
    reduced = safety.load_safety_layer(write_calibrated_dahuan_config(tmp_path))
    data = json.loads(DAHUAN_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["calibrated"] = True
    data["robot_model"]["path"] = str(
        ROBOT_CLIENT_SOURCE / data["robot_model"]["path"]
    )
    for mesh in data["collision_meshes"]:
        mesh["path"] = str(ROBOT_CLIENT_SOURCE / mesh["path"])
        mesh.pop("support_vertex_indices")
    data["surfaces"][0]["height"] = 0.0
    data["surfaces"][0]["origin"][2] = 0.0
    full_path = tmp_path / "dahuan.full-mesh.json"
    full_path.write_text(json.dumps(data), encoding="utf-8")
    full = safety.load_safety_layer(full_path)

    assert reduced.geometry.horizontal_support_exact is True
    assert reduced.realtime_guard(path_samples=2).geometry is reduced.geometry
    assert len(full.geometry.points) > len(reduced.geometry.points) * 20

    generator = np.random.default_rng(20260910)
    configurations = [
        np.zeros(24, dtype=np.float64),
    ]
    configurations.extend(generator.uniform(-1.0, 1.0, size=24) for _ in range(12))
    for configuration in configurations:
        configuration[7] = configuration[15] = 0.025
        assert reduced.minimum_z(configuration).z == pytest.approx(
            full.minimum_z(configuration).z,
            abs=1e-12,
        )
        reduced_decision = reduced.project(configuration, configuration)
        full_decision = full.project(configuration, configuration)
        assert reduced_decision.distance_to_surface == pytest.approx(
            full_decision.distance_to_surface,
            abs=1e-12,
        )
        assert reduced_decision.hard_limit_triggered is full_decision.hard_limit_triggered


def test_collision_support_vertex_indices_are_fail_closed(tmp_path: Path) -> None:
    data = json.loads(DAHUAN_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["calibrated"] = True
    data["robot_model"]["path"] = str(
        ROBOT_CLIENT_SOURCE / data["robot_model"]["path"]
    )
    for mesh in data["collision_meshes"]:
        mesh["path"] = str(ROBOT_CLIENT_SOURCE / mesh["path"])
    data["collision_meshes"][0]["support_vertex_indices"] = [999_999]
    path = tmp_path / "dahuan.bad-support-indices.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        safety.SafetyConfigurationError,
        match="out of range or duplicated",
    ):
        safety.load_safety_layer(path)


def test_dahuan_collision_mesh_origins_clear_recorded_right_hand_false_positive(
    tmp_path: Path,
) -> None:
    layer = safety.load_safety_layer(
        write_calibrated_dahuan_config(tmp_path, table_height=0.72759)
    )
    # Measured GDK state from the 2026-09-10 read-only trace. Loading the STL
    # directly in the link frame reported z=0.71997 and a collision. Applying
    # the URDF collision origin with the canonical crsB kinematics puts the
    # physical finger 70.8 mm above the table and therefore outside the
    # protected buffer.  The CRS-to-CRS-B flange correction moves the complete
    # terminal 5 mm toward the wrist, so this pose now has 74.1 mm clearance.
    measured = np.asarray(
        [
            1.34739721, -0.88714623, -1.72176039, -1.53322935, 1.04045391,
            -0.96944374, -0.42980632, 0.025, -0.86667639, -0.60837162,
            0.85045838, -1.12570918, -1.10362220, -0.80286038, 0.06631375,
            0.025, 0.0, 0.0, 0.41717678, -0.55537879, 0.81459415,
            -0.08561833, 0.01543324, 0.03192092,
        ],
        dtype=np.float64,
    )

    decision = layer.project(measured, measured)

    assert decision.hard_limit_triggered is False
    assert decision.distance_to_surface == pytest.approx(0.07410784, abs=1e-5)
    assert decision.limiting_frame == "gripper_r_inner_link1"


def test_dahuan_collision_mesh_origin_must_match_urdf(tmp_path: Path) -> None:
    data = json.loads(DAHUAN_SAFETY_TEMPLATE.read_text(encoding="utf-8"))
    data["calibrated"] = True
    data["robot_model"]["path"] = str(
        ROBOT_CLIENT_SOURCE / data["robot_model"]["path"]
    )
    for mesh in data["collision_meshes"]:
        mesh["path"] = str(ROBOT_CLIENT_SOURCE / mesh["path"])
    data["collision_meshes"][4]["origin"]["xyz"][2] += 0.01
    path = tmp_path / "dahuan.bad-origin.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        safety.SafetyConfigurationError,
        match="origin.*does not match the URDF",
    ):
        safety.load_safety_layer(path)


def test_surface_xy_bounds_only_apply_inside_configured_region() -> None:
    point = safety.CollisionPoint("finger", "tool", (0.0, 0.0, 0.0))
    inside = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry((point,)),
        surface=safety.SurfaceConstraints(
            (safety.TableConstraint(0.10, 0.02, "conveyor", (-0.1, 0.1, -0.1, 0.1)),)
        ),
        motion_indices=(0, 1),
        passthrough_indices=(2,),
    )
    outside_fk = TranslationFK()

    class ShiftedFK:
        def frame_pose(self, joint_positions: np.ndarray, frame: str) -> np.ndarray:
            pose = outside_fk.frame_pose(joint_positions, frame)
            pose[0, 3] = 0.5
            return pose

    outside = safety.ActionSafetyLayer(
        fk=ShiftedFK(),
        geometry=safety.GripperGeometry((point,)),
        surface=safety.SurfaceConstraints(
            (safety.TableConstraint(0.10, 0.02, "conveyor", (-0.1, 0.1, -0.1, 0.1)),)
        ),
        motion_indices=(0, 1),
        passthrough_indices=(2,),
    )
    configuration = np.asarray([0.05, 0.0, -0.2])

    assert inside.project(configuration, configuration).hard_limit_triggered is True
    assert outside.project(configuration, configuration).hard_limit_triggered is False


def test_safety_jsonl_contains_required_step_diagnostics(tmp_path: Path) -> None:
    layer = make_layer()
    decision = layer.project(
        np.asarray([0.03, 0.0, -0.2]),
        np.asarray([0.00, 0.0, -0.6]),
        raw_action=np.asarray([-1.0, 0.0, -0.6]),
    )
    log_path = tmp_path / "safety.jsonl"
    audit = safety.SafetyJSONLLogger(log_path)
    audit.write(
        decision,
        {
            "chunk_output_length": 50,
            "chunk_execute_length": 30,
            "policy_latency_ms": 120.0,
            "control_latency_ms": 2.0,
        },
    )
    audit.flush()
    audit.close()

    record = json.loads(log_path.read_text(encoding="utf-8"))
    required = {
        "raw_action",
        "safe_action",
        "current_z_min",
        "predicted_z_min",
        "distance_to_surface",
        "limiting_surface",
        "hard_limit_triggered",
        "soft_limit_triggered",
        "downward_scale",
        "action_scale",
        "chunk_output_length",
        "chunk_execute_length",
        "policy_latency_ms",
        "control_latency_ms",
    }
    assert required <= record.keys()
    assert record["chunk_output_length"] == 50
    assert record["hard_limit_triggered"] is True


def test_gdk_adapter_holds_after_final_limiter_before_dry_controller(tmp_path: Path) -> None:
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (safety.CollisionPoint("finger_bottom", "tool", (0.0, 0.0, 0.0)),)
        ),
        surface=safety.TableConstraint(height=0.0, hard_margin=0.02),
        motion_indices=tuple(index for index in range(24) if index not in (7, 15)),
        passthrough_indices=(7, 15),
        path_samples=8,
        search_iterations=32,
    )
    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="dry-run",
            effector_profile="omnipicker_legacy",
            servo_control_period_s=0.01,
            max_arm_velocity=1e6,
            max_arm_acceleration=1e6,
            max_effector_velocity=1e6,
            max_effector_acceleration=1e6,
            max_head_waist_velocity=1e6,
            max_head_waist_acceleration=1e6,
        )
    )
    adapter._safety_layer = layer
    adapter._safety_logger = safety.SafetyJSONLLogger(tmp_path / "adapter_safety.jsonl")
    adapter.connect()
    adapter._dry_state[0] = 0.03
    initial = adapter._dry_state.copy()
    adapter._kinematic_limiter.reset(adapter._dry_state)
    target = adapter._dry_state.copy()
    target[0] = 0.0
    target[7] = -0.6

    sent = adapter.send_action(
        target,
        safety_context={
            "raw_action": target.copy(),
            "chunk_output_length": 50,
            "chunk_execute_length": 30,
        },
    )

    np.testing.assert_array_equal(sent, initial)
    assert adapter.last_action_intervened
    adapter._safety_logger.flush()
    record = json.loads((tmp_path / "adapter_safety.jsonl").read_text(encoding="utf-8"))
    assert record["hard_limit_triggered"] is True
    assert record["control_latency_ms"] >= record["projection_latency_ms"]
    adapter.disconnect()


def test_conservative_bounds_never_report_more_clearance_than_detailed_mesh() -> None:
    points = (
        safety.CollisionPoint("lower", "tool", (0.03, 0.08, -0.04), radius=0.01),
        safety.CollisionPoint("upper", "tool", (-0.02, -0.06, 0.05), radius=0.005),
    )
    detailed = make_layer(points=points)
    guarded = detailed.realtime_guard(path_samples=2)

    for angle in np.linspace(-math.pi, math.pi, 31):
        configuration = np.asarray([0.2, angle, -0.2])
        detailed_z = detailed.minimum_z(configuration).z
        guarded_z = guarded.minimum_z(configuration).z
        assert guarded_z <= detailed_z + 1e-12


def test_realtime_guard_holds_every_dimension_while_projection_is_pending() -> None:
    guard = make_layer().realtime_guard(path_samples=2)
    current = np.asarray([0.03, 0.0, -0.2])
    target = np.asarray([0.0, 0.0, -0.6])

    decision = guard.guard(current, target)

    assert decision.hard_limit_triggered is True
    assert decision.reason == "realtime_guard_hold_pending_projection"
    np.testing.assert_array_equal(decision.safe_action, current)


def test_hold_only_controller_never_submits_or_adopts_projection() -> None:
    detailed = make_layer()
    controller = safety.RealtimeSafetyController(
        detailed.realtime_guard(path_samples=2), detailed,
        allow_projected_actions=False,
    )
    current = np.asarray([0.03, 0.0, -0.2])
    target = np.asarray([0.0, 0.0, -0.6])
    try:
        for _ in range(100):
            result = controller.protect(current, target)
            assert result.hard_limit_triggered
            np.testing.assert_array_equal(result.safe_action, current)
        assert controller._queue.empty()
        assert not controller._thread.is_alive()
        safe = controller.protect(current, current)
        assert not safe.hard_limit_triggered
    finally:
        controller.close()


def test_background_projection_is_rechecked_before_it_can_be_used() -> None:
    detailed = make_layer()
    controller = safety.RealtimeSafetyController(
        detailed.realtime_guard(path_samples=2),
        detailed,
    )
    current = np.asarray([0.03, 0.0, -0.2])
    target = np.asarray([0.0, 0.0, -0.6])
    try:
        first = controller.protect(current, target)
        assert first.reason == "realtime_guard_hold_pending_projection"
        np.testing.assert_array_equal(first.safe_action, current)

        deadline = time.monotonic() + 1.0
        while True:
            result = controller.protect(current, target)
            if result.reason == "background_projection_revalidated":
                break
            assert time.monotonic() < deadline
            time.sleep(0.001)

        assert result.hard_limit_triggered is True
        assert result.safe_action[0] == pytest.approx(0.02, abs=1e-6)
        assert result.safe_action[2] == pytest.approx(-0.6)
        # The candidate was accepted only after the conservative guard found
        # the current-to-candidate segment safe.
        rechecked = controller.guard_layer.guard(current, result.safe_action)
        assert rechecked.hard_limit_triggered is False
    finally:
        controller.close()


def test_adapter_holds_then_stops_after_consecutive_safety_deadline_misses(
    tmp_path: Path,
) -> None:
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (safety.CollisionPoint("finger_bottom", "tool", (0.0, 0.0, 0.0)),)
        ),
        surface=safety.TableConstraint(height=-1.0, hard_margin=0.02),
        motion_indices=tuple(index for index in range(24) if index not in (7, 15)),
        passthrough_indices=(7, 15),
    )
    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="dry-run",
            servo_control_period_s=0.01,
            max_safety_guard_latency_ms=1e-9,
            max_consecutive_safety_deadline_misses=2,
        )
    )
    adapter._safety_layer = layer
    adapter._safety_logger = safety.SafetyJSONLLogger(tmp_path / "deadline.jsonl")
    adapter.connect()
    measured = adapter._dry_state.copy()
    target = measured.copy()
    target[0] = 0.01
    try:
        first = adapter.send_action(target)
        np.testing.assert_array_equal(first, measured)
        with pytest.raises(RuntimeError, match="realtime deadline"):
            adapter.send_action(target)
    finally:
        adapter.disconnect()


def test_read_only_deadline_misses_keep_session_and_allow_later_safe_action(
    tmp_path: Path,
) -> None:
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (safety.CollisionPoint("finger_bottom", "tool", (0.0, 0.0, 0.0)),)
        ),
        surface=safety.TableConstraint(height=0.0, hard_margin=0.02),
        motion_indices=tuple(index for index in range(24) if index not in (7, 15)),
        passthrough_indices=(7, 15),
        path_samples=2,
    )
    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="read-only",
            servo_control_period_s=0.01,
            max_arm_velocity=1e6,
            max_arm_acceleration=1e6,
            max_effector_velocity=1e6,
            max_effector_acceleration=1e6,
            max_head_waist_velocity=1e6,
            max_head_waist_acceleration=1e6,
            max_safety_guard_latency_ms=1e-9,
            max_consecutive_safety_deadline_misses=2,
        )
    )
    current = adapter._apply_absolute_position_limits(
        np.zeros(24, dtype=np.float32),
        warning_suffix="test_setup",
    )
    current[0] = 0.03
    unsafe = current.copy()
    unsafe[0] = 0.0
    safe = current.copy()
    safe[0] = 0.04
    adapter._connected = True
    adapter._safety_layer = layer
    adapter._safety_controller = safety.RealtimeSafetyController(
        layer.realtime_guard(path_samples=2),
        layer,
    )
    adapter._safety_logger = safety.SafetyJSONLLogger(tmp_path / "read_only_deadline.jsonl")
    adapter._last_state = current.copy()
    adapter._kinematic_limiter.reset(current)
    adapter.read_state = lambda: current.copy()
    try:
        held = adapter.send_action(unsafe)
        resumed = adapter.send_action(safe)
    finally:
        adapter.disconnect()

    np.testing.assert_array_equal(held, current)
    np.testing.assert_allclose(resumed, safe)
    assert adapter._consecutive_safety_deadline_misses >= 2
    records = [
        json.loads(line)
        for line in (tmp_path / "read_only_deadline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["hard_limit_triggered"] is True
    assert records[1]["hard_limit_triggered"] is False
    assert records[1]["collision_response"] == "allow_candidate"


def test_read_only_detailed_mesh_clears_conservative_sphere_false_positive(
    tmp_path: Path,
) -> None:
    # Both detailed points remain 10 cm above the table. Their single enclosing
    # guard sphere extends down to the table and therefore reports a potential
    # hit; read-only mode must publish the detailed safe result instead.
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (
                safety.CollisionPoint("left", "tool", (-0.1, 0.0, 0.1)),
                safety.CollisionPoint("right", "tool", (0.1, 0.0, 0.1)),
            )
        ),
        surface=safety.TableConstraint(height=0.0, hard_margin=0.02),
        motion_indices=tuple(index for index in range(24) if index not in (7, 15)),
        passthrough_indices=(7, 15),
        path_samples=2,
    )
    current = np.zeros(24, dtype=np.float32)
    assert layer.realtime_guard(path_samples=2).guard(
        current, current
    ).hard_limit_triggered is True
    assert layer.project(current, current).hard_limit_triggered is False

    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="read-only",
            servo_control_period_s=0.01,
            max_arm_velocity=1e6,
            max_arm_acceleration=1e6,
            max_effector_velocity=1e6,
            max_effector_acceleration=1e6,
            max_head_waist_velocity=1e6,
            max_head_waist_acceleration=1e6,
            max_safety_guard_latency_ms=100.0,
        )
    )
    current = adapter._apply_absolute_position_limits(
        current,
        warning_suffix="test_setup",
    )
    adapter._connected = True
    adapter._safety_layer = layer
    adapter._safety_controller = safety.RealtimeSafetyController(
        layer.realtime_guard(path_samples=2),
        layer,
    )
    adapter._safety_logger = safety.SafetyJSONLLogger(tmp_path / "detailed_recheck.jsonl")
    adapter._last_state = current.copy()
    adapter._kinematic_limiter.reset(current)
    adapter.read_state = lambda: current.copy()
    try:
        adopted = adapter.send_action(current)
    finally:
        adapter.disconnect()

    np.testing.assert_array_equal(adopted, current)
    record = json.loads((tmp_path / "detailed_recheck.jsonl").read_text(encoding="utf-8"))
    assert record["hard_limit_triggered"] is False
    assert record["reason"] == "read_only_detailed_geometry_safe"
    assert record["collision_response"] == "allow_candidate"


def test_read_only_detailed_mesh_publishes_exact_unsafe_clearance(
    tmp_path: Path,
) -> None:
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (
                safety.CollisionPoint("lowest", "tool", (-0.1, 0.0, 0.0005)),
                safety.CollisionPoint("far", "tool", (0.1, 0.0, 0.1)),
            )
        ),
        surface=safety.TableConstraint(
            height=0.0,
            hard_margin=0.001,
            warning_margin=0.003,
        ),
        motion_indices=tuple(index for index in range(24) if index not in (7, 15)),
        passthrough_indices=(7, 15),
        path_samples=2,
    )
    current = np.zeros(24, dtype=np.float32)
    broad = layer.realtime_guard(path_samples=2).guard(current, current)
    exact = layer.project(current, current)
    assert broad.current_z_min < -0.04
    assert exact.current_z_min == pytest.approx(0.0005)

    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="read-only",
            servo_control_period_s=0.01,
            max_arm_velocity=1e6,
            max_arm_acceleration=1e6,
            max_effector_velocity=1e6,
            max_effector_acceleration=1e6,
            max_head_waist_velocity=1e6,
            max_head_waist_acceleration=1e6,
            max_safety_guard_latency_ms=100.0,
        )
    )
    current = adapter._apply_absolute_position_limits(
        current,
        warning_suffix="test_setup",
    )
    adapter._connected = True
    adapter._safety_layer = layer
    adapter._safety_controller = safety.RealtimeSafetyController(
        layer.realtime_guard(path_samples=2),
        layer,
    )
    log_path = tmp_path / "detailed_unsafe.jsonl"
    adapter._safety_logger = safety.SafetyJSONLLogger(log_path)
    adapter._last_state = current.copy()
    adapter._kinematic_limiter.reset(current)
    adapter.read_state = lambda: current.copy()
    try:
        adapter.send_action(current)
    finally:
        adapter.disconnect()

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["hard_limit_triggered"] is True
    assert record["soft_limit_triggered"] is True
    assert record["reason"] == "read_only_detailed_geometry_unsafe"
    assert record["current_z_min"] == pytest.approx(0.0005)
    assert record["distance_to_surface"] == pytest.approx(0.0005)


def test_adapter_calibrated_profile_uses_realtime_controller_and_flushes_audit(
    tmp_path: Path,
) -> None:
    config_path = write_calibrated_official_config(tmp_path, table_height=0.0)
    audit_path = tmp_path / "integrated_safety.jsonl"
    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="dry-run",
            startup_wait_s=0,
            servo_control_period_s=0.01,
            safety_config_path=config_path,
            safety_log_path=audit_path,
            max_safety_guard_latency_ms=100.0,
        )
    )
    assert adapter._safety_controller is not None
    assert adapter._safety_controller.projection_layer is adapter._safety_layer
    assert not adapter._safety_controller._thread.is_alive()
    adapter.connect()
    sent = adapter.send_action(np.zeros(24, dtype=np.float32))
    adapter.disconnect()

    assert sent.shape == (24,)
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["reason"] == "realtime_guard_safe"
    assert records[0]["safety_guard_latency_ms"] >= 0


def test_guard_trajectory_rejects_unsafe_second_target():
    layer = make_layer()
    current = np.array([0.1, 0.0, 0.0])
    first = np.array([0.08, 0.0, 0.0])
    second = np.array([0.0, 0.0, 0.0])
    assert not layer.guard(current, first).hard_limit_triggered
    decision = layer.guard_trajectory(current, (first, second))
    assert decision.hard_limit_triggered
    np.testing.assert_array_equal(decision.safe_action, current)
    one = layer.guard_trajectory(current, (first,))
    assert one.path_z_min == layer.guard(current, first).path_z_min


def _paired_adapter(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(gdk_adapter.time, "perf_counter", lambda: clock[0])
    adapter = gdk_adapter.G2AAdapter(gdk_adapter.AdapterConfig(
        hardware_mode="dry-run", startup_wait_s=0,
        servo_control_period_s=0.01, safety_check_fps=50,
        safety_config_path=write_calibrated_official_config(tmp_path, table_height=-2.0),
        safety_log_path=tmp_path / "paired.jsonl",
    ))
    adapter.connect()
    return adapter, clock


def test_50hz_freezes_two_limited_targets_and_audits_reuse(tmp_path, monkeypatch):
    adapter, clock = _paired_adapter(tmp_path, monkeypatch)
    target = np.zeros(24, dtype=np.float32)
    target[0] = 0.1
    try:
        first = adapter.send_action(target)
        assert adapter._certified_step is not None
        expected = adapter._certified_step["action"].copy()
        assert expected[0] > first[0]
        clock[0] += 0.01
        # A new, opposite input must NOT be passed through an old safety result.
        second = adapter.send_action(-target)
        np.testing.assert_array_equal(second, expected)
        assert adapter._certified_step is None
        clock[0] += 0.01
        adapter.send_action(target)
    finally:
        adapter.disconnect()
    records = [json.loads(line) for line in (tmp_path / "paired.jsonl").read_text().splitlines()]
    assert [r["guard_evaluated"] for r in records] == [True, False, True]
    assert records[1]["certificate_age_ms"] == pytest.approx(10)


@pytest.mark.parametrize("invalidate", ["expiry", "drift", "reset", "pause", "hold", "nan"])
def test_50hz_invalidates_pending_step(tmp_path, monkeypatch, invalidate):
    adapter, clock = _paired_adapter(tmp_path, monkeypatch)
    target = np.zeros(24, dtype=np.float32)
    target[0] = 0.1
    try:
        first = adapter.send_action(target)
        clock[0] += 0.01
        if invalidate == "expiry":
            clock[0] += 0.02
        elif invalidate == "drift":
            adapter._dry_state[0] += 0.1
        elif invalidate == "reset":
            adapter.reset_command_reference(first)
        elif invalidate == "pause":
            adapter.reanchor_measured_hold(first)
        elif invalidate == "hold":
            adapter.reset_action_limiter(first)
        elif invalidate == "nan":
            adapter.read_state = lambda: np.full(24, np.nan, dtype=np.float32)
            with pytest.raises(ValueError, match="Invalid feedback"):
                adapter.send_action(target)
            assert adapter._certified_step is None
            return
        adapter.send_action(target)
        assert not adapter._certified_dispatch
    finally:
        adapter.disconnect()


def test_50hz_first_deadline_stays_8ms_second_expires_at_20ms(tmp_path, monkeypatch):
    adapter, clock = _paired_adapter(tmp_path, monkeypatch)
    try:
        adapter._command_safety_started = clock[0]
        with pytest.raises(gdk_adapter.SafetyDispatchExpiredError):
            adapter._validate_dispatch_age(clock[0] + 0.009)
        adapter._certified_dispatch = True
        adapter._validate_dispatch_age(clock[0] + 0.019)
        with pytest.raises(gdk_adapter.SafetyDispatchExpiredError):
            adapter._validate_dispatch_age(clock[0] + 0.021)
    finally:
        adapter.disconnect()


class ManualSafetyWorker:
    """No hardware, IPC or time waits: deterministically advance worker results."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.tasks = []
        self.result = None
        self.error = None

    def submit(self, task):
        self.tasks.append(task)

    def poll(self):
        if self.error:
            raise RuntimeError(self.error)
        result, self.result = self.result, None
        return result

    def finish(self, *, started=100.0, check_ms=5.0, index=-1, measured=None):
        task = self.tasks[index]
        measured = self.adapter._dry_state.copy() if measured is None else measured.copy()
        if task.get("rolling_legacy_geometry"):
            from nexus_deploy.robot.g2a.async_safety import check_sampled_segment_window
            self.result = check_sampled_segment_window(self.adapter._safety_controller.guard_layer, measured, task)
            self.result.update(task=task, measured=measured, started=started, check_ms=check_ms)
            return
        if task.get("rolling"):
            from nexus_deploy.robot.g2a.async_safety import check_rolling_window
            self.result = check_rolling_window(self.adapter._safety_controller.guard_layer, measured,
                                              task["anchor"], task["targets"], task["raw"],
                                              task["fallback_anchor"], task["braking"],
                                              self.adapter._async_hold_coefficients)
            self.result.update(task=task, measured=measured, started=started, check_ms=check_ms)
            return
        from nexus_deploy.robot.g2a.async_safety import check_window
        checked = check_window(self.adapter._safety_controller.guard_layer,
                                      measured, task["anchor"], task["targets"], task["raw"], include_proofs=True,
                                      fallback_anchor=task["fallback_anchor"], braking=task.get("braking"),
                                      braking_coefficients=self.adapter._async_hold_coefficients)
        hold, decision, target_holds, fallback_hold = checked[:4]
        self.result = dict(task=task, measured=measured, hold=hold, decision=decision,
                           target_holds=target_holds, fallback_hold=fallback_hold,
                           target_brakes=checked[4] if len(checked) == 5 else None,
                           started=started, check_ms=check_ms)

    def close(self):
        pass


def _manual_async_adapter(tmp_path, monkeypatch):
    adapter, clock = _paired_adapter(tmp_path, monkeypatch)
    adapter._dry_state = adapter._apply_absolute_position_limits(adapter._dry_state, warning_suffix="test")
    adapter.reset_command_reference(adapter._dry_state)
    adapter._async_worker = ManualSafetyWorker(adapter)
    return adapter, clock, adapter._async_worker


def test_async_servo_never_reads_feedback_or_runs_geometry(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    adapter.read_state = lambda: pytest.fail("Servo must not read GDK feedback")
    target = np.zeros(24, dtype=np.float32)
    target[0] = 0.1
    try:
        initial = adapter.send_action(target)
        assert adapter.last_command_pending
        assert len(worker.tasks) == 1
        worker.finish()
        expected = worker.tasks[0]["targets"]
        clock[0] += 0.01
        first = adapter.send_action(-target)
        np.testing.assert_array_equal(first, expected[0])
        clock[0] += 0.01
        second = adapter.send_action(-target)
        np.testing.assert_array_equal(second, expected[1])
        assert len(worker.tasks) == 2
        assert not adapter.last_command_pending
        assert initial[0] < first[0] < second[0]
    finally:
        adapter.disconnect()


def test_async_late_worker_holds_then_expires_without_refreshing_lease(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = np.zeros(24, dtype=np.float32)
    target[0] = 0.1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += 0.01
        adapter.send_action(target)
        clock[0] += 0.01
        second = adapter.send_action(target)
        clock[0] += 0.01
        held = adapter.send_action(-target)
        np.testing.assert_array_equal(held, second)
        assert adapter._command_safety_started == 100.0
        clock[0] += 0.011
        with pytest.raises(gdk_adapter.SafetyDispatchExpiredError, match="hold expired"):
            adapter.send_action(target)
    finally:
        adapter.disconnect()


def test_async_reset_retires_inflight_result(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = np.zeros(24, dtype=np.float32)
    target[0] = 0.1
    try:
        adapter.send_action(target)
        worker.finish()
        old_result = worker.result
        adapter.reanchor_measured_hold(adapter._dry_state)
        clock[0] += 0.01
        worker.result = old_result
        adapter.send_action(target)
        assert adapter.last_command_pending
        assert not adapter._async_steps
        assert worker.tasks[-1]["generation"] != old_result["task"]["generation"]
        worker.finish(started=clock[0])
        clock[0] += 0.01
        adapter.send_action(target)
        assert not adapter.last_command_pending
    finally:
        adapter.disconnect()


def test_async_pipeline_waits_for_certified_anchor_before_adopting_next_window(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += 0.1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += 0.01
        adapter.send_action(target)
        old_second = adapter._async_steps[0]["action"].copy()
        assert len(worker.tasks) == 2  # Next window already in flight.
        np.testing.assert_array_equal(worker.tasks[-1]["anchor"], old_second)
        worker.finish(started=clock[0])
        next_first = worker.tasks[-1]["targets"][0].copy()
        clock[0] += 0.01
        np.testing.assert_array_equal(adapter.send_action(target), old_second)
        assert adapter._async_ready_result is not None
        clock[0] += 0.01
        np.testing.assert_array_equal(adapter.send_action(target), next_first)
    finally:
        adapter.disconnect()


def test_async_late_window_renews_only_hold_and_restarts_limiter_from_rest(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += 0.1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += .01
        adapter.send_action(target)
        clock[0] += .01
        anchor = adapter.send_action(target)
        clock[0] += .01
        np.testing.assert_array_equal(adapter.send_action(target), anchor)
        assert not np.any(adapter._kinematic_limiter._velocity)
        assert worker.tasks[-1]["motion_cancelled"]
        worker.finish(started=100.02, check_ms=15)
        clock[0] += .01
        np.testing.assert_array_equal(adapter.send_action(target), anchor)
        assert adapter._async_hold["result"]["hold_only"]
        assert not np.any(adapter._kinematic_limiter._velocity)
        # Newly prepared motion is limited from rest, not stale pre-hold velocity.
        planned = worker.tasks[-1]["targets"][0]
        assert abs(planned[0] - anchor[0]) <= adapter.config.max_arm_acceleration * .01**2 + 1e-7
    finally:
        adapter.disconnect()


@pytest.mark.parametrize("ready_location", ["mailbox", "staged", "inflight"])
def test_async_expiry_preserves_fresh_zero_velocity_recovery(tmp_path, monkeypatch, ready_location):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "_wait_servo_slot", lambda **kwargs: None)
    target = adapter._dry_state.copy()
    try:
        adapter.send_action(target)
        worker.finish(started=100.0)
        clock[0] = 100.025
        held = adapter.send_action(target)
        generation = adapter._async_generation
        if ready_location != "inflight":
            worker.finish(started=100.020)
        if ready_location == "staged":
            adapter._async_ready_result = worker.poll()
        clock[0] = 100.039
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert adapter._async_generation == generation
        if ready_location == "inflight":
            assert not worker.tasks[-1].get("motion_cancelled")
            worker.finish(started=100.020)
        clock[0] = 100.045
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert adapter._async_hold["result"]["started"] == 100.020
        assert not adapter._async_hold["result"]["hold_only"]
        assert not np.any(adapter._kinematic_limiter._velocity)
    finally:
        adapter.disconnect()


@pytest.mark.parametrize("fallback_valid", [True, False])
def test_async_retiring_future_anchor_only_renews_independent_hold(tmp_path, monkeypatch, fallback_valid):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "_wait_servo_slot", lambda **kwargs: None)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        adapter.send_action(target)
        worker.finish(started=100.0)
        clock[0] = 100.025
        held = adapter.send_action(target)
        assert not np.array_equal(worker.tasks[-1]["anchor"], held)
        worker.finish(started=100.020)
        if not fallback_valid:
            # A proof of any other target cannot authorize the actual hold.
            worker.result["task"]["fallback_anchor"] = target.copy()
        clock[0] = 100.039
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert adapter._async_ready_result is None
        assert adapter._async_hold["result"]["started"] == (100.020 if fallback_valid else 100.0)
        clock[0] = 100.045
        if fallback_valid:
            np.testing.assert_array_equal(adapter.send_action(target), held)
            assert adapter._async_hold["result"]["hold_only"]
            assert not np.any(adapter._kinematic_limiter._velocity)
        else:
            with pytest.raises(gdk_adapter.SafetyDispatchExpiredError, match="Checked hold expired"):
                adapter.send_action(target)
    finally:
        adapter.disconnect()


def test_async_repeated_plan_gap_holds_do_not_starve_pending_proof(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += .01
        held = adapter.send_action(target)
        generation = adapter._async_generation
        worker.finish(started=clock[0])
        for _ in range(3):
            adapter.reset_action_limiter(held)
            assert adapter._async_generation == generation
        clock[0] += .01
        np.testing.assert_array_equal(adapter.send_action(held), held)
        assert adapter._async_hold["result"]["hold_only"]
        assert not np.any(adapter._kinematic_limiter._velocity)
    finally:
        adapter.disconnect()


def test_async_late_but_valid_result_can_renew_hold_not_motion(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "_wait_servo_slot", lambda **kwargs: None)
    target = adapter._dry_state.copy()
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += .01
        held = adapter.send_action(target)
        worker.finish(started=100.015)
        clock[0] = 100.054
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert adapter._async_hold["result"]["started"] == 100.015
        assert not adapter._async_steps
        assert not np.any(adapter._kinematic_limiter._velocity)
    finally:
        adapter.disconnect()


@pytest.mark.parametrize("age_ms", [30.5, 35.0, 37.0])
def test_async_second_point_uses_available_dispatch_slot(tmp_path, monkeypatch, age_ms):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] = 100.020
        adapter.send_action(target)
        expected = adapter._async_steps[0]["action"].copy()
        clock[0] = 100.0 + age_ms / 1000
        np.testing.assert_array_equal(adapter.send_action(target), expected)
        assert not worker.tasks[-1].get("motion_cancelled")
        assert np.any(adapter._kinematic_limiter._velocity)
    finally:
        adapter.disconnect()


def test_async_slot_admission_includes_pacing_and_preparation(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    try:
        result = {"started": 100.0}
        adapter._last_servo_dispatch = 100.029
        assert not adapter._async_window_has_dispatch_slot(result, 100.031)
        adapter._last_servo_dispatch = 100.025
        assert adapter._async_window_has_dispatch_slot(result, 100.031)
        assert not adapter._async_window_has_dispatch_slot(result, 100.039)
    finally:
        adapter.disconnect()


def test_async_late_dispatch_rechecks_before_hardware_and_holds(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    sent = []
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] = 100.020
        held = adapter.send_action(target)
        clock[0] = 100.035
        worker.feedback = (adapter._dry_state.copy(), clock[0])
        def dispatch(action):
            # Fake scheduling overrun, no hardware. The motion looked usable
            # at admission but must be rejected before reaching the SDK sink.
            clock[0] = 100.041
            adapter._validate_dispatch_age(clock[0])
            sent.append(action.copy())
            return action.copy()
        monkeypatch.setattr(adapter, "_dispatch_checked_action", dispatch)
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert len(sent) == 1
        np.testing.assert_array_equal(sent[0], held)
        assert not np.any(adapter._kinematic_limiter._velocity)
    finally:
        adapter.disconnect()


def test_async_explicit_hold_cancels_even_zero_velocity_recovery(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        held = adapter.send_action(target)
        assert not np.any(worker.tasks[-1]["anchor_velocity"])
        adapter.reset_action_limiter(held)
        assert worker.tasks[-1]["motion_cancelled"]
        worker.finish()
        clock[0] += .01
        np.testing.assert_array_equal(adapter.send_action(held), held)
        assert not adapter._async_steps
        assert adapter._async_hold["result"]["hold_only"]
    finally:
        adapter.disconnect()


def test_async_hold_only_result_immediately_submits_recovery(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += .01
        adapter.send_action(target)
        clock[0] += .01
        held = adapter.send_action(target)
        clock[0] += .01
        adapter.send_action(target)  # No result: real hold invalidates velocity.
        worker.finish(started=100.020)
        clock[0] += .01
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert not adapter._async_steps  # No two-tick synthetic hold window.
        assert len(worker.tasks) == 3
        recovery = worker.tasks[-1]
        np.testing.assert_array_equal(recovery["anchor"], held)
        assert not np.any(recovery["anchor_velocity"])
        worker.finish(started=100.040)
        clock[0] += .01
        resumed = adapter.send_action(target)
        assert held[0] < resumed[0] <= held[0] + adapter.config.max_arm_acceleration * .01**2 + 1e-7
    finally:
        adapter.disconnect()


def test_async_over_budget_result_never_authorizes_new_motion(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        held = adapter.send_action(target)
        worker.finish(check_ms=21.0)
        clock[0] += .025
        np.testing.assert_array_equal(adapter.send_action(target), held)
        assert adapter._async_hold["result"]["hold_reason"] == "check_over_budget"
        assert not adapter._async_steps
        assert not np.any(worker.tasks[-1]["anchor_velocity"])
        assert len(worker.tasks) == 2
    finally:
        adapter.disconnect()


def test_feedback_clock_eliminates_second_timer_phase_slip():
    from nexus_deploy.robot.g2a.async_safety import feedback_ready_for_check
    state = np.zeros(24, dtype=np.float32)
    previous = 100.0
    task = dict(submitted_at=100.010, urgent=False)
    # Prior compute started at .004; independently waiting until .024 would
    # miss this sample. Compute may instead follow its .020 publication.
    assert feedback_ready_for_check((state, 100.0201), task, previous, 100.021)
    assert not feedback_ready_for_check((state, 100.0201), task, 100.0201, 100.022)
    assert not feedback_ready_for_check((state, 100.0), task, previous, 100.021)
    assert not feedback_ready_for_check((state, 100.0201), task, previous, 100.027)
    assert not feedback_ready_for_check((state, 100.0201), dict(submitted_at=100.030), previous, 100.021)
    assert feedback_ready_for_check((state, 100.0201), dict(task, urgent=True), 100.0201, 100.022)


def run_async_jitter_schedule(tmp_path, monkeypatch, *, policy_fps, ticks=600):
    """Real FK/limits and fake time/GDK: bounded jitter, lag and late checks."""
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    def wait_slot(*, lead_s=0.0):
        if adapter._last_servo_dispatch is not None:
            clock[0] = max(clock[0], adapter._last_servo_dispatch + .01 - lead_s)
    def simulated_dispatch(action):
        # Exercise the production pacing/expiry path against a fake sink, not
        # dry-run's normally unpaced dispatch and never a vendor SDK handle.
        adapter._pace_servo_dispatch()
        adapter._dry_state = action.copy()
        return action.copy()
    monkeypatch.setattr(adapter, "_wait_servo_slot", wait_slot)
    monkeypatch.setattr(adapter, "_dispatch_checked_action", simulated_dispatch)
    history = [adapter._dry_state.copy()]
    running = None
    last_scheduled = 0
    previous_feedback = 99.0
    costs_ms = [10.5, 11.0, 15.5, 9.5, 12.0, 10.8, 16.0]
    try:
        for tick in range(ticks):
            clock[0] = max(clock[0], 100.0 + tick * .0102 + (.001 if tick % 29 == 0 else 0))
            now = clock[0]
            feedback_stamp = 100.0 + math.floor((now - 100.0) / .02012) * .02012
            measured = history[max(0, len(history) - 4)].copy()  # ~30 ms tracking lag.
            worker.feedback = measured, feedback_stamp
            if running is not None and running["end"] <= now:
                worker.finish(started=running["start"], check_ms=running["cost"],
                              index=running["index"], measured=running["measured"])
                running = None
            target = history[0].copy()
            policy_time = math.floor((now - 100.0) * policy_fps) / policy_fps
            target[0] += .2 + .12 * math.sin(policy_time * 2)
            adapter.send_action(target)
            history.append(adapter._dry_state.copy())
            if running is None and worker.tasks and len(worker.tasks) > last_scheduled:
                last_scheduled = len(worker.tasks)
                start = max(now, previous_feedback + .02012)
                stamp = 100.0 + math.floor((start - 100.0) / .02012) * .02012
                if start - stamp > .005 or stamp <= previous_feedback + 1e-9:
                    stamp += .02012
                    start = max(start, stamp)
                previous_feedback = stamp
                cost = 23.0 if last_scheduled % 37 == 0 else costs_ms[last_scheduled % len(costs_ms)]
                running = dict(start=stamp, end=start + cost / 1000, cost=cost,
                               index=last_scheduled - 1, measured=measured)
    finally:
        adapter.disconnect()
    records = [json.loads(line) for line in (tmp_path / "paired.jsonl").read_text().splitlines()]
    dispatched = [r for r in records if r.get("record_type") == "servo_dispatch"]
    holds = [r["collision_response"] != "async_checked_motion" for r in dispatched]
    import itertools
    longest = max((len(list(group)) for held, group in itertools.groupby(holds) if held), default=0)
    return dict(ticks=len(dispatched), hold_fraction=sum(holds) / len(holds),
                longest_hold_ticks=longest, max_age_ms=max(r["certificate_age_ms"] for r in dispatched),
                records=dispatched)


@pytest.mark.parametrize("policy_fps", [15, 30])
def test_async_jitter_schedule_stays_live_without_hold_cascade(tmp_path, monkeypatch, policy_fps):
    result = run_async_jitter_schedule(tmp_path, monkeypatch, policy_fps=policy_fps)
    print({k: v for k, v in result.items() if k != "records"})
    assert result["ticks"] >= 590
    assert result["max_age_ms"] < 40
    assert result["hold_fraction"] < .15
    assert result["longest_hold_ticks"] <= 8
    # All tests use dry-run dispatch, never a physical SDK acceptance.
    assert all(r["gdk_command_accepted"] is None for r in result["records"])


@pytest.mark.parametrize("fault", ["worker_error", "expired_result", "changed_anchor"])
def test_async_rejects_unusable_results(tmp_path, monkeypatch, fault):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = np.zeros(24, dtype=np.float32)
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += 0.01
        if fault == "worker_error":
            worker.error = "Worker crashed"
            with pytest.raises(RuntimeError, match="Worker crashed"):
                adapter.send_action(target)
        else:
            if fault == "expired_result":
                worker.result["started"] = 99.9
            else:
                worker.result["task"]["anchor"] = np.ones(24, dtype=np.float32)
                worker.result["task"]["fallback_anchor"] = np.ones(24, dtype=np.float32)
            adapter.send_action(target)
            assert adapter.last_command_pending
            assert not adapter._async_steps
    finally:
        adapter.disconnect()


def test_async_geometry_process_starts_checks_and_closes(tmp_path):
    from nexus_deploy.robot.g2a.async_safety import AsyncSafetyWorker
    state = np.zeros(24, dtype=np.float32)
    worker = AsyncSafetyWorker(write_calibrated_official_config(tmp_path, table_height=-2),
                               2, lambda: state.copy())
    try:
        worker.submit(dict(generation=1, anchor=state.copy(),
                           targets=[state.copy(), state.copy()], raw=state.copy()))
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            time.sleep(0.01)
            result = worker.poll()
        assert result is not None
        assert not result.get("error")
        assert not result["decision"].hard_limit_triggered
        assert result["task"]["generation"] == 1
        timing = result["geometry_timing"]
        assert timing["child_wall_ms"] >= 0
        assert timing["child_cpu_ms"] >= 0
        assert timing["dispatch_to_child_ms"] >= 0
        assert timing["child_to_parent_ms"] >= 0
        assert result["check_ms"] == pytest.approx(
            timing["dispatch_to_child_ms"] + timing["child_wall_ms"]
            + timing["child_to_parent_ms"], abs=0.05)
    finally:
        worker.close()
    assert not worker.thread.is_alive()
    assert not worker.process.is_alive()


def test_feedback_keeps_refreshing_while_result_bridge_is_blocked(tmp_path):
    import threading
    from nexus_deploy.robot.g2a.async_safety import AsyncSafetyWorker
    state = np.zeros(24, dtype=np.float32)
    entered, release = threading.Event(), threading.Event()
    reads = [0]

    def read():
        reads[0] += 1
        return state.copy()

    def slow_consumer(result):
        entered.set()
        release.wait(2)

    worker = AsyncSafetyWorker(write_calibrated_official_config(tmp_path, table_height=-2),
                               2, read, slow_consumer)
    try:
        worker.submit(dict(generation=1, anchor=state.copy(),
                           targets=[state.copy(), state.copy()], raw=state.copy()))
        assert entered.wait(10)
        initial = reads[0]
        deadline = time.monotonic() + .3
        while reads[0] < initial + 2 and time.monotonic() < deadline:
            time.sleep(.005)
        assert reads[0] >= initial + 2
        assert not release.is_set()
    finally:
        release.set()
        worker.close()
    assert not worker.feedback_thread.is_alive()


def test_async_window_certifies_hold_and_rejects_later_collision():
    from nexus_deploy.robot.g2a.async_safety import check_window
    layer = make_layer()
    measured = np.array([0.1, 0.0, 0.0])
    anchor = np.array([0.08, 0.0, 0.0])
    targets = [np.array([0.06, 0.0, 0.0]), np.array([0.005, 0.0, 0.0])]
    hold, motion = check_window(layer, measured, anchor, targets, targets[-1])
    assert not hold.hard_limit_triggered
    assert motion.hard_limit_triggered
    np.testing.assert_array_equal(motion.safe_action, anchor)
    with pytest.raises(RuntimeError, match="no certified fallback"):
        check_window(layer, measured, targets[-1], targets, targets[-1])


def test_async_window_checks_fallback_separately_from_safe_future_anchor():
    from nexus_deploy.robot.g2a.async_safety import check_window
    layer = make_layer()
    measured = np.array([.1, 0., 0.])
    anchor = np.array([.08, 0., 0.])
    targets = [np.array([.07, 0., 0.]), np.array([.06, 0., 0.])]
    hold, motion, proofs, fallback = check_window(
        layer, measured, anchor, targets, targets[-1], include_proofs=True,
        fallback_anchor=np.array([.005, 0., 0.]),
    )
    assert not hold.hard_limit_triggered
    assert not motion.hard_limit_triggered
    assert fallback.hard_limit_triggered


def test_async_unsafe_unexecuted_future_anchor_can_hold_safe_actual_target():
    from nexus_deploy.robot.g2a.async_safety import check_window
    layer = make_layer()
    measured = np.array([.1, 0., 0.])
    future = np.array([.005, 0., 0.])
    actual = np.array([.08, 0., 0.])
    hold, motion, proofs, fallback = check_window(
        layer, measured, future, [future, future], future, include_proofs=True,
        fallback_anchor=actual,
    )
    assert hold.hard_limit_triggered and motion.hard_limit_triggered
    assert not fallback.hard_limit_triggered
    np.testing.assert_array_equal(fallback.safe_action, actual)


@pytest.mark.parametrize("mode", ["dry-run", "read-only", "execute"])
def test_async_collision_hold_then_safe_motion_uses_zero_velocity_limits(tmp_path, monkeypatch, caplog, mode):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "_wait_servo_slot", lambda **kwargs: None)
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry((safety.CollisionPoint("finger", "tool", (0., 0., 0.)),)),
        surface=safety.TableConstraint(height=0., hard_margin=.02),
        motion_indices=tuple(i for i in range(24) if i not in (7, 15)),
        passthrough_indices=(7, 15), path_samples=2,
    )
    adapter._safety_layer = layer
    adapter._safety_controller.guard_layer = layer
    adapter._async_hold_coefficients = None
    initial = adapter._dry_state.copy()
    initial[0] = .0201
    adapter._dry_state = initial.copy()
    adapter.reset_command_reference(initial)
    adapter.config.hardware_mode = mode
    adapter._execution_prepared = True
    sent = []

    def fake_gdk(action):
        adapter._validate_dispatch_age(clock[0])
        sent.append(action.copy())

    # Fake boundary only: never import/initialize actual GDK, including execute.
    adapter._send_combined_action = fake_gdk
    danger = initial.copy()
    danger[0] = -.1
    safe = initial.copy()
    safe[0] = .1
    try:
        adapter.send_action(danger)
        worker.finish(started=clock[0])
        clock[0] += .01
        np.testing.assert_array_equal(adapter.send_action(danger), initial)
        assert adapter.latest_control_telemetry["safety"]["response"] == "async_collision_hold"
        assert not np.any(adapter._kinematic_limiter._velocity)
        assert "COLLISION RISK" in caplog.text
        values = []
        for _ in range(12):
            if adapter._async_pending is not None:
                worker.finish(started=clock[0])
            clock[0] += .01
            values.append(adapter.send_action(safe).copy())
        assert values[-1][0] > initial[0]
        first_motion = next(q for q in values if q[0] > initial[0])
        assert first_motion[0] - initial[0] <= adapter.config.max_arm_acceleration * .01**2 + 1e-7
        assert all(q[0] >= initial[0] for q in values)
        assert bool(sent) == (mode == "execute")
        assert all(q[0] >= initial[0] for q in sent)
    finally:
        adapter.config.hardware_mode = "dry-run"
        adapter.disconnect()


def test_async_expiry_after_pacing_cannot_reach_fake_gdk(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    sent = []
    adapter.config.hardware_mode = "execute"
    adapter._execution_prepared = True

    def fake_send(target):
        clock[0] = 100.041
        adapter._validate_dispatch_age(clock[0])
        sent.append(target)

    adapter._send_combined_action = fake_send
    try:
        adapter.send_action(adapter._dry_state)
        worker.finish()
        clock[0] += 0.01
        with pytest.raises(gdk_adapter.SafetyDispatchExpiredError):
            adapter.send_action(adapter._dry_state)
        assert not sent
    finally:
        adapter.config.hardware_mode = "dry-run"
        adapter.disconnect()


def test_async_pause_reanchor_has_bounded_wait_for_new_certificate(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += 0.01
        adapter.send_action(target)
        adapter.reanchor_measured_hold(adapter._last_state)
        # A normal 48--72 ms geometry tail must remain a pending stationary
        # hold instead of terminating the task at the former 40 ms deadline.
        clock[0] += 0.06
        adapter.send_action(target)
        assert adapter.last_command_pending
        worker.finish(started=clock[0])
        clock[0] += 0.01
        adapter.send_action(target)
        assert not adapter.last_command_pending
    finally:
        adapter.disconnect()


def test_pause_reanchor_publishes_fresh_read_and_requests_priority_check(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    published = []
    worker.publish_feedback = lambda *args: published.append(args)
    try:
        measured = adapter.read_state()
        adapter.reanchor_measured_hold(measured)
        assert len(published) == 1
        np.testing.assert_array_equal(published[0][0], measured)
        assert published[0][1] == clock[0]
        adapter.send_action(measured)
        assert worker.tasks[-1]["urgent"]
        assert not adapter._async_urgent
    finally:
        adapter.disconnect()


def test_urdf_displacement_bound_dominates_actual_fk_changes(tmp_path):
    from nexus_deploy.robot.g2a.async_safety import hold_displacement_coefficients
    layer = gdk_adapter.load_safety_layer(write_calibrated_dahuan_config(tmp_path))
    coefficients = hold_displacement_coefficients(layer)
    assert coefficients is not None
    rng = np.random.default_rng(90911)
    for _ in range(120):
        first = rng.uniform(-1, 1, 24)
        second = first + rng.uniform(-.02, .02, 24)
        first[[7, 15]] = rng.uniform(0, .025, 2)
        second[[7, 15]] = rng.uniform(0, .025, 2)
        z1 = layer.geometry.minimum_z(first, layer.fk).z
        z2 = layer.geometry.minimum_z(second, layer.fk).z
        assert abs(z2-z1) <= np.dot(coefficients, np.abs(second-first)) + 1e-10


@pytest.mark.parametrize("model", ["official", "dahuan"])
def test_chain_displacement_bound_covers_every_vertex_not_only_minimum(tmp_path, model):
    from nexus_deploy.robot.g2a.async_safety import hold_displacement_coefficients
    config = (write_calibrated_dahuan_config(tmp_path) if model == "dahuan"
              else write_calibrated_official_config(tmp_path, table_height=0))
    layer = gdk_adapter.load_safety_layer(config)
    coefficients = hold_displacement_coefficients(layer)
    assert coefficients is not None and np.isfinite(coefficients).all()
    rng = np.random.default_rng(11092026)
    for _ in range(500):
        first = rng.uniform(-np.pi, np.pi, 24)
        second = first + rng.uniform(-.5, .5, 24)
        first[[7, 15]] = rng.uniform(0, .025, 2)
        second[[7, 15]] = rng.uniform(0, .025, 2)
        bound = np.dot(coefficients, np.abs(second-first))
        worlds = [layer.geometry.world_points(q, layer.fk) for q in (first, second)]
        for group1, group2 in zip(*worlds):
            assert np.max(np.linalg.norm(group2[2] - group1[2], axis=1)) <= bound + 1e-10


def test_bounded_hold_renewal_never_authorizes_new_motion(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        adapter.send_action(target)
        worker.finish()
        reference = worker.result["measured"].copy()
        clock[0] += .01
        adapter.send_action(target)
        clock[0] += .01
        held = adapter.send_action(target)
        clock[0] = 100.05
        worker.feedback = (reference, clock[0])
        np.testing.assert_array_equal(adapter.send_action(-target), held)
        assert adapter._command_safety_started == 100.05
        clock[0] = 100.10
        with pytest.raises(gdk_adapter.SafetyDispatchExpiredError):
            adapter.send_action(target)
    finally:
        adapter.disconnect()


def test_bounded_hold_refuses_unknown_geometry_or_feedback():
    from nexus_deploy.robot.g2a.async_safety import bounded_hold_is_safe
    q = np.zeros(24)
    assert bounded_hold_is_safe(np.ones(24), q, q, 1.0, .8)
    assert not bounded_hold_is_safe(None, q, q, 1.0, .8)
    assert not bounded_hold_is_safe(np.ones(24), q, np.ones(24), 1.0, .8)
    invalid = q.copy()
    invalid[0] = np.nan
    assert not bounded_hold_is_safe(np.ones(24), q, invalid, 1.0, .8)
    assert not bounded_hold_is_safe(np.ones(24), q, q, .8, .8)


def test_expired_motion_can_only_fall_back_to_independently_checked_hold(tmp_path, monkeypatch):
    adapter, clock, worker = _manual_async_adapter(tmp_path, monkeypatch)
    adapter.config.hardware_mode = "execute"
    adapter._execution_prepared = True
    calls, sent = [0], []

    def fake_send(action):
        calls[0] += 1
        if calls[0] == 2:
            clock[0] = 100.041
            worker.feedback = (sent[0].copy(), clock[0])
        adapter._validate_dispatch_age(clock[0])
        sent.append(action.copy())

    adapter._send_combined_action = fake_send
    target = adapter._dry_state.copy()
    target[0] += .1
    try:
        adapter.send_action(target)
        worker.finish()
        clock[0] += .01
        first = adapter.send_action(target)
        clock[0] += .01
        held = adapter.send_action(target)
        np.testing.assert_array_equal(held, first)
        assert len(sent) == 2
        np.testing.assert_array_equal(sent[0], sent[1])
        assert not np.any(adapter._kinematic_limiter._velocity)
    finally:
        adapter.config.hardware_mode = "dry-run"
        adapter.disconnect()


@pytest.mark.parametrize("failure", ["first_rejected", "second_rejected", "second_expired"])
def test_50hz_fake_hardware_does_not_retain_failed_segment(tmp_path, monkeypatch, failure):
    adapter, clock = _paired_adapter(tmp_path, monkeypatch)
    # Replace only the hardware boundary; never instantiate or call real GDK.
    adapter.config.hardware_mode = "execute"
    adapter._execution_prepared = True
    adapter.read_state = lambda: adapter._dry_state.copy()
    adapter._wait_servo_slot = lambda **kwargs: None
    sent = []
    calls = [0]

    def fake_send(action):
        calls[0] += 1
        if failure == "second_expired" and calls[0] == 2:
            clock[0] += 0.015  # Model the final GDK pacing wait.
        adapter._validate_dispatch_age(clock[0])
        if (failure == "first_rejected" and calls[0] == 1) or (
            failure == "second_rejected" and calls[0] == 2
        ):
            raise RuntimeError("Fake GDK rejection")
        sent.append(action.copy())
        adapter._dry_state = action.copy()

    adapter._send_combined_action = fake_send
    adapter.reset_command_reference(adapter._dry_state)
    target = adapter._dry_state.copy()
    target[0] = 0.1
    try:
        adapter.send_action(target)
        if failure == "first_rejected":
            assert adapter._certified_step is None
            assert not sent
            return
        assert adapter._certified_step is not None
        clock[0] += 0.01
        if failure == "second_expired":
            with pytest.raises(gdk_adapter.SafetyDispatchExpiredError):
                adapter.send_action(target)
        else:
            adapter.send_action(target)
        assert adapter._certified_step is None
        assert len(sent) == 1
    finally:
        adapter.config.hardware_mode = "dry-run"
        adapter.disconnect()


@pytest.mark.parametrize("check_fps", [50, 100])
def test_execute_collision_risk_rechecks_and_holds_previous_successful_action(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    check_fps: int,
) -> None:
    layer = safety.ActionSafetyLayer(
        fk=TranslationFK(),
        geometry=safety.GripperGeometry(
            (safety.CollisionPoint("finger_bottom", "tool", (0.0, 0.0, 0.0)),)
        ),
        surface=safety.TableConstraint(height=0.0, hard_margin=0.02),
        motion_indices=tuple(index for index in range(24) if index not in (7, 15)),
        passthrough_indices=(7, 15),
        path_samples=2,
    )
    guard = layer.realtime_guard(path_samples=2)

    class FakeController:
        guard_layer = guard

        def protect(self, current, target, *, raw_action=None):
            return self.guard_layer.guard(current, target, raw_action=raw_action)

        def close(self):
            pass

    adapter = gdk_adapter.G2AAdapter(
        gdk_adapter.AdapterConfig(
            hardware_mode="execute",
            safety_check_fps=check_fps,
            servo_control_period_s=0.01,
            max_arm_velocity=1e6,
            max_arm_acceleration=1e6,
            max_effector_velocity=1e6,
            max_effector_acceleration=1e6,
            max_head_waist_velocity=1e6,
            max_head_waist_acceleration=1e6,
            max_safety_guard_latency_ms=100.0,
        )
    )
    current = adapter._apply_absolute_position_limits(
        np.zeros(24, dtype=np.float32),
        warning_suffix="test_setup",
    )
    current[0] = 0.03
    previous = current.copy()
    previous[0] = 0.025
    target = current.copy()
    target[0] = 0.0
    sent_to_gdk: list[np.ndarray] = []
    adapter._connected = True
    adapter._execution_prepared = True
    adapter._safety_layer = layer
    adapter._safety_controller = FakeController()
    adapter._safety_logger = safety.SafetyJSONLLogger(tmp_path / "collision_hold.jsonl")
    adapter._last_successful_action = previous.copy()
    adapter._last_state = previous.copy()
    adapter._kinematic_limiter.reset(current)
    adapter.read_state = lambda: current.copy()
    adapter._send_combined_action = lambda action: sent_to_gdk.append(action.copy())

    with caplog.at_level("WARNING"):
        result = adapter.send_action(target)
    adapter.disconnect()

    np.testing.assert_allclose(result, previous)
    np.testing.assert_allclose(sent_to_gdk, [previous])
    assert "COLLISION RISK: rejected candidate" in caplog.text
    assert "response=hold_previous_successful_action" in caplog.text
    record = json.loads((tmp_path / "collision_hold.jsonl").read_text(encoding="utf-8"))
    assert record["reason"] == "collision_risk_hold_previous_safe_action"
    assert record["collision_response"] == "hold_previous_successful_action"


def test_optimized_axis_rotation_matches_rodrigues_reference():
    rng = np.random.default_rng(317)
    for _ in range(500):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-10, 10)
        x, y, z = axis
        cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        expected = np.eye(4)
        expected[:3, :3] = (
            np.eye(3) * math.cos(angle)
            + np.outer(axis, axis) * (1 - math.cos(angle))
            + cross * math.sin(angle)
        )
        np.testing.assert_allclose(safety._axis_angle_transform(axis, angle), expected, atol=1e-14)


def test_optimized_surface_query_matches_mask_reference():
    rng = np.random.default_rng(812)
    for bounds in (None, (-0.5, 0.5, -0.3, 0.3), (10, 11, 10, 11)):
        surface = safety.TableConstraint(height=0.4, hard_margin=0.002, xy_bounds=bounds)
        constraints = safety.SurfaceConstraints([surface])
        for _ in range(100):
            world = rng.normal(size=(30, 3))
            radii = rng.uniform(0, 0.03, size=30)
            names = tuple(map(str, range(30)))
            active = np.ones(30, dtype=bool)
            if bounds is not None:
                xmin, xmax, ymin, ymax = bounds
                active[:] = (
                    np.max(world[:, 0] + radii) >= xmin
                    and np.min(world[:, 0] - radii) <= xmax
                    and np.max(world[:, 1] + radii) >= ymin
                    and np.min(world[:, 1] - radii) <= ymax
                )
            indices = np.flatnonzero(active)
            result = constraints.evaluate((("tool", names, world, radii),))
            if not len(indices):
                assert result.distance is None and result.clearance == math.inf
            else:
                distances = world[indices, 2] - radii[indices] - surface.height
                i = int(np.argmin(distances))
                assert result.distance == float(distances[i])
                assert result.point_name == names[indices[i]]
                assert result.clearance == float(distances[i]) - surface.hard_margin


def test_batch_fk_preserves_frame_order_and_returned_pose_isolation(tmp_path):
    layer = safety.load_safety_layer(write_calibrated_dahuan_config(tmp_path))
    frames = tuple(group[0] for group in layer.geometry._groups)
    q = np.zeros(24)
    expected = [layer.fk.frame_pose(q, frame) for frame in frames]
    actual = layer.fk.frame_poses(q, frames)
    np.testing.assert_array_equal(actual, expected)
    actual[0][:] = 0
    np.testing.assert_array_equal(layer.fk.frame_pose(q, frames[0]), expected[0])
    with pytest.raises(safety.SafetyConfigurationError):
        layer.fk.frame_poses(q, ("missing_frame",))
    with pytest.raises(ValueError):
        layer.fk.frame_poses(np.full(24, np.nan), frames)


def test_batch_fk_shared_by_threads_keeps_configuration_snapshot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    layer = safety.load_safety_layer(write_calibrated_dahuan_config(tmp_path))
    frames = tuple(group[0] for group in layer.geometry._groups)
    rng = np.random.default_rng(954)
    configurations = [rng.uniform(-0.5, 0.5, size=24) for _ in range(8)]
    expected = [layer.fk.frame_poses(q, frames) for q in configurations]

    def check(index):
        for _ in range(40):
            actual = layer.fk.frame_poses(configurations[index], frames)
            np.testing.assert_array_equal(actual, expected[index])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, range(8)))
