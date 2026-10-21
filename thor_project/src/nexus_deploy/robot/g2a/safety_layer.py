"""Dependency-light URDF safety projection for horizontal work surfaces.

The runtime deliberately depends only on NumPy and the Python standard library.
Robot kinematics, collision geometry, joint mapping, and work surfaces are all
loaded from a versioned JSON configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import struct
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from xml.etree import ElementTree

import numpy as np


class SafetyConfigurationError(ValueError):
    """Safety geometry is incomplete, inconsistent, or not calibrated."""


class ForwardKinematics(Protocol):
    """Return a safety-base-frame pose for a named robot link/frame."""

    def frame_pose(self, joint_positions: np.ndarray, frame: str) -> np.ndarray: ...


@dataclass(frozen=True)
class CollisionPoint:
    name: str
    frame: str
    xyz: tuple[float, float, float]
    radius: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.frame or len(self.xyz) != 3:
            raise SafetyConfigurationError("Collision points require name, frame, and xyz[3]")
        if not np.isfinite(np.asarray(self.xyz, dtype=np.float64)).all():
            raise SafetyConfigurationError(f"Collision point {self.name!r} is non-finite")
        if not math.isfinite(self.radius) or self.radius < 0:
            raise SafetyConfigurationError(f"Collision point {self.name!r} has invalid radius")


@dataclass(frozen=True)
class MinimumZ:
    z: float
    point_name: str
    frame: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class RevoluteJoint:
    action_index: int
    origin: np.ndarray = field(repr=False)
    axis: np.ndarray = field(repr=False)
    multiplier: float = 1.0
    offset: float = 0.0


class SerialChain:
    """Legacy calibrated serial-chain FK retained for version-1 configs/tests."""

    def __init__(
        self,
        *,
        base_transform: np.ndarray,
        joints: Sequence[RevoluteJoint],
        frame_transform: np.ndarray,
    ) -> None:
        self.base_transform = _validate_transform(base_transform, "base_transform")
        self.joints = tuple(joints)
        self.frame_transform = _validate_transform(frame_transform, "frame_transform")
        if not self.joints:
            raise SafetyConfigurationError("A serial chain needs at least one joint")
        for joint in self.joints:
            if joint.action_index < 0:
                raise SafetyConfigurationError("Joint action indices must be non-negative")
            _validate_transform(joint.origin, "joint origin")
            _validate_axis(joint.axis, "joint axis")
            if not math.isfinite(joint.multiplier) or not math.isfinite(joint.offset):
                raise SafetyConfigurationError("Joint multiplier and offset must be finite")

    def pose(self, joint_positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(joint_positions, dtype=np.float64)
        pose = self.base_transform.copy()
        for joint in self.joints:
            if joint.action_index >= len(positions):
                raise SafetyConfigurationError(
                    f"FK action index {joint.action_index} exceeds vector length {len(positions)}"
                )
            angle = float(positions[joint.action_index]) * joint.multiplier + joint.offset
            pose = pose @ joint.origin @ _axis_angle_transform(joint.axis, angle)
        return pose @ self.frame_transform


class SerialChainForwardKinematics:
    def __init__(self, chains: Mapping[str, SerialChain]) -> None:
        self.chains = dict(chains)
        if not self.chains:
            raise SafetyConfigurationError("At least one FK frame chain is required")

    def frame_pose(self, joint_positions: np.ndarray, frame: str) -> np.ndarray:
        try:
            return self.chains[frame].pose(joint_positions)
        except KeyError as error:
            raise SafetyConfigurationError(f"No FK chain configured for frame {frame!r}") from error


@dataclass(frozen=True)
class _JointRule:
    action_index: int | None
    multiplier: float
    offset: float

    def value(self, positions: np.ndarray, name: str) -> float:
        if self.action_index is None:
            return self.offset
        if self.action_index >= len(positions):
            raise SafetyConfigurationError(
                f"URDF joint {name!r} uses action index {self.action_index}, "
                f"but action dimension is {len(positions)}"
            )
        return float(positions[self.action_index]) * self.multiplier + self.offset


@dataclass(frozen=True)
class _URDFJoint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray = field(repr=False)
    axis: np.ndarray = field(repr=False)
    mimic: tuple[str, float, float] | None = None


class URDFForwardKinematics:
    """Small, strict URDF FK engine for the links used by the safety geometry."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        root_link: str,
        required_frames: Sequence[str],
        joint_map: Mapping[str, Mapping[str, Any]],
        joint_mimics: Mapping[str, Mapping[str, Any]] | None = None,
        fixed_joint_positions: Mapping[str, float] | None = None,
        base_transform: np.ndarray | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        self.urdf_path = urdf_path.expanduser().resolve()
        if not self.urdf_path.is_file():
            raise SafetyConfigurationError(f"URDF does not exist: {self.urdf_path}")
        if expected_sha256:
            digest = hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
            if digest.lower() != str(expected_sha256).lower():
                raise SafetyConfigurationError(
                    f"URDF checksum mismatch: expected {expected_sha256}, got {digest}"
                )
        try:
            robot = ElementTree.parse(self.urdf_path).getroot()
        except ElementTree.ParseError as error:
            raise SafetyConfigurationError(f"Invalid URDF XML: {error}") from error
        if robot.tag != "robot":
            raise SafetyConfigurationError("URDF root element must be <robot>")

        links = {str(element.attrib["name"]) for element in robot.findall("link")}
        if root_link not in links:
            raise SafetyConfigurationError(f"URDF root_link {root_link!r} does not exist")
        self.root_link = root_link
        self.base_transform = _validate_transform(
            np.eye(4) if base_transform is None else base_transform, "URDF base_transform"
        )
        joints: dict[str, _URDFJoint] = {}
        parent_joint: dict[str, _URDFJoint] = {}
        for element in robot.findall("joint"):
            name = str(element.attrib["name"])
            kind = str(element.attrib.get("type", "fixed"))
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise SafetyConfigurationError(f"Unsupported URDF joint type {kind!r}: {name}")
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise SafetyConfigurationError(f"URDF joint {name!r} lacks parent/child")
            parent = str(parent_element.attrib["link"])
            child = str(child_element.attrib["link"])
            axis_element = element.find("axis")
            axis = _parse_vector(
                axis_element.attrib.get("xyz", "1 0 0") if axis_element is not None else "1 0 0",
                f"URDF joint {name} axis",
            )
            if kind != "fixed":
                axis = _validate_axis(axis, f"URDF joint {name} axis")
            origin_element = element.find("origin")
            origin = _pose_from_config(origin_element.attrib if origin_element is not None else {})
            mimic_element = element.find("mimic")
            mimic = None
            if mimic_element is not None:
                mimic = (
                    str(mimic_element.attrib["joint"]),
                    float(mimic_element.attrib.get("multiplier", 1.0)),
                    float(mimic_element.attrib.get("offset", 0.0)),
                )
            joint = _URDFJoint(name, kind, parent, child, origin, axis, mimic)
            if name in joints or child in parent_joint:
                raise SafetyConfigurationError(f"Duplicate URDF joint or child link: {name!r}")
            joints[name] = joint
            parent_joint[child] = joint
        self._joints = joints

        direct: dict[str, _JointRule] = {}
        for name, value in joint_map.items():
            if name not in joints:
                raise SafetyConfigurationError(f"joint_map references unknown URDF joint {name!r}")
            if joints[name].kind == "fixed":
                raise SafetyConfigurationError(f"joint_map cannot drive fixed joint {name!r}")
            direct[name] = _JointRule(
                int(value["action_index"]),
                float(value.get("multiplier", 1.0)),
                float(value.get("offset", 0.0)),
            )
        configured_mimics = {
            str(name): (
                str(value["joint"]),
                float(value.get("multiplier", 1.0)),
                float(value.get("offset", 0.0)),
            )
            for name, value in (joint_mimics or {}).items()
        }
        fixed = {str(name): float(value) for name, value in (fixed_joint_positions or {}).items()}
        unknown_rules = (set(configured_mimics) | set(fixed)) - joints.keys()
        if unknown_rules:
            raise SafetyConfigurationError(f"Joint rules reference unknown joints: {sorted(unknown_rules)}")

        def resolve_rule(name: str, stack: tuple[str, ...] = ()) -> _JointRule:
            if name in stack:
                raise SafetyConfigurationError(f"Cyclic URDF mimic rule: {stack + (name,)}")
            joint = joints[name]
            if joint.kind == "fixed":
                return _JointRule(None, 0.0, 0.0)
            if name in direct:
                return direct[name]
            if name in fixed:
                return _JointRule(None, 0.0, fixed[name])
            mimic = configured_mimics.get(name, joint.mimic)
            if mimic is None:
                raise SafetyConfigurationError(
                    f"Protected URDF chain has unmapped moving joint {name!r}"
                )
            source, multiplier, offset = mimic
            if source not in joints:
                raise SafetyConfigurationError(
                    f"URDF mimic joint {name!r} references unknown source {source!r}"
                )
            source_rule = resolve_rule(source, stack + (name,))
            return _JointRule(
                source_rule.action_index,
                source_rule.multiplier * multiplier,
                source_rule.offset * multiplier + offset,
            )

        required = set()
        for frame in required_frames:
            if frame not in links:
                raise SafetyConfigurationError(f"Collision frame {frame!r} is absent from URDF")
            cursor = frame
            seen: set[str] = set()
            while cursor != root_link:
                if cursor in seen or cursor not in parent_joint:
                    raise SafetyConfigurationError(
                        f"Frame {frame!r} is not connected to root_link {root_link!r}"
                    )
                seen.add(cursor)
                joint = parent_joint[cursor]
                required.add(joint.name)
                cursor = joint.parent

        pending = {name: joints[name] for name in required}
        ordered: list[_URDFJoint] = []
        available = {root_link}
        while pending:
            ready = [joint for joint in pending.values() if joint.parent in available]
            if not ready:
                raise SafetyConfigurationError("Protected URDF joint graph is disconnected or cyclic")
            for joint in ready:
                ordered.append(joint)
                available.add(joint.child)
                del pending[joint.name]
        self._ordered_joints = tuple(ordered)
        self._rules = {
            joint.name: resolve_rule(joint.name)
            for joint in ordered
            if joint.kind != "fixed"
        }
        self._required_frames = frozenset(required_frames)
        # Keep the key and its poses in one snapshot: the legacy optional
        # projector may share this FK object with the synchronous guard.
        self._cache: tuple[bytes, dict[str, np.ndarray]] | None = None

    def frame_pose(self, joint_positions: np.ndarray, frame: str) -> np.ndarray:
        return self.frame_poses(joint_positions, (frame,))[0]

    def frame_poses(
        self, joint_positions: np.ndarray, frames: Sequence[str]
    ) -> tuple[np.ndarray, ...]:
        for frame in frames:
            if frame not in self._required_frames:
                raise SafetyConfigurationError(f"Frame {frame!r} was not compiled for safety FK")
        positions = _finite_vector(joint_positions, "joint_positions")
        cache_key = positions.tobytes()
        cached = self._cache
        if cached is not None and cache_key == cached[0]:
            poses = cached[1]
        else:
            poses = {self.root_link: self.base_transform}
            for joint in self._ordered_joints:
                pose = poses[joint.parent] @ joint.origin
                if joint.kind in {"revolute", "continuous"}:
                    pose = pose @ _axis_angle_transform(
                        joint.axis, self._rules[joint.name].value(positions, joint.name)
                    )
                elif joint.kind == "prismatic":
                    pose = pose @ _translation_transform(
                        joint.axis * self._rules[joint.name].value(positions, joint.name)
                    )
                poses[joint.child] = pose
            self._cache = (cache_key, poses)
        return tuple(poses[frame].copy() for frame in frames)


class GripperGeometry:
    """Collision points/mesh vertices grouped and transformed per URDF frame."""

    def __init__(
        self,
        points: Sequence[CollisionPoint],
        *,
        horizontal_support_exact: bool = False,
    ) -> None:
        self.points = tuple(points)
        self.horizontal_support_exact = bool(horizontal_support_exact)
        if not self.points:
            raise SafetyConfigurationError("At least one gripper collision point is required")
        names = [point.name for point in self.points]
        if len(names) != len(set(names)):
            raise SafetyConfigurationError("Collision point names must be unique")
        grouped: dict[str, list[CollisionPoint]] = {}
        for point in self.points:
            grouped.setdefault(point.frame, []).append(point)
        self._groups = tuple(
            (
                frame,
                tuple(point.name for point in values),
                np.asarray([point.xyz for point in values], dtype=np.float64),
                np.asarray([point.radius for point in values], dtype=np.float64),
            )
            for frame, values in grouped.items()
        )

    def world_points(
        self, joint_positions: np.ndarray, fk: ForwardKinematics
    ) -> tuple[tuple[str, tuple[str, ...], np.ndarray, np.ndarray], ...]:
        result = []
        batch = getattr(fk, "frame_poses", None)
        frames = tuple(group[0] for group in self._groups)
        poses = (
            batch(joint_positions, frames)
            if batch is not None
            else tuple(fk.frame_pose(joint_positions, frame) for frame in frames)
        )
        if len(poses) != len(frames):
            raise SafetyConfigurationError("FK batch returned an incorrect number of frames")
        for (frame, names, xyz, radii), value in zip(self._groups, poses, strict=True):
            # FK implementations validate their static model at construction.
            # Keep only the fail-closed shape/finite check in this 100 Hz path;
            # repeated SVD-like orthonormal checks dominated projection time.
            pose = np.asarray(value, dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise SafetyConfigurationError(f"FK frame {frame!r} returned an invalid pose")
            world = xyz @ pose[:3, :3].T + pose[:3, 3]
            result.append((frame, names, world, radii))
        return tuple(result)

    def minimum_z(self, joint_positions: np.ndarray, fk: ForwardKinematics) -> MinimumZ:
        return _minimum_z_from_world(self.world_points(joint_positions, fk))

    def conservative_bounds(self) -> GripperGeometry:
        """Return one enclosing sphere per collision frame.

        The sphere contains every configured vertex/sphere in its local frame,
        so its lower world-space Z is never above the detailed mesh minimum.
        It may reject a geometrically safe pose, but it cannot approve a pose
        solely because mesh vertices were removed.  This small geometry is the
        synchronous 100 Hz guard; detailed meshes remain available to the
        lower-frequency projector.
        """
        bounds: list[CollisionPoint] = []
        for frame, _names, xyz, radii in self._groups:
            lower = np.min(xyz - radii[:, None], axis=0)
            upper = np.max(xyz + radii[:, None], axis=0)
            center = (lower + upper) * 0.5
            radius = float(np.max(np.linalg.norm(xyz - center, axis=1) + radii))
            bounds.append(
                CollisionPoint(
                    name=f"{frame}:conservative_bound",
                    frame=frame,
                    xyz=tuple(float(value) for value in center),
                    radius=radius,
                )
            )
        return GripperGeometry(bounds)


@dataclass(frozen=True)
class TableConstraint:
    height: float
    hard_margin: float = 0.01
    name: str = "table"
    xy_bounds: tuple[float, float, float, float] | None = None
    warning_margin: float | None = None
    origin: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise SafetyConfigurationError("Surface name must be non-empty")
        if not math.isfinite(self.height):
            raise SafetyConfigurationError("Surface height must be finite in the FK base frame")
        origin = (0.0, 0.0, self.height) if self.origin is None else tuple(self.origin)
        normal = np.asarray(self.normal, dtype=np.float64)
        if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
            raise SafetyConfigurationError("Surface origin must contain three finite values")
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise SafetyConfigurationError("Surface normal must contain three finite values")
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            raise SafetyConfigurationError("Surface normal must be non-zero")
        normal /= norm
        if normal[2] <= 0:
            raise SafetyConfigurationError("Surface normal must point upward in the FK base frame")
        if abs(float(origin[2]) - self.height) > 1e-6:
            # ``dataclasses.replace(surface, height=...)`` was part of the
            # legacy horizontal-surface test/tool API. Preserve that behavior
            # only for the unmistakable legacy origin; calibrated tilted
            # planes still fail closed on inconsistent metadata.
            if tuple(float(value) for value in normal) == (0.0, 0.0, 1.0) and (
                abs(float(origin[0])) <= 1e-12 and abs(float(origin[1])) <= 1e-12
            ):
                origin = (0.0, 0.0, self.height)
            else:
                raise SafetyConfigurationError(
                    "Surface origin Z must match the compatibility height field"
                )
        object.__setattr__(self, "origin", tuple(float(value) for value in origin))
        object.__setattr__(self, "normal", tuple(float(value) for value in normal))
        if not math.isfinite(self.hard_margin) or self.hard_margin <= 0:
            raise SafetyConfigurationError("Hard margin must be positive and finite")
        if self.warning_margin is not None:
            if not math.isfinite(self.warning_margin) or self.warning_margin <= 0:
                raise SafetyConfigurationError("Warning margin must be positive and finite")
            if self.warning_margin < self.hard_margin:
                raise SafetyConfigurationError(
                    "Warning margin must be greater than or equal to the hard margin"
                )
        if self.xy_bounds is not None:
            if len(self.xy_bounds) != 4 or not all(math.isfinite(v) for v in self.xy_bounds):
                raise SafetyConfigurationError("Surface xy_bounds must be finite [xmin,xmax,ymin,ymax]")
            xmin, xmax, ymin, ymax = self.xy_bounds
            if xmin >= xmax or ymin >= ymax:
                raise SafetyConfigurationError("Surface xy_bounds minima must be below maxima")

    @property
    def z_limit(self) -> float:
        """Compatibility value for legacy horizontal-plane diagnostics."""
        return self.height + self.hard_margin

    @property
    def plane_offset(self) -> float:
        return float(np.dot(np.asarray(self.normal), np.asarray(self.origin)))

    def signed_distances(self, world: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Return normal distance from each collision sphere to this plane."""
        if self.normal == (0.0, 0.0, 1.0):
            # Preserve the original exact and allocation-light hot path for
            # legacy horizontal profiles.
            return world[:, 2] - radii - self.height
        return world @ np.asarray(self.normal) - self.plane_offset - radii


@dataclass(frozen=True)
class _SurfaceEvaluation:
    clearance: float
    distance: float | None
    surface_name: str
    point_name: str
    frame: str


class SurfaceConstraints:
    """One or more calibrated planar table/conveyor constraints."""

    def __init__(self, surfaces: Sequence[TableConstraint]) -> None:
        self.surfaces = tuple(surfaces)
        if not self.surfaces:
            raise SafetyConfigurationError("At least one enabled work surface is required")
        names = [surface.name for surface in self.surfaces]
        if len(names) != len(set(names)):
            raise SafetyConfigurationError("Surface names must be unique")

    def evaluate(
        self, groups: tuple[tuple[str, tuple[str, ...], np.ndarray, np.ndarray], ...]
    ) -> _SurfaceEvaluation:
        best: _SurfaceEvaluation | None = None
        for surface in self.surfaces:
            for frame, names, world, radii in groups:
                if surface.xy_bounds is not None:
                    xmin, xmax, ymin, ymax = surface.xy_bounds
                    # Treat a collision frame as active when its complete
                    # world-space AABB overlaps the surface rectangle. Using
                    # per-vertex inclusion could miss a triangle crossing the
                    # rectangle with all three vertices outside it.
                    overlaps = (
                        float((world[:, 0] + radii).max()) >= xmin
                        and float((world[:, 0] - radii).min()) <= xmax
                        and float((world[:, 1] + radii).max()) >= ymin
                        and float((world[:, 1] - radii).min()) <= ymax
                    )
                    if not overlaps:
                        continue
                # A whole collision frame is active or inactive, never a
                # subset of vertices. Avoid allocating an all-true mask,
                # indices and advanced-indexing copies on every servo sample.
                distances = surface.signed_distances(world, radii)
                point_index = int(distances.argmin())
                distance = float(distances[point_index])
                candidate = _SurfaceEvaluation(
                    clearance=distance - surface.hard_margin,
                    distance=distance,
                    surface_name=surface.name,
                    point_name=names[point_index],
                    frame=frame,
                )
                if best is None or candidate.clearance < best.clearance:
                    best = candidate
        return best or _SurfaceEvaluation(math.inf, None, "", "", "")

    def global_clearance(
        self, groups: tuple[tuple[str, tuple[str, ...], np.ndarray, np.ndarray], ...]
    ) -> float:
        """Conservatively treat every configured finite surface as an infinite plane."""
        return min(
            float(surface.signed_distances(world, radii).min()) - surface.hard_margin
            for surface in self.surfaces
            for _frame, _names, world, radii in groups
        )


@dataclass(frozen=True)
class CollisionCapsule:
    """One conservative capsule expressed in a moving URDF frame."""

    name: str
    frame: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float

    def __post_init__(self) -> None:
        values = (*self.start, *self.end, self.radius)
        if not self.name or not self.frame or not all(math.isfinite(value) for value in values):
            raise SafetyConfigurationError("Collision capsule fields must be finite and named")
        if self.radius <= 0:
            raise SafetyConfigurationError("Collision capsule radius must be positive")


@dataclass(frozen=True)
class InterArmEvaluation:
    clearance: float
    separation: float
    left_name: str
    right_name: str
    phase: str = "narrow"


class InterArmCollisionConstraint:
    """Cross-chain capsule clearance; same-arm pairs are intentionally excluded."""

    def __init__(
        self,
        left: Sequence[CollisionCapsule],
        right: Sequence[CollisionCapsule],
        *,
        hard_margin: float = 0.008,
        warning_margin: float = 0.020,
        broad_left: Sequence[CollisionCapsule] = (),
        broad_right: Sequence[CollisionCapsule] = (),
        broad_phase_activation_margin: float = 0.030,
    ) -> None:
        self.left = tuple(left)
        self.right = tuple(right)
        self.broad_left = tuple(broad_left)
        self.broad_right = tuple(broad_right)
        self.hard_margin = float(hard_margin)
        self.warning_margin = float(warning_margin)
        self.broad_phase_activation_margin = float(broad_phase_activation_margin)
        if not self.left or not self.right:
            raise SafetyConfigurationError("Inter-arm collision requires both capsule groups")
        if bool(self.broad_left) != bool(self.broad_right):
            raise SafetyConfigurationError("Inter-arm broad phase requires both capsule groups")
        if (
            not math.isfinite(self.hard_margin)
            or not math.isfinite(self.warning_margin)
            or not math.isfinite(self.broad_phase_activation_margin)
            or self.hard_margin <= 0
            or self.warning_margin < self.hard_margin
            or self.broad_phase_activation_margin < self.warning_margin
        ):
            raise SafetyConfigurationError("Invalid inter-arm collision margins")

    @property
    def required_frames(self) -> frozenset[str]:
        return frozenset(
            item.frame
            for item in (*self.left, *self.right, *self.broad_left, *self.broad_right)
        )

    @staticmethod
    def _world_capsules(
        capsules: Sequence[CollisionCapsule],
        poses: Mapping[str, np.ndarray],
    ) -> tuple[tuple[CollisionCapsule, np.ndarray, np.ndarray], ...]:
        result = []
        for item in capsules:
            pose = poses[item.frame]
            rotation, translation = pose[:3, :3], pose[:3, 3]
            start = np.asarray(item.start) @ rotation.T + translation
            end = np.asarray(item.end) @ rotation.T + translation
            result.append((item, start, end))
        return tuple(result)

    @staticmethod
    def _evaluate_world_capsules(
        left: tuple[tuple[CollisionCapsule, np.ndarray, np.ndarray], ...],
        right: tuple[tuple[CollisionCapsule, np.ndarray, np.ndarray], ...],
        *,
        hard_margin: float,
        phase: str,
    ) -> InterArmEvaluation:
        left_start = np.asarray([item[1] for item in left])
        left_end = np.asarray([item[2] for item in left])
        right_start = np.asarray([item[1] for item in right])
        right_end = np.asarray([item[2] for item in right])
        first = left_end - left_start
        second = right_end - right_start
        aa = np.einsum("li,li->l", first, first)
        cc = np.einsum("ri,ri->r", second, second)
        if np.any(aa <= 1e-12) or np.any(cc <= 1e-12):
            distances = np.asarray(
                [
                    [
                        _segment_segment_distance(a0, a1, b0, b1)
                        for b0, b1 in zip(right_start, right_end, strict=True)
                    ]
                    for a0, a1 in zip(left_start, left_end, strict=True)
                ]
            )
        else:
            offset = left_start[:, None, :] - right_start[None, :, :]
            bb = np.einsum("li,ri->lr", first, second)
            dd = np.einsum("lri,li->lr", offset, first)
            ee = np.einsum("lri,ri->lr", offset, second)
            determinant = aa[:, None] * cc[None, :] - bb * bb
            safe_determinant = np.where(determinant > 1e-12, determinant, 1.0)
            first_parameter = np.where(
                determinant > 1e-12,
                (bb * ee - cc[None, :] * dd) / safe_determinant,
                0.0,
            )
            second_parameter = np.where(
                determinant > 1e-12,
                (aa[:, None] * ee - bb * dd) / safe_determinant,
                ee / cc[None, :],
            )
            first_parameter = np.clip(first_parameter, 0.0, 1.0)
            second_parameter = np.clip(second_parameter, 0.0, 1.0)
            first_parameter = np.clip(
                (bb * second_parameter - dd) / aa[:, None], 0.0, 1.0
            )
            second_parameter = np.clip(
                (bb * first_parameter + ee) / cc[None, :], 0.0, 1.0
            )
            delta = (
                offset
                + first_parameter[:, :, None] * first[:, None, :]
                - second_parameter[:, :, None] * second[None, :, :]
            )
            distances = np.linalg.norm(delta, axis=2)
        radii = (
            np.asarray([item[0].radius for item in left])[:, None]
            + np.asarray([item[0].radius for item in right])[None, :]
        )
        separations = distances - radii
        flat_index = int(np.argmin(separations))
        left_index, right_index = np.unravel_index(flat_index, separations.shape)
        separation = float(separations[left_index, right_index])
        return InterArmEvaluation(
            clearance=separation - hard_margin,
            separation=separation,
            left_name=left[left_index][0].name,
            right_name=right[right_index][0].name,
            phase=phase,
        )

    def evaluate(self, configuration: np.ndarray, fk: ForwardKinematics) -> InterArmEvaluation:
        frames = tuple(self.required_frames)
        values = fk.frame_poses(configuration, frames)
        poses = dict(zip(frames, values, strict=True))
        if self.broad_left:
            broad = self._evaluate_world_capsules(
                self._world_capsules(self.broad_left, poses),
                self._world_capsules(self.broad_right, poses),
                hard_margin=self.hard_margin,
                phase="broad",
            )
            if broad.separation >= self.broad_phase_activation_margin:
                return broad
        return self._evaluate_world_capsules(
            self._world_capsules(self.left, poses),
            self._world_capsules(self.right, poses),
            hard_margin=self.hard_margin,
            phase="narrow",
        )


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Shortest Euclidean distance between two finite 3-D segments."""
    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa = float(np.dot(first, first))
    bb = float(np.dot(first, second))
    cc = float(np.dot(second, second))
    dd = float(np.dot(first, offset))
    ee = float(np.dot(second, offset))
    determinant = aa * cc - bb * bb
    small = 1e-12
    if aa <= small and cc <= small:
        return float(np.linalg.norm(offset))
    if aa <= small:
        first_parameter = 0.0
        second_parameter = float(np.clip(ee / cc, 0.0, 1.0))
    elif cc <= small:
        second_parameter = 0.0
        first_parameter = float(np.clip(-dd / aa, 0.0, 1.0))
    else:
        first_numerator = bb * ee - cc * dd
        second_numerator = aa * ee - bb * dd
        if determinant <= small:
            first_parameter = 0.0
            second_parameter = float(np.clip(ee / cc, 0.0, 1.0))
        else:
            first_parameter = float(np.clip(first_numerator / determinant, 0.0, 1.0))
            second_parameter = float(np.clip(second_numerator / determinant, 0.0, 1.0))
            # Reproject once after clamping because one boundary can move the
            # other segment's closest point.
            first_parameter = float(np.clip((bb * second_parameter - dd) / aa, 0.0, 1.0))
            second_parameter = float(np.clip((bb * first_parameter + ee) / cc, 0.0, 1.0))
    delta = offset + first_parameter * first - second_parameter * second
    return float(np.linalg.norm(delta))


@dataclass(frozen=True)
class _ConfigurationEvaluation:
    minimum_z: MinimumZ
    surface: _SurfaceEvaluation
    work_surface: _SurfaceEvaluation | None = None
    inter_arm: InterArmEvaluation | None = None


@dataclass(frozen=True)
class SafetyDecision:
    raw_action: np.ndarray = field(repr=False)
    controller_target: np.ndarray = field(repr=False)
    safe_action: np.ndarray = field(repr=False)
    current_z_min: float
    predicted_z_min: float
    safe_predicted_z_min: float
    path_z_min: float
    distance_to_surface: float | None
    hard_limit_triggered: bool
    soft_limit_triggered: bool
    downward_scale: float
    action_scale: float
    limiting_point: str
    limiting_frame: str
    limiting_surface: str
    reason: str
    projection_latency_ms: float
    path_surface_clearance: float = math.inf
    work_surface_distance: float | None = None
    work_surface_clearance: float | None = None
    work_surface_name: str = ""
    work_surface_point: str = ""
    inter_arm_separation: float | None = None
    inter_arm_clearance: float | None = None
    inter_arm_left: str = ""
    inter_arm_right: str = ""

    def to_record(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        record = {
            "timestamp_ns": time.time_ns(),
            "raw_action": self.raw_action.tolist(),
            "controller_target": self.controller_target.tolist(),
            "safe_action": self.safe_action.tolist(),
            "current_z_min": self.current_z_min,
            "predicted_z_min": self.predicted_z_min,
            "safe_predicted_z_min": self.safe_predicted_z_min,
            "path_z_min": self.path_z_min,
            "path_surface_clearance": self.path_surface_clearance,
            "distance_to_surface": self.distance_to_surface,
            "hard_limit_triggered": self.hard_limit_triggered,
            "soft_limit_triggered": self.soft_limit_triggered,
            "downward_scale": self.downward_scale,
            "action_scale": self.action_scale,
            "limiting_point": self.limiting_point,
            "limiting_frame": self.limiting_frame,
            "limiting_surface": self.limiting_surface,
            "reason": self.reason,
            "projection_latency_ms": self.projection_latency_ms,
            "work_surface_distance": self.work_surface_distance,
            "work_surface_clearance": self.work_surface_clearance,
            "work_surface_name": self.work_surface_name,
            "work_surface_point": self.work_surface_point,
            "inter_arm_separation": self.inter_arm_separation,
            "inter_arm_clearance": self.inter_arm_clearance,
            "inter_arm_left": self.inter_arm_left,
            "inter_arm_right": self.inter_arm_right,
            "chunk_output_length": None,
            "chunk_execute_length": None,
            "policy_latency_ms": None,
            "control_latency_ms": None,
        }
        if context:
            record.update(context)
        return record


class ActionSafetyLayer:
    """Project absolute joint targets onto all configured surface-safe segments."""

    def __init__(
        self,
        *,
        fk: ForwardKinematics,
        geometry: GripperGeometry,
        surface: TableConstraint | SurfaceConstraints,
        motion_indices: Sequence[int],
        passthrough_indices: Sequence[int],
        inter_arm_collision: InterArmCollisionConstraint | None = None,
        path_samples: int = 8,
        search_iterations: int = 24,
        tolerance: float = 1e-6,
    ) -> None:
        self.fk = fk
        self.geometry = geometry
        self.surfaces = surface if isinstance(surface, SurfaceConstraints) else SurfaceConstraints((surface,))
        self.inter_arm_collision = inter_arm_collision
        # Compatibility for existing diagnostics. This is the highest global
        # hard limit, while actual checks still honor each surface's XY bounds.
        self.surface = max(self.surfaces.surfaces, key=lambda item: item.z_limit)
        self.motion_indices = tuple(int(index) for index in motion_indices)
        self.passthrough_indices = tuple(int(index) for index in passthrough_indices)
        self.path_samples = int(path_samples)
        self.search_iterations = int(search_iterations)
        self.tolerance = float(tolerance)
        combined = self.motion_indices + self.passthrough_indices
        if not self.motion_indices or len(combined) != len(set(combined)):
            raise SafetyConfigurationError("Motion/passthrough indices must be non-empty and disjoint")
        if min(combined) < 0:
            raise SafetyConfigurationError("Action indices must be non-negative")
        if self.path_samples < 1 or self.search_iterations < 1 or self.tolerance < 0:
            raise SafetyConfigurationError("Invalid path_samples, search_iterations, or tolerance")

    def minimum_z(self, configuration: np.ndarray) -> MinimumZ:
        return self.geometry.minimum_z(configuration, self.fk)

    def _measured_opening_geometry(
        self,
        configuration: np.ndarray,
        measured_configuration: np.ndarray,
    ) -> np.ndarray:
        """Evaluate arm candidates with the measured gripper opening only.

        Gripper commands remain passthrough outputs, but a planned opening or
        closing command must not resize the collision geometry before the
        hardware feedback reports that opening. The next measured-state sample
        naturally updates the geometry.
        """
        if not self.passthrough_indices:
            return configuration
        value = configuration.copy()
        indices = list(self.passthrough_indices)
        value[indices] = measured_configuration[indices]
        return value

    def realtime_guard(self, *, path_samples: int = 2) -> ActionSafetyLayer:
        """Build a bounded-cost guard for the servo thread.

        Profiles whose mesh vertices were reduced offline to their convex-hull
        support vertices can use that geometry directly. Plane-normal support
        and X/Y extrema are linear queries, so the hull vertices are mathematically
        equivalent to every original STL vertex. Legacy
        profiles keep the one-sphere-per-frame conservative fallback.
        """
        if path_samples < 1:
            raise SafetyConfigurationError("Realtime guard path_samples must be positive")
        return ActionSafetyLayer(
            fk=self.fk,
            geometry=(
                self.geometry
                if self.geometry.horizontal_support_exact
                else self.geometry.conservative_bounds()
            ),
            surface=self.surfaces,
            motion_indices=self.motion_indices,
            passthrough_indices=self.passthrough_indices,
            inter_arm_collision=self.inter_arm_collision,
            path_samples=path_samples,
            search_iterations=1,
            tolerance=self.tolerance,
        )

    def _evaluate(self, configuration: np.ndarray) -> _ConfigurationEvaluation:
        world = self.geometry.world_points(configuration, self.fk)
        work_surface = self.surfaces.evaluate(world)
        surface = work_surface
        inter_arm = None
        if self.inter_arm_collision is not None:
            inter_arm = self.inter_arm_collision.evaluate(configuration, self.fk)
            if inter_arm.clearance < surface.clearance:
                surface = _SurfaceEvaluation(
                    clearance=inter_arm.clearance,
                    distance=inter_arm.separation,
                    surface_name="inter_arm",
                    point_name=f"{inter_arm.left_name}<->{inter_arm.right_name}",
                    frame="dual_arm",
                )
        return _ConfigurationEvaluation(
            _minimum_z_from_world(world),
            surface,
            work_surface=work_surface,
            inter_arm=inter_arm,
        )

    def global_surface_clearance(self, configuration: np.ndarray) -> float:
        """Lower-bound clearance with finite surfaces extended to infinite planes."""
        world = self.geometry.world_points(configuration, self.fk)
        return self.surfaces.global_clearance(world)

    def _evaluate_for_feedback_reuse(self, configuration):
        """Same sampled geometry decision, plus a finite-table drift margin.

        A frame stays safe if either its plane-normal clearance or separation from the
        finite XY rectangle remains positive. This bounds changes at sampled
        poses only; it is not a continuous swept-volume certificate.
        """
        world = self.geometry.world_points(configuration, self.fk)
        margin = math.inf
        for surface in self.surfaces.surfaces:
            for _frame, _names, points, radii in world:
                separation = float(surface.signed_distances(points, radii).min()) - surface.hard_margin
                if surface.xy_bounds is not None:
                    xmin, xmax, ymin, ymax = surface.xy_bounds
                    separation = max(separation,
                        xmin-float((points[:, 0]+radii).max()),
                        float((points[:, 0]-radii).min())-xmax,
                        ymin-float((points[:, 1]+radii).max()),
                        float((points[:, 1]-radii).min())-ymax)
                margin = min(margin, separation)
        work_surface = self.surfaces.evaluate(world)
        inter_arm = (
            None
            if self.inter_arm_collision is None
            else self.inter_arm_collision.evaluate(configuration, self.fk)
        )
        limiting = work_surface
        if inter_arm is not None and inter_arm.clearance < work_surface.clearance:
            limiting = _SurfaceEvaluation(
                clearance=inter_arm.clearance,
                distance=inter_arm.separation,
                surface_name="inter_arm",
                point_name=f"{inter_arm.left_name}<->{inter_arm.right_name}",
                frame="dual_arm",
            )
        return _ConfigurationEvaluation(
            _minimum_z_from_world(world),
            limiting,
            work_surface=work_surface,
            inter_arm=inter_arm,
        ), margin

    def guard(
        self,
        current_configuration: np.ndarray,
        controller_target: np.ndarray,
        *,
        raw_action: np.ndarray | None = None,
    ) -> SafetyDecision:
        """Synchronously approve a short servo segment or hold measured pose.

        Unlike :meth:`project`, this method never performs bisection.  It
        checks the measured state and evenly spaced arm samples through the
        whole current-to-target segment. Gripper geometry stays at its measured
        opening; target gripper values remain passthrough commands. Any unsafe
        sample causes an immediate full-pose hold.
        """
        return self.guard_trajectory(current_configuration, (controller_target,), raw_action=raw_action)

    def guard_trajectory(
        self,
        current_configuration: np.ndarray,
        controller_targets: Sequence[np.ndarray],
        *,
        raw_action: np.ndarray | None = None,
    ) -> SafetyDecision:
        """Check every segment of a short, frozen servo trajectory.

        No vertex or intermediate sample is removed. Shared segment endpoints
        are evaluated once. Any unsafe segment rejects the entire trajectory.
        """
        started = time.perf_counter()
        current = _finite_vector(current_configuration, "current_configuration")
        targets = tuple(_finite_vector(value, "controller_target") for value in controller_targets)
        if not targets or any(value.shape != current.shape for value in targets):
            raise ValueError("Safety trajectory must contain matching non-empty vectors")
        target = targets[-1]
        raw = target if raw_action is None else _finite_vector(raw_action, "raw_action")
        if current.shape != target.shape or raw.shape != target.shape:
            raise ValueError(
                f"Safety vectors must share one shape: current={current.shape}, "
                f"target={target.shape}, raw={raw.shape}"
            )
        if max(self.motion_indices + self.passthrough_indices) >= len(target):
            raise SafetyConfigurationError("Safety action index exceeds action dimension")
        if set(self.motion_indices + self.passthrough_indices) != set(range(len(target))):
            raise SafetyConfigurationError(
                "Motion and passthrough indices must cover every action dimension exactly once"
            )

        current_eval = self._evaluate(current)
        if current_eval.surface.clearance < -self.tolerance:
            return self._decision(
                started,
                raw,
                target,
                current.copy(),
                current_eval,
                current_eval,
                current_eval,
                current_eval.minimum_z.z,
                True,
                0.0,
                "realtime_guard_current_unsafe_hold",
                limiting_eval=current_eval,
                path_surface_clearance=current_eval.surface.clearance,
            )

        path_z_min = current_eval.minimum_z.z
        path_surface_clearance = current_eval.surface.clearance
        target_eval = current_eval
        unsafe_eval: _ConfigurationEvaluation | None = None
        previous = current
        for waypoint in targets:
            for step in range(1, self.path_samples + 1):
                alpha = step / self.path_samples
                sample = previous + alpha * (waypoint - previous)
                sample_eval = self._evaluate(
                    self._measured_opening_geometry(sample, current)
                )
                path_z_min = min(path_z_min, sample_eval.minimum_z.z)
                path_surface_clearance = min(
                    path_surface_clearance, sample_eval.surface.clearance
                )
                target_eval = sample_eval
                if sample_eval.surface.clearance < -self.tolerance and (
                    unsafe_eval is None
                    or sample_eval.surface.clearance < unsafe_eval.surface.clearance
                ):
                    unsafe_eval = sample_eval
            previous = waypoint

        if unsafe_eval is not None:
            return self._decision(
                started,
                raw,
                target,
                current.copy(),
                current_eval,
                target_eval,
                current_eval,
                path_z_min,
                True,
                0.0,
                "realtime_guard_hold_pending_projection",
                limiting_eval=unsafe_eval,
                path_surface_clearance=path_surface_clearance,
            )
        return self._decision(
            started,
            raw,
            target,
            target.copy(),
            current_eval,
            target_eval,
            target_eval,
            path_z_min,
            False,
            1.0,
            "realtime_guard_safe",
            path_surface_clearance=path_surface_clearance,
        )

    def project(
        self,
        current_configuration: np.ndarray,
        controller_target: np.ndarray,
        *,
        raw_action: np.ndarray | None = None,
    ) -> SafetyDecision:
        started = time.perf_counter()
        current = _finite_vector(current_configuration, "current_configuration")
        target = _finite_vector(controller_target, "controller_target")
        raw = target if raw_action is None else _finite_vector(raw_action, "raw_action")
        if current.shape != target.shape or raw.shape != target.shape:
            raise ValueError(
                f"Safety vectors must share one shape: current={current.shape}, "
                f"target={target.shape}, raw={raw.shape}"
            )
        if max(self.motion_indices + self.passthrough_indices) >= len(target):
            raise SafetyConfigurationError("Safety action index exceeds action dimension")
        if set(self.motion_indices + self.passthrough_indices) != set(range(len(target))):
            raise SafetyConfigurationError(
                "Motion and passthrough indices must cover every action dimension exactly once"
            )

        current_eval = self._evaluate(current)
        predicted_eval = self._evaluate(
            self._measured_opening_geometry(target, current)
        )

        def candidate(alpha: float) -> np.ndarray:
            value = current.copy()
            indices = list(self.motion_indices)
            value[indices] = current[indices] + alpha * (target[indices] - current[indices])
            passthrough = list(self.passthrough_indices)
            value[passthrough] = current[passthrough]
            return value

        def output_candidate(alpha: float) -> np.ndarray:
            value = candidate(alpha)
            passthrough = list(self.passthrough_indices)
            value[passthrough] = target[passthrough]
            return value

        start = candidate(0.0)
        start_eval = self._evaluate(start)
        if start_eval.surface.clearance < -self.tolerance:
            safe = current.copy()
            safe_eval = current_eval
            return self._decision(
                started,
                raw,
                target,
                safe,
                current_eval,
                predicted_eval,
                safe_eval,
                min(start_eval.minimum_z.z, predicted_eval.minimum_z.z),
                True,
                0.0,
                "zero_motion_candidate_unsafe",
            )

        # The target reaching this layer has already been limited to one servo
        # tick. Match the specified projection rule: a safe predicted state is
        # sent unchanged; only an unsafe endpoint enters path search/bisection.
        # This keeps the common case well inside the 100 Hz control budget.
        if predicted_eval.surface.clearance >= -self.tolerance:
            return self._decision(
                started,
                raw,
                target,
                target.copy(),
                current_eval,
                predicted_eval,
                predicted_eval,
                min(start_eval.minimum_z.z, predicted_eval.minimum_z.z),
                False,
                1.0,
                "safe",
            )

        lower_alpha = 0.0
        upper_alpha: float | None = None
        path_z_min = start_eval.minimum_z.z
        for step in range(1, self.path_samples + 1):
            alpha = step / self.path_samples
            point_eval = self._evaluate(candidate(alpha))
            path_z_min = min(path_z_min, point_eval.minimum_z.z)
            if point_eval.surface.clearance < -self.tolerance:
                upper_alpha = alpha
                break
            lower_alpha = alpha

        if upper_alpha is None:
            raise RuntimeError("Unsafe predicted target was not found by endpoint-inclusive path search")

        for _ in range(self.search_iterations):
            midpoint = (lower_alpha + upper_alpha) * 0.5
            midpoint_eval = self._evaluate(candidate(midpoint))
            path_z_min = min(path_z_min, midpoint_eval.minimum_z.z)
            if midpoint_eval.surface.clearance >= 0:
                lower_alpha = midpoint
            else:
                upper_alpha = midpoint
        safe = output_candidate(lower_alpha)
        safe_eval = self._evaluate(candidate(lower_alpha))
        return self._decision(
            started,
            raw,
            target,
            safe,
            current_eval,
            predicted_eval,
            safe_eval,
            path_z_min,
            True,
            lower_alpha,
            "projected_to_hard_height",
        )

    def _decision(
        self,
        started: float,
        raw: np.ndarray,
        target: np.ndarray,
        safe: np.ndarray,
        current: _ConfigurationEvaluation,
        predicted: _ConfigurationEvaluation,
        safe_eval: _ConfigurationEvaluation,
        path_z_min: float,
        triggered: bool,
        alpha: float,
        reason: str,
        *,
        limiting_eval: _ConfigurationEvaluation | None = None,
        path_surface_clearance: float | None = None,
    ) -> SafetyDecision:
        limiting = (
            limiting_eval.surface
            if limiting_eval is not None
            else safe_eval.surface if triggered else predicted.surface
        )
        surface = next(
            (
                candidate
                for candidate in self.surfaces.surfaces
                if candidate.name == current.surface.surface_name
            ),
            None,
        )
        warning_triggered = bool(
            (
                surface is not None
                and surface.warning_margin is not None
                and current.surface.distance is not None
                and current.surface.distance < surface.warning_margin - self.tolerance
            )
            or (
                current.surface.surface_name == "inter_arm"
                and self.inter_arm_collision is not None
                and current.surface.distance is not None
                and current.surface.distance
                < self.inter_arm_collision.warning_margin - self.tolerance
            )
        )
        return SafetyDecision(
            raw_action=raw.copy(),
            controller_target=target.copy(),
            safe_action=safe,
            current_z_min=current.minimum_z.z,
            predicted_z_min=predicted.minimum_z.z,
            safe_predicted_z_min=safe_eval.minimum_z.z,
            path_z_min=path_z_min,
            distance_to_surface=current.surface.distance,
            hard_limit_triggered=triggered,
            soft_limit_triggered=warning_triggered,
            downward_scale=1.0,
            action_scale=alpha,
            limiting_point=limiting.point_name,
            limiting_frame=limiting.frame,
            limiting_surface=limiting.surface_name,
            reason=reason,
            projection_latency_ms=(time.perf_counter() - started) * 1000,
            path_surface_clearance=(
                min(
                    current.surface.clearance,
                    predicted.surface.clearance,
                    safe_eval.surface.clearance,
                )
                if path_surface_clearance is None
                else float(path_surface_clearance)
            ),
            work_surface_distance=(
                None if current.work_surface is None else current.work_surface.distance
            ),
            work_surface_clearance=(
                None if current.work_surface is None else current.work_surface.clearance
            ),
            work_surface_name=(
                "" if current.work_surface is None else current.work_surface.surface_name
            ),
            work_surface_point=(
                "" if current.work_surface is None else current.work_surface.point_name
            ),
            inter_arm_separation=(
                None if current.inter_arm is None else current.inter_arm.separation
            ),
            inter_arm_clearance=(
                None if current.inter_arm is None else current.inter_arm.clearance
            ),
            inter_arm_left=(
                "" if current.inter_arm is None else current.inter_arm.left_name
            ),
            inter_arm_right=(
                "" if current.inter_arm is None else current.inter_arm.right_name
            ),
        )


@dataclass(frozen=True)
class _ProjectionWork:
    sequence: int
    current: np.ndarray = field(repr=False)
    target: np.ndarray = field(repr=False)
    raw: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class _ProjectionResult:
    work: _ProjectionWork
    decision: SafetyDecision | None = None
    error: BaseException | None = None


class RealtimeSafetyController:
    """Combine a synchronous conservative guard with exact background projection."""

    def __init__(
        self,
        guard_layer: ActionSafetyLayer,
        projection_layer: ActionSafetyLayer,
        *,
        target_match_tolerance: float = 1e-5,
        allow_projected_actions: bool = True,
    ) -> None:
        if target_match_tolerance < 0 or not math.isfinite(target_match_tolerance):
            raise ValueError("target_match_tolerance must be finite and non-negative")
        self.guard_layer = guard_layer
        self.projection_layer = projection_layer
        self.target_match_tolerance = target_match_tolerance
        self.allow_projected_actions = allow_projected_actions
        self._queue: queue.Queue[_ProjectionWork | None] = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._result: _ProjectionResult | None = None
        self._pending = False
        self._sequence = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="DetailedSafetyProjector",
            daemon=True,
        )
        if self.allow_projected_actions:
            self._thread.start()

    def protect(
        self,
        current_configuration: np.ndarray,
        controller_target: np.ndarray,
        *,
        raw_action: np.ndarray | None = None,
    ) -> SafetyDecision:
        current = _finite_vector(current_configuration, "current_configuration")
        target = _finite_vector(controller_target, "controller_target")
        raw = target if raw_action is None else _finite_vector(raw_action, "raw_action")
        direct = self.guard_layer.guard(current, target, raw_action=raw)
        if not direct.hard_limit_triggered or not self.allow_projected_actions:
            return direct

        result = self._take_result()
        if result is not None:
            if result.error is not None:
                raise RuntimeError("Detailed safety projection worker failed") from result.error
            assert result.decision is not None
            if np.allclose(
                result.work.target,
                target,
                rtol=0.0,
                atol=self.target_match_tolerance,
            ):
                candidate = np.asarray(result.decision.safe_action, dtype=np.float64)
                rechecked = self.guard_layer.guard(current, candidate, raw_action=raw)
                if not rechecked.hard_limit_triggered:
                    return SafetyDecision(
                        raw_action=raw.copy(),
                        controller_target=target.copy(),
                        safe_action=np.asarray(rechecked.safe_action).copy(),
                        current_z_min=rechecked.current_z_min,
                        predicted_z_min=direct.predicted_z_min,
                        safe_predicted_z_min=rechecked.safe_predicted_z_min,
                        path_z_min=min(direct.path_z_min, rechecked.path_z_min),
                        distance_to_surface=rechecked.distance_to_surface,
                        hard_limit_triggered=True,
                        soft_limit_triggered=rechecked.soft_limit_triggered,
                        downward_scale=1.0,
                        action_scale=result.decision.action_scale,
                        limiting_point=result.decision.limiting_point,
                        limiting_frame=result.decision.limiting_frame,
                        limiting_surface=result.decision.limiting_surface,
                        reason="background_projection_revalidated",
                        projection_latency_ms=(
                            direct.projection_latency_ms
                            + result.decision.projection_latency_ms
                            + rechecked.projection_latency_ms
                        ),
                        work_surface_distance=rechecked.work_surface_distance,
                        work_surface_clearance=rechecked.work_surface_clearance,
                        work_surface_name=rechecked.work_surface_name,
                        work_surface_point=rechecked.work_surface_point,
                        inter_arm_separation=rechecked.inter_arm_separation,
                        inter_arm_clearance=rechecked.inter_arm_clearance,
                        inter_arm_left=rechecked.inter_arm_left,
                        inter_arm_right=rechecked.inter_arm_right,
                    )

        self._submit(current, target, raw)
        return direct

    def _submit(self, current: np.ndarray, target: np.ndarray, raw: np.ndarray) -> None:
        with self._result_lock:
            if self._closed or self._pending:
                return
            self._sequence += 1
            work = _ProjectionWork(
                sequence=self._sequence,
                current=current.copy(),
                target=target.copy(),
                raw=raw.copy(),
            )
            self._pending = True
        self._queue.put_nowait(work)

    def _take_result(self) -> _ProjectionResult | None:
        with self._result_lock:
            result = self._result
            self._result = None
            return result

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            if work is None:
                self._queue.task_done()
                return
            try:
                decision = self.projection_layer.project(
                    work.current,
                    work.target,
                    raw_action=work.raw,
                )
                result = _ProjectionResult(work=work, decision=decision)
            except BaseException as error:
                result = _ProjectionResult(work=work, error=error)
            with self._result_lock:
                self._result = result
                self._pending = False
            self._queue.task_done()

    def close(self, timeout_s: float = 2.0) -> None:
        with self._result_lock:
            if self._closed:
                return
            self._closed = True
        if not self.allow_projected_actions:
            return
        self._queue.put(None)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("Detailed safety projection worker did not stop")


class SafetyAuditBackpressureError(RuntimeError):
    """Safety audit cannot keep up without blocking the servo thread."""


class SafetyJSONLLogger:
    def __init__(self, path: Path, *, queue_size: int = 2048) -> None:
        if queue_size < 1:
            raise ValueError("Safety log queue_size must be positive")
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[
            tuple[SafetyDecision | None, dict[str, Any]] | None
        ] = queue.Queue(maxsize=queue_size)
        self._error_lock = threading.Lock()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="SafetyAuditWriter", daemon=True)
        self._thread.start()

    def write(self, decision: SafetyDecision, context: Mapping[str, Any] | None = None) -> None:
        self._enqueue(decision, context)

    def write_event(self, context: Mapping[str, Any]) -> None:
        """Record lifecycle events at production time, not writer-thread time."""
        self._enqueue(None, context)

    def _enqueue(self, decision: SafetyDecision | None, context: Mapping[str, Any] | None) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Safety audit logger is closed")
        record = dict(context or {})
        record.setdefault("timestamp_ns", time.time_ns())
        record.setdefault("event_monotonic", time.perf_counter())
        try:
            self._queue.put_nowait((decision, record))
        except queue.Full as error:
            raise SafetyAuditBackpressureError(
                "Safety audit queue is full; refusing to continue without a complete audit trail"
            ) from error

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_failed()

    def close(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            self._raise_if_failed()
            return
        self._raise_if_failed()
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout_s)
        except queue.Full as error:
            self._raise_if_failed()
            raise RuntimeError("Safety audit queue did not drain during shutdown") from error
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError("Safety audit writer did not stop")
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Safety audit writer failed") from error

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                while True:
                    item = self._queue.get()
                    try:
                        if item is None:
                            stream.flush()
                            return
                        decision, context = item
                        line = json.dumps(
                            context if decision is None else decision.to_record(context),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        stream.write(line + "\n")
                        stream.flush()
                    finally:
                        self._queue.task_done()
        except BaseException as error:
            with self._error_lock:
                self._error = error
            # Release any producers/flush callers waiting for queued tasks.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()


def load_safety_layer(
    path: Path,
    *,
    allow_read_only_ready: bool = False,
    asset_root: Path | None = None,
    effector_envelope_ranges: Mapping[int, tuple[float, float]] | None = None,
) -> ActionSafetyLayer:
    config_path = path.expanduser().resolve()
    asset_directory = (
        config_path.parent
        if asset_root is None
        else Path(asset_root).expanduser().resolve(strict=True)
    )
    if not asset_directory.is_dir():
        raise SafetyConfigurationError(f"Safety asset root is not a directory: {asset_directory}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if data.get("version") not in {1, 2}:
        raise SafetyConfigurationError("Safety config version must be 1 or 2")
    calibrated = data.get("calibrated") is True
    read_only_ready = data.get("read_only_ready") is True
    if not calibrated and not (allow_read_only_ready and read_only_ready):
        raise SafetyConfigurationError(
            "Safety config must explicitly set calibrated=true after FK/tool/surface validation; "
            "read_only_ready profiles require allow_read_only_ready=True"
        )
    base_frame = str(data.get("base_frame", "base"))
    if not base_frame:
        raise SafetyConfigurationError("base_frame must be non-empty")

    horizontal_support_exact = bool(data.get("collision_meshes")) and all(
        "support_vertex_indices" in item for item in data.get("collision_meshes", ())
    )
    inter_arm_data = data.get("inter_arm_collision")
    inter_arm_collision: InterArmCollisionConstraint | None = None
    inter_arm_frames: set[str] = set()
    if inter_arm_data is not None and inter_arm_data.get("enabled", True) is True:
        def load_capsules(side: str) -> tuple[CollisionCapsule, ...]:
            result = []
            for item in inter_arm_data.get(side, ()):
                result.append(
                    CollisionCapsule(
                        name=str(item["name"]),
                        frame=str(item["frame"]),
                        start=tuple(float(value) for value in item["start"]),
                        end=tuple(float(value) for value in item["end"]),
                        radius=float(item["radius"]),
                    )
                )
            return tuple(result)

        left_capsules = load_capsules("left_capsules")
        right_capsules = load_capsules("right_capsules")
        broad_left_capsules = load_capsules("broad_phase_left_capsules")
        broad_right_capsules = load_capsules("broad_phase_right_capsules")
        inter_arm_collision = InterArmCollisionConstraint(
            left_capsules,
            right_capsules,
            hard_margin=float(inter_arm_data.get("hard_margin", 0.008)),
            warning_margin=float(inter_arm_data.get("warning_margin", 0.020)),
            broad_left=broad_left_capsules,
            broad_right=broad_right_capsules,
            broad_phase_activation_margin=float(
                inter_arm_data.get("broad_phase_activation_margin", 0.030)
            ),
        )
        inter_arm_frames.update(inter_arm_collision.required_frames)
    if "robot_model" in data:
        model = data["robot_model"]
        if model.get("type") != "urdf":
            raise SafetyConfigurationError("robot_model.type must be 'urdf'")
        urdf_path = _relative_path(asset_directory, model["path"])
        points = _load_collision_points(
            data,
            asset_directory,
            urdf_path=urdf_path,
        )
        frames = sorted({point.frame for point in points} | inter_arm_frames)
        fk: ForwardKinematics = URDFForwardKinematics(
            urdf_path,
            root_link=str(model.get("root_link", base_frame)),
            required_frames=frames,
            joint_map=model.get("joint_map", {}),
            joint_mimics=model.get("joint_mimics", {}),
            fixed_joint_positions=model.get("fixed_joint_positions", {}),
            base_transform=_pose_from_config(model.get("base_transform", {})),
            expected_sha256=model.get("sha256"),
        )
    else:
        points = _load_collision_points(data, asset_directory)
        frames = sorted({point.frame for point in points} | inter_arm_frames)
        chains: dict[str, SerialChain] = {}
        for frame, chain_data in data.get("frame_chains", {}).items():
            joints = []
            for joint_data in chain_data.get("joints", ()):
                axis = _validate_axis(
                    np.asarray(joint_data["axis"], dtype=np.float64), f"frame {frame} axis"
                )
                joints.append(
                    RevoluteJoint(
                        action_index=int(joint_data["action_index"]),
                        origin=_pose_from_config(joint_data.get("origin", {})),
                        axis=axis,
                        multiplier=float(joint_data.get("multiplier", 1.0)),
                        offset=float(joint_data.get("offset", 0.0)),
                    )
                )
            chains[str(frame)] = SerialChain(
                base_transform=_pose_from_config(chain_data.get("base", {})),
                joints=joints,
                frame_transform=_pose_from_config(chain_data.get("frame", {})),
            )
        missing_frames = sorted(set(frames) - chains.keys())
        if missing_frames:
            raise SafetyConfigurationError(
                f"Collision points reference frames without FK chains: {missing_frames}"
            )
        fk = SerialChainForwardKinematics(chains)

    surfaces_data = data.get("surfaces")
    if surfaces_data is None:
        surfaces_data = [data.get("surface", {})]
    surfaces = []
    for item in surfaces_data:
        if item.get("enabled", True) is not True:
            continue
        surface_frame = str(item.get("frame", base_frame))
        if surface_frame != base_frame:
            raise SafetyConfigurationError(
                f"Surface frame {surface_frame!r} must match FK base_frame {base_frame!r}"
            )
        bounds = item.get("xy_bounds")
        surfaces.append(
            TableConstraint(
                name=str(item.get("name", "table")),
                height=float(item["height"]),
                hard_margin=float(item.get("hard_margin", 0.01)),
                xy_bounds=None if bounds is None else tuple(float(value) for value in bounds),
                warning_margin=(
                    None
                    if item.get("warning_margin") is None
                    else float(item["warning_margin"])
                ),
                origin=(
                    None
                    if item.get("origin") is None
                    else tuple(float(value) for value in item["origin"])
                ),
                normal=tuple(float(value) for value in item.get("normal", (0.0, 0.0, 1.0))),
            )
        )
    geometry = GripperGeometry(points, horizontal_support_exact=horizontal_support_exact)
    if effector_envelope_ranges:
        geometry = prismatic_opening_envelope(geometry, fk, effector_envelope_ranges)
    return ActionSafetyLayer(
        fk=fk,
        geometry=geometry,
        surface=SurfaceConstraints(surfaces),
        motion_indices=data["motion_indices"],
        passthrough_indices=data.get("passthrough_indices", ()),
        inter_arm_collision=inter_arm_collision,
        path_samples=int(data.get("path_samples", 8)),
        search_iterations=int(data.get("search_iterations", 24)),
    )


def prismatic_opening_envelope(geometry, fk, ranges):
    """Compile the full independent prismatic opening box into wrist frames.

    Every point is affine in the selected openings; its box corners contain
    the entire swept point in their convex hull. Rotating/mixed unsupported
    chains fail closed. Runtime cost does not multiply FK evaluations.
    """
    from itertools import product

    if not isinstance(fk, URDFForwardKinematics):
        raise SafetyConfigurationError("Device-position mode requires URDF prismatic-envelope FK")
    for index, bounds in ranges.items():
        if not isinstance(index, int) or index < 0 or len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]:
            raise SafetyConfigurationError("Invalid effector envelope range")
    by_child = {j.child: j for j in fk._ordered_joints}
    result, anchors, covered = [], set(), set()
    for point in geometry.points:
        chain, frame = [], point.frame
        while frame in by_child:
            joint = by_child[frame]
            chain.append(joint)
            frame = joint.parent
        chain.reverse()
        first = next((i for i, j in enumerate(chain)
                      if j.name in fk._rules and fk._rules[j.name].action_index in ranges), None)
        if first is None:
            result.append(point)
            continue
        suffix = chain[first:]
        indices = set()
        for joint in suffix:
            rule = fk._rules.get(joint.name)
            if rule is not None and rule.action_index is not None:
                if rule.action_index not in ranges or joint.kind != "prismatic":
                    raise SafetyConfigurationError("Opening envelope requires an exclusively prismatic effector suffix")
                indices.add(rule.action_index)
        indices = sorted(indices)
        covered.update(indices)
        anchor = suffix[0].parent
        anchors.add(anchor)
        for corner, values in enumerate(product(*(ranges[i] for i in indices))):
            q = np.zeros(max(ranges) + 1)
            q[indices] = values
            pose = np.eye(4)
            for joint in suffix:
                pose = pose @ joint.origin
                rule = fk._rules.get(joint.name)
                if joint.kind == "prismatic":
                    pose = pose @ _translation_transform(joint.axis * rule.value(q, joint.name))
                elif joint.kind in {"revolute", "continuous"}:
                    pose = pose @ _axis_angle_transform(joint.axis, rule.value(q, joint.name))
            xyz = pose[:3, :3] @ np.asarray(point.xyz) + pose[:3, 3]
            result.append(CollisionPoint(name=f"{point.name}:opening_box_{corner}", frame=anchor,
                                         xyz=tuple(xyz), radius=point.radius))
    if covered != set(ranges):
        raise SafetyConfigurationError("Opening envelope must include collision geometry for both moving DH effectors")
    fk._required_frames = fk._required_frames | anchors
    fk._cache = None
    return GripperGeometry(result, horizontal_support_exact=geometry.horizontal_support_exact)


def _load_collision_points(
    data: Mapping[str, Any],
    config_directory: Path,
    *,
    urdf_path: Path | None = None,
) -> list[CollisionPoint]:
    points = [
        CollisionPoint(
            name=str(item["name"]),
            frame=str(item["frame"]),
            xyz=tuple(float(value) for value in item["xyz"]),
            radius=float(item.get("radius", 0.0)),
        )
        for item in data.get("collision_points", ())
    ]
    urdf_meshes = _urdf_collision_mesh_specs(urdf_path) if urdf_path is not None else {}
    for item in data.get("collision_meshes", ()):
        mesh_path = _relative_path(config_directory, item["path"])
        expected = item.get("sha256")
        if expected:
            digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            if digest.lower() != str(expected).lower():
                raise SafetyConfigurationError(
                    f"Collision mesh checksum mismatch for {mesh_path.name}: "
                    f"expected {expected}, got {digest}"
                )
        scale = np.asarray(item.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise SafetyConfigurationError(f"Invalid collision mesh scale: {mesh_path}")
        origin = _pose_from_config(item.get("origin", {}))
        frame = str(item["frame"])
        if item.get("validate_against_urdf") is True:
            expected = urdf_meshes.get((frame, mesh_path.resolve()))
            if expected is None:
                raise SafetyConfigurationError(
                    f"Collision mesh {mesh_path.name} for frame {frame!r} "
                    "does not match a URDF collision element"
                )
            expected_scale, expected_origin = expected
            if not np.allclose(scale, expected_scale, rtol=0.0, atol=1e-12):
                raise SafetyConfigurationError(
                    f"Collision mesh scale for frame {frame!r} does not match the URDF"
                )
            if not np.allclose(origin, expected_origin, rtol=0.0, atol=1e-9):
                raise SafetyConfigurationError(
                    f"Collision mesh origin for frame {frame!r} does not match the URDF"
                )
        vertices = _read_stl_vertices(mesh_path)
        vertex_indices = np.arange(len(vertices), dtype=np.int64)
        support_indices = item.get("support_vertex_indices")
        if support_indices is not None:
            if (
                not isinstance(support_indices, list)
                or not support_indices
                or any(isinstance(index, bool) or not isinstance(index, int) for index in support_indices)
            ):
                raise SafetyConfigurationError(
                    f"Collision mesh support_vertex_indices must be a non-empty integer list: "
                    f"{mesh_path}"
                )
            indices = np.asarray(support_indices, dtype=np.int64)
            if (
                np.any(indices < 0)
                or np.any(indices >= len(vertices))
                or len(np.unique(indices)) != len(indices)
            ):
                raise SafetyConfigurationError(
                    f"Collision mesh support_vertex_indices are out of range or duplicated: "
                    f"{mesh_path}"
                )
            vertices = vertices[indices]
            vertex_indices = indices
        vertices = vertices * scale
        vertices = vertices @ origin[:3, :3].T + origin[:3, 3]
        prefix = str(item.get("name", mesh_path.stem))
        points.extend(
            CollisionPoint(
                name=f"{prefix}:vertex_{index}",
                frame=frame,
                xyz=tuple(float(value) for value in vertex),
            )
            for index, vertex in zip(vertex_indices, vertices, strict=True)
        )
    return points


def _urdf_collision_mesh_specs(
    urdf_path: Path,
) -> dict[tuple[str, Path], tuple[np.ndarray, np.ndarray]]:
    """Index mesh scale/origin pairs from URDF collision elements.

    Collision STL vertices are expressed in their mesh coordinates, not
    necessarily in the containing link frame.  Keeping this index in the
    loader lets safety profiles opt into a fail-closed consistency check
    against the same URDF used for forward kinematics.
    """
    try:
        root = ElementTree.parse(urdf_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise SafetyConfigurationError(f"Cannot inspect URDF collision meshes: {error}") from error
    result: dict[tuple[str, Path], tuple[np.ndarray, np.ndarray]] = {}
    for link in root.findall("link"):
        frame = str(link.attrib.get("name", ""))
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None or not mesh.attrib.get("filename"):
                continue
            mesh_path = _relative_path(urdf_path.parent, mesh.attrib["filename"])
            scale = _parse_vector(mesh.attrib.get("scale", "1 1 1"), "URDF mesh scale")
            origin_element = collision.find("origin")
            origin = _pose_from_config(
                {} if origin_element is None else origin_element.attrib
            )
            result[(frame, mesh_path)] = (scale, origin)
    return result


def _read_stl_vertices(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SafetyConfigurationError(f"Collision mesh does not exist: {path}")
    payload = path.read_bytes()
    vertices: np.ndarray
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + triangle_count * 50 == len(payload):
            dtype = np.dtype(
                [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
            )
            triangles = np.frombuffer(payload, dtype=dtype, count=triangle_count, offset=84)
            vertices = np.asarray(triangles["vertices"], dtype=np.float64).reshape(-1, 3)
        else:
            vertices = _read_ascii_stl_vertices(payload, path)
    else:
        vertices = _read_ascii_stl_vertices(payload, path)
    if not len(vertices) or not np.isfinite(vertices).all():
        raise SafetyConfigurationError(f"STL has no finite vertices: {path}")
    return np.unique(vertices, axis=0)


def _read_ascii_stl_vertices(payload: bytes, path: Path) -> np.ndarray:
    values = []
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise SafetyConfigurationError(f"Unsupported or corrupt STL: {path}") from error
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            values.append(tuple(float(value) for value in fields[1:]))
    return np.asarray(values, dtype=np.float64).reshape(-1, 3)


def _minimum_z_from_world(
    groups: tuple[tuple[str, tuple[str, ...], np.ndarray, np.ndarray], ...]
) -> MinimumZ:
    minimum: MinimumZ | None = None
    for frame, names, world, radii in groups:
        z_values = world[:, 2] - radii
        index = int(np.argmin(z_values))
        z = float(z_values[index])
        candidate = MinimumZ(
            z=z,
            point_name=names[index],
            frame=frame,
            position=(float(world[index, 0]), float(world[index, 1]), z),
        )
        if minimum is None or candidate.z < minimum.z:
            minimum = candidate
    assert minimum is not None
    return minimum


def _relative_path(directory: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def _parse_vector(value: str, label: str) -> np.ndarray:
    try:
        vector = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    except ValueError as error:
        raise SafetyConfigurationError(f"{label} must contain three numbers") from error
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise SafetyConfigurationError(f"{label} must contain three finite numbers")
    return vector


def _pose_from_config(value: Mapping[str, Any]) -> np.ndarray:
    xyz_value = value.get("xyz", (0.0, 0.0, 0.0))
    rpy_value = value.get("rpy", (0.0, 0.0, 0.0))
    xyz = _parse_vector(xyz_value, "Pose xyz") if isinstance(xyz_value, str) else np.asarray(xyz_value, dtype=np.float64)
    rpy = _parse_vector(rpy_value, "Pose rpy") if isinstance(rpy_value, str) else np.asarray(rpy_value, dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
        raise SafetyConfigurationError("Pose xyz and rpy must each contain three finite values")
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def _axis_angle_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    # Python scalar arithmetic avoids dozens of NumPy scalar dispatches per
    # joint. Axis validation/normalization still happens when the URDF loads.
    x, y, z = map(float, axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus = 1.0 - cosine
    return np.asarray(
        (
            (cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine, 0.0),
            (y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine, 0.0),
            (z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _translation_transform(xyz: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = xyz
    return result


def _validate_axis(value: Any, label: str) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(axis)) if axis.shape == (3,) else math.nan
    if axis.shape != (3,) or not np.isfinite(axis).all() or not math.isfinite(norm) or norm <= 0:
        raise SafetyConfigurationError(f"{label} must be one finite non-zero xyz vector")
    return axis / norm


def _validate_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise SafetyConfigurationError(f"{label} must be one finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise SafetyConfigurationError(f"{label} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise SafetyConfigurationError(f"{label} rotation is not orthonormal")
    return transform.copy()


def _finite_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite vector, got {array.shape}")
    return array
