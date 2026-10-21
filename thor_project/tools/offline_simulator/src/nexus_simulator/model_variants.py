"""受控的 G2 可视模型版本及数据集关节映射。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nexus_simulator.assets import sha256_file
from nexus_simulator.dataset import (
    LEGACY_EFFECTOR_ENCODING,
    NATIVE_DH_EFFECTOR_ENCODING,
    NATIVE_DH_EFFECTOR_UPPER_M,
)


class ModelVariantError(ValueError):
    """可视模型版本、资产或关节语义不符合已声明约束。"""


LEGACY_MODEL_ID = "omnipicker-legacy"
G2_DAHUAN_MODEL_ID = "G2_dahuan"
LEGACY_GRIPPER_LOWER_RAD = -0.7853981633974483
PRISMATIC_GRIPPER_UPPER_M = 0.025
DAHUAN_GRIPPER_MAPPING = "legacy_revolute_to_reversed_prismatic_0_025m"


@dataclass(frozen=True)
class ModelVariantSpec:
    id: str
    label: str
    relative_urdf: str
    relative_asset_root: str
    expected_urdf_sha256: str | None
    relative_manifest: str | None
    asset_file_count: int
    gripper_mapping: str
    safety_geometry_aligned: bool
    explicit_camera_frame: bool

    @staticmethod
    def _repository_root(project_root: Path) -> Path:
        root = project_root.expanduser().resolve()
        if root.parent.name == "tools" and (root.parents[1] / "src" / "nexus_deploy").is_dir():
            return root.parents[1]
        return root

    def _resolve_repository_path(self, project_root: Path, relative: str) -> Path:
        root = project_root.expanduser().resolve()
        repository_root = self._repository_root(root)
        candidate = (root / relative).resolve()
        if candidate != repository_root and repository_root not in candidate.parents:
            raise ModelVariantError(f"模型 {self.id} 的资产路径越出 Nexus 仓库: {candidate}")
        return candidate

    def urdf_path(self, project_root: Path) -> Path:
        candidate = self._resolve_repository_path(project_root, self.relative_urdf)
        if not candidate.is_file():
            raise ModelVariantError(f"模型 {self.id} 的 URDF 不存在: {candidate}")
        expected = self.expected_urdf_sha256
        if self.relative_manifest is not None:
            manifest_path = self._resolve_repository_path(project_root, self.relative_manifest)
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ModelVariantError(f"模型 {self.id} 的权威清单不可读: {exc}") from exc
            if (
                manifest.get("id") != self.id
                or manifest.get("status") != "frozen"
                or manifest.get("urdf") != candidate.name
                or manifest.get("asset_file_count") != self.asset_file_count
            ):
                raise ModelVariantError(f"模型 {self.id} 的权威清单身份或版本无效")
            expected = manifest.get("urdf_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ModelVariantError(f"模型 {self.id} 缺少有效的 URDF SHA-256")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ModelVariantError(
                f"模型 {self.id} 的 URDF 哈希不一致: "
                f"expected={expected} actual={actual}"
            )
        return candidate

    def asset_root(self, project_root: Path) -> Path:
        candidate = self._resolve_repository_path(project_root, self.relative_asset_root)
        if not candidate.is_dir():
            raise ModelVariantError(f"模型 {self.id} 的资产根目录不存在: {candidate}")
        return candidate

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "replay_gripper_mapping": self.gripper_mapping,
            "safety_geometry_aligned": self.safety_geometry_aligned,
            "explicit_camera_frame": self.explicit_camera_frame,
        }

    def map_replay_joint_values(
        self,
        values: dict[str, float],
        *,
        effector_encoding: str = LEGACY_EFFECTOR_ENCODING,
        source: str = "observation",
    ) -> dict[str, float]:
        mapped = dict(values)
        if effector_encoding == NATIVE_DH_EFFECTOR_ENCODING:
            return self._map_native_dh_grippers(mapped, source)
        if effector_encoding != LEGACY_EFFECTOR_ENCODING:
            raise ModelVariantError(f"未知数据集夹钳编码: {effector_encoding}")
        if self.gripper_mapping == "direct":
            return mapped
        if self.gripper_mapping != DAHUAN_GRIPPER_MAPPING:
            raise ModelVariantError(f"模型 {self.id} 使用未知夹爪映射: {self.gripper_mapping}")

        for inner, outer in (
            ("idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"),
            ("idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"),
        ):
            legacy_value = float(mapped[inner])
            tolerance = 1e-6
            if not LEGACY_GRIPPER_LOWER_RAD - tolerance <= legacy_value <= tolerance:
                raise ModelVariantError(
                    f"数据集夹爪 {inner}={legacy_value} 越出旧协议范围 "
                    f"[{LEGACY_GRIPPER_LOWER_RAD}, 0.0]"
                )
            fraction = (legacy_value - LEGACY_GRIPPER_LOWER_RAD) / -LEGACY_GRIPPER_LOWER_RAD
            # 新 URDF 的零位在画面上是张开，正向位移才是闭合；旧协议反之以
            # -0.785398 表示闭合、0 表示张开，因此这里必须反向映射。
            position_m = (1.0 - min(1.0, max(0.0, fraction))) * PRISMATIC_GRIPPER_UPPER_M
            mapped[inner] = position_m
            # Nexus 权威 DH URDF 由 outer mimic inner；只提交可驱动的 inner。
            mapped.pop(outer, None)
        return mapped

    def _map_native_dh_grippers(
        self, mapped: dict[str, float], source: str
    ) -> dict[str, float]:
        if source not in {"observation", "action"}:
            raise ModelVariantError("source 必须是 observation 或 action")
        tolerance = 1e-6
        for inner, outer in (
            ("idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"),
            ("idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"),
        ):
            encoded = float(mapped[inner])
            if not -tolerance <= encoded <= NATIVE_DH_EFFECTOR_UPPER_M + tolerance:
                raise ModelVariantError(
                    f"数据集夹钳 {inner}={encoded} 越出原生范围 "
                    f"[0.0, {NATIVE_DH_EFFECTOR_UPPER_M}]"
                )
            physical_opening_m = (
                encoded if source == "observation" else NATIVE_DH_EFFECTOR_UPPER_M - encoded
            )
            physical_opening_m = min(
                NATIVE_DH_EFFECTOR_UPPER_M, max(0.0, physical_opening_m)
            )
            if self.gripper_mapping == "direct":
                fraction = physical_opening_m / NATIVE_DH_EFFECTOR_UPPER_M
                mapped[inner] = LEGACY_GRIPPER_LOWER_RAD * (1.0 - fraction)
            elif self.gripper_mapping == DAHUAN_GRIPPER_MAPPING:
                urdf_position_m = NATIVE_DH_EFFECTOR_UPPER_M - physical_opening_m
                mapped[inner] = urdf_position_m
                mapped.pop(outer, None)
            else:
                raise ModelVariantError(
                    f"模型 {self.id} 使用未知夹爪映射: {self.gripper_mapping}"
                )
        return mapped


_MODEL_VARIANTS = {
    LEGACY_MODEL_ID: ModelVariantSpec(
        id=LEGACY_MODEL_ID,
        label="G2 OmniPicker · 已验证旧夹爪",
        relative_urdf="assets/g2/G2_omnipicker/G2_omnipicker_visual.urdf",
        relative_asset_root="assets/g2/G2_omnipicker",
        expected_urdf_sha256=(
            "e66ad093999c4d9675902def559e5a3a7a42c4ed0cfe5ab2f3efd3b087582814"
        ),
        relative_manifest=None,
        asset_file_count=60,
        gripper_mapping="direct",
        safety_geometry_aligned=True,
        explicit_camera_frame=True,
    ),
    G2_DAHUAN_MODEL_ID: ModelVariantSpec(
        id=G2_DAHUAN_MODEL_ID,
        label="G2 DH 夹爪 · Nexus 权威版本",
        relative_urdf=(
            "../../src/nexus_deploy/viewer/assets/g2/G2_dahuan/G2_dahuan.urdf"
        ),
        relative_asset_root="../../src/nexus_deploy/viewer/assets",
        expected_urdf_sha256=None,
        relative_manifest=(
            "../../src/nexus_deploy/viewer/assets/g2/G2_dahuan/manifest.json"
        ),
        asset_file_count=109,
        gripper_mapping=DAHUAN_GRIPPER_MAPPING,
        safety_geometry_aligned=False,
        explicit_camera_frame=False,
    ),
}


def model_variant_ids() -> tuple[str, ...]:
    return tuple(_MODEL_VARIANTS)


def model_variant_spec(model_id: str) -> ModelVariantSpec:
    try:
        return _MODEL_VARIANTS[model_id]
    except KeyError as exc:
        raise ModelVariantError(f"未知 G2 模型版本: {model_id}") from exc


def resolve_model_variant(project_root: Path, model_id: str) -> ModelVariantSpec:
    spec = model_variant_spec(model_id)
    spec.urdf_path(project_root)
    return spec
