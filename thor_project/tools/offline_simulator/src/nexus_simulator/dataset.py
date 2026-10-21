"""LeRobot v3 Episode 只读加载和 G2 24 维关节映射。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class DatasetError(ValueError):
    """数据集结构、Episode 或特征不符合本仿真的严格预期。"""


def _parquet_module():
    """Load the optional dataset dependency only when an Episode is opened."""

    try:
        from pyarrow import parquet
    except ImportError as error:
        raise DatasetError(
            "读取 LeRobot Episode 需要 offline-simulator extra；"
            "请执行 uv sync --extra offline-simulator"
        ) from error
    return parquet


G2_DATASET_NAMES = (
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
    "left_arm_joint5", "left_arm_joint6", "left_arm_joint7", "left_effector_position",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
    "right_arm_joint5", "right_arm_joint6", "right_arm_joint7", "right_effector_position",
    "head_joint1", "head_joint2", "head_joint3", "waist_joint1", "waist_joint2",
    "waist_joint3", "waist_joint4", "waist_joint5",
)

G2_URDF_JOINTS = (
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7", "idx31_gripper_l_inner_joint1", "idx61_arm_r_joint1",
    "idx62_arm_r_joint2", "idx63_arm_r_joint3", "idx64_arm_r_joint4",
    "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7",
    "idx71_gripper_r_inner_joint1", "idx11_head_joint1", "idx12_head_joint2",
    "idx13_head_joint3", "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
)

LEGACY_EFFECTOR_ENCODING = "legacy_revolute"
NATIVE_DH_EFFECTOR_ENCODING = "native_dh_0_025"
NATIVE_DH_EFFECTOR_UPPER_M = 0.025
EFFECTOR_INDICES = (7, 15)


@dataclass(frozen=True)
class EpisodeData:
    index: int
    fps: float
    task: str
    states: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray
    video_path: Path
    video_from_s: float
    video_to_s: float
    effector_encoding: str = LEGACY_EFFECTOR_ENCODING

    @property
    def frame_count(self) -> int:
        return len(self.timestamps)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps

    def vector(self, frame_index: int, source: str) -> np.ndarray:
        if frame_index < 0 or frame_index >= self.frame_count:
            raise DatasetError(f"frame_index 必须在 [0, {self.frame_count - 1}]")
        if source == "observation":
            return self.states[frame_index].copy()
        if source == "action":
            return self.actions[frame_index].copy()
        raise DatasetError("source 必须是 observation 或 action")

    def joint_values(self, frame_index: int, source: str) -> dict[str, float]:
        vector = self.vector(frame_index, source)
        return {name: float(vector[index]) for index, name in enumerate(G2_URDF_JOINTS)}

    def to_safety_vector(self, vector: np.ndarray, source: str) -> np.ndarray:
        """转换为当前 DH 安全参考层使用的夹钳命令极性。"""

        converted = np.asarray(vector, dtype=np.float64).copy()
        if self.effector_encoding == LEGACY_EFFECTOR_ENCODING:
            return converted
        if self.effector_encoding != NATIVE_DH_EFFECTOR_ENCODING:
            raise DatasetError(f"未知夹钳编码: {self.effector_encoding}")
        self._validate_native_effectors(converted)
        if source == "observation":
            converted[list(EFFECTOR_INDICES)] = (
                NATIVE_DH_EFFECTOR_UPPER_M - converted[list(EFFECTOR_INDICES)]
            )
        elif source != "action":
            raise DatasetError("source 必须是 observation 或 action")
        return converted

    def from_safety_vector(self, vector: np.ndarray, source: str) -> np.ndarray:
        """把安全层采用的姿态还原为所选数据源的夹钳编码。"""

        converted = np.asarray(vector, dtype=np.float64).copy()
        if self.effector_encoding == LEGACY_EFFECTOR_ENCODING:
            return converted
        if self.effector_encoding != NATIVE_DH_EFFECTOR_ENCODING:
            raise DatasetError(f"未知夹钳编码: {self.effector_encoding}")
        self._validate_native_effectors(converted)
        if source == "observation":
            converted[list(EFFECTOR_INDICES)] = (
                NATIVE_DH_EFFECTOR_UPPER_M - converted[list(EFFECTOR_INDICES)]
            )
        elif source != "action":
            raise DatasetError("source 必须是 observation 或 action")
        return converted

    @staticmethod
    def _validate_native_effectors(vector: np.ndarray) -> None:
        tolerance = 1e-6
        for index in EFFECTOR_INDICES:
            value = float(vector[index])
            if not -tolerance <= value <= NATIVE_DH_EFFECTOR_UPPER_M + tolerance:
                raise DatasetError(
                    f"原生 DH 夹钳值 index={index} value={value} 越界 "
                    f"[0.0, {NATIVE_DH_EFFECTOR_UPPER_M}]"
                )

    def metadata(self) -> dict[str, Any]:
        return {
            "available": True,
            "episode_index": self.index,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "task": self.task,
            "source_options": ["observation", "action"],
            "effector_encoding": self.effector_encoding,
            "video": {
                "url": "/api/replay/video/head",
                "from_s": self.video_from_s,
                "to_s": self.video_to_s,
                "width": 640,
                "height": 400,
            },
            "display_fps": 30,
        }


class LeRobotDataset:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise DatasetError(f"LeRobot info.json 不存在: {info_path}")
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self._episode_effector_encodings = self._load_effector_encodings()
        self.fps = float(self.info.get("fps", 0))
        features = self.info.get("features", {})
        if self.info.get("codebase_version") != "v3.0" or self.info.get("robot_type") != "G2A":
            raise DatasetError("只支持 LeRobot v3.0 的 G2A 数据集")
        if self.fps <= 0:
            raise DatasetError("数据集 fps 必须为正数")
        for key in ("observation.state", "action"):
            feature = features.get(key, {})
            if tuple(feature.get("shape", ())) != (24,):
                raise DatasetError(f"{key} 必须是 24 维")
            if tuple(feature.get("names") or ()) != G2_DATASET_NAMES:
                raise DatasetError(f"{key} 的关节名称或顺序与 G2 24D 协议不一致")

    def _load_effector_encodings(self) -> dict[int, str]:
        manifest_path = self.root / "conversion_manifest.json"
        if not manifest_path.is_file():
            return {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DatasetError(f"conversion_manifest.json 无效: {exc}") from exc
        result: dict[int, str] = {}
        for episode in manifest.get("episodes", []):
            if not isinstance(episode, dict) or "episode_index" not in episode:
                continue
            encoding = episode.get("effector_encoding")
            if encoding == "native":
                result[int(episode["episode_index"])] = NATIVE_DH_EFFECTOR_ENCODING
            elif encoding not in {None, "canonical"}:
                raise DatasetError(f"未知 conversion manifest 夹钳编码: {encoding}")
        return result

    def load_episode(self, episode_index: int) -> EpisodeData:
        if episode_index < 0 or episode_index >= int(self.info.get("total_episodes", 0)):
            raise DatasetError(f"未知 Episode: {episode_index}")
        parquet = _parquet_module()
        episode_table = parquet.read_table(self.root / "meta/episodes/chunk-000/file-000.parquet")
        rows = [row for row in episode_table.to_pylist() if row["episode_index"] == episode_index]
        if len(rows) != 1:
            raise DatasetError(f"Episode {episode_index} 元数据不唯一")
        meta = rows[0]
        chunk = int(meta["data/chunk_index"])
        file_index = int(meta["data/file_index"])
        data_path = self.root / f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
        table = parquet.read_table(
            data_path,
            columns=["observation.state", "action", "timestamp", "episode_index", "frame_index"],
            filters=[("episode_index", "=", episode_index)],
        )
        records = sorted(table.to_pylist(), key=lambda row: row["frame_index"])
        expected = int(meta["length"])
        if len(records) != expected:
            raise DatasetError(f"Episode 帧数不一致: metadata={expected}, data={len(records)}")
        states = np.asarray([row["observation.state"] for row in records], dtype=np.float64)
        actions = np.asarray([row["action"] for row in records], dtype=np.float64)
        timestamps = np.asarray([row["timestamp"] for row in records], dtype=np.float64)
        if states.shape != (expected, 24) or actions.shape != (expected, 24):
            raise DatasetError("Episode 轨迹不是 N×24")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise DatasetError("Episode 轨迹包含非有限值")
        effector_encoding = self._episode_effector_encodings.get(
            episode_index, LEGACY_EFFECTOR_ENCODING
        )
        if effector_encoding == NATIVE_DH_EFFECTOR_ENCODING:
            tolerance = 1e-6
            effectors = np.concatenate(
                [states[:, list(EFFECTOR_INDICES)], actions[:, list(EFFECTOR_INDICES)]]
            )
            if effectors.min() < -tolerance or effectors.max() > NATIVE_DH_EFFECTOR_UPPER_M + tolerance:
                raise DatasetError("原生 DH 夹钳轨迹越出 [0, 0.025] m")
        video_key = "observation.images.cam_head"
        video_chunk = int(meta[f"videos/{video_key}/chunk_index"])
        video_file = int(meta[f"videos/{video_key}/file_index"])
        video_path = self.root / f"videos/{video_key}/chunk-{video_chunk:03d}/file-{video_file:03d}.mp4"
        if not video_path.is_file():
            raise DatasetError(f"头部相机视频不存在: {video_path}")
        return EpisodeData(
            index=episode_index,
            fps=self.fps,
            task=str((meta.get("tasks") or [""])[0]),
            states=states,
            actions=actions,
            timestamps=timestamps,
            video_path=video_path,
            video_from_s=float(meta[f"videos/{video_key}/from_timestamp"]),
            video_to_s=float(meta[f"videos/{video_key}/to_timestamp"]),
            effector_encoding=effector_encoding,
        )
