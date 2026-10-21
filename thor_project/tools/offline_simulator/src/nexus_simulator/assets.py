"""从权威 LeRobot 目录建立带哈希的只读 G2 OmniPicker 模型资产。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AssetError(RuntimeError):
    """模型资产来源、目录或哈希不满足约束。"""


@dataclass(frozen=True)
class AssetPaths:
    project_root: Path
    source_manifest: Path
    model_root: Path
    model_manifest: Path


def default_asset_paths(project_root: Path) -> AssetPaths:
    root = project_root.resolve()
    model_root = root / "assets" / "g2" / "G2_omnipicker"
    return AssetPaths(
        project_root=root,
        source_manifest=model_root / "source_manifest.json",
        model_root=model_root,
        model_manifest=model_root / "manifest.json",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"无法读取资产清单 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssetError(f"资产清单根节点必须是对象: {path}")
    return payload


def _source_root(paths: AssetPaths, override: Path | None) -> Path:
    manifest = _load_json(paths.source_manifest)
    raw = override if override is not None else manifest.get("source_root")
    if not isinstance(raw, (str, Path)):
        raise AssetError("source_manifest.json 缺少 source_root")
    source = Path(raw).expanduser()
    if not source.is_absolute():
        source = paths.project_root / source
    source = source.resolve()
    if not source.is_dir():
        raise AssetError(f"G2 资产源目录不存在: {source}")
    return source


def _verify_declared_sources(paths: AssetPaths, source: Path) -> None:
    manifest = _load_json(paths.source_manifest)
    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise AssetError("source_manifest.json files 必须是非空数组")
    for entry in declared:
        if not isinstance(entry, dict):
            raise AssetError("source_manifest.json files 元素必须是对象")
        name = entry.get("name")
        expected = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(expected, str):
            raise AssetError("资产声明必须包含 name 和 sha256")
        candidate = (source / name).resolve()
        if source not in candidate.parents or not candidate.is_file():
            raise AssetError(f"声明资产不存在或越出源目录: {name}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise AssetError(f"权威资产哈希变化: {name}: expected={expected} actual={actual}")


def _model_hashes(model_root: Path) -> dict[str, str]:
    metadata_files = {"manifest.json", "source_manifest.json"}
    return {
        path.relative_to(model_root).as_posix(): sha256_file(path)
        for path in sorted(model_root.rglob("*"))
        if path.is_file() and path.relative_to(model_root).as_posix() not in metadata_files
    }


def sync_assets(project_root: Path, source_override: Path | None = None) -> dict[str, Any]:
    """复制完整资产树；永不修改源目录，并通过临时目录原子替换模型资产。"""

    paths = default_asset_paths(project_root)
    source = _source_root(paths, source_override)
    _verify_declared_sources(paths, source)

    temporary = paths.model_root.with_name("G2_omnipicker.pending")
    if temporary.exists():
        shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary)
        shutil.copy2(paths.source_manifest, temporary / "source_manifest.json")
        hashes = _model_hashes(temporary)
        if paths.model_root.exists():
            shutil.rmtree(paths.model_root)
        temporary.replace(paths.model_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    payload: dict[str, Any] = {
        "schema_version": 1,
        # Keep generated metadata portable after the repository is moved.
        "source_root": Path(os.path.relpath(source, paths.project_root)).as_posix(),
        "file_count": len(hashes),
        "files": hashes,
    }
    paths.model_manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verify_assets(project_root)
    return payload


def verify_assets(project_root: Path) -> dict[str, Any]:
    paths = default_asset_paths(project_root)
    if not paths.model_root.is_dir():
        raise AssetError(f"G2 OmniPicker 模型资产不存在: {paths.model_root}")
    manifest = _load_json(paths.model_manifest)
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise AssetError("G2_omnipicker/manifest.json files 必须是非空对象")
    actual = _model_hashes(paths.model_root)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )
        raise AssetError(
            f"G2 OmniPicker 模型资产不一致: "
            f"missing={missing} added={added} changed={changed}"
        )
    return manifest
