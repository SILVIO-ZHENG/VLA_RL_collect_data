"""Deterministic identity and startup verification for the standalone G2 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RUNTIME_PROVENANCE_SCHEMA_VERSION = "flowpro.g2_runtime_provenance/v1"
_ROBOT_CLIENT_SUFFIXES = frozenset({".py", ".sh", ".json", ".txt", ".urdf", ".whl"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    robot_client = root / "robot_client"
    flowpro = root / "scripts" / "flowpro"
    if not robot_client.is_dir() or not flowpro.is_dir():
        raise FileNotFoundError("G2 runtime root must contain robot_client/ and scripts/flowpro/")
    candidates.extend(
        path
        for path in robot_client.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix in _ROBOT_CLIENT_SUFFIXES
    )
    candidates.extend(
        path
        for path in flowpro.rglob("*.py")
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    )
    module_manifest = root / "g2_flowpro_modules.txt"
    if not module_manifest.is_file() or module_manifest.is_symlink():
        raise FileNotFoundError(module_manifest)
    candidates.append(module_manifest)
    return sorted(set(candidates), key=lambda path: path.relative_to(root).as_posix())


def build_runtime_provenance(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    entries = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _runtime_files(resolved)
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
    }


def write_runtime_provenance(root: Path, output: Path) -> dict[str, Any]:
    payload = build_runtime_provenance(root)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_and_verify_runtime_provenance(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"Runtime provenance must be a regular non-symlink file: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    expected_fields = {"schema_version", "source_sha256", "files"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("Runtime provenance fields differ from v1")
    if payload["schema_version"] != RUNTIME_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported G2 runtime provenance schema")
    actual = build_runtime_provenance(resolved.parent)
    if payload != actual:
        raise ValueError("G2 runtime files differ from frozen runtime provenance")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_runtime_provenance(args.root, args.output)
    print(payload["source_sha256"])


if __name__ == "__main__":
    main()
