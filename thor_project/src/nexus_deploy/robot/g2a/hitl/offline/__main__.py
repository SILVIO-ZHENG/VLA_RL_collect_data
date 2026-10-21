"""Pull one sealed G2 tablet/sidecar episode to Thor and verify its archive."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from uuid import UUID

from ..common.sidecar import MANIFEST_FINAL, validate_sealed_sidecar
from .ingest import ImmutableRawStore


def _remote_root(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Remote root must be an absolute path without '..'")
    return path


def pull_and_archive(
    *,
    session_id: str,
    ssh_target: str,
    remote_sidecar_root: str,
    remote_vendor_root: str,
    staging_root: Path,
    store_root: Path,
) -> dict[str, object]:
    session_id = str(UUID(session_id))
    if not ssh_target or ssh_target.startswith("-") or any(c.isspace() for c in ssh_target):
        raise ValueError("Invalid SSH target")
    sidecar_root = _remote_root(remote_sidecar_root)
    vendor_root = _remote_root(remote_vendor_root)
    remote_sidecar = sidecar_root / f"{session_id}.sealed"
    manifest_path = remote_sidecar / MANIFEST_FINAL
    manifest_bytes = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ssh_target, f"cat -- {shlex.quote(str(manifest_path))}"],
        check=True,
        capture_output=True,
    ).stdout
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "sealed" or manifest.get("flowpro_session_id") != session_id:
        raise ValueError("Remote sealed manifest does not match the requested session")
    episode = manifest.get("vendor_episode")
    if not isinstance(episode, dict):
        raise ValueError("Remote sealed manifest has no vendor episode")
    record_name = episode.get("record_name")
    if (not isinstance(record_name, str) or not record_name
            or PurePosixPath(record_name).name != record_name or record_name in {".", ".."}):
        raise ValueError("Remote vendor record_name is not one directory name")
    remote_episode = vendor_root / record_name
    if episode.get("episode_path") != str(remote_episode):
        raise ValueError("Remote episode path differs from the configured vendor root")

    staging_root = staging_root.expanduser().resolve()
    sidecar_destination = staging_root / "sidecars" / f"{session_id}.sealed"
    vendor_destination = staging_root / "vendor" / record_name
    sidecar_destination.mkdir(parents=True, exist_ok=True)
    vendor_destination.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        (remote_sidecar, sidecar_destination),
        (remote_episode, vendor_destination),
    ):
        subprocess.run(
            ["rsync", "-a", "--secluded-args", "-e", "ssh -o BatchMode=yes",
             f"{ssh_target}:{source}/", f"{destination}/"],
            check=True,
        )
    if (sidecar_destination / MANIFEST_FINAL).read_bytes() != manifest_bytes:
        raise ValueError("Remote sealed manifest changed while pulling the episode")
    validation = validate_sealed_sidecar(
        sidecar_destination,
        vendor_episode_path=vendor_destination,
        vendor_record_root=staging_root / "vendor",
    )
    store = ImmutableRawStore(store_root)
    receipt = store.ingest(
        sealed_sidecar=sidecar_destination,
        vendor_episode=vendor_destination,
        vendor_record_root=staging_root / "vendor",
    )
    store.verify_receipt(receipt.bundle_id)
    return {
        "session_id": session_id,
        "record_name": validation.record_name,
        "event_count": validation.event_count,
        "candidate_count": validation.candidate_count,
        "bundle_id": receipt.bundle_id,
        "object_count": len(receipt.objects),
        "store_root": str(store.root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", help="UUID of one G2 .sealed sidecar")
    parser.add_argument("--ssh-target", default="g2")
    parser.add_argument("--remote-sidecar-root", default="/data/logs/flowpro_sidecar")
    parser.add_argument("--remote-vendor-root", default="/data/record")
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    args = parser.parse_args()
    report = pull_and_archive(
        session_id=args.session_id,
        ssh_target=args.ssh_target,
        remote_sidecar_root=args.remote_sidecar_root,
        remote_vendor_root=args.remote_vendor_root,
        staging_root=args.staging_root,
        store_root=args.store_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
