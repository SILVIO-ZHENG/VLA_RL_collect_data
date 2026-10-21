"""5090-side immutable content store for sealed FlowPRO source bundles."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..common.sidecar import MANIFEST_FINAL, validate_sealed_sidecar

RECEIPT_VERSION = "flowpro.raw_store_receipt/v1"
ACK_VERSION = "flowpro.raw_store_ack/v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_file(path: Path, *, block_size: int = 4 * 1024 * 1024) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Raw-store source must be a regular non-symlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Source changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class RawStoreReceipt:
    schema_version: str
    bundle_id: str
    flowpro_session_id: str
    record_name: str
    ingested_at_utc: str
    source_manifest_sha256: str
    objects: list[dict[str, Any]]


class ImmutableRawStore:
    """Content-addressed files plus one immutable receipt and ACK per session."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.objects = self.root / "objects"
        self.receipts = self.root / "receipts"
        self.acks = self.root / "acks"
        for directory in (self.objects, self.receipts, self.acks):
            directory.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        *,
        sealed_sidecar: Path,
        vendor_episode: Path,
        vendor_record_root: Path,
    ) -> RawStoreReceipt:
        report = validate_sealed_sidecar(
            sealed_sidecar,
            vendor_episode_path=vendor_episode,
            vendor_record_root=vendor_record_root,
        )
        manifest_path = sealed_sidecar / MANIFEST_FINAL
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest_size, manifest_sha = _hash_file(manifest_path)

        sources: list[tuple[str, str, Path, int, str]] = [
            ("sidecar", MANIFEST_FINAL, manifest_path, manifest_size, manifest_sha)
        ]
        for item in manifest["sidecar_files"]:
            sources.append(
                (
                    "sidecar",
                    item["relative_path"],
                    sealed_sidecar / item["relative_path"],
                    int(item["size_bytes"]),
                    str(item["sha256"]),
                )
            )
        for item in manifest["vendor_episode"]["files"]:
            sources.append(
                (
                    "vendor",
                    item["relative_path"],
                    vendor_episode / item["relative_path"],
                    int(item["size_bytes"]),
                    str(item["sha256"]),
                )
            )

        object_records: list[dict[str, Any]] = []
        logical_keys: set[tuple[str, str]] = set()
        for namespace, relative_path, source, expected_size, expected_sha in sources:
            logical_key = (namespace, relative_path)
            if logical_key in logical_keys:
                raise ValueError(f"Duplicate raw-store logical path: {logical_key}")
            logical_keys.add(logical_key)
            object_path = self._put_object(source, expected_size=expected_size, expected_sha=expected_sha)
            object_records.append(
                {
                    "namespace": namespace,
                    "relative_path": relative_path,
                    "size_bytes": expected_size,
                    "sha256": expected_sha,
                    "object_path": object_path.relative_to(self.root).as_posix(),
                }
            )

        bundle_id = report.flowpro_session_id
        receipt = RawStoreReceipt(
            schema_version=RECEIPT_VERSION,
            bundle_id=bundle_id,
            flowpro_session_id=report.flowpro_session_id,
            record_name=report.record_name,
            ingested_at_utc=_utc_now(),
            source_manifest_sha256=manifest_sha,
            objects=sorted(object_records, key=lambda item: (item["namespace"], item["relative_path"])),
        )
        receipt_path = self.receipts / f"{bundle_id}.json"
        if receipt_path.exists():
            existing = self.load_receipt(bundle_id)
            if self._receipt_identity(existing) != self._receipt_identity(receipt):
                raise ValueError(f"Bundle ID collision with different content: {bundle_id}")
            receipt = existing
        else:
            _write_json_atomic(receipt_path, asdict(receipt))
            receipt_path.chmod(0o444)
            _fsync_directory(receipt_path.parent)

        ack = {
            "schema_version": ACK_VERSION,
            "bundle_id": bundle_id,
            "receipt_sha256": _hash_file(receipt_path)[1],
            "source_manifest_sha256": manifest_sha,
            "object_count": len(receipt.objects),
        }
        ack_path = self.acks / f"{bundle_id}.json"
        if ack_path.exists():
            with ack_path.open(encoding="utf-8") as stream:
                if json.load(stream) != ack:
                    raise ValueError(f"ACK collision for bundle {bundle_id}")
        else:
            _write_json_atomic(ack_path, ack)
            ack_path.chmod(0o444)
            _fsync_directory(ack_path.parent)
        return receipt

    def load_receipt(self, bundle_id: str) -> RawStoreReceipt:
        with (self.receipts / f"{bundle_id}.json").open(encoding="utf-8") as stream:
            value = json.load(stream)
        return RawStoreReceipt(**value)

    def verify_receipt(self, bundle_id: str) -> RawStoreReceipt:
        receipt_path = self.receipts / f"{bundle_id}.json"
        receipt = self.load_receipt(bundle_id)
        if receipt.schema_version != RECEIPT_VERSION or receipt.bundle_id != bundle_id:
            raise ValueError(f"Raw-store receipt identity/version mismatch: {bundle_id}")
        ack_path = self.acks / f"{bundle_id}.json"
        with ack_path.open(encoding="utf-8") as stream:
            ack = json.load(stream)
        expected_ack = {
            "schema_version": ACK_VERSION,
            "bundle_id": bundle_id,
            "receipt_sha256": _hash_file(receipt_path)[1],
            "source_manifest_sha256": receipt.source_manifest_sha256,
            "object_count": len(receipt.objects),
        }
        if ack != expected_ack:
            raise ValueError(f"Raw-store receipt does not match its immutable ACK: {bundle_id}")
        logical_keys: set[tuple[str, str]] = set()
        for item in receipt.objects:
            key = (str(item["namespace"]), str(item["relative_path"]))
            if key in logical_keys:
                raise ValueError(f"Raw-store receipt contains duplicate logical path: {key}")
            logical_keys.add(key)
            self.resolve(receipt, namespace=key[0], relative_path=key[1])
        manifest = self.resolve(receipt, namespace="sidecar", relative_path=MANIFEST_FINAL)
        if _hash_file(manifest)[1] != receipt.source_manifest_sha256:
            raise ValueError("Raw-store receipt source manifest hash mismatch")
        return receipt

    def resolve(self, receipt: RawStoreReceipt, *, namespace: str, relative_path: str) -> Path:
        matches = [
            item
            for item in receipt.objects
            if item["namespace"] == namespace and item["relative_path"] == relative_path
        ]
        if len(matches) != 1:
            raise KeyError(f"Receipt has no unique {namespace}:{relative_path}")
        item = matches[0]
        path = (self.root / item["object_path"]).resolve()
        expected_parent = (self.objects / item["sha256"][:2]).resolve()
        if path.parent != expected_parent or path.name != item["sha256"]:
            raise ValueError("Receipt object path is not canonical")
        size, digest = _hash_file(path)
        if size != item["size_bytes"] or digest != item["sha256"]:
            raise ValueError(f"Raw-store object failed integrity validation: {path}")
        return path

    def _put_object(self, source: Path, *, expected_size: int, expected_sha: str) -> Path:
        size, digest = _hash_file(source)
        if size != expected_size or digest != expected_sha:
            raise ValueError(f"Source no longer matches sealed hash: {source}")
        destination = self.objects / expected_sha[:2] / expected_sha
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stored_size, stored_digest = _hash_file(destination)
            if stored_size != expected_size or stored_digest != expected_sha:
                raise ValueError(f"Content-addressed object collision/corruption: {destination}")
            return destination
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        try:
            digest_builder = hashlib.sha256()
            total = 0
            with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                for block in iter(lambda: input_stream.read(4 * 1024 * 1024), b""):
                    output_stream.write(block)
                    digest_builder.update(block)
                    total += len(block)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            if total != expected_size or digest_builder.hexdigest() != expected_sha:
                raise RuntimeError(f"Source changed during raw-store copy: {source}")
            temporary.chmod(0o444)
            with suppress(FileExistsError):
                os.link(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        stored_size, stored_digest = _hash_file(destination)
        if stored_size != expected_size or stored_digest != expected_sha:
            raise ValueError(f"Stored object failed post-commit validation: {destination}")
        return destination

    @staticmethod
    def _receipt_identity(receipt: RawStoreReceipt) -> tuple[Any, ...]:
        return (
            receipt.schema_version,
            receipt.bundle_id,
            receipt.flowpro_session_id,
            receipt.record_name,
            receipt.source_manifest_sha256,
            receipt.objects,
        )


@dataclass(frozen=True)
class PullReport:
    ingested_bundle_ids: list[str]
    failed: list[dict[str, str]]


class FilesystemBundlePuller:
    """Ingest every visible sealed sidecar; transport/mount setup is external."""

    def __init__(
        self,
        *,
        sidecar_root: Path,
        vendor_record_root: Path,
        raw_store: ImmutableRawStore,
    ) -> None:
        self.sidecar_root = sidecar_root.expanduser().resolve()
        self.vendor_record_root = vendor_record_root.expanduser().resolve()
        self.raw_store = raw_store

    def pull_once(self) -> PullReport:
        ingested: list[str] = []
        failed: list[dict[str, str]] = []
        for sealed in sorted(self.sidecar_root.glob("*.sealed")):
            try:
                manifest = json.loads((sealed / MANIFEST_FINAL).read_text(encoding="utf-8"))
                # The sealed manifest keeps the original G2 absolute path as
                # provenance.  On Thor, the copied episode lives under the
                # local staging root and is selected by its validated name.
                record_name = str(manifest["vendor_episode"]["record_name"])
                if not record_name or Path(record_name).name != record_name or record_name in {".", ".."}:
                    raise ValueError("Sealed vendor record_name is not one directory name")
                vendor_episode = self.vendor_record_root / record_name
                receipt = self.raw_store.ingest(
                    sealed_sidecar=sealed,
                    vendor_episode=vendor_episode,
                    vendor_record_root=self.vendor_record_root,
                )
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                failed.append({"sealed_sidecar": str(sealed), "reason": str(error)})
            else:
                ingested.append(receipt.bundle_id)
        return PullReport(ingested_bundle_ids=ingested, failed=failed)
