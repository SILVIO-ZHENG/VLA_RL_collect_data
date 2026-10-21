"""Recreate the reviewed PI0.5 LeRobot source without using a mutable checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://github.com/huggingface/lerobot.git"
REVISION = "a9879e69edefb3c308f7025046d3d12b3cb2fe50"
PATCH = ROOT / "third_party/lerobot/pi05-deployment.patch"
PATCH_SHA256 = "8ddd844c1c2bbf1271dcd2b5ed831d0107116236085683253be34485ce567c95"
DESTINATION = ROOT / "third_party/lerobot/source"
PATCHED_FILES = (
    "src/lerobot/policies/factory.py",
    "src/lerobot/policies/pi05/configuration_pi05.py",
    "src/lerobot/policies/pi05/modeling_pi05.py",
    "src/lerobot/policies/pi05/processor_pi05.py",
)


def run(*command: str, cwd: Path | None = None) -> str:
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    return subprocess.check_output(command, cwd=cwd, env=environment, text=True).strip()


def verify(destination: Path = DESTINATION) -> None:
    if hashlib.sha256(PATCH.read_bytes()).hexdigest() != PATCH_SHA256:
        raise RuntimeError("LeRobot patch hash mismatch")
    if run("git", "rev-parse", "HEAD", cwd=destination) != REVISION:
        raise RuntimeError("LeRobot revision mismatch")
    actual = subprocess.check_output(("git", "diff", "--", *PATCHED_FILES), cwd=destination)
    if actual != PATCH.read_bytes():
        raise RuntimeError("LeRobot patch differs from the reviewed source")
    changed = set(run("git", "diff", "--name-only", cwd=destination).splitlines())
    if changed != set(PATCHED_FILES) or run("git", "diff", "--cached", "--name-only", cwd=destination):
        raise RuntimeError(f"Unexpected LeRobot modifications: {changed}")
    untracked = run("git", "ls-files", "--others", "--exclude-standard", cwd=destination)
    if untracked:
        raise RuntimeError(f"Unexpected LeRobot source files: {untracked}")


def prepare(source: str, destination: Path = DESTINATION) -> None:
    actual = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    if actual != PATCH_SHA256:
        raise RuntimeError(f"LeRobot patch hash mismatch: {actual}")
    if destination.exists():
        raise RuntimeError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        run("git", "clone", "--quiet", "--no-checkout", source, str(destination))
        run("git", "checkout", "--quiet", "--detach", REVISION, cwd=destination)
        if run("git", "rev-parse", "HEAD", cwd=destination) != REVISION:
            raise RuntimeError("LeRobot revision mismatch")
        run("git", "apply", "--check", str(PATCH), cwd=destination)
        run("git", "apply", str(PATCH), cwd=destination)
        verify(destination)
        print(f"Prepared LeRobot {REVISION} with patch {actual} at {destination}")
    except Exception:
        # Leave a failed checkout for inspection; never silently reuse it.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=UPSTREAM, help="Git URL or local mirror of the pinned upstream")
    parser.add_argument("--verify", action="store_true", help="Verify an existing prepared checkout")
    args = parser.parse_args()
    if args.verify:
        verify()
        print(f"Verified LeRobot {REVISION} with patch {PATCH_SHA256}")
    else:
        prepare(args.source)


if __name__ == "__main__":
    main()
