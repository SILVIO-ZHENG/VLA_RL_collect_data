from __future__ import annotations

import base64
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [REPO_ROOT / Path(item.decode()) for item in result.stdout.split(b"\0") if item]


def test_tracked_and_new_files_use_only_nexus_brand() -> None:
    legacy_name = bytes((97, 119, 118, 100))
    legacy_name_encoded = base64.b64encode(legacy_name)
    violations: list[str] = []

    for path in _repository_files():
        if not path.is_file():
            continue
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        if legacy_name in relative_path.lower().encode():
            violations.append(relative_path)
            continue
        content = path.read_bytes().lower()
        if legacy_name in content or legacy_name_encoded.lower() in content:
            violations.append(relative_path)

    assert not violations, f"发现未迁移的旧品牌标识: {violations}"
