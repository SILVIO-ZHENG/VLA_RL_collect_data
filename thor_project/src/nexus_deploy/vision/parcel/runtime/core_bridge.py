"""加载已验证的状态规则核心，避免部署封装复制出另一套算法。

Load the validated state-rule core instead of maintaining a second implementation.

源码包会同时携带经校验的 `rules_core`。若目录布局不同，可使用
`PARCEL_STATE_CORE_DIR` 指向包含 state_engine.py 的目录。
The source package includes the validated rules_core. Set
PARCEL_STATE_CORE_DIR when extracting it into a non-standard layout.
"""

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = (
    Path(os.environ.get("PARCEL_STATE_CORE_DIR", DEPLOYMENT_ROOT / "rules_core"))
    .expanduser()
    .resolve()
)

# 固定四个实际执行文件，保证部署调用的就是完成视频复核和42项回归测试的版本。
# Pin all executable core files to the exact revision validated by video review and 42 tests.
CORE_FILE_SHA256 = {
    "dataset_io.py": "cc5172d124c1404b221719d228480add5a1f32ffeb804fc4c266f146e6ee6fea",
    "model_adapter.py": "c5a9a5353e0e46eadd348619fd5a0b7c1038fc21c6983f0437cd2ad61ad756e8",
    "state_engine.py": "1128e277ac29c3a04f7afec09632dea6a23c34026fc825d97386ce6e57efb2a0",
    "temporal_filter.py": "01540d03a1a72eed13ef9372e7ae426155b20957e146c1464d0810bf03536f08",
}


def _sha256_file(path: Path) -> str:
    """流式计算源码 SHA256，避免读取并执行未经确认的核心代码。

    Stream a source-file SHA256 before importing and executing the core module.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for filename, expected_hash in CORE_FILE_SHA256.items():
    core_file = CORE_DIR / filename
    if not core_file.is_file():
        raise ImportError(
            "parcel-state rule core is incomplete. Keep rules_core in the parcel "
            "vision package, or set "
            f"PARCEL_STATE_CORE_DIR. Missing: {core_file}"
        )
    actual_hash = _sha256_file(core_file)
    if actual_hash != expected_hash:
        raise ImportError(
            f"unverified parcel-state core file: {core_file}; "
            f"expected SHA256 {expected_hash}, got {actual_hash}"
        )

# 旧核心最初是可直接运行的脚本目录，内部采用顶层导入。因此这里只在最前面加入
# 一个经过明确校验的目录，不扫描或猜测其他路径。
# The original core uses script-style top-level imports. Prepend only the explicit,
# validated core directory rather than searching arbitrary sys.path locations.
core_text = str(CORE_DIR)
if core_text not in sys.path:
    sys.path.insert(0, core_text)

# E402 是有意忽略：必须先把 CORE_DIR 加入 sys.path 再导入这些脚本模块。
# E402 is intentional because CORE_DIR must be registered before these imports.
from model_adapter import (  # noqa: E402
    FINAL_WEIGHT_SHA256,
    instances_from_result,
    load_final_model,
)
from state_engine import STATE_NAMES, Instance, RuleConfig, StateDecision, StateEngine  # noqa: E402
from temporal_filter import TemporalConfig, TemporalDecision, TemporalStateFilter  # noqa: E402


__all__ = [
    "CORE_DIR",
    "CORE_FILE_SHA256",
    "FINAL_WEIGHT_SHA256",
    "STATE_NAMES",
    "Instance",
    "RuleConfig",
    "StateDecision",
    "StateEngine",
    "TemporalConfig",
    "TemporalDecision",
    "TemporalStateFilter",
    "instances_from_result",
    "load_final_model",
]
