#!/usr/bin/env python3
"""Derive a v2 capsule profile without changing the approved v1 rollback profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    source: dict,
    overlay: dict,
    overlay_path: Path,
    *,
    execute_approved: bool = False,
) -> dict:
    result = copy.deepcopy(source)
    result["display_name"] = (
        "G2 大寰 · 接触拟合桌面 · 双臂多胶囊 v2 · 真机测试"
        if execute_approved
        else "G2 大寰 · 接触拟合桌面 · 双臂多胶囊 v2 · 只读验证"
    )
    result["execute_approved"] = execute_approved
    result["read_only_ready"] = True
    if execute_approved:
        approval = copy.deepcopy(source.get("execution_approval", {}))
        approval.update(
            {
                "scope": "g2a-contact-refined-multicapsule-v2-commissioning",
                "status": "operator-authorized-physical-commissioning",
                "inter_arm_geometry": "cross-chain-capsules-v2",
                "requires_runtime_identity_check": True,
                "requires_operator_double_confirmation": True,
            }
        )
        result["execution_approval"] = approval
    else:
        result.pop("execution_approval", None)
    keys = (
        "geometry",
        "validation_status",
        "hard_margin",
        "warning_margin",
        "broad_phase_activation_margin",
        "broad_phase_left_capsules",
        "broad_phase_right_capsules",
        "left_capsules",
        "right_capsules",
        "canonical_urdf",
        "canonical_urdf_sha256",
        "capsule_padding_m",
        "coverage",
    )
    validation_status = "execute-approved" if execute_approved else "read-only-candidate"
    result["inter_arm_collision"] = {
        "enabled": True,
        **{key: copy.deepcopy(overlay[key]) for key in keys},
        "validation_status": validation_status,
        "canonical_source": "G2_dahuan/collision_overlay.json",
        "canonical_source_sha256": sha256(overlay_path),
        "notes": (
            "双臂多胶囊 v2；操作员授权用于受控真机测试，保留 v1 配置用于回退。"
            if execute_approved
            else "双臂多胶囊 v2 候选；只读验收通过前禁止真机执行。"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute-approved",
        action="store_true",
        help="produce a separately named operator-authorized physical commissioning profile",
    )
    args = parser.parse_args()
    source_path = args.source.expanduser().resolve(strict=True)
    overlay_path = args.overlay.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    result = build(
        json.loads(source_path.read_text(encoding="utf-8")),
        json.loads(overlay_path.read_text(encoding="utf-8")),
        overlay_path,
        execute_approved=args.execute_approved,
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
