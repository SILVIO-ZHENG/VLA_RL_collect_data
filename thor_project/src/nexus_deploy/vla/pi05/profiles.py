#!/usr/bin/env python
"""Local checkpoint profiles for the G2A PI0.5 experiments."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PROFILE_OUTPUT_DIRS = {
    "24d_absolute": PROJECT_ROOT / "outputs/g2a_pi05_lora_r16_30k",
    "16d_absolute": PROJECT_ROOT / "outputs/g2a_pi05_16d_absolute_lora_r16_30k",
    "16d_absolute_full": PROJECT_ROOT
    / "outputs/g2a_pi05_16d_absolute_full_openpi_aug_per_sample_b512_7500_split241_7_20260818_20260820",
    "16d_absolute_full_aug": PROJECT_ROOT / "outputs/g2a_pi05_16d_absolute_full_openpi_aug_30k",
    "16d_delta": PROJECT_ROOT / "outputs/g2a_pi05_16d_armdelta_lora_r16_30k",
    "16d_delta_full": PROJECT_ROOT / "outputs/g2a_pi05_16d_armdelta_full_30k",
}

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def _is_complete_pretrained_model(path: Path) -> bool:
    if not path.is_dir() or any(not (path / filename).is_file() for filename in REQUIRED_CHECKPOINT_FILES):
        return False
    return (path / "model.safetensors").is_file() or (
        (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()
    )


def resolve_checkpoint(profile: str, checkpoint: Path | None = None) -> Path:
    """Resolve an explicit checkpoint or the latest complete checkpoint in a profile."""
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not _is_complete_pretrained_model(resolved):
            raise FileNotFoundError(f"Checkpoint is incomplete or invalid: {resolved}")
        return resolved

    if profile not in PROFILE_OUTPUT_DIRS:
        raise ValueError(f"Unknown model profile {profile!r}; choices={sorted(PROFILE_OUTPUT_DIRS)}")

    checkpoints_dir = PROFILE_OUTPUT_DIRS[profile] / "checkpoints"
    candidates: list[tuple[int, Path]] = []
    if checkpoints_dir.is_dir():
        for item in checkpoints_dir.iterdir():
            pretrained_model = item / "pretrained_model"
            if item.name.isdigit() and _is_complete_pretrained_model(pretrained_model):
                candidates.append((int(item.name), pretrained_model))
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint found under {checkpoints_dir}")
    return max(candidates, key=lambda candidate: candidate[0])[1].resolve()


def infer_profile(action_dim: int, use_relative_actions: bool, use_peft: bool) -> str:
    if action_dim == 24 and not use_relative_actions:
        return "24d_absolute"
    if action_dim == 16 and not use_relative_actions:
        return "16d_absolute" if use_peft else "16d_absolute_full"
    if action_dim == 16 and use_relative_actions:
        return "16d_delta" if use_peft else "16d_delta_full"
    return f"{action_dim}d_{'delta' if use_relative_actions else 'absolute'}"
