"""Resolve PI0.5 RTC routes from arguments and checkpoint metadata, without weights."""
import json
from collections.abc import Sequence
from pathlib import Path


def _option(args, name):
    value = None
    index = 0
    while index < len(args):
        argument = str(args[index])
        if argument == "--":
            break
        if argument.startswith(name + "="):
            value = argument.split("=", 1)[1]
        elif argument == name:
            index += 1
            value = str(args[index]) if index < len(args) else None
        index += 1
    return value


def rtc_inference_mode_for_args(args: Sequence[str]) -> str | None:
    mode = _option(args, "--rtc-inference-mode")
    if mode is None and "--rtc-inference-mode" not in args:
        mode = "guided"
    return mode if mode in {"trained", "guided"} else None


def with_rtc_inference_mode(args: Sequence[str], mode: str) -> list[str]:
    if mode not in {"trained", "guided"}:
        raise ValueError("RTC 推理模式必须为 trained 或 guided")
    result = []
    index = 0
    while index < len(args):
        argument = str(args[index])
        if argument == "--":
            result.append(f"--rtc-inference-mode={mode}")
            return result + list(args[index:])
        if argument == "--rtc-inference-mode":
            index += 2
            continue
        if not argument.startswith("--rtc-inference-mode="):
            result.append(argument)
        index += 1
    result.append(f"--rtc-inference-mode={mode}")
    return result


def rtc_metadata_for_args(args: Sequence[str], repo: Path) -> dict:
    checkpoint = _option(args, "--checkpoint")
    checkpoint_config = {}
    if checkpoint:
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(repo) / checkpoint_path
        try:
            value = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
            if isinstance(value, dict):
                checkpoint_config = value
        except (OSError, ValueError, TypeError):
            pass
    delay = checkpoint_config.get("rtc_training_max_delay")
    supported = type(delay) is int and delay > 0
    try:
        horizon = int(_option(args, "--execution-horizon") or "30")
    except ValueError:
        horizon = None
    return dict(
        rtc_inference_mode_default="trained" if supported else "guided",
        trained_rtc_supported=supported,
        rtc_modes=["trained", "guided"] if supported else ["guided"],
        rtc_training_max_delay=delay if type(delay) is int else None,
        rtc_chunk_size=checkpoint_config.get("chunk_size"),
        rtc_execution_horizon=horizon,
    )


def validate_rtc_inference_mode(mode, metadata):
    if not isinstance(mode, str) or mode not in {"trained", "guided"}:
        raise ValueError("RTC 推理模式必须为 trained 或 guided")
    if mode == "trained":
        if metadata.get("trained_rtc_supported") is not True:
            raise ValueError("Trained RTC 需要 checkpoint 元数据 rtc_training_max_delay > 0；请选择 Inference RTC")
        delay = metadata.get("rtc_training_max_delay")
        chunk = metadata.get("rtc_chunk_size")
        horizon = metadata.get("rtc_execution_horizon")
        if (type(chunk) is not int or type(horizon) is not int or
                not delay <= horizon <= chunk - delay):
            raise ValueError("Trained RTC 执行窗口须满足 max_delay <= execution_horizon <= chunk_size - max_delay")
