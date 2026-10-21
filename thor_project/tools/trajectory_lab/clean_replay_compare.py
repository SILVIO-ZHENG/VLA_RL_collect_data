"""Offline raw-vs-cleaned replay diagnostic for one dataset episode."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ARMS = np.array([*range(7), *range(8, 15)])


def metrics(points, fps=30.0):
    body = points[:, ARMS]
    velocity = np.diff(body, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    out = {}
    for name, values in (("velocity", velocity), ("acceleration", acceleration), ("jerk", jerk)):
        peak = np.max(np.abs(values), axis=1) if len(values) else np.array([0.0])
        out[name] = {
            "p50": float(np.percentile(peak, 50)),
            "p95": float(np.percentile(peak, 95)),
            "p99": float(np.percentile(peak, 99)),
            "max": float(np.max(peak)),
        }
    return out


def interpolate_100hz(points, fps=30.0, servo_fps=100.0):
    """The same deterministic linear arm interpolation for both replay arms."""
    times = np.arange(0.0, (len(points) - 1) / fps, 1.0 / servo_fps)
    phase = times * fps
    left = np.minimum(np.floor(phase + 1e-9).astype(int), len(points) - 2)
    u = (phase - left)[:, None]
    out = points[left] + (points[left + 1] - points[left]) * u
    # Device-position channels use the next source target (zero-order hold).
    out[:, [7, 15]] = points[left + 1][:, [7, 15]]
    return out


def clean_isolated_arm_spikes(points, *, min_residual_rad=0.02, max_run=1):
    """Repair only isolated reversal spikes; preserve gripper channels exactly."""
    result = np.asarray(points, dtype=float).copy()
    body = result[:, ARMS]
    repairs = []
    for i in range(1, len(body) - 1):
        prediction = 0.5 * (body[i - 1] + body[i + 1])
        residual = np.abs(body[i] - prediction)
        before = body[i] - body[i - 1]
        after = body[i + 1] - body[i]
        reversal = (before * after) < 0
        for local in np.flatnonzero((residual >= min_residual_rad) & reversal):
            joint = int(ARMS[local])
            result[i, joint] = prediction[local]
            repairs.append(
                {
                    "frame": int(i),
                    "joint": int(joint),
                    "residual_rad": float(residual[local]),
                    "replacement": float(prediction[local]),
                }
            )
    return result, repairs


def smooth_arms_savgol(points, *, window=11, order=3):
    """Offline legacy-style SG smoothing; gripper channels remain exact."""
    result = np.asarray(points, dtype=float).copy()
    result[:, ARMS] = savgol_filter(result[:, ARMS], window_length=window, polyorder=order, axis=0, mode="interp")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-residual-rad", type=float, default=0.02)
    parser.add_argument("--plot", action="store_true", help="also save trajectory comparison PNGs")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    rows = []
    for path in sorted((args.dataset / "data").glob("*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
        ids = table["episode_index"].to_numpy()
        mask = ids == args.episode
        if np.any(mask):
            frames = table["frame_index"].to_numpy()[mask]
            actions = table["action"].combine_chunks().values.to_numpy().reshape(-1, 16)[mask]
            rows.extend(zip(frames.tolist(), actions.tolist()))
    if not rows:
        raise SystemExit(f"episode {args.episode} not found")
    rows.sort(key=lambda item: item[0])
    frame = np.asarray([item[0] for item in rows])
    raw = np.asarray([item[1] for item in rows], dtype=float)
    if not np.array_equal(frame, np.arange(len(frame))):
        raise SystemExit("episode frames are not contiguous from zero")
    cleaned, repairs = clean_isolated_arm_spikes(raw, min_residual_rad=args.min_residual_rad)
    smoothed = smooth_arms_savgol(cleaned, window=11, order=3)
    # Explicit invariant: cleaning must not alter either gripper channel.
    if not np.array_equal(raw[:, [7, 15]], cleaned[:, [7, 15]]):
        raise SystemExit("gripper channels changed")
    if not np.array_equal(raw[:, [7, 15]], smoothed[:, [7, 15]]):
        raise SystemExit("gripper channels changed by SG")
    args.output.mkdir(parents=True, exist_ok=False)
    raw_100 = interpolate_100hz(raw)
    cleaned_100 = interpolate_100hz(cleaned)
    smoothed_100 = interpolate_100hz(smoothed)
    np.savez_compressed(
        args.output / "raw_and_cleaned.npz",
        raw=raw,
        cleaned=cleaned,
        raw_100hz=raw_100,
        cleaned_100hz=cleaned_100,
        smoothed_savgol=smoothed,
        smoothed_savgol_100hz=smoothed_100,
        frame=frame,
    )
    report = {
        "episode": args.episode,
        "fps": 30,
        "frames": len(raw),
        "cleaning": "isolated one-frame arm reversal spikes only",
        "min_residual_rad": args.min_residual_rad,
        "repairs": repairs,
        "repair_count": len(repairs),
        "raw_metrics": metrics(raw),
        "cleaned_metrics": metrics(cleaned),
        "savgol_metrics": metrics(smoothed),
        "raw_100hz_metrics": metrics(raw_100, fps=100),
        "cleaned_100hz_metrics": metrics(cleaned_100, fps=100),
        "savgol_100hz_metrics": metrics(smoothed_100, fps=100),
        "savgol": {"window_frames": 11, "order": 3, "mode": "interp", "arm_only": True},
        "gripper_unchanged": True,
        "replay": {
            "source": "raw_and_cleaned.npz",
            "interpolation": "same linear 30Hz-to-100Hz interval interpolation for both",
            "robot_commands": 0,
            "gdk": False,
        },
        "limitations": [
            "This is a diagnostic cleaning rule, not an accepted training preprocessing policy.",
            "No full physics or robot replay is performed by this command.",
            "Cleaning does not change gripper event frames.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [f"L{i + 1}" for i in range(7)] + [f"R{i + 1}" for i in range(7)]
        t30 = np.arange(len(raw)) / 30.0
        series = [("raw", raw), ("cleaned", cleaned), ("savgol", smoothed)]
        colors = {"raw": "#8b95a5", "cleaned": "#1976d2", "savgol": "#e65100"}

        def save_grid(kind, arrays, filename, ylabel):
            fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
            for local, ax in enumerate(axes.flat):
                joint = int(ARMS[local])
                # Draw the smoothed trace first and the raw trace last so small
                # differences remain visible instead of being completely covered.
                for label, values in reversed(arrays):
                    if kind == "position":
                        y, x = values[:, joint], t30
                    else:
                        body = values[:, ARMS]
                        if kind == "velocity":
                            y, x = np.diff(body, axis=0)[:, local] * 30, t30[1:]
                        elif kind == "acceleration":
                            y, x = np.diff(np.diff(body, axis=0), axis=0)[:, local] * 30**2, t30[2:]
                        else:
                            y, x = np.diff(np.diff(np.diff(body, axis=0), axis=0), axis=0)[:, local] * 30**3, t30[3:]
                    style = {"raw": "--", "cleaned": "-", "savgol": "-"}[label]
                    alpha = {"raw": 0.75, "cleaned": 0.9, "savgol": 0.9}[label]
                    ax.plot(x, y, lw=0.85, ls=style, alpha=alpha, color=colors[label], label=label)
                ax.set_title(names[local])
                ax.grid(alpha=0.25)
                if local in (0, 7):
                    ax.legend(loc="upper right", fontsize=8)
            fig.supxlabel("时间 (s)")
            fig.supylabel(ylabel)
            fig.suptitle(f"Episode {args.episode} - {kind}")
            fig.tight_layout()
            fig.savefig(args.output / filename, dpi=130)
            plt.close(fig)

        for kind, filename, ylabel in (
            ("position", "position_comparison.png", "关节位置 (rad)"),
            ("velocity", "velocity_comparison.png", "速度 (rad/s)"),
            ("acceleration", "acceleration_comparison.png", "加速度 (rad/s²)"),
            ("jerk", "jerk_comparison.png", "jerk (rad/s³)"),
        ):
            save_grid(kind, series, filename, ylabel)

        fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
        for local, ax in enumerate(axes.flat):
            joint = int(ARMS[local])
            delta = smoothed[:, joint] - raw[:, joint]
            ax.plot(t30, delta, color="#6a1b9a", lw=0.8)
            ax.axhline(0.0, color="#555", lw=0.5)
            ax.set_title(names[local])
            ax.grid(alpha=0.25)
        fig.supxlabel("时间 (s)")
        fig.supylabel("SG - raw (rad)")
        fig.suptitle(f"Episode {args.episode} - SG position residual")
        fig.tight_layout()
        fig.savefig(args.output / "position_residual_savgol_minus_raw.png", dpi=130)
        plt.close(fig)

        # Local zoom around repaired frames. If no repairs exist, use the
        # largest raw acceleration as the diagnostic center.
        body_raw = raw[:, ARMS]
        raw_acc_peak = np.max(np.abs(np.diff(np.diff(body_raw, axis=0), axis=0) * 30**2), axis=1)
        centers = [item["frame"] for item in repairs]
        if not centers:
            centers = [int(np.argmax(raw_acc_peak)) + 1]
        for serial, center in enumerate(dict.fromkeys(centers), start=1):
            start = max(0, center - 30)
            end = min(len(raw), center + 31)
            local_t = (np.arange(start, end) - center) / 30.0
            fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
            for local, label in ((0, "L1"), (1, "L2"), (2, "L3"), (3, "L4"), (4, "L5"), (5, "L6"), (6, "L7"),
                                 (7, "R1"), (8, "R2"), (9, "R3"), (10, "R4"), (11, "R5"), (12, "R6"), (13, "R7")):
                joint = int(ARMS[local])
                axes[0].plot(local_t, raw[start:end, joint], color="#9aa4b2", lw=0.8, alpha=0.7)
                axes[0].plot(local_t, cleaned[start:end, joint], color="#1976d2", lw=0.9, alpha=0.7)
                axes[0].plot(local_t, smoothed[start:end, joint], color="#e65100", lw=1.0, alpha=0.9)
            axes[0].set_ylabel("位置 (rad)")
            axes[0].set_title(f"Episode {args.episode} frame {center} 附近位置（全部手臂关节）")
            axes[0].legend(("raw", "cleaned", "savgol"), loc="upper right")

            for local in range(14):
                joint = int(ARMS[local])
                axes[1].plot(local_t[1:], np.diff(raw[start:end, joint]) * 30, color="#9aa4b2", lw=0.7, alpha=0.6)
                axes[1].plot(local_t[1:], np.diff(smoothed[start:end, joint]) * 30, color="#e65100", lw=0.9, alpha=0.8)
                axes[2].plot(local_t[2:], np.diff(np.diff(raw[start:end, joint])) * 30**2, color="#9aa4b2", lw=0.7, alpha=0.6)
                axes[2].plot(local_t[2:], np.diff(np.diff(smoothed[start:end, joint])) * 30**2, color="#e65100", lw=0.9, alpha=0.8)
            axes[1].set_ylabel("速度 (rad/s)")
            axes[2].set_ylabel("加速度 (rad/s²)")
            axes[2].set_xlabel("相对 frame 时间 (s)")
            for ax in axes:
                ax.axvline(0.0, color="#c62828", lw=0.8, alpha=0.7)
                ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.output / f"local_zoom_{serial:02d}_frame_{center}.png", dpi=150)
            plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
        for index, label in ((7, "left gripper"), (15, "right gripper")):
            axes[0 if index == 7 else 1].plot(t30, raw[:, index], color="#212121", lw=1)
            axes[0 if index == 7 else 1].set_title(label)
            axes[0 if index == 7 else 1].set_ylabel("position")
            axes[0 if index == 7 else 1].grid(alpha=0.25)
        for repair in repairs:
            for ax in axes:
                ax.axvline(repair["frame"] / 30.0, color="#c62828", alpha=0.35, lw=0.8)
        axes[-1].set_xlabel("时间 (s)")
        fig.suptitle(f"Episode {args.episode} - gripper events / repaired frames")
        fig.tight_layout()
        fig.savefig(args.output / "gripper_and_repairs.png", dpi=140)
        plt.close(fig)
    print(json.dumps({k: report[k] for k in ("episode", "frames", "repair_count", "raw_metrics", "cleaned_metrics")}, indent=2))


if __name__ == "__main__":
    main()
