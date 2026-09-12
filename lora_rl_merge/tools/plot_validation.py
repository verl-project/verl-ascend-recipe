#!/usr/bin/env python3
"""Plot training metrics without requiring the run to pass acceptance checks."""

import argparse
import math
import statistics
from pathlib import Path

from check_validation import VALIDATION_KEY
from log_metrics import parse_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path, help="Output image path, for example training-curves.png")
    parser.add_argument("--hardware", required=True, help="Measured device model, for example Ascend 910B3")
    parser.add_argument("--devices", type=int, required=True)
    args = parser.parse_args()
    text = args.log.read_text()
    metrics = parse_metrics(text)
    rows = [r for r in metrics if "training/global_step" in r and "timing_s/update_actor" in r]
    keys = ("critic/rewards/mean", "perf/throughput", "actor/grad_norm", "perf/time_per_step")
    if args.devices < 1 or not rows:
        parser.error("Require a positive device count and at least one training record")
    for row in rows:
        if any(key not in row or not math.isfinite(row[key]) for key in keys):
            parser.error(f"Missing or non-finite plotting metric at step {row['step']}")
    validation = [r for r in metrics if VALIDATION_KEY in r]
    if any(not math.isfinite(r[VALIDATION_KEY]) or not 0 <= r[VALIDATION_KEY] <= 1 for r in validation):
        parser.error("Invalid validation accuracy")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    reward = [r["critic/rewards/mean"] for r in rows]
    rolling = [statistics.mean(reward[i - 9 : i + 1]) for i in range(9, len(reward))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True)
    axes[0, 0].plot(steps, reward, color="#3274a1", alpha=0.35, label="Training reward")
    axes[0, 0].plot(steps[9:], rolling, color="#3274a1", linewidth=2, label="Trailing 10-step mean")
    axes[0, 0].plot(
        [v["step"] for v in validation],
        [v[VALIDATION_KEY] for v in validation],
        color="#238b45",
        marker="o",
        label="GSM8K validation accuracy",
    )
    axes[0, 0].set(ylabel="Reward / accuracy", ylim=(0, 1.05), title="Training and validation")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 1].plot(steps, [r["perf/throughput"] for r in rows], color="#a6761d")
    axes[0, 1].axhline(100, color="#777777", linestyle=":", label="100 tokens/s/NPU")
    axes[0, 1].set(ylabel="Tokens/s/NPU", title="Reported per-step throughput")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(steps, [r["actor/grad_norm"] for r in rows], color="#756bb1")
    axes[1, 0].set(ylabel="Gradient norm", title="Actor gradient norm")
    axes[1, 1].plot(steps, [r["perf/time_per_step"] for r in rows], color="#d95f02")
    axes[1, 1].set(ylabel="Seconds", title="Time measured inside each training step")
    for ax in axes.flat:
        ax.set(xlabel="Training step", xlim=(0, max(steps)))
        ax.grid(alpha=0.2)
    fig.suptitle(f"Qwen3-8B LoRA merge | FSDP2 | {args.devices} x {args.hardware}")
    fig.text(0.06, 0.035, "Throughput divides timed training-step tokens by time and all physical NPUs.", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
