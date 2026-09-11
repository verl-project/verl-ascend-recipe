#!/usr/bin/env python3
"""Plot the checked single-host training log; hardware must be supplied explicitly."""

import argparse
import statistics
from pathlib import Path

from check_validation import check_validation, parse_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path, help="Output path without the PNG/SVG suffix")
    parser.add_argument("--hardware", required=True, help="Measured device model, for example Ascend 910B3")
    parser.add_argument("--devices", type=int, required=True)
    args = parser.parse_args()
    text = args.log.read_text()
    result = check_validation(text, 100, args.devices)
    rows = [r for r in parse_metrics(text) if "training/global_step" in r and "timing_s/update_actor" in r]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    reward = [r["critic/rewards/mean"] for r in rows]
    rolling = [statistics.mean(reward[i - 9 : i + 1]) for i in range(9, len(reward))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True)
    axes[0, 0].plot(steps, reward, color="#3274a1", alpha=0.35, label="Training reward (n=8)")
    axes[0, 0].plot(steps[9:], rolling, color="#3274a1", linewidth=2, label="Trailing 10-step mean")
    axes[0, 0].plot(
        [v["step"] for v in result["validation"]],
        [v["gsm8k_accuracy"] for v in result["validation"]],
        color="#238b45",
        marker="o",
        label="GSM8K accuracy (greedy, 1,319 items)",
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
        ax.set(xlabel="Training step", xlim=(0, 100))
        ax.grid(alpha=0.2)
    fig.suptitle(f"Qwen3-8B LoRA merge | FSDP2 | {args.devices} x {args.hardware}")
    fig.text(0.06, 0.035, "Throughput divides timed training-step tokens by time and all physical NPUs.", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        output = args.output.with_suffix(suffix)
        fig.savefig(output, dpi=180)
        if suffix == ".svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
