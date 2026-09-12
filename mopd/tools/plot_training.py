"""Plot the full 100-step MOPD log; requires matplotlib in addition to the summary tool."""

import argparse
from pathlib import Path

from summarize_training import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize([(args.log, 1, 100)], 100, 4)
    if not summary["training_length_complete"]:
        parser.error("A complete 100-step log is required")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for ax in axes.flat:
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.25)
    panels = [
        (axes[0, 0], "critic/rewards/mean", "Training reward", "Reward"),
        (axes[1, 0], "perf/throughput", "Throughput (all four NPUs)", "tokens/s/NPU"),
        (axes[1, 1], "actor/grad_norm", "Actor gradient norm", "Norm"),
        (axes[2, 0], "perf/time_per_step", "Timed training step", "Seconds"),
        (axes[2, 1], "actor/distillation/loss", "Distillation loss", "Loss"),
    ]
    for ax, key, title, ylabel in panels:
        ax.plot(steps, [row[key] for row in rows], linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    windows = [sum(row["critic/rewards/mean"] for row in rows[i : i + 10]) / 10 for i in range(0, 100, 10)]
    axes[0, 0].plot([5.5 + 10 * i for i in range(10)], windows, "o-", color="darkorange", label="10-step mean")
    axes[0, 0].legend()
    axes[1, 0].axhline(100, linestyle="--", color="firebrick", label="Threshold: 100")
    axes[1, 0].legend()
    ax = axes[0, 1]
    for source, label in [("openai/gsm8k", "GSM8K"), ("DigitalLearningGmbH/MATH-lighteval", "MATH")]:
        points = []
        for validation in summary["validation"]:
            matching = [
                value
                for key, value in validation["metrics"].items()
                if key.startswith(f"val-core/{source}/") and "/mean@1" in key
            ]
            if matching:
                if len(set(matching)) != 1:
                    raise ValueError(f"Ambiguous validation result for {source}")
                points.append((validation["step"], matching[0] * 100))
        if [step for step, _ in points] != [0, 20, 40, 60, 80, 100]:
            raise ValueError(f"Missing or duplicate validation points for {source}")
        ax.plot([step for step, _ in points], [value for _, value in points], "o-", label=label)
    ax.set_title("Full validation with unchanged scoring")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.suptitle("Qwen3-0.6B MOPD | 4 x Ascend 910B3 | batch 256 | 100 steps", fontsize=14)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
