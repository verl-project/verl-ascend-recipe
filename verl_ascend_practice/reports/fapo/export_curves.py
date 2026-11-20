"""Rebuild readable logs, step metrics and figures from the archived run."""

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
raw = ROOT / "logs/train_1280184.raw.log"
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw.read_text()).replace("\r", "\n")
(ROOT / "logs/train_1280184.log").write_text(text)
steps = {}
for line in text.splitlines():
    match = re.search(r"\bstep:(\d+) - ", line)
    if match:
        row = {"step": int(match[1])}
        for item in line[match.end() :].split(" - "):
            key, separator, value = item.partition(":")
            if separator:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
        if row["step"] in steps:
            assert steps[row["step"]] == row, "Conflicting duplicate step"
        steps[row["step"]] = row
assert sorted(steps) == list(range(1, 101)), "Expected complete steps 1..100"
rows = [steps[index] for index in sorted(steps)]
keys = ["step"] + sorted(set().union(*(row.keys() for row in rows)) - {"step"})
with (ROOT / "metrics_1280184.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def mean(key, subset):
    return sum(row[key] for row in subset) / len(subset)


seconds = sum(row["timing_s/step"] for row in rows)
tokens = sum(row["perf/total_num_tokens"] for row in rows)
summary = {
    "job_id": 1280184,
    "steps": len(rows),
    "devices_total": 16,
    "raw_log_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    "sum_step_seconds": seconds,
    "mean_step_seconds": seconds / len(rows),
    "total_tokens": tokens,
    "weighted_tokens_per_second_per_device": tokens / seconds / 16,
    "weighted_tokens_per_second_all_devices": tokens / seconds,
    "reward_first10": mean("critic/rewards/mean", rows[:10]),
    "reward_last10": mean("critic/rewards/mean", rows[-10:]),
    "throughput_min": min(row["perf/throughput"] for row in rows),
    "throughput_max": max(row["perf/throughput"] for row in rows),
    "exit_code": int((ROOT / "logs/exit_code").read_text()),
}
for row in rows:
    expected = row["perf/total_num_tokens"] / row["timing_s/step"] / 16
    assert abs(row["perf/throughput"] - expected) < 1e-6
(ROOT / "summary_1280184.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
panels = [
    ("critic/rewards/mean", "Training reward", "Reward", "#19765b"),
    ("actor/loss", "Actor loss", "Loss", "#ad4a3d"),
    ("perf/throughput", "Throughput / allocated NPU", "tokens/s/NPU (16 devices)", "#316bb0"),
    ("timing_s/step", "Training step time", "seconds", "#8652a1"),
    ("actor/entropy", "Actor entropy", "Entropy", "#9a6b12"),
    ("response_length/mean", "Response length", "tokens", "#167c86"),
]
for axis, (key, title, label, color) in zip(axes.flat, panels, strict=True):
    values = [row[key] for row in rows]
    axis.plot(range(1, 101), values, color=color, alpha=0.35, linewidth=1, label="Per step")
    axis.plot(
        range(10, 101),
        [sum(values[i - 9 : i + 1]) / 10 for i in range(9, 100)],
        color=color,
        linewidth=2,
        label="Trailing 10-step mean",
    )
    axis.set(title=title, xlabel="Step", ylabel=label, xlim=(1, 100))
    axis.grid(alpha=0.18)
    axis.legend(fontsize=8)
fig.suptitle("FAPO | Qwen2.5-Math-7B + GenRM-4B | 16 x Ascend 910C | Job 1280184", fontsize=14)
fig.savefig(ROOT / "assets/training_curves.png", dpi=160)
plt.close(fig)
print(json.dumps(summary, indent=2))
