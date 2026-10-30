# Copyright 2026 Individual Contributor
# Licensed under the Apache License, Version 2.0.
"""Export a redacted complete log, per-step metrics, and reproducible curves.

Run with Python 3.10+ and matplotlib. No training framework is imported.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC = re.compile(r"(?:^| - )([A-Za-z0-9_/]+):([-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|inf|nan))(?= - |$)")


def redact(text):
    text = ANSI.sub("", text)
    text = text.replace("/user/dengxianglong/verl-sapo-repro", "/path/to/sapo-repro")
    text = re.sub(r"/user/[A-Za-z0-9_.-]+", "/user/REDACTED", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", text)
    return text


def parse(text, expected):
    rows = {}
    for line in text.splitlines():
        match = re.search(r"\bstep:(\d+) - ", line)
        if not match:
            continue
        values = {key: float(value) for key, value in METRIC.findall(line[match.start() :])}
        step = int(values["step"])
        if step in rows:
            raise ValueError(f"Duplicate step {step}")
        for key in ("timing_s/step", "perf/total_num_tokens", "perf/throughput"):
            if key not in values or not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(f"Missing/invalid {key} at step {step}")
        rows[step] = values
    if sorted(rows) != list(range(1, expected + 1)):
        raise ValueError(f"Expected steps 1..{expected}; found {sorted(rows)}")
    if not re.search(r"^EXIT_CODE=0\s*$", text, re.MULTILINE):
        raise ValueError("Missing successful exit marker")
    return [rows[n] for n in sorted(rows)]


def summarize(rows, devices):
    duration = sum(row["timing_s/step"] for row in rows)
    tokens = sum(row["perf/total_num_tokens"] for row in rows)
    anomalies = {}
    for row in rows:
        expected = row["perf/total_num_tokens"] / row["timing_s/step"] / devices
        if not math.isclose(expected, row["perf/throughput"], rel_tol=1e-6):
            raise ValueError(f"Throughput/device mismatch at step {row['step']}")
        for key, value in row.items():
            if not math.isfinite(value):
                anomalies.setdefault(key, []).append(int(row["step"]))

    def mean(key, subset):
        return sum(row[key] for row in subset) / len(subset)

    return {
        "steps": len(rows),
        "devices": devices,
        "timed_step_seconds_sum": duration,
        "tokens_sum": int(tokens),
        "aggregate_tokens_per_second": tokens / duration,
        "per_device_tokens_per_second_time_weighted": tokens / duration / devices,
        "per_device_tokens_per_second_arithmetic_mean": mean("perf/throughput", rows),
        "steady_step_seconds_mean_excluding_first": mean("timing_s/step", rows[1:]),
        "weight_sync_seconds_mean": mean("timing_s/update_weights", rows),
        "reward_first_25_mean": mean("critic/rewards/mean", rows[:25]),
        "reward_last_25_mean": mean("critic/rewards/mean", rows[-25:]),
        "reward_first": rows[0]["critic/rewards/mean"],
        "reward_last": rows[-1]["critic/rewards/mean"],
        "nonfinite_metric_steps": anomalies,
    }


def plot(rows, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in rows]
    panels = [
        ("critic/rewards/mean", "Training reward", "Mean reward"),
        ("perf/throughput", "End-to-end throughput", "tokens/s/NPU"),
        ("timing_s/step", "Step duration", "Seconds"),
        ("timing_s/update_weights", "Weight synchronization", "Seconds"),
        ("actor/loss", "SAPO actor loss", "Loss"),
        ("actor/entropy", "Actor entropy", "Entropy"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for axis, (key, title, unit) in zip(axes.flat, panels, strict=True):
        values = [row[key] for row in rows]
        axis.plot(steps, values, color="#147d92", linewidth=1, alpha=0.65, label="Per step")
        if key == "critic/rewards/mean":
            moving = [
                sum(values[max(0, i - 9) : i + 1]) / len(values[max(0, i - 9) : i + 1]) for i in range(len(values))
            ]
            axis.plot(steps, moving, color="#b54949", linewidth=2, label="Trailing mean (up to 10 steps)")
        axis.set(title=title, xlabel="Training step", ylabel=unit, xlim=(1, len(rows)))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Qwen3-30B-A3B-Base / SAPO / Megatron / 16 Ascend 910C\n"
        "Run 1276169: 100 steps; throughput includes the whole training step"
    )
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--submitted", default="2026-10-18T12:28:09.163253+00:00")
    parser.add_argument("--container-started", default="2026-10-18T12:31:20+00:00")
    parser.add_argument("--exited", default="2026-10-18T18:55:24.906574+00:00")
    args = parser.parse_args()
    source = args.log.read_bytes()
    raw = gzip.decompress(source) if args.log.suffix == ".gz" else source
    clean = redact(raw.decode("utf-8", errors="replace"))
    rows = parse(clean, args.steps)
    summary = summarize(rows, args.devices)
    summary["redacted_log_sha256"] = hashlib.sha256(clean.encode()).hexdigest()
    summary["submission_to_exit_seconds"] = (
        datetime.fromisoformat(args.exited) - datetime.fromisoformat(args.submitted)
    ).total_seconds()
    summary["container_to_exit_seconds"] = (
        datetime.fromisoformat(args.exited) - datetime.fromisoformat(args.container_started)
    ).total_seconds()
    summary["timestamps"] = {
        "submitted": args.submitted,
        "container_started": args.container_started,
        "exited": args.exited,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.redacted.log.gz").write_bytes(gzip.compress(clean.encode(), mtime=0))
    fields = ["step", *sorted(set().union(*(row.keys() for row in rows)) - {"step"})]
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(rows, args.output / "training_curves.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
