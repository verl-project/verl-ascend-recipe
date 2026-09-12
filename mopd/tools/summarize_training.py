"""Summarize explicit training-log segments without counting superseded updates.

This checks logged training evidence only. A complete result still needs process exit,
checkpoint, configuration identity, and numeric-diagnostic review.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


def parse_metrics(text):
    rows = []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        match = re.search(r"\bstep:(\d+) - (.*)", line)
        if not match:
            continue
        values = {}
        for part in match[2].split(" - "):
            if ":" not in part:
                continue
            key, value = part.rsplit(":", 1)
            value = value.strip()
            numpy_scalar = re.fullmatch(r"np\.(?:float|int)(?:32|64)\(([^()]+)\)", value)
            if numpy_scalar:
                value = numpy_scalar[1]
            try:
                values[key.strip()] = float(value)
            except ValueError:
                continue
        step = int(match[1])
        if "training/global_step" in values and values["training/global_step"] != step:
            raise ValueError(f"Inconsistent global step at {step}")
        rows.append({"step": step, **values})
    return rows


def parse_training(text):
    return [row for row in parse_metrics(text) if "training/global_step" in row and "timing_s/update_actor" in row]


def summarize(segments, expected_steps, devices):
    rows, sources, validation = [], [], []
    for path, first, last in segments:
        raw = path.read_bytes()
        parsed_metrics = parse_metrics(raw.decode(errors="replace"))
        parsed = [row for row in parsed_metrics if "training/global_step" in row and "timing_s/update_actor" in row]
        for row in parsed_metrics:
            if not (first <= row["step"] <= last or first == 1 and row["step"] == 0):
                continue
            metrics = {
                key: value if math.isfinite(value) else str(value)
                for key, value in row.items()
                if key.startswith(("val-core/", "val-aux/"))
            }
            if metrics:
                validation.append({"step": row["step"], "source": str(path), "metrics": metrics})
        selected = [row for row in parsed if first <= row["step"] <= last]
        rows.extend(selected)
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "requested_range": [first, last],
                "included_steps": [r["step"] for r in selected],
                "excluded_steps": [r["step"] for r in parsed if not first <= r["step"] <= last],
            }
        )
    steps = [row["step"] for row in rows]
    if not steps or steps != list(range(1, steps[-1] + 1)) or steps[-1] > expected_steps:
        raise ValueError("Selected segments must form a contiguous trajectory starting at step 1")
    required = (
        "actor/loss",
        "actor/grad_norm",
        "critic/rewards/mean",
        "perf/total_num_tokens",
        "perf/time_per_step",
        "perf/throughput",
    )
    nonfinite = Counter()
    for row in rows:
        for key in required:
            if key not in row:
                raise ValueError(f"Missing {key} at step {row['step']}")
        nonfinite.update(key for key, value in row.items() if not math.isfinite(value))
        tokens, duration, throughput = (row[k] for k in required[-3:])
        if not all(math.isfinite(value) and value > 0 for value in (tokens, duration, throughput)):
            raise ValueError(f"Invalid throughput inputs at step {row['step']}")
        if not math.isclose(throughput, tokens / duration / devices, rel_tol=1e-9):
            raise ValueError(f"Reported throughput has a different device denominator at step {row['step']}")
    by_step = {row["step"]: row for row in rows}

    def reward_window(first, last):
        if any(step not in by_step for step in range(first, last + 1)):
            return None
        return statistics.mean(by_step[step]["critic/rewards/mean"] for step in range(first, last + 1))

    first_reward = reward_window(1, 10)
    final_reward = reward_window(expected_steps - 9, expected_steps)
    total_tokens = sum(row["perf/total_num_tokens"] for row in rows)
    timed_seconds = sum(row["perf/time_per_step"] for row in rows)
    per_device = total_tokens / timed_seconds / devices
    return {
        "sources": sources,
        "expected_steps": expected_steps,
        "observed_steps": steps,
        "validation": validation,
        "training_length_complete": steps[-1] == expected_steps,
        "physical_device_count": devices,
        "nonfinite_metric_step_counts": dict(nonfinite),
        "actor_metrics_finite": not any(nonfinite[key] for key in ("actor/loss", "actor/grad_norm")),
        "zero_gradient_steps": [r["step"] for r in rows if r["actor/grad_norm"] == 0],
        "reward": {
            "first_10_mean": first_reward,
            "final_10_mean": final_reward,
            "increased": None if first_reward is None or final_reward is None else final_reward > first_reward,
        },
        "throughput": {
            "total_tokens": total_tokens,
            "sum_timed_step_seconds": timed_seconds,
            "weighted_tokens_per_second_per_device": per_device,
            "arithmetic_mean_tokens_per_second_per_device": statistics.mean(r["perf/throughput"] for r in rows),
            "minimum_tokens_per_second_per_device": min(r["perf/throughput"] for r in rows),
            "weighted_exceeds_100": per_device > 100,
        },
        "limitations": [
            "Only explicitly selected updates are included; excluded updates are listed per source.",
            "Timed-step throughput is not whole-job wall-clock throughput or pure generation throughput.",
            "Startup, checkpoint restoration, inter-run downtime and some evaluation time can be outside step timing.",
            "This summary cannot prove checkpoint identity, process success, "
            "sampler correctness or external acceptance.",
        ],
        "rows": [{key: value if math.isfinite(value) else str(value) for key, value in row.items()} for row in rows],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", nargs=3, action="append", required=True, metavar=("LOG", "FIRST", "LAST"))
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_steps < 10 or args.devices < 1:
        parser.error("At least ten target steps and a positive device count are required")
    result = summarize(
        [(Path(path), int(first), int(last)) for path, first, last in args.segment], args.expected_steps, args.devices
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("training_length_complete", "actor_metrics_finite", "reward", "throughput")}
        )
    )
