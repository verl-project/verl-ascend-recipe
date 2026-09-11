#!/usr/bin/env python3
"""Check one complete LoRA merge training log; this does not certify external acceptance."""

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROLLOUT_KEYS = (
    "rollout_corr/kl",
    "rollout_corr/k3_kl",
    "rollout_corr/rollout_ppl",
    "rollout_corr/rollout_log_ppl",
    "rollout_corr/log_ppl_diff",
    "rollout_corr/log_ppl_abs_diff",
    "rollout_corr/log_ppl_diff_min",
)
REQUIRED_KEYS = (
    "training/global_step",
    "timing_s/update_actor",
    "actor/loss",
    "actor/grad_norm",
    "critic/rewards/mean",
    "perf/total_num_tokens",
    "perf/time_per_step",
    "perf/throughput",
    *ROLLOUT_KEYS,
)
VALIDATION_KEY = "val-core/openai/gsm8k/acc/mean@1"


def parse_metrics(text: str) -> list[dict]:
    rows = []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        match = re.search(r"\bstep:(\d+) - (.*)", line)
        if not match:
            continue
        row = {"step": int(match[1])}
        for part in match[2].split(" - "):
            if ":" not in part:
                continue
            key, value = part.rsplit(":", 1)
            value = value.strip()
            scalar = re.fullmatch(r"np\.(?:float|int)(?:32|64)\(([^()]+)\)", value)
            try:
                row[key.strip()] = float(scalar[1] if scalar else value)
            except ValueError as exc:
                raise ValueError(f"Non-numeric metric {key.strip()} at step {row['step']}") from exc
        rows.append(row)
    return rows


def check_validation(text: str, expected_steps: int, devices: int) -> dict:
    if expected_steps < 20 or devices < 1:
        raise ValueError("Require at least 20 steps for disjoint reward windows and a positive device count")
    metrics = parse_metrics(text)
    rows = [r for r in metrics if "training/global_step" in r and "timing_s/update_actor" in r]
    if [r["step"] for r in rows] != list(range(1, expected_steps + 1)):
        raise ValueError("Require exactly one contiguous training trajectory from step 1 to the expected final step")
    for row in rows:
        for key in REQUIRED_KEYS:
            if key not in row:
                raise ValueError(f"Missing {key} at step {row['step']}")
        if row["training/global_step"] != row["step"]:
            raise ValueError(f"Inconsistent global step at step {row['step']}")
        for key, value in row.items():
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} at step {row['step']}")
        if row["actor/grad_norm"] <= 0:
            raise ValueError(f"Non-positive gradient norm at step {row['step']}")
        tokens, seconds = row["perf/total_num_tokens"], row["perf/time_per_step"]
        if tokens <= 0 or seconds <= 0:
            raise ValueError(f"Invalid throughput inputs at step {row['step']}")
        if not math.isclose(row["perf/throughput"], tokens / seconds / devices, rel_tol=1e-9):
            raise ValueError(f"Throughput device denominator mismatch at step {row['step']}")
    validation = [{"step": r["step"], "gsm8k_accuracy": r[VALIDATION_KEY]} for r in metrics if VALIDATION_KEY in r]
    if not validation or validation[0]["step"] != 0 or validation[-1]["step"] != expected_steps:
        raise ValueError("Missing initial or final GSM8K validation")
    if any(not math.isfinite(r["gsm8k_accuracy"]) or not 0 <= r["gsm8k_accuracy"] <= 1 for r in validation):
        raise ValueError("Invalid GSM8K validation accuracy")
    first = statistics.mean(r["critic/rewards/mean"] for r in rows[:10])
    last = statistics.mean(r["critic/rewards/mean"] for r in rows[-10:])
    if last <= first:
        raise ValueError(f"Reward did not increase: first ten {first}, final ten {last}")
    tokens = sum(r["perf/total_num_tokens"] for r in rows)
    seconds = sum(r["perf/time_per_step"] for r in rows)
    throughput = tokens / seconds / devices
    if throughput <= 100:
        raise ValueError(f"Timed-step throughput {throughput} tokens/s/NPU does not exceed 100")
    return {
        "training_steps": len(rows),
        "physical_device_count": devices,
        "all_logged_training_metrics_finite": True,
        "zero_gradient_steps": [],
        "reward_first_10_mean": first,
        "reward_final_10_mean": last,
        "total_training_step_tokens": tokens,
        "total_timed_step_seconds": seconds,
        "weighted_tokens_per_second_per_npu": throughput,
        "minimum_tokens_per_second_per_npu": min(r["perf/throughput"] for r in rows),
        "validation": validation,
        "limitations": "Checks logged metrics only; excludes process exit, checkpoint restore and external acceptance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-steps", type=int, default=100)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    try:
        result = check_validation(raw.decode(errors="replace"), args.expected_steps, args.devices)
    except ValueError as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    result["log_sha256"] = hashlib.sha256(raw).hexdigest()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
