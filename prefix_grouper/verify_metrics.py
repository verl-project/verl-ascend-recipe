#!/usr/bin/env python3
"""Check a run written by run_qwen3_8b_megatron_npu.sh using the Python standard library."""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import fmean


@dataclass
class Step:
    reward: float | None
    seconds: float
    tps: float
    tokens: float
    validation: dict


def metric(data, key):
    value = data.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"missing or non-finite metric {key}")
    return value


def read_steps(path, *, include_quality=True):
    steps = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                data = record["data"]
                number = record["step"]
                if not isinstance(data, dict) or type(number) is not int:
                    raise ValueError("invalid verl file-logger record")
                if number == 0 and not steps and "training/global_step" not in data:
                    continue  # Optional pre-training validation.
                if number != len(steps) + 1 or metric(data, "training/global_step") != number:
                    raise ValueError("training steps must be contiguous, unique and start at 1")
                seconds = metric(data, "perf/time_per_step")
                tps = metric(data, "perf/throughput")
                tokens = metric(data, "perf/total_num_tokens")
                if min(seconds, tps, tokens) <= 0:
                    raise ValueError("non-positive elapsed time, throughput or token count")
                reward = metric(data, "critic/rewards/mean") if include_quality else None
                validation = {
                    key: metric(data, key)
                    for key in ("val-core/openai/gsm8k/acc/mean@1", "val-core/openai/gsm8k/reward/mean@1")
                    if include_quality and key in data
                }
                steps.append(Step(reward, seconds, tps, tokens, validation))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
    if not steps:
        raise ValueError(f"{path}: no training steps")
    return steps


TREE = re.compile(
    r"\bprefix_grouper phase=train rows=(\d+) original_tokens=(\d+) packed_tokens=(\d+) saved_tokens=(\d+)\b"
)
FINISHED = re.compile(r"^prefix_grouper_run_finished steps_requested=(\d+)$")


def check_training_log(path, steps):
    rank_entries_with_savings = 0
    rank_entry_saved_tokens = 0
    requested = []
    device = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("Ascend device:"):
                device = line.partition(":")[2].strip()
            if "prefix_grouper phase=train" in line:
                match = TREE.search(line)
                if match is None:
                    raise ValueError("malformed Megatron training tree record")
                rows, original, packed, saved = map(int, match.groups())
                if rows < 1 or packed < 1 or original - packed != saved:
                    raise ValueError("inconsistent Megatron training tree token counts")
                if saved:
                    if rows < 2:
                        raise ValueError("shared prefix reported on a single-row microbatch")
                    rank_entries_with_savings += 1
                    rank_entry_saved_tokens += saved
            finished = FINISHED.fullmatch(line.strip())
            if finished:
                requested.append(int(finished.group(1)))
    if requested != [len(steps)]:
        raise ValueError("run did not finish at its requested step count")
    if not device:
        raise ValueError("missing Ascend device identity from launcher preflight")
    if not rank_entries_with_savings:
        raise ValueError("no Megatron actor training microbatch saved any prefix tokens")
    return {
        "ascend_device_name": device,
        # TP workers may emit the same microbatch. These sums describe observed log entries.
        "observed_rank_local_train_entries_with_savings": rank_entries_with_savings,
        "saved_tokens_summed_over_observed_rank_local_train_entries": rank_entry_saved_tokens,
    }


def weighted_tps(steps):
    # Compare the logged decimal values exactly at strict acceptance boundaries.
    times = [Fraction(str(step.seconds)) for step in steps]
    return sum(t * Fraction(str(step.tps)) for t, step in zip(times, steps, strict=True)) / sum(times)


def compare_a100(steps, baseline):
    if len(baseline) != len(steps):
        raise ValueError("A2 and A100 runs must cover identical training steps")
    rewards = [Fraction(str(step.reward)) for step in baseline]
    reward_scale = sum(abs(reward) for reward in rewards)
    if not reward_scale:
        raise ValueError("A100 mean absolute reward is zero; relative error is undefined")
    keys = steps[-1].validation.keys() & baseline[-1].validation.keys()
    if len(keys) != 1:
        raise ValueError("need a matching final GSM8K mean@1 validation metric in both runs")
    key = next(iter(keys))
    reference = Fraction(str(baseline[-1].validation[key]))
    if reference <= 0:
        raise ValueError("A100 downstream metric must be positive for relative error")
    absolute_error = sum(abs(Fraction(str(step.reward)) - reward) for step, reward in zip(steps, rewards, strict=True))
    reward_error = absolute_error / reward_scale
    downstream_error = abs(Fraction(str(steps[-1].validation[key])) - reference) / reference
    tps_ratio = weighted_tps(steps) / weighted_tps(baseline)
    details = {
        "mean_absolute_reward_error_relative_to_a100_mean_absolute_reward": float(reward_error),
        "final_gsm8k_metric": key,
        "final_gsm8k_relative_error": float(downstream_error),
        "a2_to_a100_tps_ratio": float(tps_ratio),
    }
    checks = {
        "average_reward_error_below_5_percent": reward_error < Fraction(1, 20),
        "downstream_error_within_5_percent": downstream_error <= Fraction(1, 20),
        "a2_throughput_at_least_0.7_of_a100": tps_ratio >= Fraction(7, 10),
    }
    return details, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="recipe run directory containing metrics.jsonl and console.log")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="check recorded steps and grouped actor execution only")
    modes.add_argument(
        "--performance-only",
        action="store_true",
        help="measure all steps of a separate no-GPU-baseline run (recipe convention: at least 10 complete steps)",
    )
    modes.add_argument("--a100-metrics", type=Path, help="matched A100 verl JSONL from the same model/data/settings")
    parser.add_argument("--device", choices=["A2", "A3"], help="confirmed Ascend product; A100 comparison needs A2")
    args = parser.parse_args()
    if args.a100_metrics and args.device != "A2":
        parser.error("A100 comparison requires --device A2 and a full run")

    metrics_path = args.run_dir / "metrics.jsonl"
    try:
        steps = read_steps(metrics_path, include_quality=not args.performance_only)
        evidence = check_training_log(args.run_dir / "console.log", steps)
        seconds = sum(step.seconds for step in steps)
        tps = weighted_tps(steps)
        report = {
            "mode": "smoke" if args.smoke else ("A100 comparison" if args.a100_metrics else "no GPU baseline"),
            "device_as_declared": args.device,
            "recorded_steps": len(steps),
            "measured_training_seconds": seconds,
            "tokens_per_second_per_device": float(tps),
            "tokens_per_second_all_devices": sum(step.tokens for step in steps) / seconds,
            **evidence,
        }
        checks = {}
        if args.performance_only:
            report["mode"] = "performance only (no GPU baseline)"
            checks["recipe_measurement_minimum_10_complete_steps"] = len(steps) >= 10
            checks["per_device_tps_above_100"] = tps > 100
        else:
            # Non-overlapping windows avoid treating a single noisy endpoint as a trend.
            window = min(10, len(steps) // 2)
            first = fmean(step.reward for step in steps[:window]) if window else None
            last = fmean(step.reward for step in steps[-window:]) if window else None
            report.update(reward_window_steps=window, first_window_mean_reward=first, last_window_mean_reward=last)
        if not args.smoke and not args.performance_only:
            checks["at_least_100_steps_or_12_training_hours"] = len(steps) >= 100 or seconds >= 12 * 3600
            checks["reward_increased"] = bool(window and last > first)
            if args.a100_metrics:
                if args.a100_metrics.samefile(metrics_path):
                    raise ValueError("A100 metrics and Ascend metrics must be separate runs")
                details, comparison = compare_a100(steps, read_steps(args.a100_metrics))
                report.update(details)
                checks.update(comparison)
            else:
                checks["per_device_tps_above_100"] = tps > 100
        report["checks"] = checks
        passed = all(checks.values())
        if args.performance_only:
            report["result"] = "PERF_PASS" if passed else "PERF_FAIL"
        else:
            report["result"] = ("SMOKE_OK" if args.smoke else "PASS") if passed else "FAIL"
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if passed else 1
    except (OSError, ValueError) as error:
        print(f"Evidence check failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
