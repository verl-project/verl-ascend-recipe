#!/usr/bin/env python3
"""Reject successful trainer exits that did not reach the requested training step."""

import argparse
from pathlib import Path

from log_metrics import parse_metrics


def check_completion(text: str, expected_step: int) -> list[int]:
    steps = []
    for row in parse_metrics(text):
        # Validation-only records cannot prove a training update.
        if "training/global_step" not in row or "timing_s/update_actor" not in row:
            continue
        if row["training/global_step"] != row["step"]:
            raise ValueError("Console step disagrees with training/global_step")
        steps.append(row["step"])
    if not steps or steps[-1] != expected_step:
        raise ValueError(f"Expected training step {expected_step}; observed {steps[-1] if steps else 'none'}")
    if steps != list(range(steps[0], expected_step + 1)):
        raise ValueError("Training steps contain gaps, duplicates, or are out of order")
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    if args.expected_step <= 0:
        parser.error("--expected-step must be positive")
    try:
        steps = check_completion(args.log.read_text(errors="replace"), args.expected_step)
    except ValueError as exc:
        parser.exit(1, f"Training incomplete: {exc}\n")
    print(f"Training reached step {steps[-1]} ({len(steps)} updates in this invocation).")


if __name__ == "__main__":
    main()
