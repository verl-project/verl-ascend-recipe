#!/usr/bin/env python3
"""Reject successful trainer exits that did not reach the requested training step."""

import argparse
import re
from pathlib import Path


def check_completion(text: str, expected_step: int) -> list[int]:
    steps = []
    for line in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines():
        match = re.search(r"\bstep:(\d+) - (.*)", line)
        if not match:
            continue
        values = dict(part.rsplit(":", 1) for part in match[2].split(" - ") if ":" in part)
        values = {key.strip(): value.strip() for key, value in values.items()}
        # Validation-only lines have a step too; they cannot prove a training update.
        if "training/global_step" not in values or "timing_s/update_actor" not in values:
            continue
        step = int(match[1])
        if float(values["training/global_step"]) != step:
            raise ValueError("Console step disagrees with training/global_step")
        steps.append(step)
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
