"""Parse numeric metrics from verl console log records."""

import re


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
