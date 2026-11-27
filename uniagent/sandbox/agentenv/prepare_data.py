"""Bind operator-supplied benchmark metadata to operator-supplied AgentENV templates."""

import argparse
import json
from pathlib import Path

from evaluation_adapter import normalize_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="JSON list of benchmark metadata rows")
    parser.add_argument("--templates", type=Path, required=True, help="instance_id -> {template, workdir} JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1, help="Explicit smoke-only repetition; default: no repetition")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    rows = json.loads(args.rows.read_text())
    templates = json.loads(args.templates.read_text())
    if not rows or len({row["instance_id"] for row in rows}) != len(rows):
        parser.error("Rows must be nonempty and have unique instance_id values")
    if set(templates) != {row["instance_id"] for row in rows}:
        parser.error("Template map must match the input instance IDs exactly")
    output = []
    for original in rows:
        row = dict(original)
        binding = templates[row["instance_id"]]
        if not isinstance(binding.get("template"), str) or not binding["template"].strip():
            parser.error("Every instance needs a nonempty template ID")
        row["workdir"] = binding["workdir"]
        normalize_metadata(row)
        for key in ("problem_statement", "test_patch"):
            if not isinstance(row.get(key), str):
                parser.error(f"Missing string metadata field: {key}")
        if isinstance(row.get("install_config"), dict):
            row["install_config"] = json.dumps(row["install_config"])
        task = {
            "name": "agentenv_swe_v2",
            "metadata": row,
            "sandbox": {"provider": "e2b", "sandbox_kwargs": {"template": binding["template"], "user": "root"}},
        }
        output.append(
            {
                "data_source": "agentenv_swe_v2",
                "prompt": [{"role": "user", "content": row["problem_statement"]}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"tools_kwargs": {"task": task}},
            }
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output * args.repeat), args.output)
    print(f"Wrote {len(output) * args.repeat} rows ({len(output)} unique instances): {args.output}")


if __name__ == "__main__":
    main()
