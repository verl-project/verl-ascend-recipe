"""Bind raw SWE-Bench rows to explicitly selected E2B templates; never runs tasks."""

import argparse
import json
from pathlib import Path

REQUIRED = (
    "instance_id",
    "repo",
    "version",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def prepare(source, manifest, output, smoke_only=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    catalog = json.loads(Path(manifest).read_text())
    if catalog.get("schema_version") != 1 or not catalog.get("instances"):
        raise ValueError("Expected schema_version=1 and a nonempty instances mapping")
    path = Path(source)
    if str(source).startswith("hf://"):
        from datasets import load_dataset

        rows = load_dataset(str(source)[5:], split="test")
    elif path.suffix == ".parquet":
        rows = pq.read_table(path).to_pylist()
        # Existing Uni-Agent datasets keep the original SWE-Bench row inside
        # reward metadata. Accept these so template bindings can be refreshed.
        if rows and "instance_id" not in rows[0]:
            extracted = []
            for row in rows:
                metadata = row.get("extra_info", {}).get("tools_kwargs", {}).get("reward", {}).get("metadata", {})
                if metadata.get("instance_id") in catalog["instances"]:
                    extracted.append(metadata)
            rows = extracted
    else:
        text = path.read_text()
        rows = json.loads(text) if path.suffix == ".json" else [json.loads(x) for x in text.splitlines() if x.strip()]
    selected = {}
    for row in rows:
        iid = row["instance_id"]
        if iid in catalog["instances"]:
            if iid in selected:
                raise ValueError(f"Duplicate dataset instance: {iid}")
            selected[iid] = row
    missing = set(catalog["instances"]) - selected.keys()
    if missing:
        raise ValueError(f"Manifest instances missing from source: {sorted(missing)}")
    splits = {"train": [], "validation": []}
    templates = set()
    for iid, binding in catalog["instances"].items():
        template = binding["template"]
        if not isinstance(template, str) or not template.strip() or template in templates:
            raise ValueError(f"Missing or reused task-specific template for {iid}")
        templates.add(template)
        split = binding["split"]
        if split == "smoke":
            if not smoke_only:
                raise ValueError("Smoke manifest requires --smoke-only; it is not a train/held-out split")
            destinations = ["train", "validation"]
        elif split in splits and not smoke_only:
            destinations = [split]
        else:
            raise ValueError(f"Invalid split/mode combination: {split}")
        row = selected[iid]
        metadata = {key: row[key] for key in REQUIRED}
        metadata["version"] = str(metadata["version"])
        for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            value = metadata[key]
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, list):
                raise ValueError(f"{iid}: {key} must be a list")
            metadata[key] = json.dumps(value)
        task = {
            "name": "swe_bench",
            "sandbox": {"provider": "e2b", "sandbox_kwargs": {"template": template, "expected_instance_id": iid}},
            "metadata": metadata,
        }
        item = {
            "data_source": "swe_bench/e2b",
            "prompt": [{"role": "user", "content": metadata["problem_statement"]}],
            "extra_info": {"tools_kwargs": {"task": task}},
        }
        for destination in destinations:
            splits[destination].append(item)
    if not splits["train"]:
        raise ValueError("At least one train manifest entry is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        pq.write_table(pa.Table.from_pylist(rows), output / f"{name}.parquet")
    (output / "binding_manifest.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "dataset_info.json").write_text(
        json.dumps(
            {
                "smoke_only": smoke_only,
                "task_count": len(catalog["instances"]),
                "counts": {k: len(v) for k, v in splits.items()},
                "source": str(path.resolve()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Raw SWE-Bench JSON, JSONL or parquet")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("templates.json")))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.manifest, args.output_dir, args.smoke_only)
