#!/usr/bin/env python3
"""Convert the rollout SFT set into verl's multi-turn VLM SFT parquet.

Source: ``dataset_vllm_rollout/geometry3k_vllm_rollout_sft.json``
Output: ``verl_start/geo3k-sft-dataset/train.parquet``

Verl's ``MultiTurnSFTDataset`` reads ``messages`` (role/content) and ``images``
(bytes/path) columns, splits user content on ``<image>`` and inserts PIL images
before applying the chat template. The human turn must keep its leading
``<image>`` marker and ``images`` order must match the markers. Images are
embedded as raw bytes, so the dataset survives moving; ``path`` is kept only
for traceability. The SFT json's ``human``/``gpt`` roles are renamed to verl's
``user``/``assistant``, same as examples/data_preprocess/pokemon.py.

Run:
    python gen_verl_dataset_from_sft.py
"""

import argparse
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SFT = HERE / "dataset_vllm_rollout/geometry3k_vllm_rollout_sft.json"
DEFAULT_OUT = HERE / "verl_start/geo3k-sft-dataset"

ROLE_MAP = {"human": "user", "gpt": "assistant"}

SCHEMA = pa.schema(
    [
        (
            "messages",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        (
            "images",
            pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ),
    ]
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", type=pathlib.Path, default=DEFAULT_SFT)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    records = json.loads(args.sft.read_text(encoding="utf-8"))
    rows = []
    for i, rec in enumerate(records):
        conv = rec["conversations"]
        if [c["from"] for c in conv] != ["human", "gpt"]:
            raise SystemExit(f"record {i}: expected [human, gpt], got {[c['from'] for c in conv]}")

        messages = [{"role": ROLE_MAP[c["from"]], "content": c["value"]} for c in conv]
        if not messages[0]["content"].startswith("<image>"):
            raise SystemExit(f"record {i}: human turn has no leading <image> marker")

        images = []
        for path in rec["images"]:
            p = pathlib.Path(path)
            images.append({"bytes": p.read_bytes(), "path": str(p)})

        if messages[0]["content"].count("<image>") != len(images):
            raise SystemExit(f"record {i}: <image> markers != images")
        rows.append({"messages": messages, "images": images})

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "train.parquet"
    pq.write_table(table, str(dst))

    print(f"sft   : {args.sft}  ({len(records)} records)")
    print(f"wrote : {dst}  ({table.num_rows} rows, {dst.stat().st_size / 1e6:.1f} MB)")
    print(f"        roles={[m['role'] for m in rows[0]['messages']]}, image bytes={len(rows[0]['images'][0]['bytes'])}")


if __name__ == "__main__":
    main()
