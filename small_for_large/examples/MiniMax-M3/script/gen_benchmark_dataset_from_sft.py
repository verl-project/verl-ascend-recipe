#!/usr/bin/env python3
"""Build an ais_bench Geometry3K eval parquet from the vLLM-rollout SFT set.

Source: the SFT json plus the rollout jsonl it came from (for original golds).
Output: ``benchmark_test_after/geo3k/test.parquet`` (pass to
``ais_bench --dataset-path``).

``Geometry3KDataset`` expects the hiyouga geometry3k fields
``problem``/``answer``/``images``, with no instruction suffix in ``problem``.
The SFT json only has the model's ``\\boxed{...}`` answer, often a labelled
form, so the original gold is recovered by matching image bytes to the rollout
row (one image per sample, so md5 is a safe key). Also normalises the feature
metadata ``_type: "List"`` to ``"Sequence"`` for the ais_bench runtime.

Run:
    python gen_benchmark_dataset_from_sft.py
"""

import argparse
import hashlib
import json
import pathlib

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Sequence, Value

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SFT = HERE / "dataset_vllm_rollout/geometry3k_vllm_rollout_sft.json"
DEFAULT_SRC = HERE / "outputs/default/geo3k_rollout/predictions" / "vllm-api-general-chat/geometry3k.jsonl"
DEFAULT_OUT = HERE / "dataset_vllm_rollout/benchmark_test_after"

IMAGE_PLACEHOLDER = "<image>"
INSTRUCTION = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _md5_file(path: pathlib.Path) -> str:
    return _md5_bytes(path.read_bytes())


def _source_index(src: pathlib.Path) -> dict:
    """(md5(image bytes), prompt text) -> {id, gold} from the rollout jsonl.

    Neither key alone is unique: distinct rows can share a byte-identical
    figure (e.g. ``67.png``) or the same question wording.  The pair is unique
    across this run, and the SFT record carries both.
    """
    index = {}
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        parts = row["origin_prompt"][0]["prompt"]
        texts = [p["text"] for p in parts if p.get("type") == "text"]
        urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
        if len(texts) != 1 or len(urls) != 1 or not urls[0].startswith("file://"):
            raise SystemExit(f"row {row.get('id')}: unexpected origin_prompt shape")
        image_path = pathlib.Path(urls[0][len("file://") :])
        key = (_md5_file(image_path), texts[0])
        if key in index:
            raise SystemExit(f"duplicate (image, prompt) in source: row {row.get('id')}")
        index[key] = {"id": row["id"], "gold": str(row["gold"])}
    return index


def _normalize_feature_types(path: pathlib.Path) -> None:
    """Rewrite HF feature metadata from ``List`` to ``Sequence`` (datasets <= 3.x)."""
    table = pq.read_table(str(path))
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    if not raw:
        return

    def _fix(node):
        if isinstance(node, dict):
            if node.get("_type") == "List":
                node["_type"] = "Sequence"
            for value in node.values():
                _fix(value)
        elif isinstance(node, list):
            for value in node:
                _fix(value)

    info = json.loads(raw)
    _fix(info.get("info", {}).get("features", {}))
    metadata[b"huggingface"] = json.dumps(info).encode()
    pq.write_table(table.replace_schema_metadata(metadata), str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", type=pathlib.Path, default=DEFAULT_SFT)
    ap.add_argument(
        "--src",
        type=pathlib.Path,
        default=DEFAULT_SRC,
        help="rollout jsonl the SFT set was built from (for the gold answers)",
    )
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    records = json.loads(args.sft.read_text(encoding="utf-8"))
    index = _source_index(args.src)

    problems, answers, images = [], [], []
    seen_ids = set()
    for i, rec in enumerate(records):
        human = rec["conversations"][0]["value"]
        if not human.startswith(IMAGE_PLACEHOLDER):
            raise SystemExit(f"record {i}: human turn does not start with {IMAGE_PLACEHOLDER}")
        prompt_text = human[len(IMAGE_PLACEHOLDER) :]

        image_path = pathlib.Path(rec["images"][0])
        image_bytes = image_path.read_bytes()
        meta = index.get((_md5_bytes(image_bytes), prompt_text))
        if meta is None:
            raise SystemExit(f"record {i}: (image, prompt) not found in {args.src}")

        if not prompt_text.endswith(" " + INSTRUCTION):
            raise SystemExit(f"record {i}: prompt does not end with ' ' + the Geo3K instruction")
        # Drop exactly the joining space, not the question's own trailing
        # whitespace: 11 questions end with "\r\n", and the loader rebuilds the
        # prompt as f"{problem} {instruction}", so those must be preserved.
        problem = prompt_text[: -len(INSTRUCTION) - 1]

        seen_ids.add(meta["id"])
        problems.append(problem)
        answers.append(meta["gold"])
        images.append([{"bytes": image_bytes, "path": None}])

    features = Features(
        {
            "problem": Value("string"),
            "answer": Value("string"),
            "images": Sequence(Image()),
        }
    )
    dataset = Dataset.from_dict({"problem": problems, "answer": answers, "images": images}, features=features)

    dst = args.out / "geo3k" / "test.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(dst))
    _normalize_feature_types(dst)

    print(f"sft     : {args.sft}  ({len(records)} records)")
    print(f"source  : {args.src}  ({len(index)} rows, {len(seen_ids)} matched)")
    print(f"wrote   : {dst}  ({len(dataset)} rows)")
    print(f"          images will be written to {args.out / 'geometry3k_images'} at load time")
    print(f"sample  : problem={problems[0][:70]!r} answer={answers[0]!r}")


if __name__ == "__main__":
    main()
