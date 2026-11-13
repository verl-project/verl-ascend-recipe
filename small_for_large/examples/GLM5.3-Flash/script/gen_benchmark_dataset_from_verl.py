#!/usr/bin/env python
"""Convert a verl/RL-style dataset parquet into the layout ais_bench expects.

Dataset-agnostic (``--dataset``, default ``geo3k``). The source parquet uses
the verl RL schema (``images``/``prompt``/``reward_model``/``extra_info``);
ais_bench's ``Geometry3KDataset`` needs the HF ``problem``/``answer``/``images``
layout. The instruction suffix is appended by ``Geometry3KDataset`` itself, so
``problem`` must not contain it: we take the raw question from
``extra_info.question`` rather than the already-suffixed ``prompt``.

Output: ``<dataset>_bench/<dataset>/test.parquet`` (plus a sibling
``geometry3k_images/`` written by ``Geometry3KDataset`` at load time).

Usage:
    python gen_benchmark_dataset_from_verl.py                            # geo3k, test only
    python gen_benchmark_dataset_from_verl.py --splits train test        # one <dataset>_bench_<split>/ tree each
    python gen_benchmark_dataset_from_verl.py --dataset foo              # any other verl-style dataset
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, Features, Image, Sequence, Value

HERE = Path(__file__).resolve().parent

# Geometry3KDataset does ``problem.replace("<image>", "", 1).lstrip()``.
IMAGE_PLACEHOLDER = "<image>"


def _normalize_feature_types(path: Path) -> None:
    """Rewrite the embedded HF feature metadata in the canonical "Sequence" form.

    ``datasets`` >= 4.x serialises ``Sequence(Image())`` as ``_type: "List"``,
    but ``datasets`` <= 3.x (e.g. the 3.6.0 in the ais_bench runtime env) only
    knows ``"Sequence"`` and raises ``Feature type 'List' not found`` on read.
    Both generations read ``"Sequence"``, and it is what the upstream
    hiyouga/geometry3k parquet uses, so normalise to it.  No-op on old
    ``datasets``.
    """
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

    # Layout is {"info": {"features": {...}}} -- `datasets` nests it this way.
    info = json.loads(raw)
    _fix(info.get("info", {}).get("features", {}))
    metadata[b"huggingface"] = json.dumps(info).encode()
    pq.write_table(table.replace_schema_metadata(metadata), str(path))


def convert_split(src: Path, dst: Path) -> int:
    """Read one parquet split and write it in geometry3k/HF layout.

    Returns the number of converted rows.
    """
    problems, answers, images = [], [], []

    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            extra_info = row.get("extra_info") or {}

            # Raw question, without the instruction suffix baked into `prompt`.
            problem = extra_info.get("question") or ""
            problem = problem.replace(IMAGE_PLACEHOLDER, "", 1).lstrip()

            # `reward_model.ground_truth` is the answer; `ground_truth` (top level)
            # is only the multiple-choice letter and must not be used.
            reward_model = row.get("reward_model") or {}
            answer = reward_model.get("ground_truth")
            if answer is None or str(answer).strip() == "":
                answer = extra_info.get("answer", "")

            # Embedded image bytes -> keep the encoded form so `datasets` decodes
            # them into PIL on read, exactly like the original geometry3k parquet.
            row_images = [
                {"bytes": img["bytes"], "path": None} for img in (row.get("images") or []) if img.get("bytes")
            ]

            problems.append(problem)
            answers.append(str(answer))
            images.append(row_images)

    features = Features(
        {
            "problem": Value("string"),
            "answer": Value("string"),
            "images": Sequence(Image()),
        }
    )
    dataset = Dataset.from_dict(
        {"problem": problems, "answer": answers, "images": images},
        features=features,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(dst))
    _normalize_feature_types(dst)

    missing = sum(1 for imgs in images if not imgs)
    if missing:
        print(f"  WARNING: {missing}/{len(images)} rows have no image")
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="geo3k", help="dataset name; also used to build the <dataset>_bench output tree"
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Directory holding the <dataset>.parquet files (default: <out-dir>/<dataset>)",
    )
    parser.add_argument("--out-dir", type=Path, default=HERE, help="Directory to write the ais_bench variants into")
    parser.add_argument("--splits", nargs="+", default=["test"], help="Which splits to convert")
    args = parser.parse_args()

    src_dir = args.src_dir or args.out_dir / args.dataset

    # Geometry3KDataset writes decoded PNGs to
    # ``Path(parquet_file).parent.parent / "geometry3k_images"``, naming them
    # ``{index}.png`` after the position within the split.  That grandparent dir
    # is therefore the only thing that separates two splits' images, so a single
    # split lands in ``<dataset>_bench/`` while converting several gets one tree
    # per split -- otherwise the index-based names would overwrite each other.
    multi = len(args.splits) > 1

    for split in args.splits:
        src = src_dir / f"{split}.parquet"
        if not src.is_file():
            raise FileNotFoundError(f"source split not found: {src}")

        root = args.out_dir / (f"{args.dataset}_bench_{split}" if multi else f"{args.dataset}_bench")
        dst = root / args.dataset / f"{split}.parquet"

        count = convert_split(src, dst)
        print(f"{split}: {count} rows -> {dst}")
        print(f"         images will be written to {root / 'geometry3k_images'} at load time")


if __name__ == "__main__":
    main()
