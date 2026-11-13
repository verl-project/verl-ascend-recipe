#!/usr/bin/env python3
"""Cast a MindSpeed-MM training export (all-F32) to bf16, tensor-wise.

Per-tensor policy (decided by ``--dtype-ref``, an already-converted bf16
checkpoint of the same architecture):

* tensors that are **F32 in the reference** stay F32 -- the numerically
  sensitive ``hc_*_base/scale`` (hyper-connections), KDA ``A_log``/``dt_bias``
  and MoE ``e_score_correction_bias``;
* every other F32 tensor is cast to **BF16** (round-to-nearest-even via
  ``torch.Tensor.to``).

Sharding is preserved 1:1 (``model-XXXXX-of-YYYYY.safetensors`` names are
reused), every non-shard file in ``--src`` is copied verbatim, and the output
index's ``total_size`` is recomputed from the actual bf16 tensor bytes.

Exits 2 when ``--dst`` exists and is non-empty unless ``--overwrite`` is
given (checked before anything is written, including in ``--dry-run``).

The ``read_header``/``scan_dtypes`` helpers are importable for reuse by other
tooling.

Usage::

    python3 convert_ckpt_bf16.py --src ITER_DIR --dst OUT_DIR --dtype-ref REF
    python3 convert_ckpt_bf16.py --src ... --dst ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

# CPU-only torch: must be set before torch is first imported (this module is
# also exec'd by the driver, which sets it too, but be self-sufficient).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

ELEMENT_SIZE = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
SHARD_GLOB = "model-*-of-*.safetensors"


# ---------------------------------------------------------------- helpers
def read_header(path: str) -> dict:
    """Parse a safetensors file header without reading tensor data."""
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(header_len))


def scan_dtypes(directory: str) -> dict[str, str]:
    """{tensor_name: dtype} over every shard in ``directory``."""
    out: dict[str, str] = {}
    for shard in sorted(Path(directory).glob(SHARD_GLOB)):
        for name, meta in read_header(str(shard)).items():
            if name != "__metadata__":
                out[name] = meta["dtype"]
    return out


def _shards(directory: Path) -> list[Path]:
    return sorted(directory.glob(SHARD_GLOB))


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


# ---------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tensor-wise F32->bf16 cast preserving the reference's F32 set.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--dtype-ref", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    ref: Path = args.dtype_ref.resolve()

    for label, path in (("src", src), ("--dtype-ref", ref)):
        if not path.is_dir():
            print(f"error: {label} is not a directory: {path}", file=sys.stderr)
            return 1

    # guard BEFORE anything else, including --dry-run
    if dst.exists() and any(dst.iterdir()) and not args.overwrite:
        print(f"error: --dst exists and is not empty: {dst} (pass --overwrite to replace)", file=sys.stderr)
        return 2
    if src == dst:
        print("error: --dst must differ from --src", file=sys.stderr)
        return 2

    src_shards = _shards(src)
    if not src_shards:
        print(f"error: no {SHARD_GLOB} in {src}", file=sys.stderr)
        return 1

    src_dtypes = scan_dtypes(str(src))
    ref_dtypes = scan_dtypes(str(ref))
    keep_f32 = {n for n, d in ref_dtypes.items() if d == "F32"} & set(src_dtypes)

    n_bf16 = len(src_dtypes) - len(keep_f32)
    print(f"src        : {src}")
    print(f"dst        : {dst}")
    print(f"dtype-ref  : {ref}")
    print(f"tensors    : {len(src_dtypes)} total -> {n_bf16} BF16, {len(keep_f32)} F32 preserved")
    print(f"shards     : {len(src_shards)}")

    odd = {n: d for n, d in src_dtypes.items() if d != "F32"}
    if odd:
        print(
            f"error: src contains non-F32 tensors (expected an all-F32 export): {list(odd.items())[:3]}",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("dry-run: no weights written.")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    done_bytes = 0
    for i, shard in enumerate(src_shards, 1):
        tensors = load_file(str(shard))
        out = {}
        for name, tensor in tensors.items():
            if tensor.dtype == torch.float32 and name not in keep_f32:
                out[name] = tensor.to(torch.bfloat16)
            else:
                out[name] = tensor
        out_path = dst / shard.name
        save_file(out, str(out_path), metadata={"format": "pt"})
        written += 1
        done_bytes += out_path.stat().st_size
        print(
            f"  [{i:3d}/{len(src_shards)}] {shard.name}  "
            f"({done_bytes / 2**30:.2f} GiB written, {time.time() - t0:.0f}s)",
            flush=True,
        )
        del tensors, out

    # copy every non-shard file verbatim (config/tokenizer/index/aux)
    for entry in sorted(src.iterdir()):
        if entry.is_file() and not entry.name.startswith("model-"):
            shutil.copy2(entry, dst / entry.name)

    # recompute index total_size from the converted tensors
    index_path = dst / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        total = 0
        for shard in _shards(dst):
            for name, meta in read_header(str(shard)).items():
                if name != "__metadata__":
                    total += ELEMENT_SIZE[meta["dtype"]] * _numel(meta["shape"])
        index.setdefault("metadata", {})["total_size"] = total
        index_path.write_text(json.dumps(index, indent=2))

    print(f"done: {written} shards, {done_bytes / 2**30:.2f} GiB, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
