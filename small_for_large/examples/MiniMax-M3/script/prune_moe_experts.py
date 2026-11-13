#!/usr/bin/env python3
"""Prune MiniMax-M3 routed MoE experts, keeping every layer.

Modelled on the parent directory's ``expert_pruned.py`` (the GLM-5.3 path), but
MiniMax-M3's checkpoint uses a different layout, so the name patterns had to change:

    GLM-5.3   model.language_model.layers.<L>.mlp.experts.<E>.{gate,up,down}_proj.weight
              model.language_model.layers.<L>.mlp.gate.weight

    MiniMax   language_model.model.layers.<L>.block_sparse_moe.experts.<E>.w{1,2,3}.weight
              language_model.model.layers.<L>.block_sparse_moe.gate.weight
              language_model.model.layers.<L>.block_sparse_moe.e_score_correction_bias

What gets rewritten, per MoE layer:

* ``...experts.<E>.w{1,2,3}.weight`` -- kept experts renumbered to ``0..top_k-1``
* ``...block_sparse_moe.gate.weight``  -- sliced on dim 0 (one row per expert)
* ``...block_sparse_moe.e_score_correction_bias`` -- sliced on dim 0 (one entry per expert)
* ``config.json`` -- ``num_local_experts`` rewritten to ``top_k``, and
  ``num_experts_per_tok`` clamped to ``top_k`` (a top-4 router cannot select 4
  experts out of 2, so leaving it alone would produce a model whose gate is a
  no-op). Override with ``--experts-per-tok``.

Everything else is copied verbatim: the dense layers, ``shared_experts.*``, the
sparse-attention indexer, the vision tower, ``lm_head``, the tokenizer/processor
assets and the custom ``*.py`` the checkpoint carries.

The output keeps the *same naming convention as the input* rather than the newer
transformers>=5.12 layout, because that is what VeOmni's
``MiniMaxM3VLCheckpointTensorConverter`` recognises on load.

``<model_dir>/MiniMax-M3`` is 795.5 GiB of bf16 tensors (60 layers, 57 MoE,
128 experts each); routed experts are ~97% of it, so ``--top-k`` is the size
lever. Every MoE layer must keep the same ``--top-k``; see the README for the
size table.

Usage:
    python prune_moe_experts.py --dry-run --top-k 2
    python prune_moe_experts.py --top-k 2 \\
        --output-dir <output_dir>/MiniMax-M3-e2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

DEFAULT_MODEL_DIR = "<model_dir>/MiniMax-M3"
DEFAULT_SHARD_SIZE_GB = 4.5
INDEX_NAME = "model.safetensors.index.json"

# ``<anything>.layers.<L>.block_sparse_moe.experts.<E>.w{1,2,3}.weight``
EXPERT_RE = re.compile(
    r"^(?P<stem>.+\.layers\.(?P<layer>\d+)\.block_sparse_moe)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.weight$"
)
# Router tensors for the same layer: one row (or entry) per expert, sliced on dim 0.
ROUTER_RE = re.compile(r"^.+\.layers\.(?P<layer>\d+)\.block_sparse_moe\.(?:gate\.weight|e_score_correction_bias)$")
# MiniMax's sparse-attention indexer (``self_attn.index_{q,k}_{proj,norm}.weight``).
# It is not expert-structured and must never be renamed or sliced; the guard below
# asserts that rather than relying on the patterns simply not overlapping.
INDEXER_RE = re.compile(r"\.self_attn\.index_")
# Keys in config.json that encode the routed-expert count (nested under text_config).
CONFIG_EXPERT_KEYS = ("num_local_experts", "num_experts_per_tok")
# Files describing the *old* weights; never copy these into the output.
SKIP_COPY_SUFFIXES = (".safetensors", ".index.json.bak", ".index.json")

_DTYPE_SIZE = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.2f} TiB"


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _dtype_size(dtype) -> int:
    key = str(dtype).rsplit(".", 1)[-1]
    if key not in _DTYPE_SIZE:
        raise SystemExit(f"unknown dtype {dtype!r}")
    return _DTYPE_SIZE[key]


def _config_ints(node: Any, key: str):
    """Yield (key, value) for every int ``key`` anywhere in the config tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, int):
                yield k, v
            yield from _config_ints(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _config_ints(v, key)


def _config_lists(node: Any, key: str):
    """Yield (key, value) for every list ``key`` anywhere in the config tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, list):
                yield k, v
            yield from _config_lists(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _config_lists(v, key)


def _discover(model_dir: Path, index_name: str | None) -> tuple[list[Path], dict[str, str]]:
    """Return (shard paths, weight_map), handling both a sharded index and one flat file."""
    index_path = model_dir / (index_name or INDEX_NAME)
    if index_path.is_file():
        weight_map = _load_json(index_path)["weight_map"]
        return sorted({model_dir / n for n in weight_map.values()}), weight_map

    flat = model_dir / "model.safetensors"
    if not flat.is_file():
        raise SystemExit(f"found neither {INDEX_NAME} nor model.safetensors under {model_dir}")
    with safe_open(str(flat), framework="pt") as f:
        names = list(f.keys())
    return [flat], {n: "model.safetensors" for n in names}


def _scan_experts(weight_map: dict[str, str]) -> dict[int, set[int]]:
    per_layer: dict[int, set[int]] = {}
    for name in weight_map:
        m = EXPERT_RE.match(name)
        if m:
            per_layer.setdefault(int(m.group("layer")), set()).add(int(m.group("expert")))
    return per_layer


def _interleave(ids: list[int], stage_length: int) -> list[int]:
    """Spread kept experts so high-ranked ones do not all land on low compact ids.

    Mirrors the GLM tool's ``--stage-length``.
    """
    if stage_length <= 0 or len(ids) <= stage_length:
        return ids
    stages = [ids[i : i + stage_length] for i in range(0, len(ids), stage_length)]
    out: list[int] = []
    for col in range(max(len(s) for s in stages)):
        out.extend(stage[col] for stage in stages if col < len(stage))
    return out


def _parse_topk_json(path: Path, top_k: int, num_experts: int, stage_length: int) -> dict[int, list[int]]:
    """GLM-compatible per-layer topk_ids file. Keys only need to contain a layer number."""
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must be a JSON object keyed by layer")
    out: dict[int, list[int]] = {}
    for key, value in raw.items():
        m = re.search(r"(?:layers\.|layer_|^)(\d+)", str(key))
        if m is None:
            continue
        ids = value.get("topk_ids") if isinstance(value, dict) else value
        if not isinstance(ids, list):
            raise SystemExit(f"{key}: expected a list or {{'topk_ids': [...]}}")
        if len(ids) < top_k:
            raise SystemExit(f"{key}: has {len(ids)} ids, fewer than --top-k {top_k}")
        chosen = [int(i) for i in ids[:top_k]]
        bad = [i for i in chosen if not 0 <= i < num_experts]
        if bad:
            raise SystemExit(f"{key}: ids outside [0, {num_experts}): {bad[:8]}")
        out[int(m.group(1))] = _interleave(chosen, stage_length)
    if not out:
        raise SystemExit(f"{path}: no key parsed as a layer number")
    return out


def _rank_by_score_bias(handles, model_dir: Path, weight_map: dict[str, str], layer: int) -> list[int]:
    """Most-used expert first, via the aux-loss-free load-balancing bias.

    ``e_score_correction_bias`` is nudged *down* for experts the router picks too
    often, so ascending bias ~ descending popularity. A heuristic read off the
    pretrained router, not a measured activation frequency.
    """
    target = f".layers.{layer}.block_sparse_moe.e_score_correction_bias"
    name = next((n for n in weight_map if n.endswith(target)), None)
    if name is None:
        raise SystemExit(f"--rank-by score_bias: no tensor ending in {target!r}")
    bias = handles[str(model_dir / weight_map[name])].get_tensor(name).float()
    return sorted(range(bias.numel()), key=lambda i: float(bias[i]))


def _apply_config_update(cfg: dict, updates: dict[str, int]) -> tuple[dict, dict[str, int]]:
    """Rewrite the given keys, but only inside ``text_config``.

    Scoped on purpose. The top-level config also carries a ``vision_config`` that
    has its own ``num_hidden_layers`` -- 32 for the vision tower vs the text
    tower's 60. A whole-tree rewrite silently clobbers it, and then vLLM builds a
    60-layer vision tower and dies with "Following weights were not initialized
    from checkpoint". All three keys we touch live under ``text_config`` anyway.
    """
    hits: dict[str, int] = {}

    def walk(n: Any) -> Any:
        if isinstance(n, dict):
            out = {}
            for k, v in n.items():
                if k in updates and isinstance(v, int):
                    out[k] = updates[k]
                    hits[k] = hits.get(k, 0) + 1
                else:
                    out[k] = walk(v)
            return out
        if isinstance(n, list):
            return [walk(v) for v in n]
        return n

    out = dict(cfg)
    if isinstance(out.get("text_config"), dict):
        out["text_config"] = walk(out["text_config"])
    return out, hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="input HF model directory")
    ap.add_argument("--output-dir", help="output HF model directory (required unless --dry-run)")
    ap.add_argument("--top-k", type=int, required=True, help="experts to keep per MoE layer")
    ap.add_argument("--topk-json", help="GLM-compatible per-layer topk_ids JSON; overrides --rank-by")
    ap.add_argument(
        "--rank-by",
        choices=("index", "score_bias"),
        default="index",
        help="how to pick experts when --topk-json is absent (default: index = keep 0..N-1)",
    )
    ap.add_argument(
        "--experts-per-tok",
        type=int,
        default=None,
        help="override num_experts_per_tok in the output config (default: min(original, --top-k))",
    )
    ap.add_argument(
        "--stage-length",
        type=int,
        default=0,
        help="interleave kept ids by this stage length, like the GLM tool (0 = off)",
    )
    ap.add_argument("--shard-size-gb", type=float, default=DEFAULT_SHARD_SIZE_GB)
    ap.add_argument("--index-name", default=None, help=f"input index filename (default: {INDEX_NAME})")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report sizes and the plan; write nothing")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} is not a directory")

    shard_paths, weight_map = _discover(model_dir, args.index_name)
    experts_by_layer = _scan_experts(weight_map)
    if not experts_by_layer:
        raise SystemExit("no ``...block_sparse_moe.experts.<E>.w{1,2,3}.weight`` tensors found")

    indexer_names = [n for n in weight_map if INDEXER_RE.search(n)]
    clash = [n for n in indexer_names if EXPERT_RE.match(n) or ROUTER_RE.match(n)]
    if clash:
        raise SystemExit(f"refusing to run: the pruning patterns would touch indexer tensors: {clash[:3]}")

    cfg_path = model_dir / "config.json"
    cfg = _load_json(cfg_path) if cfg_path.is_file() else {}
    num_experts = next((v for _k, v in _config_ints(cfg, "num_local_experts")), None)
    experts_per_tok = next((v for _k, v in _config_ints(cfg, "num_experts_per_tok")), None)

    detected = {len(ids) for ids in experts_by_layer.values()}
    print(f"model dir           : {model_dir}")
    print(f"shards / tensors    : {len(shard_paths)} / {len(weight_map)}")
    print(f"num_local_experts   : {num_experts}")
    print(f"num_experts_per_tok : {experts_per_tok}")
    print(f"MoE layers          : {sorted(experts_by_layer)}")
    if indexer_names:
        print(f"indexer tensors     : {len(indexer_names)} (protected -- copied verbatim)")

    if num_experts is None:
        num_experts = max(detected)
        print(f"  (config has no num_local_experts; using {num_experts} from the tensors)")
    elif detected != {num_experts}:
        raise SystemExit(f"config says {num_experts} experts but tensors show {sorted(detected)} per layer")

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if args.top_k > num_experts:
        raise SystemExit(f"--top-k {args.top_k} exceeds the {num_experts} experts present")

    # A top-N router cannot pick N distinct experts out of fewer than N, so the
    # count has to come down with --top-k or the gate degenerates.
    if args.experts_per_tok is not None:
        experts_per_tok_out = args.experts_per_tok
        why = "from --experts-per-tok"
    elif experts_per_tok is not None:
        experts_per_tok_out = min(experts_per_tok, args.top_k)
        why = "clamped to --top-k; pass --experts-per-tok to override"
    else:
        experts_per_tok_out = None
        why = ""
    if experts_per_tok is not None and experts_per_tok_out != experts_per_tok:
        print(f"num_experts_per_tok   : {experts_per_tok} -> {experts_per_tok_out} ({why})")
    if experts_per_tok_out is not None and experts_per_tok_out > args.top_k:
        raise SystemExit(
            f"--experts-per-tok {experts_per_tok_out} > --top-k {args.top_k}: cannot select that many experts"
        )

    with ExitStack() as stack:
        handles = {str(p): stack.enter_context(safe_open(str(p), framework="pt")) for p in shard_paths}

        # ---- which experts survive ----------------------------------------
        if args.topk_json:
            keep_by_layer = _parse_topk_json(Path(args.topk_json), args.top_k, num_experts, args.stage_length)
            missing = sorted(set(experts_by_layer) - set(keep_by_layer))
            if missing:
                raise SystemExit(f"{args.topk_json} has no entry for MoE layer(s) {missing}")
            print(f"expert ids          : from {args.topk_json}")
        else:
            keep_by_layer = {}
            for layer in experts_by_layer:
                if args.rank_by == "score_bias":
                    ranks = _rank_by_score_bias(handles, model_dir, weight_map, layer)
                else:
                    ranks = list(range(num_experts))
                keep_by_layer[layer] = _interleave(ranks[: args.top_k], args.stage_length)
            print(f"expert ids          : keep {args.top_k} of {num_experts} (rank-by={args.rank_by})")
        distinct = {tuple(ids) for ids in keep_by_layer.values()}
        if len(distinct) == 1:
            print(f"  all {len(keep_by_layer)} MoE layers: {next(iter(distinct))}")
        else:
            for layer in sorted(keep_by_layer):
                print(f"  layer {layer}: {keep_by_layer[layer]}")

        remap = {layer: {old: new for new, old in enumerate(ids)} for layer, ids in keep_by_layer.items()}

        # ---- size accounting ----------------------------------------------
        total_bytes = kept_bytes = 0
        for name, shard in weight_map.items():
            sl = handles[str(model_dir / shard)].get_slice(name)
            nbytes = _numel(sl.get_shape()) * _dtype_size(sl.get_dtype())
            total_bytes += nbytes

            m = EXPERT_RE.match(name)
            if m:
                if int(m.group("expert")) in remap.get(int(m.group("layer")), {}):
                    kept_bytes += nbytes
                continue
            if ROUTER_RE.match(name):
                first_dim = sl.get_shape()[0]
                kept_bytes += round(nbytes / max(first_dim, 1) * args.top_k)
                continue
            kept_bytes += nbytes

        print(f"\noriginal            : {_human(total_bytes)}")
        print(
            f"pruned              : {_human(kept_bytes)}  "
            f"(dropped {_human(total_bytes - kept_bytes)}, {100 * kept_bytes / total_bytes:.1f}% remains)"
        )

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return

        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir is None:
            raise SystemExit("--output-dir is required unless --dry-run")
        if output_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"{output_dir} exists; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # ---- stream tensors into shards ------------------------------------
        shard_limit = int(args.shard_size_gb * 1024**3)
        buffer: dict[str, Any] = {}
        buffer_bytes = 0
        shard_names: list[str] = []
        new_weight_map: dict[str, str] = {}
        written = 0

        def flush() -> None:
            nonlocal buffer, buffer_bytes
            if not buffer:
                return
            # Placeholder name; the final ``-of-<total>`` suffix needs the shard
            # count, which is only known once everything is written (see below).
            fname = f"model-{len(shard_names) + 1:05d}.safetensors"
            safetensors_save_file(buffer, str(output_dir / fname), metadata={"format": "pt"})
            new_weight_map.update({k: fname for k in buffer})
            shard_names.append(fname)
            buffer, buffer_bytes = {}, 0

        for name, shard in weight_map.items():
            m = EXPERT_RE.match(name)
            if m:
                new_id = remap.get(int(m.group("layer")), {}).get(int(m.group("expert")))
                if new_id is None:
                    continue
                new_name = f"{m.group('stem')}.experts.{new_id}.{m.group('proj')}.weight"
                tensor = handles[str(model_dir / shard)].get_tensor(name)
            else:
                r = ROUTER_RE.match(name)
                if r is None:
                    new_name = name
                    tensor = handles[str(model_dir / shard)].get_tensor(name)
                else:
                    import torch

                    ids = keep_by_layer[int(r.group("layer"))]
                    tensor = (
                        handles[str(model_dir / shard)]
                        .get_tensor(name)
                        .index_select(0, torch.tensor(ids, dtype=torch.long))
                        .contiguous()
                    )
                    new_name = name

            nbytes = tensor.numel() * tensor.element_size()
            if buffer and buffer_bytes + nbytes > shard_limit:
                flush()
            buffer[new_name] = tensor
            buffer_bytes += nbytes
            written += nbytes

        flush()

        # Rename to the HF convention ``model-<idx>-of-<total>.safetensors``.
        # VeOmni's ``shard_index_from_filename`` does ``int(name.split("-")[1])``,
        # which reads ``model-00001.safetensors`` as ``"00001.safetensors"`` and
        # raises, so the ``-of-<total>`` field has to be present.
        total = len(shard_names)
        renamed = {}
        for old in shard_names:
            idx = old.split("-", 1)[1].split(".", 1)[0]
            new = f"model-{idx}-of-{total:05d}.safetensors"
            (output_dir / old).rename(output_dir / new)
            renamed[old] = new
        new_weight_map = {k: renamed[v] for k, v in new_weight_map.items()}
        shard_names = [renamed[s] for s in shard_names]

    _write_json(output_dir / INDEX_NAME, {"metadata": {"total_size": written}, "weight_map": new_weight_map})

    # ---- the non-weight side ----------------------------------------------
    for src in sorted(model_dir.iterdir()):
        if src.is_dir() or src.name == "config.json":
            continue
        if any(src.name.endswith(s) for s in SKIP_COPY_SUFFIXES):
            continue
        shutil.copy2(src, output_dir / src.name)

    cfg_note = "config.json copied as-is"
    if cfg_path.is_file():
        updates = {"num_local_experts": args.top_k}
        if experts_per_tok_out is not None:
            updates["num_experts_per_tok"] = experts_per_tok_out
        # MiniMax-M3's public config carries a stale ``num_hidden_layers`` (12) while
        # ``moe_layer_freq`` / ``sparse_attention_freq`` and the weight shards all
        # describe 60. VeOmni builds exactly ``range(config.num_hidden_layers)``
        # layers, so the stale value makes the checkpoint unloadable. Derive it from
        # the per-layer lists whenever they disagree.
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
        derived = next((len(v) for _k, v in _config_lists(tc, "moe_layer_freq")), None)
        current = next((v for _k, v in _config_ints(tc, "num_hidden_layers")), None)
        if derived is not None and derived != current:
            print(f"num_hidden_layers   : {current} -> {derived} (derived from len(moe_layer_freq))")
            updates["num_hidden_layers"] = derived
        updated, hits = _apply_config_update(cfg, updates)
        _write_json(output_dir / "config.json", updated)
        for key in updates:
            if hits.get(key, 0) == 0:
                print(f"  !! config.json had no {key!r} key -- not written")
        cfg_note = "config.json (" + ", ".join(f"{k}={v}" for k, v in updates.items()) + ")"

    side = len([p for p in output_dir.iterdir() if p.name not in shard_names and p.name != INDEX_NAME])
    print(f"\nwrote {output_dir}")
    print(f"  {len(shard_names)} shard(s), {_human(written)}, {len(new_weight_map)} tensors")
    print(f"  {INDEX_NAME} + {cfg_note} + {side} side files")


if __name__ == "__main__":
    main()
