#!/usr/bin/env python
"""Post-pruning verification gate for MoE checkpoints produced by ``expert_pruned.py``.

The prune step in the ``small_for_large`` recipe drops routed experts and
renumbers the kept ones. That step currently verifies nothing about its own
output, and a wrong or partially-collected ``topk_experts.json`` is silently
accepted: any layer missing from the JSON falls back to "keep experts
``0..N-1``" without a single warning. This companion tool closes that gap.

What it verifies
----------------
1. Source integrity: the source model directory still matches a fingerprint
   snapshot taken before pruning, so the prune provably never touched the
   original weights.
2. Kept-expert fidelity: for every pruned layer, every kept expert tensor in
   the pruned checkpoint is byte-equal (``torch.equal``) to the mapped slice
   of the source tensor, applying exactly the same remapping
   ``expert_pruned.py`` performs (sort the keep list, renumber by position).
3. Router fidelity: ``gate.weight`` / ``gate.bias`` rows of kept experts are
   byte-equal to the source rows selected by the keep list.
4. Coverage: every prunable layer is present in the topk JSON. A missing
   layer means the activation collection was incomplete; this tool reports
   it as a failure (downgradable to a warning with ``--allow-missing-layers``
   to reproduce the current silent fallback).
5. Config diff: only the routed-expert-count key(s) changed.
6. Stale expert-dim heuristic: passthrough tensors in pruned layers that
   still carry a dimension equal to the old expert count (for example a
   score-correction bias that was never sliced) are reported for review.
7. ``PROVENANCE.json``: a machine-readable record of all checks above,
   written next to the pruned checkpoint.

Exit code is 0 only when every enabled check passes, so the tool can be used
as a hard gate inside an automated pipeline.

Usage
-----
# 1) before pruning: snapshot the source directory fingerprint
python verify_pruned.py --source /path/model-bf16 \
    --snapshot source_fp.json --fingerprint-only

# 2) prune as usual (expert_pruned.py ...)

# 3) after pruning: verify
python verify_pruned.py --source /path/model-bf16 \
    --pruned /path/model-bf16-top64-pruned \
    --json-path top64_experts.json --top-k 64 \
    --snapshot source_fp.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import torch
from safetensors import safe_open

# Files larger than this are fingerprinted by (size, mtime_ns) only; smaller
# files get a full sha256. Byte-comparing every shard of a 500GB model would
# take hours, and the per-tensor checks below already cover shard content.
SMALL_FILE_HASH_LIMIT = 64 * 1024 * 1024

EXPERT_KEY_RE = re.compile(
    r"^(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.experts\.(\d+)\."
    r"(?:w[123]|gate_proj|up_proj|down_proj)\.weight$"
)
GATE_KEY_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.(?:ffn|mlp)\.gate\.(weight|bias)$")
LAYER_IN_KEY_RE = re.compile(r"layers\.(\d+)")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_dir(model_dir: str) -> dict:
    """Fingerprint a model directory: full sha256 for small files, (size, mtime_ns) for big ones."""
    files = {}
    for root, _, names in os.walk(model_dir):
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, model_dir).replace("\\", "/")
            st = os.stat(path)
            entry = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            if st.st_size <= SMALL_FILE_HASH_LIMIT:
                entry["sha256"] = sha256_file(path)
            files[rel] = entry
    return {
        "dir": os.path.abspath(model_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def diff_fingerprint(recorded: dict, current: dict) -> list[str]:
    """Return a human-readable list of differences between two fingerprints."""
    diffs = []
    for rel, entry in sorted(recorded["files"].items()):
        cur = current["files"].get(rel)
        if cur is None:
            diffs.append(f"deleted: {rel}")
            continue
        for field in entry:
            if entry[field] != cur.get(field):
                diffs.append(f"changed: {rel} ({field})")
    for rel in sorted(current["files"]):
        if rel not in recorded["files"]:
            diffs.append(f"added: {rel}")
    return diffs


def load_keep_dict(json_path: str, top_k: int) -> dict[int, list[int]]:
    """Load and normalize the keep list, mirroring expert_pruned.py semantics.

    expert_pruned.py keeps ``sorted(set(topk_ids))`` per layer and renumbers
    experts by position in that sorted list, so the same normalization is
    applied here before any comparison.
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    keep: dict[int, list[int]] = {}
    for key, val in raw.items():
        m = LAYER_IN_KEY_RE.search(key)
        if not m:
            print(f"[warn] unparseable layer key in JSON: {key}")
            continue
        layer = int(m.group(1))
        ids = val["topk_ids"] if isinstance(val, dict) else val
        ids = sorted({int(x) for x in ids})
        if len(ids) != top_k:
            sys.exit(f"[fail] layer {layer}: keep list has {len(ids)} experts, expected {top_k}")
        keep[layer] = ids
    return keep


def build_pruned_layer_plans(
    num_layers: int,
    keep: dict[int, list[int]],
    skip_layers: set[int],
    old_n: int,
    top_k: int,
    allow_missing: bool,
) -> tuple[dict[int, list[int]], list[int]]:
    """Resolve the effective keep list per layer and flag missing layers."""
    missing = [i for i in range(num_layers) if i not in skip_layers and i not in keep]
    if missing and not allow_missing:
        sys.exit(
            "[fail] layers missing from the topk JSON "
            f"(expert_pruned.py silently keeps experts 0..{top_k - 1} for these): {missing}"
        )
    plans = {}
    for i in range(num_layers):
        if i in skip_layers:
            plans[i] = list(range(old_n))  # identity: no remap, all experts kept
        else:
            plans[i] = keep.get(i, list(range(top_k)))
    return plans, missing


def source_key_for(pruned_key: str, layer: int, plan: list[int]) -> str | None:
    """Map a pruned expert key back to the source key it must be byte-equal to.

    Returns None when the pruned expert id is out of range of the plan, which
    is itself a failure the caller reports.
    """
    m = EXPERT_KEY_RE.match(pruned_key)
    if not m:
        return None
    new_idx = int(m.group(2))
    if new_idx >= len(plan):
        return None
    old_idx = plan[new_idx]
    return re.sub(r"experts\.\d+", f"experts.{old_idx}", pruned_key, count=1)


def slice_by_size(tensor: torch.Tensor, size: int, idx: list[int]):
    """Slice the first dimension whose length equals ``size`` (gate weight/bias)."""
    for dim, length in enumerate(tensor.shape):
        if length == size:
            sl = [slice(None)] * tensor.dim()
            sl[dim] = idx
            return tensor[tuple(sl)]
    if tensor.numel() == size:
        return tensor[idx]
    raise ValueError(f"no dimension of length {size} to slice")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="original (unpruned) model directory, read-only")
    parser.add_argument("--pruned", help="pruned model directory produced by expert_pruned.py")
    parser.add_argument("--json-path", help="topk_experts.json used for pruning")
    parser.add_argument("--top-k", type=int, default=64, help="expert count kept per pruned layer")
    parser.add_argument(
        "--skip-layers",
        default="0,1,2",
        help="comma-separated layers kept unpruned (hash routing)",
    )
    parser.add_argument(
        "--config-keys",
        default="n_routed_experts",
        help="config keys allowed to change",
    )
    parser.add_argument("--snapshot", help="source fingerprint JSON taken before pruning (step 1)")
    parser.add_argument(
        "--allow-missing-layers",
        action="store_true",
        help="downgrade missing-layer failure to warning",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=1,
        help="verify every Nth expert of unpruned layers",
    )
    parser.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="only write the source fingerprint snapshot",
    )
    parser.add_argument(
        "--provenance",
        help="explicit PROVENANCE.json path (default: <pruned>/PROVENANCE.json)",
    )
    args = parser.parse_args()

    skip_layers = {int(x) for x in args.skip_layers.split(",") if x != ""}

    if args.fingerprint_only:
        if not args.snapshot:
            sys.exit("[fail] --fingerprint-only requires --snapshot <path>")
        fp = fingerprint_dir(args.source)
        with open(args.snapshot, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=2)
        print(f"[ok] source fingerprint written: {args.snapshot} ({len(fp['files'])} files)")
        return

    for flag in ("pruned", "json-path"):
        if not getattr(args, flag.replace("-", "_")):
            sys.exit(f"[fail] --{flag} is required unless --fingerprint-only is set")

    report: dict = {
        "tool": "verify_pruned.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"dir": os.path.abspath(args.source)},
        "pruned": {"dir": os.path.abspath(args.pruned)},
        "keep_json": {
            "path": os.path.abspath(args.json_path),
            "sha256": sha256_file(args.json_path),
        },
        "checks": {},
    }
    failures: list[str] = []

    # ---- 1. source integrity -------------------------------------------------
    if args.snapshot and os.path.isfile(args.snapshot):
        with open(args.snapshot, encoding="utf-8") as f:
            recorded = json.load(f)
        diffs = diff_fingerprint(recorded, fingerprint_dir(args.source))
        report["checks"]["source_untouched"] = not diffs
        report["checks"]["source_diffs"] = diffs[:50]
        if diffs:
            failures.append(f"source directory changed since snapshot: {diffs[:5]}")
        print(f"[{'ok' if not diffs else 'FAIL'}] source integrity: {len(diffs)} difference(s)")
    else:
        report["checks"]["source_untouched"] = None
        print("[warn] no pre-prune fingerprint snapshot given; source integrity recorded but not proven")

    # ---- 2. config diff ------------------------------------------------------
    with open(os.path.join(args.source, "config.json"), encoding="utf-8") as f:
        cfg_src = json.load(f)
    with open(os.path.join(args.pruned, "config.json"), encoding="utf-8") as f:
        cfg_pruned = json.load(f)
    allowed = {k.strip() for k in args.config_keys.split(",") if k.strip()}
    cfg_diff = {
        k: [cfg_src.get(k), cfg_pruned.get(k)]
        for k in sorted(set(cfg_src) | set(cfg_pruned))
        if cfg_src.get(k) != cfg_pruned.get(k)
    }
    bad_keys = sorted(set(cfg_diff) - allowed)
    report["checks"]["config_diff"] = cfg_diff
    if bad_keys:
        failures.append(f"config changed keys outside the allowlist {sorted(allowed)}: {bad_keys}")
    print(f"[{'ok' if not bad_keys else 'FAIL'}] config diff: {cfg_diff} (allowed: {sorted(allowed)})")

    old_n = cfg_src.get("n_routed_experts", cfg_src.get("moe_num_experts"))
    num_layers = cfg_src["num_hidden_layers"]
    if old_n is None:
        sys.exit("[fail] cannot find the routed-expert count in the source config.json")

    # ---- 3. coverage ---------------------------------------------------------
    keep = load_keep_dict(args.json_path, args.top_k)
    plans, missing = build_pruned_layer_plans(num_layers, keep, skip_layers, old_n, args.top_k, args.allow_missing)
    report["checks"]["layers_total"] = num_layers
    report["checks"]["layers_unpruned"] = sorted(skip_layers)
    report["checks"]["layers_missing_from_json"] = missing
    if missing:
        print(f"[{'warn' if args.allow_missing_layers else 'FAIL'}] layers missing from JSON: {missing}")
    else:
        print(f"[ok] coverage: all {num_layers - len(skip_layers)} prunable layers present in the JSON")

    # ---- 4. tensor-level verification ----------------------------------------
    with open(os.path.join(args.pruned, "model.safetensors.index.json"), encoding="utf-8") as f:
        pruned_index = json.load(f)["weight_map"]
    with open(os.path.join(args.source, "model.safetensors.index.json"), encoding="utf-8") as f:
        source_index = json.load(f)["weight_map"]

    stats = {
        "experts_checked": 0,
        "experts_byte_equal": 0,
        "gate_checked": 0,
        "gate_byte_equal": 0,
        "passthrough_checked": 0,
        "passthrough_byte_equal": 0,
        "stale_expert_dim_keys": [],
        "mismatches": [],
    }
    shard_cache: dict[str, safe_open] = {}

    def open_shard(model_dir: str, filename: str) -> safe_open:
        path = os.path.join(model_dir, filename)
        if path not in shard_cache:
            shard_cache[path] = safe_open(path, framework="pt", device="cpu")
        return shard_cache[path]

    # Iterate the pruned index: every tensor it declares must be accounted for.
    for key in sorted(pruned_index):
        layer_m = LAYER_IN_KEY_RE.search(key)
        layer = int(layer_m.group(1)) if layer_m else None
        pruned_tensor = open_shard(args.pruned, pruned_index[key]).get_tensor(key)
        plan = plans.get(layer)
        unpruned_layer = layer is None or layer in skip_layers

        expert_m = EXPERT_KEY_RE.match(key)
        gate_m = GATE_KEY_RE.match(key)
        if expert_m:
            if unpruned_layer:
                if int(expert_m.group(2)) % max(args.sample_step, 1) != 0:
                    continue
                src_key = key
            elif plan is None:
                failures.append(f"layer {layer} not covered by any plan: {key}")
                continue
            else:
                src_key = source_key_for(key, layer, plan)
                if src_key is None:
                    failures.append(f"pruned expert id out of range of the keep plan: {key}")
                    continue
            src_tensor = open_shard(args.source, source_index[src_key]).get_tensor(src_key)
            equal = src_tensor.shape == pruned_tensor.shape and torch.equal(src_tensor, pruned_tensor)
            stats["experts_checked"] += 1
            if equal:
                stats["experts_byte_equal"] += 1
            else:
                stats["mismatches"].append(f"expert mismatch: {key} != {src_key}")
        elif gate_m:
            if unpruned_layer or plan is None:
                src_key = key
                src_tensor = open_shard(args.source, source_index[src_key]).get_tensor(src_key)
            else:
                src_key = key
                src_full = open_shard(args.source, source_index[src_key]).get_tensor(src_key)
                src_tensor = slice_by_size(src_full, old_n, plan)
            equal = src_tensor.shape == pruned_tensor.shape and torch.equal(src_tensor, pruned_tensor)
            stats["gate_checked"] += 1
            if equal:
                stats["gate_byte_equal"] += 1
            else:
                stats["mismatches"].append(f"gate mismatch: {key}")
        else:
            src_key = key
            src_tensor = open_shard(args.source, source_index[src_key]).get_tensor(src_key)
            equal = src_tensor.shape == pruned_tensor.shape and torch.equal(src_tensor, pruned_tensor)
            stats["passthrough_checked"] += 1
            if equal:
                stats["passthrough_byte_equal"] += 1
            else:
                stats["mismatches"].append(f"passthrough mismatch: {key}")
            if not unpruned_layer and any(s == old_n for s in pruned_tensor.shape):
                stats["stale_expert_dim_keys"].append(key)

    report["checks"]["tensors"] = {k: v for k, v in stats.items() if k != "mismatches"}
    report["checks"]["mismatches"] = stats["mismatches"][:50]
    if stats["mismatches"]:
        failures.append(f"{len(stats['mismatches'])} tensor mismatch(es)")

    print(
        f"[{'ok' if not stats['mismatches'] else 'FAIL'}] tensors: experts "
        f"{stats['experts_byte_equal']}/{stats['experts_checked']} byte-equal, gate "
        f"{stats['gate_byte_equal']}/{stats['gate_checked']}, passthrough "
        f"{stats['passthrough_byte_equal']}/{stats['passthrough_checked']}"
    )
    if stats["stale_expert_dim_keys"]:
        print(
            f"[warn] passthrough tensors still carrying a dim == {old_n} in pruned layers "
            f"(possible unsliced expert-shaped tensors): {stats['stale_expert_dim_keys'][:10]}"
        )

    # ---- 5. provenance --------------------------------------------------------
    report["checks"]["failures"] = failures
    report["result"] = "PASS" if not failures else "FAIL"
    out = args.provenance or os.path.join(args.pruned, "PROVENANCE.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {report['result']} — PROVENANCE.json written to {out}")
    if failures:
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
