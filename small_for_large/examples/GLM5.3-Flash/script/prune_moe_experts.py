#!/usr/bin/env python3
"""Prune HF MoE expert weights by per-layer top-k expert ids.

The script rewrites a sharded safetensors model directory. For every tensor named
``<stem>.layers.<layer>.mlp.experts.<expert>.<suffix>``, the original expert id is
looked up in the per-layer top-k list and renamed to the new compact id
``0..top_k-1``. Router/gate tensors for the same layer are sliced on dimension 0
with the same order. If ``--stage-length`` is set, the selected top-k list is
deterministically interleaved after truncation so high-frequency experts are not
all assigned to the earliest compact ids. Quantization tensors are treated like
normal tensors.

Layer-container prefixes are matched generically so the same script handles both

* GLM-5.2 text checkpoints: ``model.layers.<L>.mlp.experts.<E>``
* GLM-5.3 multimodal checkpoints: ``model.language_model.layers.<L>.mlp.experts.<E>``

Only ``mlp.experts.<digits>`` tensors are renamed and only ``mlp.gate`` / ``mlp.router``
tensors are sliced. Everything else -- notably ``mlp.shared_experts.*``, the dense
layer ``mlp.gate_proj`` weights, and the ``model.visual.*`` tower -- is copied
through untouched.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from safetensors import safe_open

try:
    import torch
    from safetensors.torch import save_file as torch_save_file
except ImportError:
    torch = None
    torch_save_file = None


DEFAULT_INDEX_NAME = "model.safetensors.index.json"
LEGACY_INDEX_NAME = "index.json"
INDEX_OUTPUT_CANDIDATES = (DEFAULT_INDEX_NAME, LEGACY_INDEX_NAME)
DEFAULT_TOP_K = 64
DEFAULT_STAGE_LENGTH = 16

# ``...<prefix>.layers.<L>.mlp.experts.<E>.<suffix>`` -- the whole path before the
# expert id is captured verbatim so the new name can be rebuilt losslessly.
EXPERT_TENSOR_RE = re.compile(
    r"^(?P<stem>.+\.layers\.(?P<layer>\d+)\.mlp\.experts)\.(?P<expert>\d+)\.(?P<suffix>.+)$"
)
# ``mlp.gate.weight`` / ``mlp.gate.e_score_correction_bias`` / ``mlp.router.weight``.
# The trailing ``(?:\.|$)`` is what keeps the dense layers' ``mlp.gate_proj.weight``
# out of this match.
ROUTER_TENSOR_RE = re.compile(r"^.+\.layers\.(?P<layer>\d+)\.mlp\.(?:gate|router)(?:\.|$)")
# Top-k JSON keys: ``model.language_model.layers.3.mlp.experts``,
# ``model.layers.3.mlp.experts.topk_ids``, ``layer_3``, ``layer3`` ...
TOPK_LAYER_RE = re.compile(
    r"^(?:"
    r".+\.layers\.(?P<layer>\d+)\.mlp\.experts(?:\.topk_ids)?"
    r"|"
    r"layers?_(?P<layer_n>\d+)(?:\.topk_ids)?"
    r")$"
)
# Any tensor belonging to a decoder layer, used by ``--drop-layers``.
LAYER_TENSOR_RE = re.compile(r"^.+\.layers\.(?P<layer>\d+)\.")

CONFIG_EXPERT_KEYS = {
    "moe_num_experts",
    "n_routed_experts",
    "num_experts",
    "num_local_experts",
}
CONFIG_PER_TOK_KEY = "num_experts_per_tok"
CONFIG_NUM_LAYERS_KEY = "num_hidden_layers"

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".h5", ".msgpack")
COPIED_AS_IS_SKIP = {
    ".git",
    "__pycache__",
    "config.json",  # rewritten by _update_config
    "sizes.json",  # rebuilt from the pruned shards
    "local.sha256",  # rebuilt from the pruned shards
    "expert_prune_summary.json",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _select_safetensors_backend():
    if torch is not None and torch_save_file is not None:
        return "pt", torch_save_file
    raise RuntimeError("torch and safetensors.torch are required to rewrite bfloat16/quantized safetensors")


def _take_first_dim(tensor: Any, selected_ids: list[int]) -> Any:
    index = torch.tensor(selected_ids, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index)


def _tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return int(tensor.numel() * tensor.element_size())


def _human_bytes(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _read_safetensors_header(shard_path: Path) -> dict[str, dict[str, Any]]:
    """Return the safetensors JSON header without reading any tensor data."""
    with shard_path.open("rb") as f:
        raw_length = f.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{shard_path} is too short to be a safetensors file")
        header_length = struct.unpack("<Q", raw_length)[0]
        payload = f.read(header_length)
        if len(payload) != header_length:
            raise ValueError(f"{shard_path} has a truncated safetensors header")
    header = json.loads(payload.decode("utf-8"))
    return {name: spec for name, spec in header.items() if name != "__metadata__"}


def _resolve_index_path(model_dir: Path, index_name: str | None) -> Path:
    """Auto-detect the index, preferring the canonical HF name."""
    if index_name:
        index_path = model_dir / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"index file not found: {index_path}")
        return index_path

    for candidate in INDEX_OUTPUT_CANDIDATES:
        index_path = model_dir / candidate
        if index_path.is_file():
            return index_path

    candidates = sorted(model_dir.glob("*.index.json"))
    if not candidates:
        raise FileNotFoundError(f"no *.index.json found in {model_dir}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"multiple safetensors index files found; pass --index-name explicitly: {names}")
    return candidates[0]


def _resolve_output_index_names(model_dir: Path, index_names: str | None, source_index_name: str) -> list[str]:
    """Write the pruned index under the source name, plus any sibling index the input had."""
    if index_names:
        names = [name.strip() for name in index_names.split(",") if name.strip()]
        if not names:
            raise ValueError("--index-names must contain at least one filename")
        return list(dict.fromkeys(names))

    names = [source_index_name]
    for candidate in INDEX_OUTPUT_CANDIDATES:
        if candidate != source_index_name and (model_dir / candidate).is_file():
            names.append(candidate)
    return names


def _shuffle_by_stage(expert_ids: list[int], stage_length: int) -> list[int]:
    if stage_length < 0:
        raise ValueError(f"stage_length must be non-negative, got {stage_length}")
    if stage_length == 0 or stage_length >= len(expert_ids):
        return list(expert_ids)

    chunks = [expert_ids[start : start + stage_length] for start in range(0, len(expert_ids), stage_length)]
    shuffled_ids: list[int] = []
    for column in range(stage_length):
        for chunk in chunks:
            if column < len(chunk):
                shuffled_ids.append(chunk[column])
    return shuffled_ids


def _parse_topk_map(topk_path: Path, top_k: int, stage_length: int, original_num_experts: int) -> dict[int, list[int]]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    raw_topk = _load_json(topk_path)
    if not isinstance(raw_topk, dict):
        raise ValueError(f"{topk_path} must contain a JSON object keyed by layer")

    layer_topk: dict[int, list[int]] = {}

    for key, value in raw_topk.items():
        match = TOPK_LAYER_RE.match(str(key))
        if match is None:
            continue

        topk_ids = value.get("topk_ids") if isinstance(value, dict) else value
        if not isinstance(topk_ids, list):
            raise ValueError(f"{key} must contain a topk_ids list")

        if len(topk_ids) < top_k:
            raise ValueError(f"{key} contains {len(topk_ids)} expert ids, fewer than requested top_k={top_k}")

        selected_ids = [int(expert_id) for expert_id in topk_ids[:top_k]]
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f"{key} contains duplicate expert ids in the selected top-k list")

        out_of_range = [expert_id for expert_id in selected_ids if not 0 <= expert_id < original_num_experts]
        if out_of_range:
            raise ValueError(
                f"{key} references expert ids outside [0, {original_num_experts}): {sorted(out_of_range)[:8]}"
            )

        layer_id = int(match.group("layer") if match.group("layer") is not None else match.group("layer_n"))
        if layer_id in layer_topk:
            raise ValueError(f"duplicate top-k entries for layer {layer_id}")
        layer_topk[layer_id] = _shuffle_by_stage(selected_ids, stage_length)

    if not layer_topk:
        raise ValueError(f"no top-k entries matching a '.layers.<N>.mlp.experts' key found in {topk_path}")

    return layer_topk


def _build_remap(layer_topk: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    return {
        layer_id: {old_expert_id: new_expert_id for new_expert_id, old_expert_id in enumerate(expert_ids)}
        for layer_id, expert_ids in layer_topk.items()
    }


def _scan_model_moe_layers(weight_map: dict[str, str]) -> tuple[set[int], int]:
    """Find every layer that actually holds routed experts, and how many it has."""
    layers: set[int] = set()
    max_expert_id = -1
    for name in weight_map:
        match = EXPERT_TENSOR_RE.match(name)
        if match is None:
            continue
        layers.add(int(match.group("layer")))
        max_expert_id = max(max_expert_id, int(match.group("expert")))

    if not layers:
        raise ValueError("no '.layers.<N>.mlp.experts.<E>' tensors found in the weight map")

    return layers, max_expert_id + 1


def _resolve_fallback_layer(
    model_moe_layers: set[int],
    layer_topk: dict[int, list[int]],
    num_hidden_layers: int | None,
    skip_layers: set[int],
    drop_layers: set[int],
) -> tuple[int, list[int]]:
    """Pick the fallback layer used by MTP layers, and reject genuinely missing layers.

    Layers at or beyond ``text_config.num_hidden_layers`` are the MTP / next-token
    prediction heads. They are allowed to reuse the last decoder layer's top-k so the
    pruned checkpoint stays consistent with the single global ``n_routed_experts`` in
    config.json. A decoder layer missing from the top-k JSON is always an error.
    """
    exempt = skip_layers | drop_layers
    decoder_layers = sorted(layer for layer in layer_topk if num_hidden_layers is None or layer < num_hidden_layers)
    if not decoder_layers:
        raise ValueError("the top-k JSON contains no decoder layer entries")
    fallback_layer_id = decoder_layers[-1]

    missing: list[int] = []
    fallback_users: list[int] = []
    for layer_id in sorted(model_moe_layers):
        if layer_id in layer_topk or layer_id in exempt:
            continue
        is_beyond_decoder = (
            layer_id >= num_hidden_layers if num_hidden_layers is not None else layer_id > fallback_layer_id
        )
        if is_beyond_decoder:
            fallback_users.append(layer_id)
        else:
            missing.append(layer_id)

    if missing:
        raise ValueError(
            "the top-k JSON has no entry for MoE layer(s) "
            f"{missing}; it covers layers {sorted(layer_topk)}. "
            "Pass --skip-layers to leave them unpruned instead."
        )

    return fallback_layer_id, fallback_users


def _remap_expert_tensor_name(
    name: str,
    expert_remap: dict[int, dict[int, int]],
    skip_layers: set[int] | None = None,
    *,
    fallback_layer_id: int | None = None,
) -> str | None:
    match = EXPERT_TENSOR_RE.match(name)
    if match is None:
        return name

    layer_id = int(match.group("layer"))
    if skip_layers is not None and layer_id in skip_layers:
        return name

    old_expert_id = int(match.group("expert"))
    if layer_id not in expert_remap:
        if fallback_layer_id is not None and fallback_layer_id in expert_remap:
            # Use the fallback layer's top-k mapping (e.g. the last decoder layer for MTP)
            remap = expert_remap[fallback_layer_id]
        else:
            raise ValueError(f"missing top-k expert list for layer {layer_id}; tensor: {name}")
    else:
        remap = expert_remap[layer_id]

    new_expert_id = remap.get(old_expert_id)
    if new_expert_id is None:
        return None

    return f"{match.group('stem')}.{new_expert_id}.{match.group('suffix')}"


def _maybe_prune_router_tensor(
    name: str,
    tensor: Any,
    layer_topk: dict[int, list[int]],
    skip_layers: set[int] | None = None,
    *,
    fallback_layer_id: int | None = None,
) -> Any:
    match = ROUTER_TENSOR_RE.match(name)
    if match is None:
        return tensor

    layer_id = int(match.group("layer"))
    if skip_layers is not None and layer_id in skip_layers:
        return tensor

    selected_ids = layer_topk.get(layer_id)
    if selected_ids is None:
        if fallback_layer_id is not None and fallback_layer_id in layer_topk:
            selected_ids = layer_topk[fallback_layer_id]
        else:
            return tensor

    if tensor.ndim == 0:
        return tensor

    if max(selected_ids, default=-1) >= tensor.shape[0]:
        raise ValueError(
            f"cannot prune router tensor {name}: selected expert id exceeds first dimension {tensor.shape[0]}"
        )

    return _take_first_dim(tensor, selected_ids)


def _plan_shard_tensors(
    tensor_names: list[str],
    expert_remap: dict[int, dict[int, int]],
    skip_layers: set[int],
    drop_layers: set[int],
    fallback_layer_id: int | None,
) -> tuple[list[tuple[str, str]], int]:
    """Decide which tensors survive in a shard. Shared by --dry-run and the real pass."""
    kept: list[tuple[str, str]] = []
    dropped = 0
    for tensor_name in tensor_names:
        if drop_layers:
            layer_match = LAYER_TENSOR_RE.match(tensor_name)
            if layer_match is not None and int(layer_match.group("layer")) in drop_layers:
                dropped += 1
                continue

        new_name = _remap_expert_tensor_name(
            tensor_name, expert_remap, skip_layers=skip_layers, fallback_layer_id=fallback_layer_id
        )
        if new_name is None:
            dropped += 1
            continue

        kept.append((tensor_name, new_name))
    return kept, dropped


def _copy_non_weight_files(model_dir: Path, output_dir: Path, skip_names: set[str]) -> None:
    """Copy tokenizer/processor metadata, but never the old checkpoint or index files."""
    for source_path in sorted(model_dir.iterdir()):
        if source_path.name in COPIED_AS_IS_SKIP or source_path.name in skip_names:
            continue

        if source_path.is_file():
            if source_path.suffix in WEIGHT_SUFFIXES or fnmatch.fnmatch(source_path.name, "*.index.json"):
                continue
            shutil.copy2(source_path, output_dir / source_path.name)
            continue

        if source_path.is_dir():
            shutil.copytree(
                source_path,
                output_dir / source_path.name,
                dirs_exist_ok=True,
                # ignore_patterns takes globs, so the weight suffixes need a leading '*'
                ignore=shutil.ignore_patterns(".git", "__pycache__", *(f"*{s}" for s in WEIGHT_SUFFIXES)),
            )


def _update_config(source_config: Path, output_config: Path, top_k: int) -> int:
    """Rewrite expert-count keys wherever they appear (GLM-5.3 nests them in text_config)."""
    if not source_config.is_file():
        return 0

    replaced = 0

    def update_value(value: Any) -> Any:
        nonlocal replaced
        if isinstance(value, dict):
            updated: dict[str, Any] = {}
            for key, item in value.items():
                if key in CONFIG_EXPERT_KEYS and isinstance(item, int):
                    updated[key] = top_k
                    replaced += 1
                else:
                    updated[key] = update_value(item)
            return updated
        if isinstance(value, list):
            return [update_value(item) for item in value]
        return value

    _write_json(output_config, update_value(_load_json(source_config)))
    return replaced


def _find_config_int(node: Any, key: str) -> int | None:
    """Depth-first search for the first int value under ``key`` (handles text_config nesting)."""
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        for item in node.values():
            found = _find_config_int(item, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_config_int(item, key)
            if found is not None:
                return found
    return None


def _validate_output_dir(model_dir: Path, output_dir: Path, overwrite: bool) -> None:
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise ValueError("output directory must not be the input model directory or a child of it")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def _rebuild_sizes_json(
    output_dir: Path,
    source_sizes_path: Path,
    kept_sizes: dict[str, dict[str, Any]],
) -> int:
    """Rewrite sizes.json for the pruned checkpoint, keeping the source key order.

    Note: the source file stores ``size`` as the element *count*, not bytes; that
    convention is preserved so downstream consumers keep working.
    """
    original_order: list[str] = []
    if source_sizes_path.is_file():
        raw = _load_json(source_sizes_path)
        if isinstance(raw, dict):
            original_order = [str(name) for name in raw]

    ordered: dict[str, dict[str, Any]] = {}
    for name in original_order:
        if name in kept_sizes:
            ordered[name] = kept_sizes[name]
    for name, entry in kept_sizes.items():
        ordered.setdefault(name, entry)

    _write_json(output_dir / "sizes.json", ordered)
    return len(ordered)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_local_sha256(output_dir: Path, shard_names: list[str]) -> None:
    """Re-hash every written shard. Hashing releases the GIL, so threads actually help."""
    names = sorted(shard_names)
    workers = max(1, min(8, len(names)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(lambda name: _sha256_file(output_dir / name), names))
    lines = [f"{digest}  {name}\n" for digest, name in zip(digests, names)]
    (output_dir / "local.sha256").write_text("".join(lines), encoding="utf-8")


def _estimate_router_bytes(
    tensor_name: str,
    spec: dict[str, Any],
    nbytes: int,
    layer_topk: dict[int, list[int]],
    skip_layers: set[int],
    fallback_layer_id: int,
) -> int:
    """Scale a router/gate tensor's byte footprint down to the number of kept experts."""
    match = ROUTER_TENSOR_RE.match(tensor_name)
    if match is None:
        return nbytes

    layer_id = int(match.group("layer"))
    if layer_id in skip_layers:
        return nbytes

    selected_ids = layer_topk.get(layer_id)
    if selected_ids is None:
        selected_ids = layer_topk.get(fallback_layer_id)
    if not selected_ids:
        return nbytes

    shape = spec.get("shape") or []
    if not shape or not shape[0]:
        return nbytes

    return int(nbytes * len(selected_ids) / shape[0])


def _dry_run_report(
    *,
    model_dir: Path,
    output_dir: Path,
    index_path: Path,
    output_index_names: list[str],
    tensors_by_shard: dict[str, list[str]],
    expert_remap: dict[int, dict[int, int]],
    layer_topk: dict[int, list[int]],
    skip_layers: set[int],
    drop_layers: set[int],
    fallback_layer_id: int,
    fallback_users: list[int],
    model_moe_layers: set[int],
    original_num_experts: int,
    top_k: int,
    stage_length: int,
    skip_output_dir: bool,
) -> dict[str, Any]:
    """Validate the plan and estimate the output size without writing any weights."""
    kept_tensors = 0
    dropped_tensors = 0
    total_size = 0
    empty_shards: list[str] = []
    missing_shards: list[str] = []

    for shard_name, tensor_names in sorted(tensors_by_shard.items()):
        source_shard = model_dir / shard_name
        if not source_shard.is_file():
            missing_shards.append(shard_name)
            continue

        kept, dropped = _plan_shard_tensors(
            tensor_names, expert_remap, skip_layers, drop_layers, fallback_layer_id
        )
        dropped_tensors += dropped
        kept_tensors += len(kept)
        if not kept:
            empty_shards.append(shard_name)
            continue

        # Read only the header: exact per-tensor byte sizes, no weight data.
        header = _read_safetensors_header(source_shard)
        for tensor_name, new_name in kept:
            spec = header.get(tensor_name)
            if spec is None:
                raise KeyError(f"{tensor_name} is listed in the index but missing from {source_shard}")

            start, end = spec["data_offsets"]
            nbytes = int(end) - int(start)
            nbytes = _estimate_router_bytes(tensor_name, spec, nbytes, layer_topk, skip_layers, fallback_layer_id)
            total_size += nbytes

    warnings: list[str] = []
    if empty_shards:
        warnings.append(f"{len(empty_shards)} shard(s) would become empty and be skipped: {empty_shards[:5]}")
    if skip_layers:
        warnings.append(
            f"layers {sorted(skip_layers)} stay at {original_num_experts} experts while config.json "
            f"would advertise n_routed_experts={top_k}"
        )
    if fallback_users:
        warnings.append(
            f"layer(s) {fallback_users} reuse layer {fallback_layer_id}'s top-k (MTP / beyond num_hidden_layers)"
        )
    if missing_shards:
        warnings.append(f"{len(missing_shards)} shard file(s) referenced by the index are missing")

    free_bytes = None
    if not skip_output_dir:
        try:
            free_bytes = shutil.disk_usage(output_dir if output_dir.exists() else output_dir.parent).free
        except OSError:
            free_bytes = None
    if free_bytes is not None and free_bytes < total_size:
        warnings.append(
            f"insufficient space at the output location: need {_human_bytes(total_size)}, "
            f"have {_human_bytes(free_bytes)}"
        )

    return {
        "dry_run": True,
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "source_index": index_path.name,
        "output_index_names": output_index_names,
        "model_moe_layers": sorted(model_moe_layers),
        "topk_layers": sorted(layer_topk),
        "fallback_layer_id": fallback_layer_id,
        "fallback_layers": fallback_users,
        "original_num_experts": original_num_experts,
        "top_k": top_k,
        "stage_length": stage_length,
        "skipped_layers": sorted(skip_layers),
        "dropped_layers": sorted(drop_layers),
        "shard_count": len(tensors_by_shard),
        "kept_tensors": kept_tensors,
        "dropped_tensors": dropped_tensors,
        "empty_shards": empty_shards,
        "missing_shards": missing_shards,
        "estimated_total_size": total_size,
        "estimated_total_size_human": _human_bytes(total_size),
        "free_bytes": free_bytes,
        "warnings": warnings,
    }


def prune_model(
    model_dir: str | Path,
    topk_path: str | Path,
    output_dir: str | Path,
    top_k: int = DEFAULT_TOP_K,
    stage_length: int = DEFAULT_STAGE_LENGTH,
    index_name: str | None = None,
    index_names: str | None = None,
    original_num_experts: int | None = None,
    overwrite: bool = False,
    skip_layers: set[int] | None = None,
    drop_layers: set[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    topk_path = Path(topk_path)
    output_dir = Path(output_dir)
    skip_layers = set() if skip_layers is None else set(skip_layers)
    drop_layers = set() if drop_layers is None else set(drop_layers)
    overlap = skip_layers & drop_layers
    if overlap:
        raise ValueError(f"layers cannot be in both skip_layers and drop_layers: {sorted(overlap)}")

    index_path = _resolve_index_path(model_dir, index_name)
    output_index_names = _resolve_output_index_names(model_dir, index_names, index_path.name)
    index_data = _load_json(index_path)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} must contain a weight_map object")

    config = _load_json(model_dir / "config.json") if (model_dir / "config.json").is_file() else {}
    num_hidden_layers = _find_config_int(config, CONFIG_NUM_LAYERS_KEY)
    num_experts_per_tok = _find_config_int(config, CONFIG_PER_TOK_KEY)

    model_moe_layers, detected_num_experts = _scan_model_moe_layers(weight_map)
    if original_num_experts is None:
        original_num_experts = detected_num_experts
    elif original_num_experts != detected_num_experts:
        raise ValueError(
            f"--original-num-experts={original_num_experts} does not match the checkpoint's "
            f"{detected_num_experts} experts per MoE layer"
        )

    if num_experts_per_tok is not None and top_k < num_experts_per_tok:
        raise ValueError(
            f"top_k={top_k} is smaller than config's {CONFIG_PER_TOK_KEY}={num_experts_per_tok}; "
            "the router could not select that many experts"
        )

    # original_num_experts is detected from the checkpoint above, so out-of-range ids in the
    # top-k JSON are rejected here rather than silently producing a sliced-to-nothing router.
    layer_topk = _parse_topk_map(topk_path, top_k, stage_length, original_num_experts)
    fallback_layer_id, fallback_users = _resolve_fallback_layer(
        model_moe_layers,
        layer_topk,
        num_hidden_layers,
        skip_layers,
        drop_layers,
    )
    expert_remap = _build_remap(layer_topk)

    warnings: list[str] = []
    if skip_layers:
        warnings.append(
            f"layers {sorted(skip_layers)} are left with {original_num_experts} experts while "
            f"config.json will advertise n_routed_experts={top_k}"
        )
    if fallback_users:
        warnings.append(
            f"layer(s) {fallback_users} beyond num_hidden_layers={num_hidden_layers} "
            f"reuse layer {fallback_layer_id}'s top-k"
        )

    tensors_by_shard: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in weight_map.items():
        tensors_by_shard[str(shard_name)].append(tensor_name)

    if dry_run:
        return _dry_run_report(
            model_dir=model_dir,
            output_dir=output_dir,
            index_path=index_path,
            output_index_names=output_index_names,
            tensors_by_shard=tensors_by_shard,
            expert_remap=expert_remap,
            layer_topk=layer_topk,
            skip_layers=skip_layers,
            drop_layers=drop_layers,
            fallback_layer_id=fallback_layer_id,
            fallback_users=fallback_users,
            model_moe_layers=model_moe_layers,
            original_num_experts=original_num_experts,
            top_k=top_k,
            stage_length=stage_length,
            skip_output_dir=not output_dir.exists(),
        )

    _validate_output_dir(model_dir, output_dir, overwrite)
    _copy_non_weight_files(model_dir, output_dir, set(output_index_names))
    replaced_keys = _update_config(model_dir / "config.json", output_dir / "config.json", top_k)
    if not replaced_keys:
        warnings.append("no expert-count key was found in config.json; it was copied unchanged")

    source_sizes_path = model_dir / "sizes.json"
    rebuild_sizes = source_sizes_path.is_file()
    rebuild_sha256 = (model_dir / "local.sha256").is_file()

    safetensors_framework, save_safetensors_file = _select_safetensors_backend()

    new_weight_map: dict[str, str] = {}
    kept_sizes: dict[str, dict[str, Any]] = {}
    total_size = 0
    kept_tensors = 0
    dropped_tensors = 0
    written_shards: list[str] = []
    skipped_empty_shards: list[str] = []

    shard_items = sorted(tensors_by_shard.items())
    for position, (shard_name, tensor_names) in enumerate(shard_items, start=1):
        source_shard = model_dir / shard_name
        if not source_shard.is_file():
            raise FileNotFoundError(f"shard referenced by index is missing: {source_shard}")

        kept, dropped = _plan_shard_tensors(
            tensor_names, expert_remap, skip_layers, drop_layers, fallback_layer_id
        )
        dropped_tensors += dropped
        if not kept:
            skipped_empty_shards.append(shard_name)
            print(f"[{position}/{len(shard_items)}] {shard_name} -> empty, skipped", flush=True)
            continue

        output_tensors: dict[str, Any] = {}
        with safe_open(source_shard, framework=safetensors_framework) as shard:
            metadata = shard.metadata()
            available_keys = set(shard.keys())
            for tensor_name, new_name in kept:
                if tensor_name not in available_keys:
                    raise KeyError(f"{tensor_name} is listed in the index but missing from {source_shard}")
                if new_name in output_tensors:
                    raise ValueError(f"pruning produced a duplicate tensor name: {new_name}")

                tensor = shard.get_tensor(tensor_name)
                tensor = _maybe_prune_router_tensor(
                    tensor_name, tensor, layer_topk, skip_layers=skip_layers, fallback_layer_id=fallback_layer_id
                )
                output_tensors[new_name] = tensor
                new_weight_map[new_name] = shard_name
                kept_sizes[new_name] = {"size": int(tensor.numel()), "shape": list(tensor.shape)}
                total_size += _tensor_nbytes(tensor)
                kept_tensors += 1

        target_shard = output_dir / shard_name
        target_shard.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors_file(output_tensors, target_shard, metadata=metadata)
        written_shards.append(shard_name)
        shard_bytes = sum(_tensor_nbytes(tensor) for tensor in output_tensors.values())
        del output_tensors
        print(
            f"[{position}/{len(shard_items)}] {shard_name} -> {len(kept)} tensors, "
            f"{_human_bytes(shard_bytes)} ({dropped} dropped)",
            flush=True,
        )

    if not written_shards:
        raise RuntimeError("pruning produced no tensors at all; check --drop-layers and the top-k JSON")

    new_index = dict(index_data)
    new_metadata = dict(new_index.get("metadata") or {})
    new_metadata["total_size"] = total_size
    new_metadata["expert_prune"] = {
        "top_k": top_k,
        "stage_length": stage_length,
        "original_num_experts": original_num_experts,
        "model_moe_layers": sorted(model_moe_layers),
        "topk_layers": sorted(layer_topk),
        "fallback_layer_id": fallback_layer_id,
        "fallback_layers": fallback_users,
        "skipped_layers": sorted(skip_layers),
        "dropped_layers": sorted(drop_layers),
        "pruned_shards": len(written_shards),
        "skipped_empty_shards": skipped_empty_shards,
        "topk_source": str(topk_path),
    }
    new_index["metadata"] = new_metadata
    new_index["weight_map"] = new_weight_map
    for output_index_name in output_index_names:
        _write_json(output_dir / output_index_name, new_index)

    aux_files: dict[str, Any] = {}
    if rebuild_sizes:
        count = _rebuild_sizes_json(output_dir, source_sizes_path, kept_sizes)
        aux_files["sizes.json"] = count
    if rebuild_sha256:
        _write_local_sha256(output_dir, written_shards)
        aux_files["local.sha256"] = len(written_shards)

    summary = {
        "dry_run": False,
        "output_dir": str(output_dir),
        "source_index": index_path.name,
        "output_index_names": output_index_names,
        "model_moe_layers": sorted(model_moe_layers),
        "topk_layers": sorted(layer_topk),
        "fallback_layer_id": fallback_layer_id,
        "fallback_layers": fallback_users,
        "original_num_experts": original_num_experts,
        "top_k": top_k,
        "stage_length": stage_length,
        "skipped_layers": sorted(skip_layers),
        "dropped_layers": sorted(drop_layers),
        "kept_tensors": kept_tensors,
        "dropped_tensors": dropped_tensors,
        "written_shards": len(written_shards),
        "skipped_empty_shards": skipped_empty_shards,
        "total_size": total_size,
        "aux_files": aux_files,
        "warnings": warnings,
    }
    _write_json(output_dir / "expert_prune_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune and remap sharded HF MoE expert safetensors by per-layer top-k expert ids. "
            "Handles GLM-5.2 (model.layers.*) and GLM-5.3 (model.language_model.layers.*) checkpoints."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Input HF model directory.")
    parser.add_argument(
        "--topk-json",
        required=True,
        help="JSON file with per-layer topk_ids, keyed by any of "
        "'.layers.<N>.mlp.experts' / 'layer_<N>'; values may be a list or {'topk_ids': [...]}.",
    )
    parser.add_argument("--output-dir", help="Output HF model directory. Required unless --dry-run.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of experts to keep per MoE layer.")
    parser.add_argument(
        "--stage-length",
        type=int,
        default=DEFAULT_STAGE_LENGTH,
        help=(
            "Deterministically interleave the truncated top-k ids by this stage length. "
            "Use 0 to keep the original count-sorted order."
        ),
    )
    parser.add_argument(
        "--index-name",
        default="",
        help="Safetensors index filename to read. Empty (default) auto-detects "
        "model.safetensors.index.json, then index.json.",
    )
    parser.add_argument(
        "--index-names",
        default="",
        help="Comma-separated index filenames to write. Empty (default) mirrors the input directory.",
    )
    parser.add_argument(
        "--original-num-experts",
        type=int,
        default=None,
        help="Original number of routed experts. Default: auto-detected from the checkpoint.",
    )
    parser.add_argument(
        "--skip-layers",
        default="",
        help="Comma-separated layer ids to keep unpruned. WARNING: config.json is still rewritten globally.",
    )
    parser.add_argument(
        "--drop-layers",
        default="",
        help="Comma-separated layer ids to drop entirely (tensors removed, not copied).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the top-k JSON and the layer coverage, then report sizes. Writes nothing.",
    )
    return parser.parse_args()


def _parse_layer_list(value: str) -> set[int]:
    if not value:
        return set()

    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layers.add(int(item))
    return layers


def main() -> None:
    args = _parse_args()
    if not args.dry_run and not args.output_dir:
        raise SystemExit("--output-dir is required unless --dry-run is set")

    summary = prune_model(
        model_dir=args.model_dir,
        topk_path=args.topk_json,
        output_dir=args.output_dir or args.model_dir,
        top_k=args.top_k,
        stage_length=args.stage_length,
        index_name=args.index_name or None,
        index_names=args.index_names or None,
        original_num_experts=args.original_num_experts,
        overwrite=args.overwrite,
        skip_layers=_parse_layer_list(args.skip_layers),
        drop_layers=_parse_layer_list(args.drop_layers),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()