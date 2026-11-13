#!/usr/bin/env python3
"""Convert a VeOmni checkpoint to HF safetensors in the *release* layout.

Input can be either layout VeOmni writes:
  * a DCP step dir (``global_step_N/`` with ``__*.distcp``), or
  * ``global_step_N/hf_ckpt/`` (the safetensors ``save_hf`` exported, which is
    still in the VeOmni naming layout).

Usage:
    python dcp_to_release_hf.py \
        --ckpt CKPT/Moe8-ckpt-minimax_m3_vl_sft/checkpoints/global_step_500 \
        --out  CKPT/hf_release/global_step_500 \
        --reference Minimax-M3-moe8        # verifies the name set matches

    python dcp_to_release_hf.py \
        --ckpt CKPT/Moe8-ckpt-minimax_m3_vl_sft-3/checkpoints/global_step_5400/hf_ckpt \
        --out  CKPT/hf_release/global_step_5400

Why this exists
---------------
dcp_to_hf.py uses VeOmni's own `_normalize_key`, which only strips one leading
`model.`. The result is the *VeOmni/transformers>=5.12* layout, which is meant for
round-tripping back into VeOmni -- vLLM cannot load it:

  * `model.multi_modal_projector.*` matches no rule in vLLM's
    `MiniMaxM3VLForConditionalGeneration.hf_to_vllm_mapper`, so it falls through
    and vLLM dies with "There is no module or parameter named
    'model.multi_modal_projector'".
  * `model.vision_tower.layers.N.*` maps to itself, but vLLM's parameter is
    `model.vision_tower.vision_model.encoder.layers.N.*`.

This script inverts `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`,
producing exactly the layout the released weights (and our `Minimax-M3-moe8/`) use,
which is what vLLM expects. It also undoes the four fusions that converter performs
(`gate` first, then `up`, concatenated on the output dim; experts stacked by id).

Everything is verifiable without serving: `--reference` re-checks that the output
name set equals the reference model's name set.
"""

import argparse
import json
import pathlib
import re
import shutil
import time
from collections import OrderedDict

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "Minimax-M3-moe8"
MAX_PIXELS = 602112

_LM_PREFIX = "model.language_model."
_VT_PREFIX = "model.vision_tower."
_MMP_PREFIX = "model.multi_modal_projector."

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_VT_LAYER_RE = re.compile(r"^model\.vision_tower\.layers\.(\d+)\.(.+)$")

# suffixes under mlp. that are renamed rather than passed through
_EXPERT_REGEX = re.compile(r"^(.+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def _split_gate_up(tensor):
    """Undo ``torch.cat([gate, up], dim=-2)``: the fused dim is second-to-last."""
    half = tensor.shape[-2] // 2
    return tensor[..., :half, :].contiguous(), tensor[..., half:, :].contiguous()


def invert(name, tensor):
    """Yield (release_name, tensor) for one model-layout tensor; fusions expand."""
    # ---- language model: prefix swap + a few renames under mlp. -------------
    m = _LAYER_RE.match(name)
    if m:
        layer, rest = m.group(1), m.group(2)
        base = f"language_model.model.layers.{layer}"

        if rest == "mlp.gate_up_proj.weight":  # dense layers 0-2
            gate, up = _split_gate_up(tensor)
            yield f"{base}.mlp.gate_proj.weight", gate
            yield f"{base}.mlp.up_proj.weight", up
            return

        m2 = _EXPERT_REGEX.match(f"{base}.{rest}")
        if m2:
            prefix = f"{base}.block_sparse_moe.experts"
            if rest.endswith("gate_up_proj"):  # [E, 2I, H]
                gate, up = _split_gate_up(tensor)
                for j in range(tensor.shape[0]):
                    yield f"{prefix}.{j}.w1.weight", gate[j].contiguous()
                    yield f"{prefix}.{j}.w3.weight", up[j].contiguous()
            else:  # down_proj [E, H, I] -> w2
                for j in range(tensor.shape[0]):
                    yield f"{prefix}.{j}.w2.weight", tensor[j].contiguous()
            return

        if rest == "mlp.shared_experts.gate_up_proj.weight":
            gate, up = _split_gate_up(tensor)
            yield f"{base}.block_sparse_moe.shared_experts.gate_proj.weight", gate
            yield f"{base}.block_sparse_moe.shared_experts.up_proj.weight", up
            return
        if rest == "mlp.shared_experts.down_proj.weight":
            yield f"{base}.block_sparse_moe.shared_experts.down_proj.weight", tensor
            return
        if rest == "mlp.gate.weight":
            yield f"{base}.block_sparse_moe.gate.weight", tensor
            return
        if rest == "mlp.gate.e_score_correction_bias":
            yield f"{base}.block_sparse_moe.e_score_correction_bias", tensor
            return
        if rest.startswith("self_attn.indexer."):
            yield f"{base}.self_attn.index_{rest.removeprefix('self_attn.indexer.')}", tensor
            return
        yield f"{base}.{rest}", tensor
        return

    if name.startswith(_LM_PREFIX):
        yield "language_model.model." + name.removeprefix(_LM_PREFIX), tensor
        return
    if name == "lm_head.weight":
        yield "language_model.lm_head.weight", tensor
        return

    # ---- vision tower ------------------------------------------------------
    m = _VT_LAYER_RE.match(name)
    if m:
        yield f"vision_tower.vision_model.encoder.layers.{m.group(1)}.{m.group(2)}", tensor
        return
    if name.startswith(_VT_PREFIX):
        suffix = name.removeprefix(_VT_PREFIX)
        suffix = suffix.replace("embeddings.proj.", "embeddings.patch_embedding.")
        yield f"vision_tower.vision_model.{suffix}", tensor
        return

    # ---- multimodal projector ---------------------------------------------
    if name.startswith(_MMP_PREFIX):
        suffix = name.removeprefix(_MMP_PREFIX)
        if suffix.startswith("merge_linear_"):
            yield "patch_merge_mlp." + suffix.replace("merge_linear_", "linear_", 1), tensor
        else:
            yield "multi_modal_projector." + suffix, tensor
        return

    raise SystemExit(f"unmapped tensor name: {name!r}")


def _load_source(path: pathlib.Path) -> "OrderedDict[str, object]":
    """Read a DCP step dir or an ``hf_ckpt`` safetensors dir into model layout."""
    if list(path.glob("*.distcp")):
        from veomni.checkpoint.checkpointer import ckpt_to_state_dict

        return ckpt_to_state_dict(save_checkpoint_path=str(path), ckpt_manager="dcp")

    index = path / "model.safetensors.index.json"
    if index.is_file():
        from safetensors.torch import load_file

        weight_map = json.load(open(index))["weight_map"]
        src = OrderedDict()
        for shard in sorted(set(weight_map.values())):
            src.update(load_file(str(path / shard), device="cpu"))
        return src

    raise SystemExit(f"{path} is neither a DCP dir (__*.distcp) nor an HF dir (model.safetensors.index.json)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument(
        "--reference",
        type=pathlib.Path,
        default=DEFAULT_MODEL,
        help="release-layout model whose name set the output must match",
    )
    ap.add_argument(
        "--assets-from",
        type=pathlib.Path,
        default=DEFAULT_MODEL,
        help="model dir to copy config.json / tokenizer / custom *.py from",
    )
    ap.add_argument("--save-dtype", default="bfloat16")
    ap.add_argument("--shard-size", type=int, default=5_000_000_000)
    args = ap.parse_args()

    if not args.ckpt.is_dir():
        raise SystemExit(f"{args.ckpt} is not a directory")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")

    # --- 1. read the checkpoint (CPU only, single process) ------------------
    t0 = time.time()
    src = _load_source(args.ckpt)
    print(f"loaded : {len(src)} tensors in model layout, {time.time() - t0:.1f}s")

    # --- 2. invert names / split fusions ------------------------------------
    out = OrderedDict()
    for name, tensor in src.items():
        for new_name, new_tensor in invert(name, tensor):
            if new_name in out:
                raise SystemExit(f"duplicate output name {new_name!r}")
            out[new_name] = new_tensor

    print(f"mapped : {len(src)} -> {len(out)} tensors (+{len(out) - len(src)} from undoing fusions)")
    print(f"  sample: {list(out)[:2]}")

    # --- 3. verify names and shapes against the reference -------------------
    ref_index = args.reference / "model.safetensors.index.json"
    if ref_index.is_file():
        ref_wm = json.load(open(ref_index))["weight_map"]
        ref_names = set(ref_wm)
        missing = sorted(ref_names - set(out))
        extra = sorted(set(out) - ref_names)
        print(f"check  : reference {args.reference.name} has {len(ref_names)} names")
        if missing or extra:
            print(f"  missing vs reference: {len(missing)} {missing[:4]}")
            print(f"  extra   vs reference: {len(extra)} {extra[:4]}")
            raise SystemExit("name set does not match the reference release layout")

        # Shapes catch a split done on the wrong axis or with the wrong halves.
        from safetensors import safe_open

        bad = []
        for shard in sorted(set(ref_wm.values())):
            with safe_open(str(args.reference / shard), framework="pt") as sf:
                for k in sf.keys():
                    if tuple(sf.get_slice(k).get_shape()) != tuple(out[k].shape):
                        bad.append((k, tuple(sf.get_slice(k).get_shape()), tuple(out[k].shape)))
                    if len(bad) >= 5:
                        break
            if len(bad) >= 5:
                break
        print(f"  shape mismatches: {len(bad)} {bad[:2]}")
        if bad:
            raise SystemExit("tensor shapes do not match the reference release layout")
        print("  names + shapes match the reference release layout")

    # --- 4. write -----------------------------------------------------------
    t1 = time.time()
    from veomni.data.chat_template import build_chat_template
    from veomni.models.auto import build_processor
    from veomni.models.module_utils import save_model_weights

    processor = build_processor(str(args.assets_from), max_pixels=MAX_PIXELS)
    save_model_weights(
        str(args.out),
        out,
        save_dtype=args.save_dtype,
        shard_size=args.shard_size,
        model_assets=[processor, build_chat_template("minimax_m3_vl", processor)],
    )
    print(f"wrote  : {args.out}  ({time.time() - t1:.1f}s)")

    # save_model_weights never writes config.json / the custom *.py, and vLLM
    # rejects a directory without config.json.
    copied = []
    for s in sorted(args.assets_from.iterdir()):
        if s.is_dir() or s.name.endswith((".safetensors", ".index.json", ".index.json.bak")):
            continue
        if (args.out / s.name).exists():
            continue
        shutil.copy2(s, args.out / s.name)
        copied.append(s.name)
    print(f"side   : +{len(copied)} files (config.json, custom *.py, tokenizer assets)")


if __name__ == "__main__":
    main()
