#!/usr/bin/env python
"""Build a small SFT dataset from teacher (full-model) outputs for a pruned MoE model.

Given a pool of per-sample teacher generations and the source dataset (parquet
with embedded images), this tool produces a messages-format SFT jsonl plus a
complete audit trail (``selection_stats.json``):

    pool -> filter chain (blacklist / finish_reason / token budget)
         -> fixed-seed shuffle + sample N
         -> image re-join from the source parquet
         -> optional render verification through the trainer's own processor
         -> sft<N>.jsonl (messages + images + source_id for GT rejoin)

The render verification step is what makes the dataset trustworthy: every
selected sample is rendered through the *same* processor + chat template the
trainer will use, and any sample whose rendered length exceeds the cap is
rejected before training ever sees it. The render function is pluggable via
``--render-module`` so the tool itself stays free of trainer-specific imports.

Input sample format (one JSON file per sample, or a single jsonl):

    {
      "index": 12,               # row in the source parquet
      "question": "...",
      "content": "<teacher chain-of-thought>",
      "metrics": {"prompt_tokens": 123, "completion_tokens": 456,
                  "finish_reason": "stop"},
      "blacklisted": false       # set by the scoring stage
    }

Output record format:

    {"messages": [{"role": "user", "content": "<image>\n<question>"},
                  {"role": "assistant", "content": "<content>"}],
     "images": ["images/0012.png"],
     "source_id": "train:12"}

``source_id`` carries the parquet row so evaluation can rejoin ground truth
with zero drift later.
"""

import argparse
import glob
import json
import os
import random
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", required=True, help="directory of sample-*.json files, or a single .jsonl file")
    p.add_argument(
        "--parquet",
        default=None,
        help="source dataset parquet (with embedded images); optional for text-only datasets",
    )
    p.add_argument("--out-dir", required=True, help="output directory (jsonl, images/, stats)")
    p.add_argument("--dataset-name", default="sft", help="output jsonl basename (default: sft)")
    p.add_argument("--num-samples", type=int, default=200, help="how many samples to select")
    p.add_argument("--seed", type=int, default=42, help="fixed seed for the shuffle+sample")
    p.add_argument(
        "--max-filter-tokens", type=int, default=4000, help="teacher prompt+completion token budget during filtering"
    )
    p.add_argument(
        "--max-final-len", type=int, default=4096, help="rendered-length cap; candidates above it are rejected"
    )
    p.add_argument(
        "--image-placeholder",
        default="<image>",
        help="placeholder prepended to the user message when the sample has images",
    )
    p.add_argument(
        "--render-module",
        default=None,
        help="python module exposing render_length(messages, image_paths) -> int; "
        "when omitted, no render verification is done (NOT recommended)",
    )
    return p.parse_args()


def load_pool(samples_path: str, max_filter_tokens: int) -> tuple[list[dict], dict]:
    """Load teacher samples and apply the filter chain, returning (pool, skipped-counts)."""
    pool, skipped = [], {"missing_file": 0, "blacklisted": 0, "finish_reason": 0, "empty": 0, "too_long": 0}
    records = []
    if os.path.isdir(samples_path):
        records = [
            json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(samples_path, "sample-*.json")))
        ]
    elif samples_path.endswith(".jsonl"):
        with open(samples_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = [json.load(open(samples_path, encoding="utf-8"))]

    for d in records:
        if d.get("blacklisted"):
            skipped["blacklisted"] += 1
            continue
        m = d.get("metrics") or {}
        if m.get("finish_reason") not in (None, "stop"):
            skipped["finish_reason"] += 1
            continue
        content = (d.get("content") or "").strip()
        if not content:
            skipped["empty"] += 1
            continue
        total = int(m.get("prompt_tokens", 0)) + int(m.get("completion_tokens", 0))
        if total >= max_filter_tokens:
            skipped["too_long"] += 1
            continue
        pool.append({"index": int(d["index"]), "question": d["question"], "content": content, "eval_tokens": total})
    return pool, skipped


def build_render_fn(render_module: str):
    """Import --render-module and return render_length(messages, image_paths) -> int.

    Returns None when no module is given (render verification disabled).
    """
    if not render_module:
        return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(render_module)) or ".")
    import importlib.util

    spec = importlib.util.spec_from_file_location("render_hook", render_module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "render_length", None)
    if not callable(fn):
        sys.exit(f"[fail] {render_module} does not expose render_length(messages, image_paths) -> int")
    return fn


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    pool, skipped = load_pool(args.samples, args.max_filter_tokens)
    print(f"eligible pool: {len(pool)} (skipped: {skipped})")

    rng = random.Random(args.seed)
    order = pool[:]
    rng.shuffle(order)
    render_length = build_render_fn(args.render_module)

    # Lazy parquet loading: text-only datasets (the upstream gsm8k case) never
    # touch it, so neither pandas nor pyarrow is a hard dependency there.
    df = None

    def extract_images(index: int) -> list[str]:
        nonlocal df
        if args.parquet is None:
            return []
        if df is None:
            import pandas as pd

            df = pd.read_parquet(args.parquet)
        row = df.iloc[index]
        imgs = row["images"] if "images" in df.columns else []
        paths = []
        for j, im in enumerate(imgs):
            suffix = f"{index:04d}" if len(imgs) == 1 else f"{index:04d}-{j}"
            p = os.path.join(args.out_dir, "images", f"{suffix}.png")
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(im["bytes"])
            paths.append(p)
        return paths

    def user_content(cand: dict) -> str:
        """User message text: image placeholder only when the sample has images."""
        if cand["images"]:
            return f"{args.image_placeholder}\n{cand['question']}"
        return cand["question"]

    selected, rejected = [], []
    for cand in order:
        if len(selected) >= args.num_samples:
            break
        try:
            cand["images"] = extract_images(cand["index"])
            if render_length is not None:
                messages = [
                    {"role": "user", "content": user_content(cand)},
                    {"role": "assistant", "content": cand["content"]},
                ]
                cand["final_len"] = render_length(messages, cand["images"])
                if cand["final_len"] >= args.max_final_len:
                    rejected.append({"index": cand["index"], "reason": f"final_len={cand['final_len']}"})
                    continue
            else:  # no render hook: fall back to the teacher token count
                cand["final_len"] = cand["eval_tokens"]
        except Exception as e:  # corrupt image / render issue
            rejected.append({"index": cand["index"], "reason": f"{type(e).__name__}: {e}"})
            continue
        selected.append(cand)

    out_jsonl = os.path.join(args.out_dir, f"{args.dataset_name}{len(selected)}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in selected:
            rec = {
                "messages": [
                    {"role": "user", "content": user_content(r)},
                    {"role": "assistant", "content": r["content"]},
                ],
                "images": r["images"],
                "source_id": f"train:{r['index']}",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lens = sorted(r["final_len"] for r in selected)
    stats = {
        "seed": args.seed,
        "filter": (f"prompt_tokens+completion_tokens < {args.max_filter_tokens}, finish_reason==stop, not blacklisted"),
        "render_module": args.render_module,
        "render_verification": render_length is not None,
        "pool_size": len(pool),
        "skipped": skipped,
        "selected": len(selected),
        "rejected_during_verify": rejected,
        "final_len": {
            "min": lens[0] if lens else None,
            "max": lens[-1] if lens else None,
            "p50": lens[len(lens) // 2] if lens else None,
            "p95": lens[int(len(lens) * 0.95)] if lens else None,
            "mean": (sum(lens) / len(lens)) if lens else None,
        },
        "max_final_len_cap": args.max_final_len,
        "jsonl": out_jsonl,
        "image_dir": os.path.join(args.out_dir, "images"),
    }
    stats_path = os.path.join(args.out_dir, "selection_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_jsonl} and {stats_path}")


if __name__ == "__main__":
    main()
