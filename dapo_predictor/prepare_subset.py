"""Select a reproducible prompt-only subset without inspecting response lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def prompt_token_count(tokenizer, messages) -> int:
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    return len(tokens)


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("subset output/manifest already exists")
    if args.count < 1:
        raise ValueError("count must be positive")
    table = pq.read_table(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    selected = []
    identities = set()
    for index, messages in enumerate(table["prompt"].to_pylist()):
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        identity = hashlib.sha256(canonical.encode()).hexdigest()
        if identity in identities:
            continue
        identities.add(identity)
        token_count = prompt_token_count(tokenizer, messages)
        if token_count <= args.max_prompt_length:
            priority = hashlib.sha256(f"{args.seed}:{identity}".encode()).hexdigest()
            selected.append((priority, index, identity, token_count))
    selected = sorted(selected)[: args.count]
    if len(selected) != args.count:
        raise ValueError("insufficient unique prompts within the prompt-token cap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.take(pa.array([row[1] for row in selected])), args.output)
    manifest = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "seed": args.seed,
        "count": args.count,
        "model": args.model,
        "max_prompt_length": args.max_prompt_length,
        "selection": "seeded prompt hash, unique identities, prompt-token cap; no response-length selection",
        "prompts": [{"source_row": row[1], "sha256": row[2], "prompt_tokens": row[3]} for row in selected],
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "count": args.count,
                "sha256": manifest["output_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
