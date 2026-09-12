"""Prepare a separate prompt candidate while preserving every non-prompt dataset field."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


FORMAT_INSTRUCTION = (
    "\nWrite a short solution. End your response with a separate final line in exactly "
    "this format: #### <number>. Replace <number> with the numeric answer only. "
    "Do not put currency symbols, units, words, LaTeX, or Markdown around the number. "
    "Do not write anything after that final line."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("--source and --output must be different directories")
    if args.output.exists():
        parser.error("Candidate directory already exists; inspect it before reuse")
    instruction = FORMAT_INSTRUCTION
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    records = []
    args.output.mkdir(parents=True)
    for dataset in ["gsm8k", "math"]:
        for split in ["train", "test"]:
            source = args.source / dataset / f"{split}.parquet"
            before_sha = sha256(source)
            table = pq.read_table(source)
            rows = table.to_pylist()
            changed = []
            lengths = []
            for row in rows:
                output = dict(row)
                if dataset == "gsm8k":
                    prompt = [dict(m) for m in row["prompt"]]
                    if not prompt or prompt[-1]["role"] != "user":
                        raise ValueError(f"Expected a final user message in {source}")
                    if instruction in prompt[-1]["content"]:
                        raise ValueError(f"The format instruction is already present in {source}")
                    prompt[-1]["content"] += instruction
                    output["prompt"] = prompt
                changed.append(output)
                tokens = tokenizer.apply_chat_template(
                    output["prompt"],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    enable_thinking=False,
                )
                if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                    raise TypeError(f"Tokenizer returned invalid token IDs for {source}")
                lengths.append(len(tokens))
            target = args.output / dataset / f"{split}.parquet"
            target.parent.mkdir(exist_ok=True)
            if dataset == "math":
                target.write_bytes(source.read_bytes())
            else:
                pq.write_table(pa.Table.from_pylist(changed, schema=table.schema), target)
            reread = pq.read_table(target).to_pylist()
            if reread != changed or len(reread) != len(rows):
                raise RuntimeError(f"Written rows differ from the prepared rows in {target}")
            for original, candidate in zip(rows, reread, strict=True):
                original_fields = {key: value for key, value in original.items() if key != "prompt"}
                candidate_fields = {key: value for key, value in candidate.items() if key != "prompt"}
                if original_fields != candidate_fields:
                    raise RuntimeError(f"A non-prompt field changed in {target}")
            if sha256(source) != before_sha:
                raise RuntimeError(f"Source file changed during preparation: {source}")
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source": str(source),
                    "source_sha256": before_sha,
                    "output": str(target),
                    "output_sha256": sha256(target),
                    "rows": len(rows),
                    "within_1024_token_limit": sum(n <= 1024 for n in lengths),
                    "max_prompt_tokens": max(lengths),
                }
            )
    manifest = {
        "instruction": instruction,
        "preparation_script_sha256": sha256(Path(__file__)),
        "template_kwargs": {"enable_thinking": False},
        "files": records,
        "unchanged": [
            "question text",
            "ground truth",
            "row order",
            "data source",
            "all non-prompt fields",
            "original source files",
            "MATH file bytes",
        ],
        "state": "prepared_candidate_not_training_evidence",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
