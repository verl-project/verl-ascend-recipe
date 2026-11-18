#!/usr/bin/env python3
"""Build a VeOmni-loadable geo3k SFT dataset from a vLLM rollout.

Pipeline: vLLM rollout JSONL → grade answers → keep correct + think →
          copy images → repeat to target rows → write ShareGPT JSON.

Source : <run>/predictions/vllm-api-general-chat/geometry3k.jsonl
Output : <this dir>/dataset/geo3k_sft/
           images/<idx>.png     copies of the source images
           geo3k_sft.json       ShareGPT-shaped records, repeated N times

Only rows whose final \\boxed{} answer grades correct against gold are kept,
and the assistant turn preserves the full CoT: ``\u23f8\\n{reasoning}\\n</think>\\n\\n\\boxed{X}``.
Truncated rows (no closing </think>) are dropped automatically.

``mathruler`` + ``pylatexenc`` are required (``pip install mathruler pylatexenc``).

Run:
    python gen_sft_dataset.py --dry-run
    python gen_sft_dataset.py
    python gen_sft_dataset.py --max-tokens 4096
"""

import argparse
import json
import math
import pathlib
import re
import shutil
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "outputs/default/geo3k_rollout/predictions" / "vllm-api-general-chat/geometry3k.jsonl"
DEFAULT_OUT_DIR = HERE / "dataset" / "geo3k_sft"
DEFAULT_OUT_NAME = "geo3k_sft.json"
DEFAULT_MODEL = HERE / "Minimax-M3-moe8"
DEFAULT_TARGET_ROWS = 2000 * 16

THINK_CLOSE = "</think>"
MERGE_OPEN = "\u23f8\n"
MERGE_CLOSE = "\n</think>\n\n"
IMAGE_PLACEHOLDER = "<image>"


# ── Boxed extraction ────────────────────────────────────────────────────
def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` (brace-matched), or None."""
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None
    body_start = start + len(r"\boxed{")
    depth = 0
    for pos in range(body_start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return text[body_start:pos].strip()
            depth -= 1
    return None  # unbalanced, i.e. truncated mid-answer


# ── Answer normalisation ────────────────────────────────────────────────
_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|°")
_LATEX_CMD_SPACE_RE = re.compile(r"(\\[A-Za-z]+)\s*(?=\{)")
_LATEX_STRUCT_SPACE_RE = re.compile(r"\s*([{}(),=+\-*/^_])\s*")
_TEXT_UNIT_RE = re.compile(r"\\text\{[^{}]*\}(\s*\^\{?[0-9]+\}?)?")


def _normalize_candidate(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\[!,;:]", "", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = _DEGREE_RE.sub("", text)
    text = text.replace("\\%", "%")
    text = _TEXT_UNIT_RE.sub("", text)
    text = text.strip()
    if text.count("=") == 1:
        text = text.split("=", 1)[1].strip()
    text = _LATEX_CMD_SPACE_RE.sub(r"\1", text)
    text = _LATEX_STRUCT_SPACE_RE.sub(r"\1", text)
    return re.sub(r"\s+", "", text)


def answer_variants(answer: str) -> list[str]:
    parts = answer.split("\\approx")
    variants = [parts[0], parts[1]] if len(parts) == 2 else [answer]
    out = []
    for variant in variants:
        normalized = _normalize_candidate(variant)
        if normalized:
            out.append(normalized)
    return out


def _make_grader():
    try:
        from mathruler.grader import grade_answer
    except ImportError as error:
        raise SystemExit(f"mathruler is required (pip install mathruler pylatexenc):\n  {error}") from error

    def is_correct(boxed: str, gold: str) -> bool:
        if grade_answer(boxed, gold):
            return True
        gold_variants = answer_variants(gold)
        if not gold_variants:
            return False
        for given in answer_variants(boxed):
            for expected in gold_variants:
                if given == expected or grade_answer(given, expected):
                    return True
        return False

    return is_correct


# ── Rollout parsing ─────────────────────────────────────────────────────
def split_prediction(prediction: str, content: str) -> tuple[str, str]:
    """Return ``(reasoning, answer_line)`` or ``("", "")`` when truncated."""
    idx = prediction.rfind(THINK_CLOSE)
    if idx == -1:
        return "", ""
    reasoning = prediction[:idx]
    answer = content.strip() if content.strip() else prediction[idx + len(THINK_CLOSE) :].strip()
    reasoning = re.sub(r"</?think>", "", reasoning).strip("\n")
    answer = re.sub(r"</?think>", "", answer).strip()
    return reasoning, answer


def extract_question_and_image(row: dict) -> tuple[str, str]:
    prompt = row.get("origin_prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("origin_prompt is not a non-empty list")
    parts = prompt[0].get("prompt")
    if not isinstance(parts, list) or not parts:
        raise ValueError("origin_prompt[0] has no 'prompt' list")
    texts = [p["text"] for p in parts if p.get("type") == "text" and p.get("text")]
    images = [p for p in parts if p.get("type") == "image_url"]
    if len(texts) != 1:
        raise ValueError(f"expected exactly 1 text part, got {len(texts)}")
    if len(images) != 1:
        raise ValueError(f"expected exactly 1 image part, got {len(images)}")
    url = images[0].get("image_url", {}).get("url", "")
    if not url.startswith("file://"):
        raise ValueError(f"image url is not a file:// path: {url!r}")
    return texts[0], url[len("file://") :]


def merge_think(reasoning: str, answer: str) -> str:
    if not reasoning.strip():
        return answer
    return MERGE_OPEN + reasoning.strip("\n") + MERGE_CLOSE + answer


# ── Token length filter (optional) ──────────────────────────────────────
def _token_lengths(records, model_path):
    from veomni.data.chat_template import build_chat_template
    from veomni.data.data_transform import build_data_transform
    from veomni.models.auto import build_processor

    processor = build_processor(str(model_path), max_pixels=602112)
    chat = build_chat_template("minimax_m3_vl", processor)
    transform = build_data_transform("minimax_m3_vl", processor=processor, chat_template=chat, position_id_func=None)
    out = []
    for rec in records:
        sample = {**rec, "source_name": "sharegpt4v_sft"}
        out.append(transform(sample, source_name="sharegpt4v_sft")[0]["input_ids"].numel())
    return out


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    ap.add_argument("--dry-run", action="store_true", help="report what would be kept without writing anything")
    ap.add_argument("--max-tokens", type=int, default=None, help="drop samples longer than this; needs the veomni env")
    ap.add_argument("--model-path", default=DEFAULT_MODEL, help="for --max-tokens tokenization")
    ap.add_argument(
        "--repeat", type=int, default=None, help="epochs; defaults to the smallest N reaching --target-rows"
    )
    ap.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help="rows the run consumes = train.max_steps * train.global_batch_size",
    )
    args = ap.parse_args()

    # ── 1. Parse rollout, grade answers, keep correct + think ───────────
    rows = [json.loads(line) for line in args.src.open(encoding="utf-8") if line.strip()]
    rows.sort(key=lambda row: row["id"])
    is_correct = _make_grader()

    dropped: Counter = Counter()
    kept: list[tuple[str, str, str, str]] = []  # question, image_path, reasoning, answer

    for row in rows:
        if row.get("success") is False:
            dropped["failed_request"] += 1
            continue
        prediction = row.get("prediction") or ""
        gold = row.get("gold")
        if not isinstance(prediction, str) or not prediction.strip():
            dropped["empty_prediction"] += 1
            continue
        if gold in (None, ""):
            dropped["missing_gold"] += 1
            continue

        try:
            question, image_path = extract_question_and_image(row)
        except ValueError:
            dropped["malformed_prompt"] += 1
            continue

        reasoning, answer_line = split_prediction(prediction, row.get("content") or "")
        if not answer_line:
            dropped["truncated_no_final_answer"] += 1
            continue

        boxed = extract_last_boxed(answer_line)
        if boxed is None or not boxed.strip():
            dropped["no_boxed_answer"] += 1
            continue

        if not is_correct(boxed, str(gold)):
            dropped["incorrect_answer"] += 1
            continue

        if not pathlib.Path(image_path).is_file():
            dropped["missing_image"] += 1
            continue

        kept.append((question, image_path, reasoning, f"\\boxed{{{boxed}}}"))

    print(f"source : {args.src}")
    print(f"  {len(rows)} rows")
    print(f"kept   : {len(kept)}")
    print(f"dropped: {len(rows) - len(kept)}")
    for reason, count in dropped.most_common():
        print(f"    - {reason:28s} {count}")

    if not kept:
        raise SystemExit("nothing left after filtering; refusing to write an empty dataset")
    if args.dry_run:
        print("\ndry-run: nothing written.")
        return

    # ── 2. Build ShareGPT records ───────────────────────────────────────
    epoch = []
    for question, image_path, reasoning, answer in kept:
        epoch.append(
            {
                "conversations": [
                    {"from": "human", "value": f"{IMAGE_PLACEHOLDER}{question}"},
                    {"from": "gpt", "value": merge_think(reasoning, answer)},
                ],
                "images": [image_path],
            }
        )

    # ── 3. Optional token-length filter ─────────────────────────────────
    if args.max_tokens is not None:
        lengths = _token_lengths(epoch, args.model_path)
        kept_rec = [r for r, n in zip(epoch, lengths, strict=False) if n <= args.max_tokens]
        dropped_n = sorted((n for n in lengths if n > args.max_tokens), reverse=True)
        if not kept_rec:
            raise SystemExit(f"--max-tokens {args.max_tokens} dropped every sample (longest {max(lengths)})")
        print(f"--max-tokens {args.max_tokens}: kept {len(kept_rec)}/{len(epoch)}, dropped {len(dropped_n)}")
        if dropped_n:
            print(f"  dropped lengths (desc): {dropped_n[:10]}{' ...' if len(dropped_n) > 10 else ''}")
        epoch = kept_rec

    # ── 4. Copy images locally ──────────────────────────────────────────
    img_dir = args.out_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    for i, rec in enumerate(epoch):
        name = f"{i:04d}.png"
        shutil.copy2(rec["images"][0], img_dir / name)
        rec["images"] = [str(img_dir / name)]

    # ── 5. Repeat to target rows ────────────────────────────────────────
    repeat = args.repeat if args.repeat is not None else max(1, math.ceil(args.target_rows / len(epoch)))
    records = epoch * repeat

    out_json = args.out_dir / args.out_name
    out_json.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")

    n_img = len(list(img_dir.glob("*.png")))
    size = sum(p.stat().st_size for p in img_dir.glob("*.png"))
    print(f"\nwrote  : {out_json}")
    print(f"  {len(records)} records ({len(epoch)} x {repeat} epochs), {n_img} images ({size / 1e6:.1f} MB)")
    print(
        f"  train_steps = floor({len(records)} / dp_size / dataloader_batch_size) "
        f"-> {len(records) // 16} at 16 cards (capped by train.max_steps)"
    )


if __name__ == "__main__":
    main()
