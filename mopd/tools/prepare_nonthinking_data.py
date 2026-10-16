"""Append the measured GSM8K answer-format instruction; preserve MATH and other fields."""

import argparse
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FORMAT_INSTRUCTION = (
    "\nWrite a short solution. End your response with a separate final line in exactly "
    "this format: #### <number>. Replace <number> with the numeric answer only. "
    "Do not put currency symbols, units, words, LaTeX, or Markdown around the number. "
    "Do not write anything after that final line."
)


def prepare_data(source: Path, output: Path) -> None:
    # Refuse to overwrite existing datasets, including the source itself.
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for dataset in ("gsm8k", "math"):
        for split in ("train", "test"):
            src = source / dataset / f"{split}.parquet"
            if not src.is_file():
                raise FileNotFoundError(src)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary)
        for dataset in ("gsm8k", "math"):
            (staging / dataset).mkdir()
            for split in ("train", "test"):
                src = source / dataset / f"{split}.parquet"
                dst = staging / dataset / f"{split}.parquet"
                if dataset == "math":
                    shutil.copyfile(src, dst)
                    continue
                table = pq.read_table(src)
                rows = table.to_pylist()
                for row in rows:
                    prompt = row["prompt"]
                    if not prompt or prompt[-1]["role"] != "user":
                        raise ValueError(f"Expected a final user message in {src}")
                    if FORMAT_INSTRUCTION in prompt[-1]["content"]:
                        raise ValueError(f"The format instruction is already present in {src}")
                    prompt[-1]["content"] += FORMAT_INSTRUCTION
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), dst)

        # Reserve the destination exclusively after preparation; preserve a directory
        # created by another process while we were working.
        output.mkdir()
        try:
            for dataset in ("gsm8k", "math"):
                (staging / dataset).rename(output / dataset)
        except BaseException:
            shutil.rmtree(output)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_data(args.source, args.output)


if __name__ == "__main__":
    main()
