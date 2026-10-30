# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export natural tokenized rollout prefixes for offline calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import digest, read_bank, write_json


def build(input_path: Path, output: Path) -> int:
    if output.exists() or output.with_suffix(output.suffix + ".receipt.json").exists():
        raise FileExistsError("context bank or its receipt already exists")
    rows, seen = [], set()
    with input_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row["context_id"]
            prompt, response = row["prompt_token_ids"], row.get("response_token_ids", [])
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("context_id must be a unique nonempty string")
            if not isinstance(prompt, list) or not isinstance(response, list) or not prompt:
                raise ValueError("tokenized rollout requires a nonempty prompt and a response token list")
            tokens = prompt + response
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("invalid token ID")
            seen.add(identity)
            rows.append({"context_id": identity, "token_ids": tokens})
    if not rows:
        raise ValueError("tokenized rollout input is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    read_bank(output)
    write_json(
        output.with_suffix(output.suffix + ".receipt.json"),
        {
            "schema_version": 1,
            "source_sha256": digest(input_path),
            "context_bank_sha256": digest(output),
            "contexts": len(rows),
        },
    )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"contexts": build(args.input, args.output), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
