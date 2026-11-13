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

"""Convert measured probe output into the stable adaptive-spec schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from adaptive_spec.phase_map import PhaseMap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    source: Mapping[str, Any],
    *,
    topology_fingerprint: str,
    excluded_ks: set[int] | None = None,
) -> dict[str, Any]:
    excluded = set() if excluded_ks is None else {int(value) for value in excluded_ks}
    cells = []
    global_depth = max(
        (int(k) for state in source["states"] for k in state["allowed_ks"] if int(k) not in excluded), default=0
    )
    for state in source["states"]:
        allowed = tuple(sorted(int(value) for value in state["allowed_ks"] if int(value) not in excluded))
        if not allowed:
            raise ValueError(f"phase {state.get('phase_id')} has no safe K after exclusions")
        fallback = int(state.get("regret10_min_k", min(allowed)))
        if fallback not in allowed:
            fallback = min(allowed)
        raw_prior = state["prior_survival_by_depth"]
        max_depth = global_depth
        prior = [float(raw_prior[str(depth)]) for depth in range(1, max_depth + 1)]
        costs = {str(k): float(state["iteration_latency_ns_by_k"][str(k)]) for k in allowed}
        cells.append(
            {
                "phase_id": str(state.get("phase_id", f"b{int(state['b'])}-l{int(state['l'])}")),
                "batch_size": int(state["b"]),
                "context_length": int(state["l"]),
                "allowed_ks": list(allowed),
                "iteration_latency_ns_by_k": costs,
                "prior_survival": prior,
                "fallback_k": fallback,
            }
        )
    result = {
        "schema_version": 1,
        "model_fingerprint": str(source["model_fingerprint"]),
        "topology_fingerprint": str(topology_fingerprint),
        "cells": cells,
    }
    PhaseMap.from_dict(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology-fingerprint", required=True)
    parser.add_argument("--exclude-k", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite phase map: {args.output}")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    result = convert(
        source,
        topology_fingerprint=args.topology_fingerprint,
        excluded_ks=set(args.exclude_k),
    )
    result["source"] = {
        "path": str(args.input.resolve()),
        "sha256": _sha256(args.input),
    }
    if source.get("payload_sha256"):
        result["source"]["payload_sha256"] = str(source["payload_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(result["cells"])}))


if __name__ == "__main__":
    main()
