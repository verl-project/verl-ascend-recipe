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

"""Offline B/L/K collection, reduction and controller-ready artifacts.

The device backend is imported only by an explicitly launched probe subprocess.
Replay validates the artifact pipeline; it is never labelled a device measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import AdaptiveSpecConfig
from .phase_map import PhaseMap


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_bank(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    identities = set()
    for row in rows:
        identity = row["context_id"]
        tokens = row["token_ids"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("context_id must be unique and nonempty")
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("token_ids must be a nonempty list of nonnegative integers")
        identities.add(identity)
    if not rows:
        raise ValueError("context bank is empty")
    return rows


def validate_plan(raw: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    plan = dict(raw)
    if plan.get("schema_version") != 1:
        raise ValueError("calibration plan requires schema_version=1")
    for key in ("model", "model_fingerprint", "topology_fingerprint", "context_bank"):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError(f"missing calibration {key}")
    defaults = {
        "repeats": 3,
        "seed": 42,
        "probe_tokens": 256,
        "warmup_steps": 2,
        "minimum_steps": 8,
        "length_window_tokens": 256,
        "regret_fraction": 0.1,
        "prior_pseudocount": 0.5,
        "probe_timeout_seconds": 3600,
    }
    for key, value in defaults.items():
        plan.setdefault(key, value)
    for key in ("repeats", "probe_tokens", "minimum_steps", "length_window_tokens", "probe_timeout_seconds"):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(plan["warmup_steps"]) is not int or plan["warmup_steps"] < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if type(plan["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not 0 <= plan["seed"] <= 2**32 - plan["repeats"]:
        raise ValueError("paired sampling seeds must fit the unsigned 32-bit range")
    for key in ("regret_fraction", "prior_pseudocount"):
        if not isinstance(plan[key], (float, int)) or not math.isfinite(plan[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 <= plan["regret_fraction"] < 1 or plan["prior_pseudocount"] < 0:
        raise ValueError("invalid regret_fraction or prior_pseudocount")
    ks = plan.get("k_values")
    if not isinstance(ks, list) or not ks or any(type(k) is not int or k < 1 or k == 11 for k in ks):
        raise ValueError("k_values must contain positive depths excluding K11")
    if len(set(ks)) != len(ks):
        raise ValueError("duplicate K values")
    plan["k_values"] = sorted(ks)
    states = plan.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a nonempty B/L grid")
    seen = set()
    for state in states:
        for key in ("batch_size", "context_length"):
            if type(state.get(key)) is not int or state[key] < 1:
                raise ValueError(f"state {key} must be positive")
        pair = (state["batch_size"], state["context_length"])
        if pair in seen:
            raise ValueError("duplicate B/L states")
        seen.add(pair)
    args = dict(plan.get("engine_args", {}))
    fixed = {
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "data_parallel_size": 1,
    }
    for key, value in fixed.items():
        if key in args and args[key] != value:
            raise ValueError(f"calibration requires {key}={value}")
    if "speculative_config" in args or "model" in args:
        raise ValueError("model and speculative_config are controlled by the plan")
    args.update(fixed)
    maximum_b = max(s["batch_size"] for s in states)
    args.setdefault("max_num_seqs", maximum_b)
    if type(args["max_num_seqs"]) is not int or args["max_num_seqs"] < maximum_b:
        raise ValueError("max_num_seqs is smaller than the B grid")
    required_length = max(s["context_length"] for s in states) + plan["probe_tokens"] + max(ks) + 1
    if type(args.get("max_model_len")) is not int or args["max_model_len"] < required_length:
        raise ValueError(f"max_model_len must provide probe/draft headroom: >= {required_length}")
    plan["engine_args"] = args
    sampling = dict(plan.get("sampling", {}))
    if set(sampling) - {"temperature", "top_p", "top_k", "min_p", "repetition_penalty"}:
        raise ValueError("unsupported calibration sampling field")
    plan["sampling"] = sampling
    bank_path = Path(plan["context_bank"])
    if not bank_path.is_absolute():
        bank_path = plan_path.resolve().parent / bank_path
    bank = read_bank(bank_path)
    plan["context_bank"] = str(bank_path.resolve())
    plan["context_bank_sha256"] = digest(bank_path)
    # Selection never depends on K: all depths see the same prefixes in a repeat.
    selections = []
    for repeat in range(plan["repeats"]):
        for state_index, state in enumerate(states):
            eligible = [row for row in bank if len(row["token_ids"]) >= state["context_length"]]
            rng = random.Random(plan["seed"] + repeat * 100003 + state_index)
            if len(eligible) < state["batch_size"]:
                raise ValueError(f"not enough distinct natural contexts for state {state}")
            chosen = rng.sample(eligible, state["batch_size"])
            selections.append(
                {
                    "repeat": repeat,
                    "state_index": state_index,
                    "context_ids": [row["context_id"] for row in chosen],
                    "prefix_sha256": [
                        hashlib.sha256(
                            json.dumps(row["token_ids"][: state["context_length"]], separators=(",", ":")).encode()
                        ).hexdigest()
                        for row in chosen
                    ],
                    "sampling_seed": plan["seed"] + repeat,
                }
            )
    plan["selections"] = selections
    return plan


def reduce_records(plan: dict[str, Any], records: list[dict[str, Any]], *, backend: str) -> dict[str, Any]:
    cells = []
    max_k = max(plan["k_values"])
    for state_index, state in enumerate(plan["states"]):
        costs, progress = {}, {}
        exposures, successes = [0] * max_k, [0] * max_k
        receipts = []
        for k in plan["k_values"]:
            repeat_costs, repeat_progress = [], []
            for repeat in range(plan["repeats"]):
                group = [r for r in records if (r["state_index"], r["k"], r["repeat"]) == (state_index, k, repeat)]
                if len(group) < plan["minimum_steps"]:
                    raise ValueError(f"insufficient clean steps: state={state_index}, K={k}, repeat={repeat}")
                latency, emitted = [], []
                for record in group:
                    if record.get("clean") is not True or record["batch_size"] != state["batch_size"]:
                        raise ValueError("dirty step or changed B in calibration records")
                    lengths = record["context_lengths"]
                    accepted = record["accepted_draft_tokens"]
                    if len(lengths) != state["batch_size"] or len(accepted) != state["batch_size"]:
                        raise ValueError("calibration row count mismatch")
                    if any(
                        type(length) is not int
                        or not state["context_length"]
                        <= length
                        <= state["context_length"] + plan["length_window_tokens"]
                        for length in lengths
                    ):
                        raise ValueError("step lies outside the declared context window")
                    cost = record["iteration_latency_ns"]
                    if (
                        isinstance(cost, bool)
                        or not isinstance(cost, (int, float))
                        or not math.isfinite(cost)
                        or cost <= 0
                    ):
                        raise ValueError("invalid iteration cost")
                    if any(type(a) is not int or not 0 <= a <= k for a in accepted):
                        raise ValueError("invalid acceptance count")
                    latency.append(cost)
                    emitted.append(statistics.mean(1 + a for a in accepted))
                    for a in accepted:
                        for depth in range(k):
                            exposures[depth] += 1
                            successes[depth] += a > depth
                repeat_costs.append(statistics.median(latency))
                repeat_progress.append(statistics.mean(emitted))
                receipts.append({"k": k, "repeat": repeat, "clean_steps": len(group)})
            costs[k] = statistics.median(repeat_costs)
            progress[k] = statistics.median(repeat_progress)
        alpha = plan["prior_pseudocount"]
        prior = []
        for success, exposure in zip(successes, exposures, strict=True):
            if exposure == 0:
                raise ValueError("unexposed draft depth cannot be filled with a fabricated prior")
            value = (success + alpha) / (exposure + 2 * alpha)
            prior.append(min(value, prior[-1]) if prior else value)
        scores = {k: progress[k] / costs[k] for k in costs}
        cutoff = max(scores.values()) * (1 - plan["regret_fraction"])
        allowed = sorted(k for k, score in scores.items() if score >= cutoff)
        cells.append(
            {
                "phase_id": f"b{state['batch_size']}-l{state['context_length']}",
                **state,
                "allowed_ks": allowed,
                "iteration_latency_ns_by_k": {str(k): costs[k] for k in allowed},
                "prior_survival": prior,
                "fallback_k": min(allowed),
                "measurement": {
                    "cost_ns_all_ks": {str(k): costs[k] for k in costs},
                    "score_all_ks": {str(k): scores[k] for k in scores},
                    "exposures_by_depth": exposures,
                    "accepted_prefix_by_depth": successes,
                    "repeats": receipts,
                },
            }
        )
    result = {
        "schema_version": 1,
        "model_fingerprint": plan["model_fingerprint"],
        "topology_fingerprint": plan["topology_fingerprint"],
        "cells": cells,
        "collection": {
            "backend": backend,
            "device_measurement": backend == "vllm",
            "context_bank_sha256": plan["context_bank_sha256"],
            "cost_definition": (
                "sum of synchronized execute_model and sample_tokens wall times; prefill/warmup/drain excluded"
            ),
            "engine_args": plan["engine_args"],
            "sampling": plan["sampling"],
            "regret_fraction": plan["regret_fraction"],
            "prior_pseudocount": plan["prior_pseudocount"],
        },
    }
    PhaseMap.from_dict(result)
    return result


def _stop_probe_group(process: subprocess.Popen) -> None:
    """Stop the session created for one probe, including surviving workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # wait() only reaps the leader. Workers may survive even when it returns.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def collect(plan: dict[str, Any], output: Path, *, backend: str, replay: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.resolved.json", plan)
    records = []
    probe_receipts = []
    try:
        if backend == "replay":
            if replay is None:
                raise ValueError("replay requires --records")
            records = [json.loads(line) for line in replay.read_text(encoding="utf-8").splitlines() if line.strip()]
            write_json(output / "replay_source.json", {"sha256": digest(replay), "device_measurement": False})
            (output / "raw_steps.jsonl").write_text(replay.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            if os.name != "posix":
                raise RuntimeError(
                    "device collection runs in the Linux Ascend environment; use --plan-only or replay locally"
                )
            # One fresh fixed-K Engine per subprocess. No adaptive controller runs during calibration.
            order = list(plan["k_values"])
            random.Random(plan["seed"]).shuffle(order)
            for k in order:
                probe_dir = output / f"k{k}"
                probe_dir.mkdir()
                args = [
                    sys.executable,
                    "-m",
                    "adaptive_spec.calibration_vllm",
                    "--plan",
                    str((output / "plan.resolved.json").resolve()),
                    "--output",
                    str(probe_dir.resolve()),
                    "--k",
                    str(k),
                ]
                with (probe_dir / "run.log").open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        process.wait(timeout=plan["probe_timeout_seconds"])
                    except BaseException:
                        _stop_probe_group(process)
                        raise
                if process.returncode:
                    _stop_probe_group(process)
                    raise RuntimeError(f"fixed-K probe failed: K={k}; see {probe_dir / 'run.log'}")
                receipt = json.loads((probe_dir / "result.json").read_text(encoding="utf-8"))
                if receipt.get("status") != "complete":
                    raise RuntimeError(f"probe did not complete: K={k}")
                probe_receipts.append(receipt)
                for path in sorted(probe_dir.glob("steps.*.jsonl")):
                    records.extend(
                        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                    )
            with (output / "raw_steps.jsonl").open("x", encoding="utf-8") as stream:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        phase_map = reduce_records(plan, records, backend=backend)
        phase_map["collection"]["backend_probes"] = probe_receipts
        phase_map["collection"]["plan_sha256"] = digest(output / "plan.resolved.json")
        phase_map["collection"]["raw_steps_sha256"] = digest(output / "raw_steps.jsonl")
        write_json(output / "phase_map.json", phase_map)
        runtime = {
            "schema_version": 1,
            "enabled": backend == "vllm",
            "phase_map_path": str((output / "phase_map.json").resolve()),
            "model_fingerprint": plan["model_fingerprint"],
            "topology_fingerprint": plan["topology_fingerprint"],
        }
        # Actual consumer loads the generated map, including identity/candidate guards.
        validation = {**runtime, "enabled": True}
        AdaptiveSpecConfig.from_dict(validation).build_controller()
        write_json(output / "runtime.json", runtime)
        write_json(
            output / "status.json",
            {
                "status": "complete",
                "backend": backend,
                "device_measurement": backend == "vllm",
                "cells": len(phase_map["cells"]),
                "phase_map_sha256": digest(output / "phase_map.json"),
            },
        )
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("vllm", "replay"), default="vllm")
    parser.add_argument("--records", type=Path, help="replay-only JSONL; not a device benchmark")
    parser.add_argument(
        "--plan-only", action="store_true", help="validate data/grid and save selections without loading a model"
    )
    args = parser.parse_args()
    if args.records and args.backend != "replay":
        parser.error("--records is only valid with --backend replay")
    plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8")), args.plan)
    if args.plan_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "plan.resolved.json", plan)
    else:
        collect(plan, args.output_dir, backend=args.backend, replay=args.records)
    print(
        json.dumps({"output_dir": str(args.output_dir.resolve()), "plan_only": args.plan_only, "backend": args.backend})
    )


if __name__ == "__main__":
    main()
