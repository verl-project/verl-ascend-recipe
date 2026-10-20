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

"""Explicit fixed-K device probe subprocess for offline calibration."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from .calibration import digest, read_bank, write_json


def run_probe(plan: dict, output: Path, k: int) -> None:
    if os.environ.get("VERL_ADAPTIVE_SPEC_CONFIG"):
        raise RuntimeError("unset VERL_ADAPTIVE_SPEC_CONFIG before offline calibration")
    if os.environ.get("VERL_MTP_KV_PREEMPT_MODE") not in (None, "", "original"):
        raise RuntimeError("calibration uses the native fixed-K scheduler; unset KV adapter mode")
    os.environ["VERL_ADAPTIVE_SPEC_CALIBRATION_DIR"] = str(output.resolve())
    os.environ["VERL_ADAPTIVE_SPEC_CALIBRATION_PLAN"] = json.dumps(
        {
            "k": k,
            "states": plan["states"],
            "warmup_steps": plan["warmup_steps"],
            "length_window_tokens": plan["length_window_tokens"],
        }
    )
    plugins = [p.strip() for p in os.environ.get("VLLM_PLUGINS", "ascend").split(",") if p.strip()]
    if "ascend" not in plugins:
        plugins.insert(0, "ascend")
    if "adaptive_spec_calibration_plugin" not in plugins:
        plugins.append("adaptive_spec_calibration_plugin")
    os.environ["VLLM_PLUGINS"] = ",".join(plugins)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from .vllm_plugin.compatibility import installed_adapter_name

    adapter, vllm_version, ascend_version = installed_adapter_name()
    if adapter != "v0_23":
        raise RuntimeError(f"calibration requires the supported 0.23 stack; got {vllm_version}/{ascend_version}")
    from vllm import LLM, SamplingParams

    if digest(Path(plan["context_bank"])) != plan["context_bank_sha256"]:
        raise RuntimeError("context bank changed after the selection was frozen")
    bank = {row["context_id"]: row["token_ids"] for row in read_bank(Path(plan["context_bank"]))}
    llm = LLM(
        model=plan["model"], speculative_config={"method": "mtp", "num_speculative_tokens": k}, **plan["engine_args"]
    )
    counts = []
    for selection in plan["selections"]:
        state_index, repeat = selection["state_index"], selection["repeat"]
        state = plan["states"][state_index]
        prompts = [
            {"prompt_token_ids": bank[identity][: state["context_length"]]} for identity in selection["context_ids"]
        ]
        params = SamplingParams(
            **plan["sampling"],
            max_tokens=plan["probe_tokens"],
            ignore_eos=True,
            seed=selection["sampling_seed"],
            extra_args={"adaptive_spec_calibration_state": state_index, "adaptive_spec_calibration_repeat": repeat},
        )
        generated = llm.generate(prompts, params, use_tqdm=False)
        if len(generated) != state["batch_size"] or any(
            len(row.outputs) != 1
            or len(row.outputs[0].token_ids) != plan["probe_tokens"]
            or row.outputs[0].finish_reason != "length"
            for row in generated
        ):
            raise RuntimeError(f"incomplete calibration continuation for state={state_index}, repeat={repeat}")
        counts.append({"state_index": state_index, "repeat": repeat, "responses": len(generated)})
    if not list(output.glob("steps.*.jsonl")):
        raise RuntimeError("calibration observer did not publish any clean device steps")
    write_json(
        output / "result.json",
        {
            "status": "complete",
            "k": k,
            "responses": counts,
            "vllm_version": vllm_version,
            "vllm_ascend_version": ascend_version,
            "recipe_version": importlib.metadata.version("verl-adaptive-spec"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.k not in plan["k_values"] or args.k == 11:
        raise ValueError("probe K is not an approved plan candidate")
    run_probe(plan, args.output, args.k)


if __name__ == "__main__":
    main()
