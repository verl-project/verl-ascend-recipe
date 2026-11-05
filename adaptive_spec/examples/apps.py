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

"""Prepare APPS, generate through verl, and score the saved responses."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import random
import time
from pathlib import Path


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(args):
    if args.limit < 0:
        raise ValueError("limit must be zero (all rows) or positive")
    selected, checksum, rng = [], hashlib.sha256(), random.Random(args.seed)
    total = 0
    with args.input.open("rb") as stream:
        for line in stream:
            checksum.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            entry = (total, row)
            total += 1
            if not args.limit or len(selected) < args.limit:
                selected.append(entry)
            else:
                slot = rng.randrange(total)
                if slot < args.limit:
                    selected[slot] = entry
    if not total or args.limit > total:
        raise ValueError(f"requested {args.limit} rows from a source containing {total}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for ordinal, row in sorted(selected):
            cases = json.loads(row.get("input_output") or "{}")
            starter = row.get("starter_code") or ""
            interface = (
                f"Implement the function {cases['fn_name']} using the supplied interface."
                if cases.get("fn_name")
                else "Read input from stdin and write the answer to stdout."
            )
            prompt = f"Write a Python solution. {interface}\nReturn the solution in a ```python code block.\n\n"
            prompt += row["question"]
            if starter:
                prompt += "\n\nStarter code:\n```python\n" + starter + "\n```"
            result = {
                "id": f"apps-{args.split}-{row['id'] if 'id' in row else row['problem_id']}",
                "row_index": ordinal,
                "messages": [{"role": "user", "content": prompt}],
                "test_cases": cases,
                "difficulty": row["difficulty"],
            }
            stream.write(json.dumps(result) + "\n")
    dump(
        args.output.with_suffix(".selection.json"),
        {
            "source_sha256": checksum.hexdigest(),
            "source_rows": total,
            "selected_rows": len(selected),
            "seed": args.seed,
        },
    )


def response_seed(base, row_index, response_index, responses):
    seed = base + row_index * responses + response_index
    if not 0 <= seed < 2**32:
        raise ValueError("response seed is outside the unsigned 32-bit range")
    return seed


def generated_tokens(tokens, mask):
    length = sum(mask)
    if len(tokens) != len(mask) or mask != [1] * length + [0] * (len(mask) - length):
        raise ValueError("single-turn response mask must contain a token prefix followed by padding")
    return tokens[:length]


def close_rollout_engine(server):
    from vllm.utils.async_utils import cancel_task_threadsafe

    engine = server.engine
    cancel_task_threadsafe(getattr(engine, "output_handler", None))
    engine.shutdown(timeout=60)


async def rollout(args):
    started = time.monotonic()
    import numpy as np
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import verl
    from adaptive_spec.verl_integration import AdaptiveSpecAgentLoopManager, AdaptiveSpecAgentLoopWorker
    from verl.protocol import DataProto
    from verl.workers.rollout.llm_server import LLMServerManager

    class SeededWorker(AdaptiveSpecAgentLoopWorker):
        async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
            params = {**sampling_params, "seed": int(kwargs.pop("sampling_seed"))}
            return await super()._run_agent_loop(params, trajectory, **kwargs)

    class SeededManager(AdaptiveSpecAgentLoopManager):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.agent_loop_workers_class = ray.remote(SeededWorker)

    if min(args.devices, args.tp, args.responses, args.batch_prompts) < 1 or args.devices % args.tp:
        raise ValueError("device count must be a positive multiple of TP; batch and response counts must be positive")
    if args.k < 1 or args.k == 11:
        raise ValueError("K must be positive and different from K11")
    if args.arm == "adaptive":
        if args.config is None:
            raise ValueError("adaptive rollout requires --config")
        from adaptive_spec.config import AdaptiveSpecConfig

        runtime = AdaptiveSpecConfig.from_path(args.config)
        if runtime.build_controller() is None:
            raise ValueError("adaptive rollout requires an enabled cost table")
        with Path(runtime.phase_map_path).open(encoding="utf-8") as stream:
            phase_map = json.load(stream)
        if max(k for cell in phase_map["cells"] for k in cell["allowed_ks"]) > args.k:
            raise ValueError("startup K must cover every depth in the cost table")
        os.environ["VERL_ADAPTIVE_SPEC_CONFIG"] = str(args.config.resolve())
        os.environ["VLLM_PLUGINS"] = "ascend,adaptive_spec_plugin"
    else:
        os.environ.pop("VERL_ADAPTIVE_SPEC_CONFIG", None)
        os.environ["VLLM_PLUGINS"] = "ascend,adaptive_spec_plugin"
    os.environ["VERL_ASCEND_CLEANUP_GRACE"] = "1"
    records = read_rows(args.input)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("input must contain unique nonempty problem IDs")
    for record in records:
        response_seed(args.seed, record["row_index"], args.responses - 1, args.responses)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    args.output.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(version_base=None, config_dir=str(Path(verl.__file__).parent / "trainer/config")):
        cfg = compose(config_name="ppo_trainer")
    settings = {
        "trainer.device": "npu",
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": args.devices,
        "actor_rollout_ref.model.path": str(args.model.resolve()),
        "actor_rollout_ref.model.mtp.enable": True,
        "actor_rollout_ref.model.mtp.enable_rollout": True,
        "actor_rollout_ref.model.mtp.num_speculative_tokens": args.k,
        "data.apply_chat_template_kwargs": {"enable_thinking": args.thinking},
        "data.max_prompt_length": args.prompt_length,
        "data.max_response_length": args.response_length,
    }
    rollout_settings = {
        "name": "vllm",
        "nnodes": 1,
        "n_gpus_per_node": args.devices,
        "tensor_model_parallel_size": args.tp,
        "n": args.responses,
        "seed": args.seed,
        "load_format": "auto",
        "dtype": "bfloat16",
        "max_num_seqs": 16,
        "max_model_len": args.prompt_length + args.response_length,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.5,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "enable_sleep_mode": False,
        "calculate_log_probs": args.logprobs,
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        "temperature": args.temperature,
        "top_p": 1.0,
        "agent.num_workers": 1,
        "engine_kwargs.vllm": {"async_scheduling": True},
    }
    if args.router:
        rollout_settings["router_config_path"] = str(args.router.resolve())
    for key, value in rollout_settings.items():
        settings["actor_rollout_ref.rollout." + key] = value
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    OmegaConf.save(cfg, args.output / "verl_config.yaml")
    ray.init(address="local", include_dashboard=False, num_cpus=args.cpus, resources={"NPU": args.devices})
    totals, elapsed, completed = 0, 0.0, 0
    servers = None
    try:
        servers = await LLMServerManager.create(cfg)
        manager = await SeededManager.create(cfg, servers.get_client())
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
        with (args.output / "responses.jsonl").open("x", encoding="utf-8") as responses:
            with (args.output / "token_records.jsonl").open("x", encoding="utf-8") as contexts:
                for step, start in enumerate(range(0, len(records), args.batch_prompts)):
                    samples = [
                        (r, i) for r in records[start : start + args.batch_prompts] for i in range(args.responses)
                    ]
                    raw = np.empty(len(samples), dtype=object)
                    raw[:] = [r["messages"] for r, _ in samples]
                    indices = [r["id"] for r, _ in samples]
                    seeds = [response_seed(args.seed, r["row_index"], i, args.responses) for r, i in samples]
                    batch = DataProto.from_dict(
                        non_tensors={
                            "raw_prompt": raw,
                            "index": np.array(indices, dtype=object),
                            "sampling_seed": np.array(seeds, dtype=np.int64),
                        },
                        meta_info={"global_steps": step, "validate": False},
                    )
                    begin = time.monotonic()
                    output = await manager.generate_sequences(batch)
                    duration = time.monotonic() - begin
                    if output.non_tensor_batch["index"].tolist() != indices:
                        raise RuntimeError("returned sample order differs from the requested order")
                    elapsed += duration
                    for n, (record, reply) in enumerate(samples):
                        tokens = generated_tokens(
                            output.batch["responses"][n].tolist(), output.batch["response_mask"][n].tolist()
                        )
                        prompt_mask = output.batch["attention_mask"][n][: output.batch["prompts"].shape[1]].bool()
                        prompt = output.batch["prompts"][n][prompt_mask].tolist()
                        identity = f"{record['id']}-r{reply}"
                        row = {
                            "id": record["id"],
                            "response_index": reply,
                            "seed": seeds[n],
                            "text": tokenizer.decode(tokens, skip_special_tokens=True),
                            "output_tokens": len(tokens),
                            "length_cap_hit": len(tokens) == args.response_length,
                        }
                        if args.logprobs:
                            row["logprobs"] = output.batch["rollout_log_probs"][n][: len(tokens)].tolist()
                        responses.write(json.dumps(row, allow_nan=False) + "\n")
                        contexts.write(
                            json.dumps(
                                {"context_id": identity, "prompt_token_ids": prompt, "response_token_ids": tokens}
                            )
                            + "\n"
                        )
                        totals += len(tokens)
                        completed += 1
                    responses.flush()
                    contexts.flush()
                    dump(args.output / f"step_{step}.json", {"responses": len(samples), "rollout_seconds": duration})
    finally:
        try:
            if servers is not None:
                await asyncio.gather(
                    *[handle.__ray_call__.remote(close_rollout_engine) for handle in servers.server_handles]
                )
        finally:
            ray.shutdown()
    if completed != len(records) * args.responses:
        raise RuntimeError("incomplete rollout")
    dump(
        args.output / "summary.json",
        {
            "arm": args.arm,
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "prompts": len(records),
            "responses": completed,
            "responses_per_prompt": args.responses,
            "output_tokens": totals,
            "rollout_seconds": elapsed,
            "rollout_response_tok_s": totals / elapsed,
            "driver_wall_seconds": time.monotonic() - started,
            "seed": args.seed,
            "startup_k": args.k,
        },
    )


def score_one(pair):
    from verl.utils.reward_score.prime_code.utils import check_correctness

    response, problem, timeout = pair
    cases = problem["test_cases"]
    if not cases.get("inputs") or len(cases["inputs"]) != len(cases.get("outputs", [])):
        return {"id": response["id"], "response_index": response["response_index"], "scorable": False}
    solution = response["text"].split("```python")[-1].split("```")[0]
    result, _ = check_correctness(cases, solution, timeout=timeout, debug=False)
    return {
        "id": response["id"],
        "response_index": response["response_index"],
        "scorable": True,
        "correct": len(result) == len(cases["inputs"]) and set(result) == {True},
    }


def score(args):
    problems = {row["id"]: row for row in read_rows(args.input)}
    responses = read_rows(args.responses)
    with args.responses.with_name("summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary["input_sha256"] != hashlib.sha256(args.input.read_bytes()).hexdigest():
        raise ValueError("scoring input differs from the rollout input")
    identities = [(r["id"], r["response_index"]) for r in responses]
    expected = {(identity, reply) for identity in problems for reply in range(summary["responses_per_prompt"])}
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError("scoring requires every expected response exactly once")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(score_one, [(r, problems[r["id"]], args.timeout) for r in responses]))
    scorable = sum(row["scorable"] for row in rows)
    correct = sum(row.get("correct", False) for row in rows)
    dump(args.output, {"responses": len(rows), "scorable": scorable, "correct": correct, "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--split", choices=["train", "test"], required=True)
    prepare_parser.add_argument("--limit", type=int, default=32, help="randomly selected prompts; 0 uses every row")
    prepare_parser.add_argument("--seed", type=int, default=42)
    run = commands.add_parser("rollout")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--arm", choices=["fixed", "adaptive"], required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--router", type=Path)
    run.add_argument("--k", type=int, default=12, help="startup MTP depth/capacity")
    run.add_argument("--devices", type=int, default=8)
    run.add_argument("--tp", type=int, default=2)
    run.add_argument("--responses", type=int, default=8)
    run.add_argument("--batch-prompts", type=int, default=32)
    run.add_argument("--prompt-length", type=int, default=4096)
    run.add_argument("--response-length", type=int, default=65536)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=20261103)
    run.add_argument("--cpus", type=int, default=24)
    run.add_argument("--thinking", action="store_true")
    run.add_argument("--logprobs", action="store_true")
    judge = commands.add_parser("score")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--responses", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--workers", type=int, default=8)
    judge.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "rollout":
        asyncio.run(rollout(args))
    else:
        score(args)


if __name__ == "__main__":
    main()
