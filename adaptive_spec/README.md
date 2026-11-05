# AdaptiveSpec

AdaptiveSpec adjusts each inference Engine's MTP draft depth K using the active batch size, context length, and acceptance evidence from other responses to the same prompt. It uses the model's existing draft weights and acceptance/rejection procedure.

| Capability | Fixed-K MTP | AdaptiveSpec |
|---|---|---|
| Draft depth | Fixed K | Adjusted by generation phase and acceptance evidence |
| Decision inputs | Manual configuration | Offline cost table and online sibling-response evidence |
| Multiple requests in one Engine | One fixed K | Maximum of the request-level K recommendations |
| Switching control | None | Two confirmations, 64-round dwell, one ladder step per switch |
| Phase control | None | Phase fallback K before and during saturation; online recommendations during drain |
| TP control messages | No dynamic control | Leader decision broadcast through a reusable fixed-size message |
| Request scheduling | Native routing | Optional Elastic interleaving of responses to the same prompt |

Compared with fixed K, throughput improves by **13.12%** on 9B DAPO-Math and **11.02%** across two pooled 9B APPS runs. See [Performance](PERFORMANCE.md) for configurations and individual runs. The target workload is a finite rollout with multiple responses per prompt, mixed response lengths, and a batch that gradually drains.

## Required verl version

[REQUIRED_VERL.txt](REQUIRED_VERL.txt) records the fixed verl commit and its installation command. The repository's `install_verl.sh` reads this file when installing the recipe's verl dependency.

## Installation and deployment

For a new server, follow [BACKEND.md](BACKEND.md) to start the fixed CANN environment and install the backend. With that environment ready, run from the repository root:

```bash
bash ./install_verl.sh --recipe adaptive_spec
python -m pip install -e ./adaptive_spec
adaptive-spec-check-backend
```

Install the package on both the driver and rollout workers.

See [BACKEND.md](BACKEND.md) for backend versions. The adapter supports eager execution, serial MTP, and PP1/CP1. Candidate K values are positive integers within the startup capacity, excluding K11. The verl integration targets synchronous rollout jobs; vLLM request scheduling can be asynchronous.

1. Collect a cost table for the model and TP configuration to produce `phase_map.json` and `runtime.json`, as described under "Collecting the cost table" below.
2. Set absolute paths, `enabled=true`, and a job-specific evidence-sharing directory in the generated `runtime.json`. Example values illustrate the format.
3. Enable the plugin on the driver and rollout workers:

```bash
export VERL_ADAPTIVE_SPEC_CONFIG=/absolute/runtime.json
export VLLM_PLUGINS=ascend,adaptive_spec_plugin
```

4. Add these overrides to the existing verl launch command:

```text
actor_rollout_ref.rollout.agent.agent_loop_manager_class=adaptive_spec.verl_integration.AdaptiveSpecAgentLoopManager
actor_rollout_ref.rollout.enforce_eager=true
actor_rollout_ref.model.mtp.enable=true
actor_rollout_ref.model.mtp.enable_rollout=true
actor_rollout_ref.model.mtp.num_speculative_tokens=12
```

Startup capacity 12 must cover the largest K in the cost table. Keep the existing model, data, reward, and training configuration.

5. To enable Elastic, also add:

```text
actor_rollout_ref.rollout.router_config_path=/absolute/elastic_interleave.router.yaml
```

Use the [router example](examples/elastic_interleave.router.yaml), with capacity matching each Engine's `max_num_seqs`. Omitting this setting uses native routing.

## Qwen3.5-9B/APPS example

Run these commands inside the installed environment, from the repository root. The example uses 32 test prompts with 8 responses each. `prepare --limit 0` selects the entire split; rollout processes it in batches of 32 prompts.

Download the [model, including its MTP weights](https://huggingface.co/Qwen/Qwen3.5-9B), and [APPS](https://huggingface.co/datasets/codeparrot/apps) at fixed revisions:

```bash
export AS_WORK="$PWD/work-adaptive-spec"
mkdir -p "$AS_WORK"
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir "$AS_WORK/model"
hf download codeparrot/apps train.jsonl test.jsonl --repo-type dataset \
  --revision 21e74ddf8de1a21436da12e3e653065c5213e9d1 --local-dir "$AS_WORK/apps"
python adaptive_spec/examples/apps.py prepare --input "$AS_WORK/apps/train.jsonl" \
  --split train --limit 64 --seed 314159 --output "$AS_WORK/calibration.jsonl"
python adaptive_spec/examples/apps.py prepare --input "$AS_WORK/apps/test.jsonl" \
  --split test --limit 32 --seed 42 --output "$AS_WORK/evaluation.jsonl"
```

Generate natural calibration contexts with fixed K on the training prompts:

```bash
python adaptive_spec/examples/apps.py rollout --model "$AS_WORK/model" \
  --input "$AS_WORK/calibration.jsonl" --output "$AS_WORK/bootstrap" \
  --arm fixed --k 5 --devices 8 --tp 2 --responses 1 --response-length 8192
adaptive-spec-build-contexts --input "$AS_WORK/bootstrap/token_records.jsonl" \
  --output "$AS_WORK/contexts.jsonl"
```

Create a cost-collection plan using the available natural context lengths:

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
work = Path(os.environ["AS_WORK"])
plan = json.loads(Path("adaptive_spec/examples/calibration.plan.example.json").read_text())
lengths = [len(json.loads(line)["token_ids"]) for line in (work / "contexts.jsonl").open()]
plan.update(model=str(work / "model"), context_bank=str(work / "contexts.jsonl"),
            model_fingerprint=hashlib.sha256((work / "model/config.json").read_bytes()).hexdigest(),
            topology_fingerprint="ascend-tp2-eager-bf16-b16")
plan["states"] = [{"batch_size": b, "context_length": length}
                  for b in (1, 8, 16) for length in (512, 1024, 2048, 4096)
                  if sum(n >= length for n in lengths) >= b]
if not any(s["batch_size"] == 16 for s in plan["states"]):
    raise ValueError("Collect more natural calibration responses to cover batch size 16")
plan["engine_args"].update(dtype="bfloat16", gpu_memory_utilization=0.5)
(work / "plan.json").write_text(json.dumps(plan, indent=2))
PY
unset VERL_ADAPTIVE_SPEC_CONFIG
export VLLM_PLUGINS=ascend,adaptive_spec_calibration_plugin
adaptive-spec-collect --plan "$AS_WORK/plan.json" --output-dir "$AS_WORK/plan-check" --plan-only
adaptive-spec-collect --plan "$AS_WORK/plan.json" --output-dir "$AS_WORK/costs"
```

The generated runtime enables maximum-K aggregation, packed TP control and drain-stage control. Set its job-specific evidence directory, then run both arms on the same test prompts and response seeds:

```bash
python - <<'PY'
import json, os
from pathlib import Path
work = Path(os.environ["AS_WORK"])
path = work / "costs/runtime.json"
runtime = json.loads(path.read_text())
runtime["evidence_exchange_directory"] = str(work / "adaptive-evidence")
path.write_text(json.dumps(runtime, indent=2))
PY
for arm in fixed adaptive; do
  options=()
  if [ "$arm" = adaptive ]; then options+=(--config "$AS_WORK/costs/runtime.json"); fi
  python adaptive_spec/examples/apps.py rollout --model "$AS_WORK/model" \
    --input "$AS_WORK/evaluation.jsonl" --output "$AS_WORK/$arm" \
    --arm "$arm" --k 12 --devices 8 --tp 2 --responses 8 --seed 20261103 \
    --router "$PWD/adaptive_spec/examples/elastic_interleave.router.yaml" "${options[@]}"
done
```

This comparison uses fixed K12 and adaptive startup capacity 12. For a best-fixed-K comparison, select K on independent calibration rollouts and pass that K to the fixed arm; keep the adaptive capacity large enough for its cost table.

The rollout example and cost collector give Ascend workers 30 seconds to finish resource cleanup before native termination handling. Both comparison arms use the same cleanup setting.

Each output directory contains `summary.json` (response tokens, rollout time, throughput and driver wall time), `responses.jsonl`, `token_records.jsonl`, and the resolved verl configuration. Per-response seeds use the source row index and response index, so they remain paired across arms and batch boundaries. Outputs stay in the original sample order.

APPS scoring executes the generated Python code in a CPU-only container. After rollout finishes, leave the inference container and run the following on the host, from the repository root. The installed environment and rollout files are mounted read-only; only the scoring output directory is writable:

```bash
mkdir -p "$PWD/work-adaptive-spec/scores"
docker run --rm --network none --read-only --cpus 8 --memory 32g --pids-limit 512 \
  --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:size=16g \
  -v "$PWD":/workspace:ro \
  -v "$PWD/work-adaptive-spec/scores":/workspace/work-adaptive-spec/scores:rw \
  -w /workspace --entrypoint /bin/bash "$AS_IMAGE" -lc '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /workspace/venv/bin/activate
    export AS_WORK=/workspace/work-adaptive-spec
    export TMPDIR=/tmp
    for arm in fixed adaptive; do
      python adaptive_spec/examples/apps.py score --input "$AS_WORK/evaluation.jsonl" \
        --responses "$AS_WORK/$arm/responses.jsonl" --output "$AS_WORK/scores/$arm.json"
    done
  '
```

The scorer runs the upstream verl APPS test executor, preserves function-style test cases, and reports correct, scorable and total response counts. Install on every worker, then use the same manager and router overrides above to integrate the scheme into a training job.

## Collecting the cost table

For the initial deployment, measure costs across batch size B, context length L, and draft depth K for the model and TP configuration. Recollect after changing the model, TP, or backend settings that affect execution cost.

1. Generate token records from an independent calibration set and save them as JSONL. Each row contains `context_id`, `prompt_token_ids`, and `response_token_ids`.
2. Build a context bank, copy the [plan example](examples/calibration.plan.example.json), and fill in the model, context bank, B/L/K, sampling parameters, and TP settings.

```bash
adaptive-spec-build-contexts --input /absolute/calibration_rollouts.jsonl --output /absolute/contexts.jsonl
adaptive-spec-collect --plan /absolute/plan.json --output-dir /absolute/plan-check --plan-only
```

3. Enable the collection plugin in the calibration environment and run the measurements:

```bash
unset VERL_ADAPTIVE_SPEC_CONFIG
export VLLM_PLUGINS=ascend,adaptive_spec_calibration_plugin
adaptive-spec-collect --plan /absolute/plan.json --output-dir /absolute/calibration-run
```

Collection runs fixed-depth MTP in separate processes for each K, using the same contexts and seed across K values. Use a fresh output directory. Outputs include raw measurements, `phase_map.json`, and `runtime.json`. For online rollout, use the generated configuration and switch back to `VLLM_PLUGINS=ascend,adaptive_spec_plugin`.

## Main parameters

| Parameter | Configuration | Purpose |
|---|---|---|
| `engine_aggregation` | `max` | Maximum request-level K recommendation |
| `tp_control_transport` | `packed` | Reusable fixed-size TP control message |
| `phase_stage_control` | `drain_triggered` | Enable online recommendations during drain |
| `phase_stage_max_batch_size` | Actual Engine capacity, e.g. 16 | Detect saturation |
| `phase_stage_below_max_confirm_steps` | 4 | Enter drain after four consecutive below-capacity observations following saturation |
| `confirmations` | 2 | Require two consecutive recommendations in the same direction before switching |
| `minimum_dwell_rounds` | 64 | Minimum dwell rounds after a switch |
| `evidence_exchange_directory` | Job-specific shared directory | Share same-prompt evidence across Engines; null keeps evidence local |
| `prompt_credit` (router YAML) | 2 | Preferred limit on concurrent responses to the same prompt |
| `elastic_delay_ms` (router YAML) | 20 | Wait before admitting more siblings into idle capacity |

The cost table supplies the fallback K for each phase. During drain, each request uses evidence from other responses to the same prompt to recommend K. The Engine takes the maximum recommendation and applies confirmation and dwell control. A response does not consume its own evidence. Evidence is scoped by model, sampling configuration, prompt, phase, and rollout step.

Use the generated configuration and set these parameters explicitly. The phase capacity must match the deployment.

## Integration and validation

The code loads through vLLM's general plugin mechanism and hooks runner methods in the process. It does not rewrite installed vLLM or vLLM-Ascend source files. verl integration uses a custom manager and an optional router. See [UPGRADING.md](UPGRADING.md) for upgrade dependencies and [VALIDATION.md](VALIDATION.md) for test configurations and results.
