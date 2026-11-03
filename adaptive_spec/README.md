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

Run from the repository root in an existing Ascend inference environment:

```bash
./install_verl.sh --recipe adaptive_spec
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
