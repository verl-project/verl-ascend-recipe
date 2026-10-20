# AdaptiveSpec: same-prompt evidence control for MTP rollout

AdaptiveSpec selects a physical serial MTP depth for each Engine during a synchronous RL rollout.
It combines an offline phase prior, acceptance evidence from responses to the same prompt,
and a stable Engine decision. Elastic prompt interleaving is an independent, optional router.

This recipe preserves the historical policy: Engine batch size and context-length P90 select
one phase; each active response queries that phase; the Engine takes the lower median and
stabilizes the result. It contains no intersection-based controller or online cost learner.
The runtime example is **disabled by default**. Example phase-map numbers are synthetic
schema examples, not measured performance data or a usable calibration for another deployment.

## Required verl version and backend contract

See [REQUIRED_VERL.txt](REQUIRED_VERL.txt). From this repository's root:

```bash
./install_verl.sh --recipe adaptive_spec --show
./install_verl.sh --recipe adaptive_spec
python -m pip install -e ./adaptive_spec
```

Install the recipe on the driver and all rollout workers in the existing, compatible Ascend
environment. This package does not install, replace, or upgrade torch, CANN, vLLM or vLLM-Ascend.

The adapter is limited to the historical 0.23.0 vLLM/vLLM-Ascend family, **serial MTP**, eager
execution, PP=1 and context parallelism=1, with the native padded-drafter raw-count path.
Model-runner and scheduler private methods also have exact source-hash guards. A version label
alone does not establish compatibility; vendor rebuilds with different source are refused.
The supported depth range is positive K up to the fixed startup capacity; K=11 is excluded.
This recipe does not enable K=0, parallel drafting, DSD, graph execution or 0.24 adapters.
Use the public revisions and source-install instructions in [BACKEND.md](BACKEND.md), then run
`adaptive-spec-check-backend` on each worker before calibration or rollout.

## Policy and defaults

For candidate K, the per-response score is `(1 + sum(posterior_survival[:K])) / phase_iteration_cost[K]`.
Evidence exposes only depths actually drafted; full acceptance at K does not expose K+1.
The requesting response cannot consume its own evidence. Completed siblings are kept in a
16-response window; active siblings become eligible at 64 blocks. Bayesian prior strength is 8,
minimum eligible siblings is 1, and response decay is 1.0. A recommendation must improve on
the phase fallback score by at least 1%; ties choose the smaller K.

Engine aggregation uses a lower median, two same-direction confirmations, a 64-round dwell
after a switch, and one step on the **global** candidate ladder. During a transition, the
physical K can be an intermediate global candidate outside the current phase's candidate
subset. This is historical behavior; the phase subset is a recommendation set, not an
instantaneous hard constraint. Startup capacity must cover the global ladder.
Within each Engine, the TP leader owns the controller and evidence lifecycle. It broadcasts
the final K to all TP ranks before drafting; followers do not independently poll evidence or
publish duplicate response statistics.

Evidence observations retain historical per-response B/L phase attribution; the Engine
queries its common B/P90 phase. Evidence is therefore not shared approximately across lengths.
Keys contain exact prompt tokens, model identity, sampling semantics, rollout step and phase.
Seed and request ID are intentionally excluded from the sampling fingerprint. Hashes are
cached by request identity and exact phase; changes to a live request identity are rejected.
Finish, discard and step reset clean the cache. No approximate prompt or phase matching is added.

## Deployment integration

1. Automatically collect a phase map for the actual model and deployment. Follow
   [the collection guide](CALIBRATION.md) to prepare natural tokenized contexts,
   freeze the B/L/K plan, run fixed-K probes and load the generated configuration:

   ```bash
   adaptive-spec-build-contexts --input /absolute/tokenized_rollouts.jsonl --output /absolute/natural_contexts.jsonl
   adaptive-spec-collect --plan /absolute/calibration.plan.json --output-dir /absolute/plan-check --plan-only
   adaptive-spec-collect --plan /absolute/calibration.plan.json --output-dir /absolute/calibration-run
   ```

   The output contains measured `phase_map.json` and controller-ready `runtime.json`.
   Offline measurement is separate from normal rollout. It requires the supported
   source-guarded Linux Ascend environment and refuses K11, dirty/incomplete probes
   and existing output directories. Previous measured maps can still be converted:

   ```bash
   python adaptive_spec/scripts/convert_phase_map.py      --input /absolute/measured-phase-map.json      --output /absolute/runtime-phase-map.json      --topology-fingerprint YOUR_TOPOLOGY_ID --exclude-k 11
   ```

   The output must not already exist. All phases must have priors covering the deepest global
   candidate, and every allowed K needs a finite positive measured cost. Model and topology
   identifiers must match the runtime config; they are caller-provided identities, not automatic
   verification of model-weight bytes. Recalibrate after changing the deployment.

2. Use the collector's generated `runtime.json`, or copy
   [runtime.schema-v1.example.json](examples/runtime.schema-v1.example.json) to a private
   config, set `enabled=true`, use an **absolute** phase-map path and real fingerprints. For
   cross-Engine evidence, set a writable shared directory dedicated to this job. Leave it null
   to use local evidence only. Completed compact counts are published; polling defaults to
   every 64 Engine decisions. Use a fresh directory for each job; no old files are deleted.

3. Enable the general plugin, keeping the existing Ascend plugin enabled:

   ```bash
   export VERL_ADAPTIVE_SPEC_CONFIG=/absolute/runtime.json
   export VLLM_PLUGINS=ascend,adaptive_spec_plugin
   ```

4. Add these overrides to your existing, working verl PPO launch command:

   ```text
   actor_rollout_ref.rollout.agent.agent_loop_manager_class=adaptive_spec.verl_integration.AdaptiveSpecAgentLoopManager
   actor_rollout_ref.rollout.enforce_eager=true
   actor_rollout_ref.model.mtp.enable=true
   actor_rollout_ref.model.mtp.enable_rollout=true
   actor_rollout_ref.model.mtp.num_speculative_tokens=12
   ```

   The pinned verl configuration inherits rollout MTP settings from `model.mtp`.
   Set both `enable` and `enable_rollout` there; overriding only `rollout.mtp`
   can replace that inherited configuration without enabling MTP.

   `12` is an example startup capacity; use a capacity covering your phase map. Keep your model,
   data, actor, reward, batch and parallel settings. The custom manager only propagates the
   training-step scope through SamplingParams; it does not implement an RL trainer or reward.
   Internal async model-runner bookkeeping does not imply support for asynchronous RL training.

5. To opt into Elastic, add
   `actor_rollout_ref.rollout.router_config_path=/absolute/elastic_interleave.router.yaml`.
   See [the example](examples/elastic_interleave.router.yaml). Capacity is per rollout server;
   set it consistently with the engine's configured capacity. Prompt credit limits simultaneous
   siblings; after the configured delay, idle capacity may admit additional siblings. Release
   requires the same request ID as acquire. Routing uses free/least-loaded placement only.
   Omit this override to preserve the upstream router.

6. For the audited scheduler's long-context KV pressure case, the optional compatibility repair
   is `VERL_MTP_KV_PREEMPT_MODE=targeted`. It preempts the allocation-failing request and preserves
   the upstream same-step resume guard. This is a generic KV scheduling repair, not a K=11
   workaround or part of the K selection algorithm. It is off when unset; `original` keeps
   original scheduling. It rejects unaudited source and enabled Ascend balance scheduling.
   Apply the same mode to comparison arms. No installed source file is rewritten.

Unset `VERL_ADAPTIVE_SPEC_CONFIG`, `VERL_MTP_KV_PREEMPT_MODE` and the router override to disable
the features for a **newly started** job. Do not change modes in a live Engine.

## Correctness bridges

The 0.23 adapter binds each verification result to the preceding physical draft K and request
snapshot, subtracts the target/bonus token, skips discarded rows, and consumes the snapshot once.
It uses the raw valid sampled-token count and its synchronization event; Mamba's state-advance
count is not an acceptance statistic. Variable-width draft copy/read and worker-local optimistic
schedule reconciliation retain necessary stream/event ordering. Private-method source guards
fail rather than silently adapting an unknown backend. The plugin defers runner import to avoid
circular initialization. All patches are process-local; no vendor checkout is edited.

## Tests and verification boundary

```bash
python -m pip install -e './adaptive_spec[test]'
pytest adaptive_spec/tests -q
ruff check adaptive_spec
ruff format --check adaptive_spec
python -m build adaptive_spec
```

Tests are CPU regression tests, including historical policy behavior, censoring, exact key-cache
identity, preceding-K attribution, raw-count selection, guarded private APIs and router lifecycle.
The TP fix also passed a real two-process Gloo regression and a native Ascend TP2 short-generation
test on the public backend source, including physical K5→K7→K5 transitions. See
[VALIDATION.md](VALIDATION.md). Full verl/Ray training and representative rollout performance
are separate validation steps.
Historical experimental speedups are not performance measurements of this delivery revision.
The online control path contains no profiling, decision journals or experimental controller flags.
The explicit offline collector writes calibration artifacts only to its requested output directory.
No benchmark queues, deployment model paths, server credentials or research result files are shipped.
