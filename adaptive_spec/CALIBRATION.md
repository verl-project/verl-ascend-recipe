# Automatic phase-map collection

Collection is an explicit **offline** job in the supported Linux Ascend environment.
It starts native fixed-K MTP Engines, measures clean B/L decode states and emits the
phase-map schema consumed by the historical controller. It does not run adaptive
depth selection or Elastic during calibration and adds no measurement calls to
ordinary rollout jobs.

Prepare the [pinned public backend](BACKEND.md) and run `adaptive-spec-check-backend`
before starting device probes. A version string alone does not identify the guarded source.

## 1. Prepare representative natural contexts

Use token IDs from an existing natural calibration rollout made with the same
model/tokenizer and prompt format. Do not repeat a short prompt to manufacture a
long context. Keep calibration and evaluation data provenance distinct; do not
select contexts according to the performance of the candidate controller.

The tokenized rollout input is JSONL, one response per line:

```json
{"context_id":"prompt-001-response-0","prompt_token_ids":[101,102],"response_token_ids":[103,104]}
```

These short token lists explain the schema only. Supply real tokenized rollouts,
including enough distinct eligible contexts for the largest B at each target L.
The builder concatenates prompt and response token IDs without retokenization:

```bash
adaptive-spec-build-contexts \
  --input /absolute/tokenized_calibration_rollouts.jsonl \
  --output /absolute/natural_contexts.jsonl
```

It writes a context bank plus a source/hash receipt and refuses to overwrite them.
Alternatively, supply a bank directly with `context_id` and `token_ids` fields.

## 2. Freeze the collection plan

Copy `adaptive_spec/examples/calibration.plan.example.json` and edit:

- The model location, context-bank location and deployment identity strings.
- The B/L states needed for the target rollout, the supported positive K values,
  sampling parameters and Engine settings. K11 is refused before loading a model.
- Repeat count, continuation length and minimum required clean decode steps.

Relative context-bank paths are resolved against the plan file. Model and topology
identifiers are caller-provided identities, not automatic hashes of weight bytes.
Use an identity that includes backend build, TP, sampling, memory and execution
settings; recalibrate after changing those settings.

Each repeat selects the same distinct context prefixes and seed for every K.
Prefixes are exactly L tokens long. The resulting selection IDs and prefix hashes
are frozen in `plan.resolved.json` before device work starts. The collector runs
one fixed-K Engine at a time in a separate subprocess; it reuses that Engine over
the configured states and repeats. K order is seed-shuffled. This is not a fully
counterbalanced experiment, so use additional independent calibration jobs if
thermal or temporal drift is material.

The supported probe uses eager serial MTP, TP as configured, DP1/PP1/CP1,
synchronous scheduling and disabled prefix caching. Its synchronized iteration
cost is a calibration prior, not an end-to-end asynchronous-rollout benchmark.
`max_model_len` must cover L, the bounded continuation and the deepest draft.

First validate data selection and capacity **without loading the model**:

```bash
adaptive-spec-collect \
  --plan /absolute/calibration.plan.json \
  --output-dir /absolute/calibration-plan-check \
  --plan-only
```

Each command requires a fresh output directory. Plan-only creates no usable phase
map and never loads vLLM or allocates NPU model resources.

## 3. Collect on the assigned devices

Use only devices assigned to this job, in an otherwise idle compatible environment:

```bash
unset VERL_ADAPTIVE_SPEC_CONFIG
unset VERL_MTP_KV_PREEMPT_MODE
unset VERL_ADAPTIVE_SPEC_CALIBRATION_DIR
unset VERL_ADAPTIVE_SPEC_CALIBRATION_PLAN
export VLLM_PLUGINS=ascend,adaptive_spec_calibration_plugin
export VLLM_WORKER_MULTIPROC_METHOD=spawn

adaptive-spec-collect \
  --plan /absolute/calibration.plan.json \
  --output-dir /absolute/calibration-run
```

Select devices with the normal deployment mechanism before starting the command.
The collector does not discover, reserve or stop other jobs and never restarts
devices or edits installed backend sources. The supported runner's source hash
must match the same audited 0.23 build used by the runtime adapter.

Measurement includes target verification, sampling, the next MTP draft and native
runner bookkeeping: 0.23 executes target forward and sampling/drafting in separate
calls, and their synchronized wall times are summed. Inter-call driver gaps and
JSONL writing are excluded. The observer does not use Mamba state-advance counts;
it reads the synchronous valid sampled-token lists and subtracts the target/bonus
token to obtain accepted draft-prefix lengths.

Prefill, partial draft batches, configured warmup steps, discarded rows, changed B
and contexts outside the declared L window are excluded. Any KV preemption fails
the probe rather than turning recomputation into a measured decode cost.
`ignore_eos=true` is used only for bounded calibration continuations; it does not
change production rollout EOS handling or response limits.

If a state cannot collect enough clean steps for **every K and repeat**, collection
fails and preserves logs and partial records. Increase probe length or adjust an
infeasible calibration grid in a new plan/job; do not invent missing measurements.
Timeout stops only the subprocess group launched for that probe. There is no
automatic device reset, hidden retry or resume into an old result directory.

## 4. Load the generated artifacts

A successful output directory contains:

```text
calibration-run/
├── plan.resolved.json
├── k5/                         # One directory per measured K
│   ├── run.log
│   ├── result.json
│   └── steps.<pid>.jsonl
├── raw_steps.jsonl
├── phase_map.json
├── runtime.json
└── status.json
```

Per-K costs use the median of repeat-level median iteration costs. Acceptance
priors pool actual per-position exposure/success counts, apply the configured
symmetric pseudocount and enforce non-increasing survival. Draft depth k exposes
only positions 1..k; deeper positions are never filled as rejections.

The candidate interval retains K whose measured progress/cost score is at least
`1 - regret_fraction` times the best measured score in that cell. Default tolerance
is 10%; the fallback is the lowest retained K, preserving historical minimum-fallback
behavior. All measured K costs, scores and counts remain in the artifact for audit.
The collector does not claim to produce a phase map numerically identical to the
historical private experimental map.

`phase_map.json` is directly loadable: no separate conversion step is required.
`runtime.json` has matching identities, the absolute map path and the historical
controller defaults. The collector validates it with the actual controller loader
before declaring completion. Check `status.json` and the preserved source records.

For a **new** production rollout job, remove offline-observer variables and switch
to the online plugin:

```bash
unset VERL_ADAPTIVE_SPEC_CALIBRATION_DIR
unset VERL_ADAPTIVE_SPEC_CALIBRATION_PLAN
export VERL_ADAPTIVE_SPEC_CONFIG=/absolute/calibration-run/runtime.json
export VLLM_PLUGINS=ascend,adaptive_spec_plugin
```

Use the existing working verl launch with the AdaptiveSpec manager override, eager
MTP and a startup capacity covering the map's global K ladder. Optional Elastic is
configured separately, as described in the recipe README. Calibration and online
control are mutually exclusive within a process.

## Replay and legacy import

For CPU validation of artifact reduction only:

```bash
adaptive-spec-collect \
  --plan /absolute/calibration.plan.json \
  --output-dir /absolute/calibration-replay \
  --backend replay \
  --records /absolute/recorded_steps.jsonl
```

Replay requires the declared minimum clean samples for all state/K/repeat groups.
Its artifacts explicitly set `device_measurement=false`, and its generated runtime
config is disabled. This is not an NPU benchmark or a replacement for device smoke.

Previously measured phase-map payloads can still be imported with
`scripts/convert_phase_map.py`. That converter does not launch collection.
