# DAPO length predictor and EWS

This recipe adds two orthogonal components to DAPO rollout generation:

1. a prompt-side response-length predictor; and
2. an Event-driven Waiting-pool Scheduler (EWS).

EWS has a deterministic fail-closed mode. Before the selected predictor passes its activation gates, EWS admits requests in stable FCFS order. After activation it admits the longest predicted work first and refills the bounded rollout window whenever one request completes. Actual inference-server placement remains owned by verl's existing `GlobalRequestLoadBalancer`.

The implementation uses verl's public `agent_loop_manager_class` extension point. It does not patch vLLM or vLLM-Ascend and does not require either project as a Python dependency of the scheduler package.

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). The tested pin is `bcb638649a50e58494a8ddd92085ad1174f674b8`.

## Tested environment and dependency boundary

| Component | Tested value | Dependency role |
| --- | --- | --- |
| Python | 3.10 / 3.11 | Provided by the selected verl runtime image. |
| verl | commit `bcb638649a50e58494a8ddd92085ad1174f674b8` (`0.8.0.dev`) | Required and machine-pinned in `REQUIRED_VERL.txt`. |
| PyTorch / torch-npu | The versions bundled by the tested Ascend runtime | Inherited from verl; this recipe does not repin them. |
| vLLM | `0.19.1` | Tested rollout runtime, not imported by this package. |
| vLLM-Ascend | `0.19.1rc1` | Tested Ascend rollout runtime, not imported by this package. |
| Ascend image | `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler` | End-to-end evaluation environment. |
| Hardware | Ascend 910 | End-to-end evaluation hardware. |

The scheduler package intentionally has no direct Python dependency on vLLM or vLLM-Ascend. Runtime compatibility with newer rollout stacks is therefore expected to follow verl's public agent-loop contract, but only the versions above are claimed as tested. Core Python dependencies (`torch`, `numpy`, `ray`, Hydra/OmegaConf, and verl utilities) come from the pinned verl environment. SciPy is optional and is used only to emit the Kendall-tau training metric; training and scheduling continue without it.

## Source-tree placement

This is a recipe overlay, not a standalone wheel. Install or clone the pinned verl revision (including its `recipe` submodule), then place this directory at `recipe/dapo_predictor` in that source checkout:

```bash
git clone --recurse-submodules https://github.com/verl-project/verl.git
cd verl
git checkout bcb638649a50e58494a8ddd92085ad1174f674b8
git submodule update --init --recursive recipe
cp -a /path/to/verl-ascend-recipe/dapo_predictor recipe/dapo_predictor
pip install -e .
```

Run the commands below from the verl checkout with that checkout on `PYTHONPATH`. The upstream DAPO recipe remains the owner of the base trainer configuration; this overlay adds predictor and EWS behavior through the documented extension point.

## Product boundary

Included in the first contribution:

- no-anchor EWS with deterministic FCFS fallback;
- D1 `Linear/ListMLE` as the default online backend;
- train-history-only PAVA calibration from ranking score to token-scale work;
- D1 epoch/sample gates, and coarse rank-first quality admission/monitoring for D2;
- predictor-only checkpoint save/resume;
- D2 right-censored LogNormal inference as an optional frozen backend;
- strict finite/range/schema checks and prediction provenance.

Not enabled or submitted as product behavior:

- forced K=1/K=2 anchor rollouts;
- sibling/history correction;
- live decode-progress polling;
- automatic D1-to-D2 switching;
- TailGate/OOD heuristics;
- vLLM/vLLM-Ascend source patches;
- exact-trace replay helpers used by experiments.

## Runtime flow (`scheduler=epws`)

1. The trainer repeats each prompt by `rollout.n` as usual.
2. Until the selected predictor passes its activation gates, no scheduling predictor forward is run and EWS behaves as stable FCFS. D2 also requires an independently validated candidate report.
3. Once active, the actor extracts one prompt representation per contiguous `rollout.n` sibling group. Siblings share that prediction; identical prompts in separate groups are not deduplicated.
4. D1 maps the scalar ListMLE score to token work using a train-history-only monotone PAVA map. D2 uses `median + risk_weight * (p90 - median)`.
5. `EPWSAgentLoopManager` keeps a bounded set of rollout requests in flight. Each completion immediately admits the next request; active mode is longest-predicted-work-first.
6. verl's existing global load balancer chooses the inference server.
7. After the actor update, D1 learns from the completed rollout batch and refits calibration for future batches. The batch being scheduled never uses its own completion lengths.
8. D2 never trains in the worker. At completed RL epochs 10, 15, 20, ... the trainer either evaluates an external candidate (while inactive) or checks a five-epoch monitoring window (while active). An admitted head stays frozen; no scheduled refitting occurs.

Admission priorities are computed before each rollout batch. Completion events refill capacity; they do not refresh predictions, migrate running requests, or re-rank the batch using live decode progress.

## Modules

| Module | Responsibility |
| --- | --- |
| `main_dapo_predictor_reorder.py` | DAPO entry point; selects the custom actor worker and installs EWS through verl's manager extension. |
| `predictor_dapo_trainer.py` | Lifecycle gates, prompt-side scoring, predictor update, and predictor checkpoint state. |
| `predictor_worker.py` | Hidden-state extraction, D1 online training/calibration, and optional D2 inference. |
| `epws_manager.py` | Event-driven bounded admission and output-order restoration. |
| `length_scheduler/calibration.py` | Weighted PAVA and monotone score-to-token interpolation. |
| `length_scheduler/scheduler.py` | Stable FCFS / longest-predicted-work-first waiting pool. |
| `length_scheduler/predictor.py` | Backend-neutral prediction API, D1/D2 adapters, provenance, and validation. |
| `length_scheduler/checkpoint.py` | Backend-aware lifecycle metadata and frozen-D2 checkpoint binding. |
| `length_scheduler/distribution.py` | Right-censored LogNormal inference primitives. |
| `length_scheduler/d2_quality.py` | Rank-first D2 candidate admission, coarse monitoring, censor-aware safeguards, and resumable state. |
| `validate_d2.py` | External CPU utility to publish a checkpoint-bound held-out validation report. |
| `predictor_utils.py` | Legacy static-snake helper retained for backward compatibility. |

## Backends

### D1: Linear/ListMLE (default)

D1 is a bias-free linear ranking head over the final prompt-token hidden state. It is trained online with ListMLE. Since a ranking score has no token unit, the worker retains a bounded training-history replay and refits a monotone PAVA map after each update. PAVA sees only already-completed training rows. Until at least two valid calibration observations exist, the predictor reports not ready and EWS remains FCFS.

### D2: censored LogNormal (optional)

D2 consumes the hidden-state tap schema declared by a frozen checkpoint and exposes unstandardized `mu`, `sigma`, expected length, median, p90, p95, and exceedance probability. Checkpoints are fitted independently; this recipe does not implement the external feature-collection/training pipeline or train D2 online. The default D2 path stays FCFS until a hash-bound validation report passes rank-first admission. `d2_checkpoint` may be omitted during warm-up; all actor ranks load the admitted candidate synchronously before scheduling begins.

## Configuration

The scheduler is named EWS. Existing `epws` configuration keys, Python identifiers (including `EPWSAgentLoopManager`), and metric names are retained. The naming change itself does not rename configuration keys; D2 quality admission is additional behavior described below.

The entry point mirrors `trainer.predictor_reorder` into the actor worker config.

```yaml
trainer:
  predictor_reorder:
    enable: true
    scheduler: epws
    backend: linear_listmle
    activation:
      min_epoch: 10
      min_samples: 2560
    epochs: 10
    batch_size: 32
    lr: 3.0e-5
    weight_decay: 1.0e-4
    seed: 1
    calibration_max_samples: 4096
    predictor_keep_actor_loaded: false
    epws:
      slots_per_server: 8
      max_concurrent_requests: null
```

`max_concurrent_requests: null` derives the admission window as `rollout server count * slots_per_server`. Set it explicitly when the inference deployment has a different safe concurrency limit.

For D2, replace the backend and publish an external candidate report:

```yaml
trainer:
  predictor_reorder:
    backend: censored_lognormal
    risk_weight: 0.5
    activation:
      min_epoch: 10
      min_samples: 2560
    d2_lifecycle:
      enable: true
      interval_epochs: 5
      candidate_report: /absolute/shared/path/d2-candidate.json
      quality:
        min_prompts: 32
        min_pairs: 100
        max_rank_pairs: 20000
        min_rank_accuracy: 0.60
        rank_relative_gap: 0.10
        max_normalized_mae: 1.0
        min_scale_ratio: 0.25
        max_scale_ratio: 4.0
        min_p90_coverage: 0.70
        max_p90_ratio: 4.0
        bad_windows_before_fallback: 2
```

The backend is selected before training and never changes mid-run.

### D2 external fitting and coarse lifecycle

`activation.min_epoch=10` means **ten completed RL dataset passes**, so the earliest admission is before pass 11. It is not a cap on predictor optimizer steps, nor proof that ten passes suffice for every workload. At a boundary without a usable report the trainer continues FCFS; it does not block waiting for an external fit. Publish the candidate before a boundary or it will be considered five passes later. While inactive, collect additional data externally and refit/revalidate every five passes as needed. Once admitted, only monitor every five passes; do not refit routinely.

Use a fixed prompt population with held-out *future outputs*, not necessarily unseen prompt identities. For example, fit a candidate on passes 1–9 and validate on pass 10; if it fails, fit the next candidate through pass 14 and validate on pass 15. If fitting includes pass 10, that candidate must instead be validated on later outputs: its fitting rows cannot also certify admission. Feature projection/normalization/head fitting must all respect the cutoff. Collect validation forecasts before seeing their responses, using the matching actor/model/tokenizer/tap schema; a post-hoc feature readout is not online validation. These provenance obligations remain with the external pipeline; this utility checks the declared temporal boundary and file hashes, not the truth of those declarations.

Write one JSONL row per held-out response (siblings share the prompt identity):

```json
{"prompt_id":"stable-tokenized-prompt-hash","epoch":10,"observed":1200,"expected":1300,"median":1100,"p90":2400,"censored":false}
```

All epoch numbers in rows are one-based. Mark cap-truncated responses `censored: true`, including ambiguous EOS-at-cap cases conservatively. Actual lengths alone cannot distinguish those cases. Publish with:

```bash
python -m recipe.dapo_predictor.validate_d2 \
  --checkpoint /absolute/shared/path/d2-candidate-v1.pt \
  --rows /absolute/path/heldout-pass10.jsonl \
  --trained-through-epoch 9 --validation-epoch 10 \
  --output /absolute/shared/path/d2-candidate.json
```

The utility streams the input rows and atomically replaces the report; it rejects an output path that resolves to the checkpoint or input rows. The candidate weight file itself must be immutable and visible at the same absolute path on every actor rank. The trainer recomputes metrics from the report's grouped sufficient statistics, checks the checkpoint SHA-256 and matching `risk_weight`, and rejects stale/future/overlapping validation. It never trusts a supplied `passed` flag. Publishing is a trusted local handoff, not a cryptographic proof of how an external trainer used its data.

The defaults above are **configurable engineering starting points**, not experimentally calibrated universal thresholds or a statistical significance claim:

| Role | Definition and default |
| --- | --- |
| Primary ranking | Pairwise order agreement of actual per-prompt mean response length with the actual EWS risk score, `median + risk_weight * (p90 - median)`, averaged per prompt. Require >=60%; predicted ties score 0.5, actual pairs differing by <=10% of the larger mean are excluded. This is **not Spearman correlation**. |
| Evidence floor | At least 32 distinct exact prompt identities and 100 comparable pairs. Repeated epochs/siblings do not create new independent identities; pair counts are descriptive, not independent statistical samples. Above 20,000 possible pairs, use a deterministic sample of at most 20,000 before excluding near-ties, bounding CPU overhead without changing global RNG state. Small datasets may explicitly lower the evidence floor, with correspondingly weaker evidence. |
| Gross length/scale protection | Prompt-balanced normalized MAE = `sum(abs(predicted_mean - observed_mean)) / sum(observed_mean)` <=1.0; aggregate predicted/observed mean ratio within [0.25, 4]. This is deliberately a gross-error veto, not a 35% point-accuracy target. |
| Tail protection | Prompt-balanced p90 coverage lower bound >=70% and mean predicted p90 / observed mean <=4. Nominal p90 coverage is 90%; 70% is only a severe-undercoverage guard, not a target. High coverage alone cannot qualify an inflated forecast. |

Any prompt identity with a censored response is excluded from exact mean/rank/MAE calculations. Coverage uses lower/upper bounds: a censored length below predicted p90 has unknown coverage, not a success. Insufficient exact data or bounds straddling the protection threshold cannot certify initial admission. Use held-out workload evidence to choose thresholds before deployment, not to retrofit a favorable result on a completed experiment. The exploratory post-hoc pilot is not a production D2 threshold calibration.

Active monitoring reuses the pre-rollout forecasts (no extra per-epoch fit or post-hoc actor forward), pairs them with completed responses **before DAPO reward filtering**, and pools each five-pass window by prompt identity. One sufficiently supported gross scale/tail violation triggers FCFS at the boundary; ranking failure requires two consecutive supported bad windows by default. Missing/insufficient monitoring evidence also suspends scheduling at the boundary; invalid numeric rows suspend it immediately. No fallback rewrites the completed batch. A suspended checkpoint cannot automatically reactivate: externally fit and validate a new checkpoint, publish a new report, and wait for the next boundary. There is no D1/D2 backend switch.

Monitoring state, incomplete-window statistics, boundary counters, the selected checkpoint path/hash, and the in-progress RL epoch are saved with the trainer. Resume validates counters, grouped statistics, readiness consistency, temporal boundaries, the same quality configuration, and the weight hash **before** loading a candidate onto workers. Missing legacy quality state returns to FCFS admission rather than trusting an old activation flag; a backend mismatch is rejected. Checkpoint snapshots do not alias a live monitor window. Do not overwrite admitted weights in place; saving a changed admitted file also fails explicitly.

The current runtime monitor identifies prompts by unpadded text token IDs. Nonempty `multi_modal_data` suspends quality-mode D2 and uses FCFS before predictor forward, rather than merging different images under the same text identity. Multimodal quality admission is not claimed. A cross-rank candidate-load RPC failure aborts training with readiness cleared; it does not silently continue with mixed heads.

For an already validated matching candidate, `activation.min_epoch=0` and `min_samples=0` explicitly permit admission at startup; the report's epochs then describe the completed historical fitting/validation run. This historical exception is only available at the explicit zero-warm-up startup boundary; later candidate checks must use recent completed epochs of the current run. For short training runs, **20 total RL epochs is a recommendation**, not a hard disable gate: the recipe only warns below 20. Decide whether warm-up and forward overhead can be amortized. D1 fitting and calibration are unchanged.

For exact legacy epoch/sample-only D2 behavior, explicitly set `d2_lifecycle.enable=false` and provide `d2_checkpoint`; no new quality protection is then claimed. Existing D2 launch configurations must either provide a report or select this compatibility mode. D1's `predictor_reorder.epochs` remains the number of ListMLE fitting iterations per update; it is unrelated to these RL epoch boundaries.

## Launch

Use the same DAPO data/model/rollout overrides as `recipe.dapo`, plus:

```bash
PYTHONPATH=/workspace/verl python recipe/dapo_predictor/main_dapo_predictor_reorder.py \
  +trainer.predictor_reorder.enable=true \
  +trainer.predictor_reorder.scheduler=epws \
  +trainer.predictor_reorder.backend=linear_listmle \
  +trainer.predictor_reorder.activation.min_epoch=10 \
  +trainer.predictor_reorder.activation.min_samples=2560 \
  +trainer.predictor_reorder.calibration_max_samples=4096 \
  +trainer.predictor_reorder.epws.slots_per_server=8 \
  +trainer.predictor_reorder.epws.max_concurrent_requests=null
```

## Correctness and fallback guarantees

- Missing, nonnumeric, non-finite, negative, or incomplete predicted-work values fail closed to stable FCFS for the entire EWS batch. Structurally misaligned protocol batches still raise explicit errors.
- Dynamic micro-batch reordering is reversed before predictions are attached to prompts.
- Generated outputs are restored to original batch order before the trainer unions them with training rows.
- D1 calibration never reads the completion lengths of the rollout batch currently being scheduled.
- D1 weights and calibration are stored beside the normal trainer checkpoint. D2 remains external and frozen; its SHA-256 is stored with the shared lifecycle state and verified on resume.
- If lifecycle state is absent on resume, the scheduler restarts in FCFS fallback mode instead of using stale predictions. A backend or D2 checkpoint mismatch fails explicitly.
- D2 validates hidden tap names/dimensions, finite inputs, positive sigma, monotone quantiles, and checkpoint provenance.

## Validation scope

Unit tests cover lifecycle gates, PAVA, scheduler fallback/priority, event-driven refill, numerical invariants, D1/D2 reload, and error inputs. Integration validation should use the pinned verl commit and the target Ascend rollout stack. See the PR validation receipt for the exact commands and environment used for the contribution.
