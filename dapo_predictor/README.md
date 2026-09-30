# DAPO length predictor and EPWS

This recipe adds two orthogonal components to DAPO rollout generation:

1. a prompt-side response-length predictor; and
2. an event-driven pending/waiting-pool scheduler (EPWS).

EPWS has a deterministic fail-closed mode. Before the selected predictor passes its activation gates, EPWS admits requests in stable FCFS order. After activation it admits the longest predicted work first and refills the bounded rollout window whenever one request completes. Actual inference-server placement remains owned by verl's existing `GlobalRequestLoadBalancer`.

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

Run the commands below from the verl checkout with that checkout on `PYTHONPATH`. The upstream DAPO recipe remains the owner of the base trainer configuration; this overlay adds predictor and EPWS behavior through the documented extension point.

## Product boundary

Included in the first contribution:

- no-anchor EPWS with deterministic FCFS fallback;
- D1 `Linear/ListMLE` as the default online backend;
- train-history-only PAVA calibration from ranking score to token-scale work;
- a two-part activation gate (`min_epoch` and `min_samples`);
- predictor-only checkpoint save/resume;
- D2 right-censored LogNormal inference as an opt-in experimental backend;
- strict finite/range/schema checks and prediction provenance.

Not enabled or submitted as product behavior:

- forced K=1/K=2 anchor rollouts;
- sibling/history correction;
- live decode-progress polling;
- automatic D1-to-D2 switching;
- TailGate/OOD heuristics;
- vLLM/vLLM-Ascend source patches;
- exact-trace replay helpers used by experiments.

## Runtime flow

1. The trainer repeats each prompt by `rollout.n` as usual.
2. If the selected predictor has not passed both activation gates, no predictor forward is run and EPWS behaves as stable FCFS.
3. Once active, the actor performs one prompt-side forward per unique prompt. All sibling rollouts share that prompt prediction.
4. D1 maps the scalar ListMLE score to token work using a train-history-only monotone PAVA map. D2 uses `median + risk_weight * (p90 - median)`.
5. `EPWSAgentLoopManager` keeps a bounded set of rollout requests in flight. Each completion immediately admits the next request; active mode is longest-predicted-work-first.
6. verl's existing global load balancer chooses the inference server.
7. After the actor update, D1 learns from the completed rollout batch and refits calibration for future batches. The batch being scheduled never uses its own completion lengths.

## Modules

| Module | Responsibility |
| --- | --- |
| `main_dapo_predictor_reorder.py` | DAPO entry point; selects the custom actor worker and installs EPWS through verl's manager extension. |
| `predictor_dapo_trainer.py` | Lifecycle gates, prompt-side scoring, predictor update, and predictor checkpoint state. |
| `predictor_worker.py` | Hidden-state extraction, D1 online training/calibration, and optional D2 inference. |
| `epws_manager.py` | Event-driven bounded admission and output-order restoration. |
| `length_scheduler/calibration.py` | Weighted PAVA and monotone score-to-token interpolation. |
| `length_scheduler/scheduler.py` | Stable FCFS / longest-predicted-work-first waiting pool. |
| `length_scheduler/predictor.py` | Backend-neutral prediction API, D1/D2 adapters, provenance, and validation. |
| `length_scheduler/distribution.py` | Right-censored LogNormal inference primitives. |
| `predictor_utils.py` | Legacy static-snake helper retained for backward compatibility. |

## Backends

### D1: Linear/ListMLE (default)

D1 is a bias-free linear ranking head over the final prompt-token hidden state. It is trained online with ListMLE. Since a ranking score has no token unit, the worker retains a bounded training-history replay and refits a monotone PAVA map after each update. PAVA sees only already-completed training rows. Until at least two valid calibration observations exist, the predictor reports not ready and EPWS remains FCFS.

### D2: censored LogNormal (experimental)

D2 consumes the hidden-state tap schema declared by a frozen checkpoint and exposes unstandardized `mu`, `sigma`, expected length, median, p90, p95, and exceedance probability. It is useful when downstream systems need a distribution rather than only an ordering. D2 checkpoints must be produced independently and supplied with `d2_checkpoint`; this recipe does not train D2 online.

## Configuration

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

For D2, replace the backend and supply the frozen checkpoint:

```yaml
trainer:
  predictor_reorder:
    backend: censored_lognormal
    d2_checkpoint: /absolute/path/to/d2.pt
    risk_weight: 0.5
```

The backend is selected before training and never changes mid-run.

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

- Missing, non-finite, negative, or incomplete predictions fail closed to FCFS.
- Dynamic micro-batch reordering is reversed before predictions are attached to prompts.
- Generated outputs are restored to original batch order before the trainer unions them with training rows.
- D1 calibration never reads the completion lengths of the rollout batch currently being scheduled.
- Predictor state is stored beside the normal trainer checkpoint. If it is absent on resume, the scheduler restarts in FCFS fallback mode instead of using stale predictions.
- D2 validates hidden tap names/dimensions, finite inputs, positive sigma, monotone quantiles, and checkpoint provenance.

## Validation scope

Unit tests cover lifecycle gates, PAVA, scheduler fallback/priority, event-driven refill, numerical invariants, D1/D2 reload, and error inputs. Integration validation should use the pinned verl commit and the target Ascend rollout stack. See the PR validation receipt for the exact commands and environment used for the contribution.
