# DAPO length predictor and EWS

EWS (Event-driven Waiting-pool Scheduler) admits rollout requests from a shared waiting pool. Before a selected predictor is ready, requests follow FCFS. Once active, longer predicted work starts first, so long responses can overlap with shorter requests instead of being left until the end of a batch.

## What changes

| Path | Admission order | Refill | Prediction |
| --- | --- | --- | --- |
| Standard DAPO rollout | Existing manager order | Existing manager | None |
| Community predictor / static Snake | Scores packed across DP ranks before generation | Existing manager | Linear/ListMLE ranking scores |
| EWS + D1 (default backend) | Predicted long first, FCFS during warm-up | One global bounded window, refilled on completion | Linear/ListMLE + historical PAVA calibration |
| EWS + D2 (optional backend) | Risk-adjusted predicted long first, FCFS until admission | Same EWS window | Frozen LogNormal distribution + quality admission/monitoring |

Server placement stays with verl's `GlobalRequestLoadBalancer`. Priorities are fixed for each batch: EWS does not poll decode progress, migrate running requests or revise predictions on completion. There are no forced anchor rollouts or sibling corrections.

D1 learns online from completed training batches. PAVA maps its unitless ranking score to token-scale work. D2 returns median, p90, p95, expected length, mu/sigma and exceedance probabilities; its scheduling score is `median + risk_weight * (p90 - median)`. The backend is selected at launch and remains fixed.

## Requirements

| Component | Configuration |
| --- | --- |
| Platform | Linux, Ascend NPU for the documented device run |
| Actor | FSDP or FSDP2; text-only D2 quality monitoring |
| verl | `0.8.0.dev`, commit `bcb638649a50e58494a8ddd92085ad1174f674b8` |
| Rollout stack used in evaluation | vLLM `0.19.1`, vLLM-Ascend `0.19.1rc1` |
| Runtime image used in evaluation | `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler` |
| Python | Device environment 3.11.14; CI 3.10 |
| Libraries | PyTorch/torch-npu and Ray from the runtime; the pinned verl dependencies |

This is a source-tree recipe, not a standalone wheel. [REQUIRED_VERL.txt](REQUIRED_VERL.txt) is read by the repository installer to select the core verl revision. The scheduler package has no vLLM/vLLM-Ascend imports or source patches. SciPy is optional for the D1 Kendall-tau metric.

For backend upgrades, see [UPGRADING.md](UPGRADING.md): device-free tests and the integration files to review are listed separately from device validation.

## Installation

Start inside a working Ascend runtime whose driver/CANN/PyTorch/torch-npu stack matches the host. Keep this checkout and its dependencies in your own workspace.

The commands below fetch this PR, install its pinned verl and copy both recipe directories. The overlay copies directory contents rather than creating a nested `dapo_predictor/dapo_predictor` layout.

```bash
export WORK_ROOT="$PWD/ews-work"
mkdir -p "$WORK_ROOT"
git clone https://github.com/verl-project/verl-ascend-recipe.git "$WORK_ROOT/recipes"
cd "$WORK_ROOT/recipes"
git fetch origin pull/142/head
git switch --detach FETCH_HEAD
export RECIPE_SRC="$PWD"

./install_verl.sh --recipe dapo_predictor --method git \
  --dest "$WORK_ROOT/verl" --yes

export VERL_SRC="$WORK_ROOT/verl"
mkdir -p "$VERL_SRC/recipe/dapo" "$VERL_SRC/recipe/dapo_predictor"
cp -a "$RECIPE_SRC/dapo/." "$VERL_SRC/recipe/dapo/"
cp -a "$RECIPE_SRC/dapo_predictor/." "$VERL_SRC/recipe/dapo_predictor/"
cd "$VERL_SRC"
export PYTHONPATH="$VERL_SRC:${PYTHONPATH:-}"
```

Use fresh destination directories for a first install; compare existing files before applying this overlay to a working checkout. Copying this directory alone does not supply the upstream `recipe.dapo` trainer.

## Data and model

Prepare the public DAPO-Math and AIME parquet files with the included data script:

```bash
cd "$VERL_SRC"
VERL_HOME="$VERL_SRC" bash recipe/dapo/prepare_dapo_data.sh
export TRAIN_FILE="$VERL_SRC/data/dapo-math-17k.parquet"
export TEST_FILE="$VERL_SRC/data/aime-2024.parquet"
```

The parquet rows contain chat messages in `prompt`, `data_source`, and a rule-based `reward_model.ground_truth`. Use separate training and validation outputs. A model download requires access to its provider and acceptance of any applicable license. For a small integration run, for example:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-0.6B", local_dir="models/Qwen3-0.6B")'
export MODEL_PATH="$VERL_SRC/models/Qwen3-0.6B"
```

Choose a model supported by the selected rollout backend. The small-model integration workload is separate from the Qwen3.5-9B performance results.

## Run D1 + EWS

From the pinned verl checkout, on devices allocated to this job:

```bash
export NPU_COUNT=2
export TP_SIZE=1
export OUTPUT_DIR="$WORK_ROOT/runs/d1-ews"
bash recipe/dapo_predictor/run_ews.sh
```

The script sets up a single-node, text-only DAPO run with console logging and no W&B. Defaults are 8 prompts/batch, 2 responses/prompt, 512 prompt tokens, 1024 response tokens and 12 RL epochs. Set `PROMPT_LENGTH`, `RESPONSE_LENGTH`, `BATCH_SIZE`, `ROLLOUT_N`, `TOTAL_EPOCHS`, `MIN_EPOCH`, `MIN_SAMPLES` and `SLOTS_PER_SERVER` for your workload. The caller controls device allocation; the script does not choose or stop devices.

`NPU_COUNT` counts logical devices visible to PyTorch, not necessarily physical boards. On A3 hardware a physical board can expose two dies.

Check the composed command without starting training:

```bash
bash recipe/dapo_predictor/run_ews.sh --dry-run
```

For a short integration check of cold-start, update and active prediction:

```bash
MIN_EPOCH=0 MIN_SAMPLES=0 TOTAL_EPOCHS=2 \
  OUTPUT_DIR="$WORK_ROOT/runs/d1-smoke" \
  bash recipe/dapo_predictor/run_ews.sh \
  data.train_max_samples=16 data.val_max_samples=8
```

D1 still needs calibration observations before activation. These reduced gates are for integration checks, not evidence that an untrained head is useful. In a normal run the defaults are ten completed RL dataset passes and 2560 observed rollout requests. `predictor_reorder.epochs=10` is the number of predictor fitting iterations per update, not the RL warm-up.

For a matched standard-DAPO baseline, use `PREDICTOR_ENABLE=false` with the same model, seed, dataset, capacity and lengths. Give each run its own `OUTPUT_DIR`; do not compare an untrained smoke with an activated performance run. Pass extra Hydra overrides after the script arguments. To resume, retain the same output directory and pass `trainer.resume_mode=auto`.

## Predictor configuration

The legacy `epws` configuration keys, Python class names and metric prefixes remain compatible with the EWS display name.

| Option under `trainer.predictor_reorder` | Default | Meaning |
| --- | --- | --- |
| `enable` | false in the library; true in run_ews.sh | Opt in to predictor worker and scheduling |
| `scheduler` | static_snake in the library; epws in run_ews.sh | Community static packing or EWS |
| `backend` | linear_listmle | D1, or censored_lognormal for D2 |
| `activation.min_epoch` | 10 | Completed RL dataset passes before activation |
| `activation.min_samples` | 2560 | Observed rollout-request count |
| `epochs / batch_size / lr / weight_decay` | 10 / 32 / 3e-5 / 1e-4 | D1 optimizer settings |
| `calibration_max_samples` | 4096 | D1 historical prompt/length replay capacity |
| `epws.slots_per_server` | 8 | Admission capacity per server |
| `epws.max_concurrent_requests` | null | Explicit global window; null derives servers × slots |
| `risk_weight` | 0.5 | D2 tail-risk weight |
| `d2_lifecycle.enable / interval_epochs` | true / 5 | D2 quality admission and monitoring cadence |

D1 uses the maximum completed sibling length as each prompt's fitting/calibration target. Prediction extracts one actor representation per contiguous `rollout.n` sibling group, reuses the trainer's tokenized inputs, then expands the prediction to all siblings.

D2 requires an externally fitted checkpoint and a held-out candidate report; it is not an online D2 trainer:

```bash
BACKEND=censored_lognormal \
D2_CANDIDATE_REPORT=/shared/ews/d2-candidate.json \
TOTAL_EPOCHS=20 OUTPUT_DIR="$WORK_ROOT/runs/d2-ews" \
  bash recipe/dapo_predictor/run_ews.sh
```

Missing or rejected candidates leave FCFS active. The default D2 checks occur after completed passes 10, 15, 20, ...; admitted heads remain frozen. Read [D2_LIFECYCLE.md](D2_LIFECYCLE.md) for the checkpoint schema, external collection/fitting obligations, validation JSONL, thresholds, monitoring and resume. For explicit epoch/sample-only compatibility, use `++trainer.predictor_reorder.d2_lifecycle.enable=false` and supply `+trainer.predictor_reorder.d2_checkpoint=/path/to/head.pt`.

## Design and integration

| File | Responsibility |
| --- | --- |
| `config.py`, `main_dapo_predictor_reorder.py` | Validate opt-in settings, choose the worker and register the manager |
| `epws_manager.py` | Bound requests in flight, refill after completion and restore original output order |
| `predictor_dapo_trainer.py` | Activation, scoring, DAPO training-loop integration and checkpoint lifecycle |
| `predictor_worker.py` | Actor hidden-state extraction, online D1 fitting/calibration, frozen D2 inference and RPCs |
| `length_scheduler/` | Waiting pool, async execution, calibration, distribution API and D2 quality state |
| `validate_d2.py` | Publish an immutable-checkpoint-bound external validation report |
| `run_ews.sh` | Single-node launch example |
| `tests/` | Numeric, configuration, worker, manager and lifecycle regressions |

The trainer generates responses, computes rewards/advantages and updates the actor through the pinned DAPO flow. D1 fitting follows the actor update and uses completed responses for future batches. D2 monitoring pairs pre-rollout forecasts with outputs before DAPO filtering. Returned rows are restored to their original sample order before reward/training alignment.

Missing or invalid work values select stable FCFS for the whole batch. Backend and checkpoint identity are checked on resume; failed cross-rank D2 loading aborts before mixed predictors can schedule. EWS has no live decode-progress interface.

## Results and verification

Representative Qwen3.5-9B / DAPO-Math rollout measurements:

| Host | FCFS, tokens/s | D1 + EWS, tokens/s | D2 + EWS, tokens/s |
| --- | ---: | ---: | ---: |
| 232 | 487.54 | 514.65 (+5.56%) | 547.07 (+12.21%) |
| 222 | 480.61 | 525.11 (+9.26%) | 547.68 (+13.96%) |

[PERFORMANCE.md](PERFORMANCE.md) records the frozen trace, measured dispatch configuration, timing scope and transfer results. [VALIDATION.md](VALIDATION.md) separates CPU checks, integration coverage and device checks.

For local CPU checks:

```bash
python -m pip install pytest numpy hydra-core codetiming ruff==0.14.10
python -m ruff check dapo_predictor
python -m ruff format --check dapo_predictor
python -m pytest tests/ dapo_predictor/tests/ -q -rs
```

Torch-dependent tests need a CPU PyTorch installation; the DataProto/manager integration cases also need the pinned verl and its dependencies. The CI predictor job installs CPU PyTorch and executes the worker/lifecycle tests without an NPU.
