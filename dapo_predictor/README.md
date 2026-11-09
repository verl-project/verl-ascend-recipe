# DAPO length predictor and EWS

EWS (Event-driven Waiting-pool Scheduler) admits rollout requests from a shared waiting pool. Before a selected predictor is ready, requests follow FCFS. Once active, longer predicted work starts first, so long responses can overlap with shorter requests instead of being left until the end of a batch.

## What changes

| Path | Admission order | Refill | Prediction |
| --- | --- | --- | --- |
| Standard DAPO rollout | Existing manager order | Existing manager | None |
| Community predictor / static Snake | Scores packed across DP ranks before generation | Existing manager | Linear/ListMLE ranking scores |
| EWS + D2 (default backend) | Risk-adjusted predicted long first after warm-up | One global bounded window, refilled on completion | Online censored LogNormal; asynchronous CPU fitting |
| EWS + D1 (optional backend) | Predicted long first, FCFS during warm-up | Same EWS window | Linear/ListMLE + historical PAVA calibration |

Server placement stays with verl's `GlobalRequestLoadBalancer`. Priorities are fixed for each batch: EWS does not poll decode progress, migrate running requests or revise predictions on completion. There are no forced anchor rollouts or sibling corrections.

D2 collects actor hidden taps before each rollout and pairs them with completed response lengths. One background CPU process incrementally fits the head while RL continues. During the first ten completed RL epochs it does not affect admission; after warm-up, EWS uses the latest published head. A fitting job still in progress does not stall rollout or replace a head mid-batch. The scheduling score is `median + risk_weight * (p90 - median)`.

D1 learns online from completed training batches. PAVA maps its unitless ranking score to token-scale work. The backend is selected at launch and remains fixed. Neither backend changes the actor's RL objective.

## Requirements

| Component | Configuration |
| --- | --- |
| Platform | Linux, Ascend NPU for the documented device run |
| Actor | FSDP or FSDP2 with legacy DataProto worker RPCs; text-only D2 |
| verl | `0.8.0.dev`, commit `bcb638649a50e58494a8ddd92085ad1174f674b8` |
| Rollout stack used in evaluation | vLLM `0.19.1`, vLLM-Ascend `0.19.1rc1` |
| Runtime image used in evaluation | `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler` |
| Python | Device environment 3.11.14; CI 3.10 |
| Libraries | PyTorch/torch-npu and Ray from the runtime; the pinned verl dependencies |

This is a source-tree recipe, not a standalone wheel. [REQUIRED_VERL.txt](REQUIRED_VERL.txt) is read by the repository installer to select the core verl revision. The scheduler package has no vLLM/vLLM-Ascend imports or source patches. SciPy is optional for the D1 Kendall-tau metric.

For backend upgrades, see [UPGRADING.md](UPGRADING.md): device-free tests and the integration files to review are listed separately from device validation.

## Installation

Start inside a working Ascend runtime whose driver/CANN/PyTorch/torch-npu stack matches the host. Keep this checkout and its dependencies in your own workspace.

For the image above, initialize its toolkit environment with `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. The example launcher selects SDPA for the actor, so it does not require a separate FlashAttention installation. It sets the same per-server sequence capacity for FCFS and EWS.

The predictor worker uses the pinned FSDP DataProto interface. Enabling it selects `trainer.use_legacy_worker_impl=enable`; the example launcher uses that same implementation in the matched FCFS baseline. This avoids mixing TensorDict engine-worker calls with DataProto worker methods.

The commands below fetch this PR, install its pinned verl and copy both recipe directories. The overlay copies directory contents rather than creating a nested `dapo_predictor/dapo_predictor` layout.

```bash
export WORK_ROOT="$PWD/ews-work"
mkdir -p "$WORK_ROOT"
git clone https://github.com/verl-project/verl-ascend-recipe.git "$WORK_ROOT/recipes"
cd "$WORK_ROOT/recipes"
git fetch origin pull/142/head
git switch --detach FETCH_HEAD
export RECIPE_SRC="$PWD"
export PIP_CONSTRAINT="$RECIPE_SRC/dapo_predictor/constraints-ascend.txt"

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

`constraints-ascend.txt` aligns dataset, Arrow and telemetry dependencies for the documented image. It does not install or replace PyTorch, CANN or the inference backend. For offline hosts, use a read-only cache of the same public core archive and ARM64 wheels, extract the pinned core into a new directory, and run `python -m pip install --no-index --find-links "$WHEEL_CACHE" --no-build-isolation -c "$PIP_CONSTRAINT" -e "$VERL_SRC"`. Apply the two recipe overlays as above. A cached source archive must identify the same core commit; copying another environment's site-packages is not an installation step.

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
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-1.7B", local_dir="models/Qwen3-1.7B")'
export MODEL_PATH="$VERL_SRC/models/Qwen3-1.7B"
```

Choose a model supported by the selected rollout backend. The small-model integration workload is separate from the Qwen3.5-9B performance results.

For a reproducible small question population, select by prompt hash before generating responses:

```bash
python -m recipe.dapo_predictor.prepare_subset \
  --input "$TRAIN_FILE" --output "$WORK_ROOT/data/dapo32.parquet" \
  --model "$MODEL_PATH" --count 32 --max-prompt-length 1024 --seed 1
```

The command writes prompt IDs and input/output hashes beside the parquet. It does not inspect completion lengths. Reuse that exact file for every compared run. A 20-epoch acceptance run can use 32 prompts, 8 responses/prompt, batches of 16 and a fixed response cap: 256 responses per pass and 5120 total. This is a complete run over the selected population, not over all 17K DAPO prompts. Validation may be disabled while measuring future responses on this same population; using it as the loader's validation file is not an independent validation set.

## Run D2 + EWS

From the pinned verl checkout, on devices allocated to this job:

```bash
export NPU_COUNT=2
export TP_SIZE=1
export OUTPUT_DIR="$WORK_ROOT/runs/d2-ews"
bash recipe/dapo_predictor/run_ews.sh
```

The script sets up a single-node, text-only DAPO run with console logging and no W&B. Defaults are D2 + EWS, 8 prompts/batch, 2 responses/prompt, 512 prompt tokens, 1024 response tokens and 20 RL epochs. Set `PROMPT_LENGTH`, `RESPONSE_LENGTH`, `BATCH_SIZE`, `ROLLOUT_N`, `TOTAL_EPOCHS`, `MIN_EPOCH`, `MIN_SAMPLES` and `SLOTS_PER_SERVER` for your workload. The caller controls device allocation; the script does not choose or stop devices.

`NPU_COUNT` counts logical devices visible to PyTorch, not necessarily physical boards. On A3 hardware a physical board can expose two dies.

Check the composed command without starting training:

```bash
bash recipe/dapo_predictor/run_ews.sh --dry-run
```

For a short integration check, reduce both activation gates and the initial fitting evidence floor:

```bash
MIN_EPOCH=0 MIN_SAMPLES=0 TOTAL_EPOCHS=3 \
  OUTPUT_DIR="$WORK_ROOT/runs/d2-smoke" \
  bash recipe/dapo_predictor/run_ews.sh \
  data.train_max_samples=16 data.val_max_samples=8 \
  ++trainer.predictor_reorder.d2_online.min_fit_prompts=8
```

D2 needs a completed CPU fit before activation. These reduced gates check integration, not predictor quality. If the CPU fit takes longer than a tiny smoke, extend the smoke rather than blocking RL to manufacture activation. In a normal run the gates are ten completed RL dataset passes, 2560 observed rollout requests and an available head. With only a few RL epochs, the default gate can leave FCFS active for the entire run. Choose epoch thresholds for the workload; there is no step-based activation mode.

Select the community backend with `BACKEND=linear_listmle`. Its `predictor_reorder.epochs=10` counts fitting iterations per update, not RL warm-up epochs.

For a matched standard-DAPO baseline, use `PREDICTOR_ENABLE=false` with the same model, seed, dataset, capacity and lengths. This uses upstream DAPO training, worker and manager behavior. Both paths normalize NumPy reward metadata before writing JSONL; training rewards are unchanged. Give each run its own `OUTPUT_DIR`; do not compare early and late RL epochs as though actor behavior were unchanged. Pass extra Hydra overrides after the script arguments. To resume, retain the same output directory and pass `trainer.resume_mode=auto`.

## Predictor configuration

Use `scheduler=ews` and the `ews` capacity settings. Legacy `epws` configurations and imports remain accepted; conflicting old and new capacity settings are rejected.

| Option under `trainer.predictor_reorder` | Default | Meaning |
| --- | --- | --- |
| `enable` | false in the library; true in run_ews.sh | Opt in to predictor worker and scheduling |
| `scheduler` | ews when enabled | EWS; static_snake remains an explicit D1 option |
| `backend` | censored_lognormal when enabled | D2; linear_listmle selects D1 |
| `activation.min_epoch` | 10 | Completed RL dataset passes before activation |
| `activation.min_samples` | 2560 | Observed rollout-request count |
| `epochs / batch_size / lr / weight_decay` | 10 / 32 / 3e-5 / 1e-4 | D1 optimizer settings |
| `calibration_max_samples` | 4096 | D1 historical prompt/length replay capacity |
| `ews.slots_per_server` | 8 | Admission capacity per server |
| `ews.max_concurrent_requests` | null | Explicit global window; null derives servers × slots |
| `risk_weight` | 0.5 | D2 tail-risk weight |
| `d2_online.enable` | true for D2 without external checkpoint/report | Background incremental fitting |
| `d2_online.min_fit_prompts` | 32 | Prompt observations before the first fit |
| `d2_online.max_prompt_observations` | 512 | Bounded recent replay; each observation includes sibling lengths |
| `d2_online.fit_steps / batch_size / lr` | 50 / 256 / 1e-3 | CPU fitting budget per completed snapshot |
| `d2_online.projection_dim` | 64, bounded by initial sample/hidden dimension | Per-tap dimension, fixed after the first fit |

D1 uses the maximum completed sibling length as each prompt's fitting/calibration target. Prediction extracts one actor representation per contiguous `rollout.n` sibling group, reuses the trainer's tokenized inputs, then expands the prediction to all siblings.

D2 trains from scratch by default. No external report or pretrained head is required. CPU fitting runs at most one job at a time; new observations are coalesced into the next snapshot. Projection and log-length normalization stay fixed after the first fit, while head weights and optimizer are warm-started. Failed fits keep the last valid head, or FCFS if none exists. Finite-value/schema checks protect publication; there is no accuracy-triggered retraining threshold.

For explicit external frozen-D2 compatibility:

```bash
BACKEND=censored_lognormal \
D2_CANDIDATE_REPORT=/shared/ews/d2-candidate.json \
TOTAL_EPOCHS=20 OUTPUT_DIR="$WORK_ROOT/runs/d2-ews" \
  bash recipe/dapo_predictor/run_ews.sh
```

Supplying `D2_CANDIDATE_REPORT` explicitly selects the external frozen workflow. Its quality admission is separate from default online fitting. Read [D2_LIFECYCLE.md](D2_LIFECYCLE.md) for online state, pre-response quality metrics and external compatibility.

## Design and integration

| File | Responsibility |
| --- | --- |
| `config.py`, `main_dapo_predictor_reorder.py` | Validate opt-in settings, choose the worker and register the manager |
| `ews_manager.py` | Bound requests in flight, refill after completion and restore original output order |
| `predictor_dapo_trainer.py` | Activation, scoring, DAPO training-loop integration and checkpoint lifecycle |
| `predictor_worker.py` | Pre-rollout hidden taps, D1 fitting/calibration, D2 inference and all-rank loading RPCs |
| `length_scheduler/d2_online.py` | Bounded replay, asynchronous CPU fitting, optimizer and publication |
| `length_scheduler/` | Waiting pool, async execution, calibration, distribution API and D2 quality state |
| `validate_d2.py` | Publish an immutable-checkpoint-bound external validation report |
| `run_ews.sh` | Single-node launch example |
| `tests/` | Numeric, configuration, worker, manager and lifecycle regressions |

The trainer generates responses, computes rewards/advantages and updates the actor through the pinned DAPO flow. D2 pairs pre-rollout features/forecasts with outputs before DAPO filtering or actor updates, evaluating forecasts before fitting on their labels. D1 fitting follows the actor update. Returned rows are restored to their original sample order before reward/training alignment.

Missing or invalid work values select stable FCFS for the whole batch. Online checkpoints include replay, head, optimizer, fixed transforms, version and counters. Backend/configuration identity is checked on resume; failed cross-rank D2 loading aborts before mixed predictors can schedule. EWS has no live decode-progress interface.

## Results and verification

With Qwen3-1.7B on 32 fixed DAPO-Math prompts, both online D2 + EWS and native FCFS completed 20 RL epochs (5120 responses each). D2 fitted on CPU during warm-up and first scheduled requests in epoch 11. Across epochs 11–20, rollout throughput was **743.76 vs 730.51 tokens/s (+1.81%)**; response-token throughput over full training steps improved **1.66%**. This is one natural-generation run per arm; output lengths and actor trajectories differ.

Separate Qwen3.5-9B / DAPO-Math exact-length rollout replay measurements:

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
