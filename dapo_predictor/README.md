# DAPO length prediction and EWS

EWS (Event-driven Waiting-pool Scheduler) starts rollout requests with longer predicted responses first and refills a shared request window as requests finish. This reduces the wait for late-starting long responses at the end of a batch. The default predictor is online D2; the community-based D1 predictor is also available.

The implementation stays in the recipe and integrates with verl's FSDP/FSDP2 workers and rollout manager. It requires no vLLM or vLLM-Ascend source patches.

## 1. How it works

### EWS scheduling

- During cold start, requests are admitted in FCFS order.
- Once the predictor meets its activation conditions, requests with more predicted work are admitted first.
- Each completed request frees a slot for another request from the shared waiting pool.
- Generated rows are restored to their original batch order before returning to the trainer.

EWS controls admission order. Inference-server placement remains with verl's `GlobalRequestLoadBalancer`. Priorities are fixed for each batch; the scheduler does not poll live decode progress or force anchor rollouts.

### D2: default predictor

D2 combines prompt hidden states from multiple actor layers and learns a censored LogNormal response-length distribution. It provides both length estimates and tail-risk information. Responses that reach the generation cap are treated as right-censored observations rather than completed answers of exactly that length.

The default workflow is:

1. Extract prompt features before rollout and collect the resulting response lengths.
2. Fit the predictor incrementally in a background CPU process, using the latest completed head for inference.
3. Keep FCFS admission during cold start.
4. Activate predictive admission after **10 completed RL epochs**, **2560 observed responses**, and an available fitted head. The epoch and sample thresholds are configurable through `activation.min_epoch` and `activation.min_samples`.
5. Continue collecting samples and updating the predictor after activation. Save and restore predictor state with training checkpoints.

No external predictor checkpoint is required by default. A fit in progress does not block rollout or replace a head mid-batch. Projection and log-length normalization are fixed after the first fit; subsequent fits warm-start the head and optimizer.

The scheduling score is:

`median + risk_weight * (p90 - median)`

`risk_weight` defaults to `0.5` and is configurable. Increasing it gives more weight to the predicted upper tail.

### D1: optional predictor

D1 follows [DAPO Predictor Reorder (verl-recipe PR #100)](https://github.com/verl-project/verl-recipe/pull/100): a bias-free linear head uses the last prompt token's hidden state and learns response-length ordering with the ListMLE loss. The fitting target is the maximum response length within each prompt's rollout group.

This recipe adds PAVA monotone calibration to map the ranking score to a token-length estimate and uses that estimate for EWS admission.

The backend is selected at launch and remains fixed during training. Neither backend changes the actor's RL objective.

## 2. Requirements and installation

| Component | Supported or tested configuration |
| --- | --- |
| Platform | Linux with an Ascend runtime for device execution |
| Actor | FSDP/FSDP2 using DataProto worker RPCs; text-only online D2 |
| verl | `0.8.0.dev`, commit `bcb638649a50e58494a8ddd92085ad1174f674b8` |
| Tested rollout stack | vLLM `0.19.1`, vLLM-Ascend `0.19.1rc1` |
| Tested runtime image | `quay.io/ascend/vllm-ascend:v0.19.1rc1-a3-openeuler` |
| Tested device environment | Python 3.11.14, PyTorch/torch-npu 2.9.0, Transformers 5.5.3 |

Use a runtime whose driver, CANN, PyTorch and torch-npu versions match the host. For the image above, initialize the toolkit with `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. The example launcher uses SDPA for the actor and does not require a separate FlashAttention installation.

[REQUIRED_VERL.txt](REQUIRED_VERL.txt) is the repository installer's input for selecting the core verl revision. This recipe is installed into the verl source tree, not as a standalone wheel. The scheduler package does not import vLLM or vLLM-Ascend; those are rollout runtime dependencies.

The following commands fetch this PR, install its pinned verl revision and copy both required recipe directories:

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

Use fresh destination directories for a first install. When integrating into an existing checkout, compare files before applying the overlays. Both directories are needed: `dapo_predictor` imports the DAPO trainer from `recipe.dapo`.

`constraints-ascend.txt` aligns dataset, Arrow and telemetry dependencies for the tested image. It does not replace PyTorch, CANN or the inference backend. Enabling the predictor selects `trainer.use_legacy_worker_impl=enable`, matching its DataProto worker interface.

For framework or backend upgrades, see [UPGRADING.md](UPGRADING.md).

## 3. Model and data

Prepare the public DAPO-Math training data and AIME validation data:

```bash
cd "$VERL_SRC"
VERL_HOME="$VERL_SRC" bash recipe/dapo/prepare_dapo_data.sh
export TRAIN_FILE="$VERL_SRC/data/dapo-math-17k.parquet"
export TEST_FILE="$VERL_SRC/data/aime-2024.parquet"
```

Parquet rows contain chat messages in `prompt`, `data_source`, and a rule-based `reward_model.ground_truth`. Keep training and validation datasets separate.

Choose a model supported by your rollout backend. For example:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-1.7B", local_dir="models/Qwen3-1.7B")'
export MODEL_PATH="$VERL_SRC/models/Qwen3-1.7B"
```

Model downloads require access to the provider and acceptance of the model license.

## 4. Launch and configuration

Run from the pinned verl checkout. Set `NPU_COUNT` to the number of devices visible to PyTorch; `TP_SIZE` must divide it.

```bash
export NPU_COUNT=2
export TP_SIZE=1

# Default: online D2 + EWS.
OUTPUT_DIR="$WORK_ROOT/runs/d2-ews" bash recipe/dapo_predictor/run_ews.sh

# Optional: D1 + EWS.
BACKEND=linear_listmle OUTPUT_DIR="$WORK_ROOT/runs/d1-ews" \
  bash recipe/dapo_predictor/run_ews.sh

# Native DAPO/FCFS comparison, without the predictor.
PREDICTOR_ENABLE=false OUTPUT_DIR="$WORK_ROOT/runs/fcfs" \
  bash recipe/dapo_predictor/run_ews.sh
```

The launcher is a single-node example. Use separate output directories and keep model, data, sampling, response caps and device resources matched for comparisons. Standard verl settings remain configurable through launcher variables such as `PROMPT_LENGTH`, `RESPONSE_LENGTH`, `BATCH_SIZE`, `ROLLOUT_N` and `TOTAL_EPOCHS`, or through additional Hydra overrides.

For an existing DAPO setup, retain its model, data, sampling and optimizer settings, use `recipe.dapo_predictor.main_dapo_predictor_reorder` as the entry point, and add:

```yaml
trainer:
  predictor_reorder:
    enable: true
    scheduler: ews
    backend: censored_lognormal
    risk_weight: 0.5
    activation:
      min_epoch: 10
      min_samples: 2560
    ews:
      slots_per_server: 8
      max_concurrent_requests: null
```

Use `backend: linear_listmle` for D1, or `enable: false` for the native path.

| Option under `trainer.predictor_reorder` | Default | Meaning |
| --- | --- | --- |
| `enable` | false in the library; true in the launcher | Enable the predictor and scheduler |
| `backend` | `censored_lognormal` | D2; `linear_listmle` selects D1 |
| `scheduler` | `ews` when enabled | Event-driven waiting-pool scheduling |
| `activation.min_epoch` | 10 | Completed RL epochs required for activation |
| `activation.min_samples` | 2560 | Observed responses required for activation |
| `risk_weight` | 0.5 | D2 upper-tail weight |
| `ews.slots_per_server` | 8 | Admission capacity per inference server |
| `ews.max_concurrent_requests` | null | Global admission limit; null derives server count × slots |

Epoch, sample-count and head-readiness conditions must all be met. For short training runs or large datasets with long epochs, adjust the thresholds to the workload; otherwise admission may remain FCFS throughout training. Activation is currently epoch-based.

The launcher exposes `MIN_EPOCH`, `MIN_SAMPLES` and `SLOTS_PER_SERVER`. D1's `predictor_reorder.epochs` controls predictor fitting iterations, not the RL activation epoch.

Legacy `epws` configurations and imports remain accepted; conflicting old and new capacity settings are rejected. Online training state and the optional external frozen-D2 workflow are described in [D2_LIFECYCLE.md](D2_LIFECYCLE.md).

## 5. Code structure

| File or directory | Responsibility |
| --- | --- |
| `config.py`, `main_dapo_predictor_reorder.py` | Validate configuration and connect the training entry point |
| `predictor_worker.py` | Prompt features, predictor fitting/inference and head loading |
| `predictor_dapo_trainer.py` | Connect RL training, length observations and predictor lifecycle |
| `ews_manager.py` | Bounded request admission, completion-driven refill and output reordering |
| `length_scheduler/predictor.py` | Predictor interfaces and backends |
| `length_scheduler/d2_online.py` | Replay buffer, background CPU training and head publication |
| `length_scheduler/distribution.py` | Length-distribution calculations |
| `length_scheduler/calibration.py` | D1 PAVA calibration |
| `length_scheduler/lifecycle.py` | Activation conditions and state |
| `length_scheduler/scheduler.py`, `event_loop.py` | Waiting-pool ordering and asynchronous execution |
| `length_scheduler/checkpoint.py` | Predictor checkpoint support |
| `tests/` | Unit, integration and regression tests |

## 6. Results and validation

In historical fixed-trace rollout tests, D2 + EWS improved throughput over FCFS by **12.21%** and **13.96%** on two hosts.

D2's Spearman correlation with actual mean response length per prompt was **0.7595** and **0.7401** on two evaluation groups, compared with **0.3229** and **0.4545** for D1 ranking scores.

These measurements use the historical experiment configuration. [PERFORMANCE.md](PERFORMANCE.md) describes the workload, dispatch settings and transfer results.

Online D2 has completed cold-start, background fitting, activation, checkpoint/resume and shutdown checks. See [VALIDATION.md](VALIDATION.md) for functional validation.
