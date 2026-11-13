# Qwen3-8B SFT on Ascend NPU (FSDP backend)

Supervised fine-tuning of **Qwen3-8B** on Ascend A2/A3 NPUs using the VeRL
**FSDP** backend, with optional **Ulysses sequence parallelism** and **LoRA
(PEFT)** switches. This recipe targets
[issue #74](https://github.com/verl-project/verl-ascend-recipe/issues/74)
(Q3 community task 5).

Reference script in the upstream VeRL repo:
`examples/sft/gsm8k/run_qwen3_8b_fsdp.sh`

## 1. Environment

The eight-card A2 validation used verl 0.9.1, CANN 9.1.0, PyTorch 2.10.0,
torch_npu 2.10.0.post4, and Transformers 5.5.4. Earlier image references:

- A2: `quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1`
- A3: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1`

## 2. Data

Prepare the GSM8K SFT parquet files (see `verl_ascend_practice` for the
preprocessing pipeline), then point the script at them:

```bash
export TRAIN_DATA=$HOME/data/gsm8k_sft/train.parquet
export TEST_DATA=$HOME/data/gsm8k_sft/test.parquet
```

## 3. Usage

```bash
cd sft

# Full-parameter SFT, no sequence parallel
USE_PEFT=0 SP_SIZE=1 bash run_qwen3_8b_fsdp_sft_npu.sh

# Ulysses sequence parallel (SP=2) + LoRA (default demo)
bash run_qwen3_8b_fsdp_sft_npu.sh
```

### Key toggles

| Env var | Default | Description |
|---|---|---|
| `SP_SIZE` | 2 | Ulysses sequence parallel size; `1` disables SP |
| `USE_PEFT` | 1 | `1` enables LoRA, `0` for full-parameter SFT |
| `LORA_RANK` / `LORA_ALPHA` | 32 / 16 | LoRA hyperparameters |
| `MICRO_BATCH_SIZE_PER_GPU` | 2 | Per-NPU micro batch size; tune for memory |
| `NPROC_PER_NODE` | 8 | 8 for A2, 16 for A3 |

Other knobs (model path, LR, epochs, max length, save dir, ...) are exposed as
env vars at the top of the script and can also be overridden directly.

## 4. Verification

- Loss decreases and converges over a 12-hour run (or 100 training steps).
- With the verl 0.9.1 FSDP engine, the default checkpoint includes model,
  optimizer, and extra-state shards even with LoRA enabled. Check for all
  ranks under `SAVE_PATH/global_step_<step>`.

### Eight-card A2 verification (2026-11-12)

See the [validation report and both training logs](../verl_ascend_practice/reports/qwen3_8b_sft/README.md).

Both runs used eight 910B4 NPUs with 32 GiB each, Qwen3-8B, and the
7,473 training records from GSM8K. Each sample contained a user question and
an assistant answer. The validation loss used a subset of the GSM8K test split.

| Mode | Steps | Global batch | Validation samples | Final training loss | Validation loss | Exit code |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA rank 32, SP=2 | 100 | 32 | 128 | 0.2753 | 0.3928 | 0 |
| Full parameters, SP=2 | 100 | 32 | 128 | 0.4211 | 0.4737 | 0 |

For the LoRA run, average training loss fell from 1.0664 over steps 1–10 to
0.3276 over steps 91–100. Steps 11–90 processed 487,529 global training tokens
in approximately 370 seconds (1,318 tokens/s across eight cards). The full-parameter run processed the same token count in 1,009 seconds
(483.2 tokens/s across eight cards). Both measurements use
elapsed times from the training progress log. Peak allocated NPU memory
reported by the trainer was 13.13 GiB for LoRA and 22.22 GiB for full training.
GSM8K generation accuracy and A100-relative throughput remain unmeasured.

To reproduce the LoRA run in the validation environment, activate the verl
environment, set the model, train, test and output paths, and run:

```bash
# This image needed jemalloc for clean process teardown.
export LD_PRELOAD=/lib/aarch64-linux-gnu/libjemalloc.so.2
SP_SIZE=2 USE_PEFT=1 MICRO_BATCH_SIZE_PER_GPU=1 TRAIN_BATCH_SIZE=32 MAX_LENGTH=1024 \
  bash run_qwen3_8b_fsdp_sft_npu.sh trainer.total_training_steps=100 \
    'trainer.logger=[console]' trainer.resume_mode=disable trainer.n_gpus_per_node=8 \
    data.num_workers=2 data.val_max_samples=128 data.max_token_len_per_gpu=2048
```
