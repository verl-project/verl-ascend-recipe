# Qwen3-8B SFT on Ascend NPU (FSDP backend)

Supervised fine-tuning of **Qwen3-8B** on Ascend A2/A3 NPUs using the VeRL
**FSDP** backend, with optional **Ulysses sequence parallelism** and **LoRA
(PEFT)** switches. This recipe closes
[issue #74](https://github.com/verl-project/verl-ascend-recipe/issues/74)
(Q3 community task 5).

Reference script in the upstream VeRL repo:
`examples/sft/gsm8k/run_qwen3_8b_fsdp.sh`

## 1. Environment

Use the official VeRL Ascend images (CANN 8.5.0, VeRL 0.7.1):

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
- With LoRA enabled, only adapter weights are saved under `SAVE_PATH`; set
  `USE_PEFT=0` for full checkpoint saving.
