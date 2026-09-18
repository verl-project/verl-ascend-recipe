#!/usr/bin/env bash
# SFT | GSM8K | FSDP engine | Ascend A2/A3 NPU
# Qwen3-8B supervised fine-tuning on the FSDP backend, with optional
# Ulysses sequence parallel and LoRA (PEFT) toggles controlled by env vars.
#
# Examples:
#   # plain full-parameter SFT on 8 NPUs
#   USE_PEFT=0 SP_SIZE=1 bash run_qwen3_8b_fsdp_sft_npu.sh
#
#   # Ulysses sequence parallel = 2 + LoRA (default)
#   bash run_qwen3_8b_fsdp_sft_npu.sh

set -xeuo pipefail

# ================== Hardware configuration setting ==================
## For Ascend A2 (8 NPUs per node)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
## For Ascend A3 (16 NPUs per node), uncomment the two lines below
#export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
#NPROC_PER_NODE=16

MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-65535 -n 1)}
# ====================================================================

# ==================== User adjustable parameters ====================
# Data path
TRAIN_DATA=${TRAIN_DATA:-"${HOME}/data/gsm8k_sft/train.parquet"}
TEST_DATA=${TEST_DATA:-"${HOME}/data/gsm8k_sft/test.parquet"}
# Model and output path
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-8B"}
SAVE_PATH=${SAVE_PATH:-"sft_outputs"}
# Training hyperparameters
SP_SIZE=${SP_SIZE:-2}                 # Ulysses sequence parallel size, 1 disables SP
USE_PEFT=${USE_PEFT:-1}               # 1 enables LoRA, 0 for full-parameter SFT
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGETS=${LORA_TARGETS:-all-linear}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
LR=${LR:-1e-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
MAX_LENGTH=${MAX_LENGTH:-4096}
# Other
PROJECT_NAME=${PROJECT_NAME:-"sft-gsm8k"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_8b_fsdp_sft"}
# ====================================================================

extra_args=()
if [ "${USE_PEFT}" = "1" ]; then
    extra_args+=(
        "model.lora_rank=${LORA_RANK}"
        "model.lora_alpha=${LORA_ALPHA}"
        "model.lora_dropout=${LORA_DROPOUT}"
        "model.target_modules=${LORA_TARGETS}"
    )
fi

# Run SFT training
torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} \
  -m verl.trainer.sft_trainer \
  data.train_files=${TRAIN_DATA} \
  data.val_files=${TEST_DATA} \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.truncation=right \
  data.max_length=${MAX_LENGTH} \
  data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
  data.ignore_input_ids_mismatch=True \
  optim.lr=${LR} \
  engine=fsdp \
  engine.ulysses_sequence_parallel_size=${SP_SIZE} \
  model.path="${MODEL_PATH}" \
  model.trust_remote_code=True \
  model.use_remove_padding=True \
  "${extra_args[@]}" \
  trainer.default_local_dir="${SAVE_PATH}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.logger='["console","wandb"]'
