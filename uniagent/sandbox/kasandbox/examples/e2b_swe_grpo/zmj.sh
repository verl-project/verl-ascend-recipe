#!/usr/bin/env bash
# Adapted from the colleague 9B Dense / 32-NPU recipe; source with set -a.
# Deployment paths and gateway_count are adapted; training values are pinned.
# Pass deliberate Hydra overrides as arguments to run_reproduce.sh.
REPO_ROOT="${REPO_ROOT:-$ROOT}"
cd "${REPO_ROOT}"

RUNTIME_DIR="${RUNTIME_DIR:-$ROOT/.artifacts}"
MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/.artifacts/data/train.parquet}"
TEST_FILE="${TEST_FILE:-$ROOT/.artifacts/data/validation.parquet}"
TASK_CONFIG="${TASK_CONFIG:-$ROOT/examples/e2b_swe_grpo/task_config.yaml}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8260}"

PROJECT_NAME="${PROJECT_NAME:-Uni-Agent-Qwen3.5-9B-megatron}"
EXP_NAME="${EXP_NAME:-e2b_django_9b_n8_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_DIR}}"
CKPTS_DIR="${CKPTS_DIR:-${OUTPUT_ROOT}/ckpts/${PROJECT_NAME}/${EXP_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${OUTPUT_ROOT}/logs/${PROJECT_NAME}/${EXP_NAME}}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$REPO_ROOT/.artifacts/rollouts/$EXP_NAME}"
CONSOLE_LOG_DIR="${CONSOLE_LOG_DIR:-${OUTPUT_ROOT}/logs/qwen3p5-4b-uniagent}"
CONSOLE_LOG_FILE="${CONSOLE_LOG_FILE:-${CONSOLE_LOG_DIR}/${EXP_NAME}.log}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# Group filtering is disabled, matching the executable reference setting.
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"

case ":${PYTHONPATH:-}:" in
    *":$REPO_ROOT:"*) ;;
    *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}";;
esac
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT="${TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT:-2400}"

# -------- Ascend runtime / vLLM optimizations --------
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE="${VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-10}"

# -------- Qwen3.5 GDN backend --------
# The default Torch GDN rejects packed cu_seqlens with remove padding.
# The reference Megatron config inherits the actor transformer override.

# Colocate async: both roles share all 32 NPUs.
NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-16}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-2}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-16}"
PHYSICAL_NPUS="${PHYSICAL_NPUS:-32}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"

# 32 prompts x 8 responses produce 256 trajectories per training step.
# Keep this no larger than the filtered dataset size; trainer v1 computes
# steps_per_epoch with integer division and cannot run when that value is zero.
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"

# Match the reference 139072-token context envelope.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-131072}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# Trainer TP4 x PP2 x CP4 gives DP1; rollout TP4 gives eight replicas.
ROLLOUT_TP="${ROLLOUT_TP:-4}"
TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_PP="${TRAIN_PP:-2}"
TRAIN_CP="${TRAIN_CP:-4}" # CP4: 139072-token envelope / 4

GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-256}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-True}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-longest}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"

# -------- GRPO algorithm settings --------
# Match the GPU Qwen3.5 Dense baseline: GRPO advantage estimation with the
# vanilla PPO clipped policy loss instead of GSPO's sequence-level objective.
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_COEF="${KL_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
BYPASS_MODE="${BYPASS_MODE:-True}"
BYPASS_LOSS_TYPE="${BYPASS_LOSS_TYPE:-ppo_clip}"
ROLLOUT_IS="${ROLLOUT_IS:-null}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
ROLLOUT_IS_BATCH_NORMALIZE="${ROLLOUT_IS_BATCH_NORMALIZE:-False}"
ROLLOUT_RS="${ROLLOUT_RS:-null}"
ROLLOUT_RS_THRESHOLD="${ROLLOUT_RS_THRESHOLD:-null}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
LOSS_MODE="${LOSS_MODE:-vanilla}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"

ACTOR_PPO_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))
LOG_PROB_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))
TOTAL_NPUS=$((NNODES * NGPUS_PER_NODE))
TRAIN_MODEL_PARALLEL_SIZE=$((TRAIN_TP * TRAIN_PP * TRAIN_CP))
ROLLOUT_NPUS=$((ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))
ROLLOUT_REPLICAS=$((ROLLOUT_NPUS / ROLLOUT_TP))
