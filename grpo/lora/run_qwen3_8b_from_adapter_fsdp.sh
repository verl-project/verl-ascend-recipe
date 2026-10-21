#!/usr/bin/env bash
# GRPO + LoRA from adapter | Qwen3-8B | FSDP | vLLM-Ascend

set -xeuo pipefail

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export USE_OPTIMIZED_MODEL=${USE_OPTIMIZED_MODEL:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

########################### user-adjustable ###########################

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:?must set LORA_ADAPTER_PATH}
TRAIN_FILE=${TRAIN_FILE:?must set TRAIN_FILE}
VAL_FILE=${VAL_FILE:?must set VAL_FILE}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

ACTOR_LR=${ACTOR_LR:-3e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-512}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16,32]'}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}
RESUME_MODE=${RESUME_MODE:-auto}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PWD/checkpoints/qwen3_8b_lora_from_adapter}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_lora_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_lora_from_adapter_fsdp_vllm_ascend}

########################### adapter validation ###########################

ADAPTER_CONFIG=${LORA_ADAPTER_PATH}/adapter_config.json
if [[ ! -f ${ADAPTER_CONFIG} ]]; then
    echo "Missing adapter config: ${ADAPTER_CONFIG}" >&2
    exit 1
fi
if [[ ! -f ${LORA_ADAPTER_PATH}/adapter_model.safetensors && ! -f ${LORA_ADAPTER_PATH}/adapter_model.bin ]]; then
    echo "Missing adapter weights under: ${LORA_ADAPTER_PATH}" >&2
    exit 1
fi

read -r ADAPTER_LORA_RANK ADAPTER_LORA_ALPHA < <(
    python3 -c 'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print(int(c["r"]), int(c.get("lora_alpha", c["r"])))' "${ADAPTER_CONFIG}"
)
LORA_RANK=${LORA_RANK:-${ADAPTER_LORA_RANK}}
LORA_ALPHA=${LORA_ALPHA:-${ADAPTER_LORA_ALPHA}}
if [[ ${LORA_RANK} -ne ${ADAPTER_LORA_RANK} || ${LORA_ALPHA} -ne ${ADAPTER_LORA_ALPHA} ]]; then
    echo "LORA_RANK/LORA_ALPHA must match adapter_config.json (${ADAPTER_LORA_RANK}/${ADAPTER_LORA_ALPHA})." >&2
    exit 1
fi

for required_file in "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -f ${required_file} ]]; then
        echo "Missing data file: ${required_file}" >&2
        exit 1
    fi
done

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.dataloader_num_workers=0
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.lora_adapter_path="${LORA_ADAPTER_PATH}"
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.use_shm=False
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    +actor_rollout_ref.actor.checkpoint.save_lora_only=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=${CUDAGRAPH_CAPTURE_SIZES}
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode=${RESUME_MODE}
    trainer.max_actor_ckpt_to_keep=2
)

########################### launch ###########################

LOG_DIR=${LOG_DIR:-$PWD/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${LOG_FILE}")"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    "$@" 2>&1 | tee "${LOG_FILE}"
