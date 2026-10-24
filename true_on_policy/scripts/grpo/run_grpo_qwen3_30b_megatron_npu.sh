#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B (MoE) | vLLM rollout | Megatron training | Ascend NPU
# Knobs:
#   ENABLE_TRUE_ON_POLICY=1  enable train-infer consistency (true on-policy)
#   ROLLOUT_PER_REQUEST_SEED=true|false  enable per-request rollout seed (default: follows ENABLE_TRUE_ON_POLICY)
#   ROLLOUT_SEED=42  base seed used when per-request rollout seed is enabled

set -xeuo pipefail

# Source Ascend CANN/ATB environment before running, e.g.:
# source ${ASCEND_HOME}/set_env.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export VERL_USE_EXTERNAL_MODULES=verl_ascend_recipe.true_on_policy.patch

# ---- train-infer consistency (true on-policy) ----
ENABLE_TRUE_ON_POLICY=${ENABLE_TRUE_ON_POLICY:-1}
if [ "${ENABLE_TRUE_ON_POLICY}" = "1" ]; then
    export VLLM_BATCH_INVARIANT=1
fi
rollout_per_request_seed_raw=${ROLLOUT_PER_REQUEST_SEED:-${ENABLE_TRUE_ON_POLICY}}
case "${rollout_per_request_seed_raw}" in
    1 | true | True | TRUE | yes | Yes | YES) rollout_per_request_seed=true ;;
    0 | false | False | FALSE | no | No | NO) rollout_per_request_seed=false ;;
    *)
        echo "ROLLOUT_PER_REQUEST_SEED must be true/false or 1/0, got: ${rollout_per_request_seed_raw}" >&2
        exit 1
        ;;
esac
rollout_seed=${ROLLOUT_SEED:-42}

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}

actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-8}
actor_ep=${ACTOR_EP:-1}
actor_etp=${ACTOR_ETP:-1}

rollout_tp=${ROLLOUT_TP:-1}
rollout_pp=${ROLLOUT_PP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_grpo_qwen3_moe}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_grpo_vllm_megatron}

train_file=${TRAIN_FILE:-$HOME/data/dapo-math-17k.parquet}
val_file=${VAL_FILE:-$HOME/data/dapo-math-17k.parquet}

# ---- end user-adjustable ----
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=3.0
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.position_embedding_type=rope
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.per_request_seed=${rollout_per_request_seed}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.pipeline_parallel_size=${rollout_pp}
)
if [ "${rollout_per_request_seed}" = "true" ] || [ -n "${ROLLOUT_SEED:-}" ]; then
    ROLLOUT+=(actor_rollout_ref.rollout.seed=${rollout_seed})
fi

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    model_engine=megatron
)

# ---- true on-policy megatron config (train-infer consistency) ----
TRUE_ON_POLICY_CONFIG=()
if [ "${ENABLE_TRUE_ON_POLICY}" = "1" ]; then
    TRUE_ON_POLICY_CONFIG+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn_npu_batch_invariant=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.batch_invariant_mode=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_batch_invariant_ops=True
        +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN
    )
fi
TRUE_ON_POLICY_CONFIG+=(
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
)

########################### launch ###########################
mkdir -p ./logs
script_path="${BASH_SOURCE[0]:-$0}"
logfile="./logs/test_qwen3_30b_grpo_megatron_$(date +%Y%m%d_%H%M%S).log"
{
    echo "===== Script Content: $(realpath "$script_path") ====="
    cat "$script_path"
    echo "===== End Script Content ====="
    echo ""

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        "${DATA[@]}" \
        "${MODEL[@]}" \
        "${ACTOR[@]}" \
        "${ROLLOUT[@]}" \
        "${REF[@]}" \
        "${TRAINER[@]}" \
        "${EXTRA[@]}" \
        "${TRUE_ON_POLICY_CONFIG[@]}" \
        "$@"

} 2>&1 | tee "$logfile"