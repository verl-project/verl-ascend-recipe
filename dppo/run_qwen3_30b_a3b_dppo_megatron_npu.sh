#!/usr/bin/env bash
# DPPO | Qwen3-30B-A3B (MoE) | Megatron actor/reference | vLLM-Ascend rollout

set -xeuo pipefail

export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export VLLM_MQ_MAX_CHUNK_BYTES_MB=${VLLM_MQ_MAX_CHUNK_BYTES_MB:-1}

export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export DISABLE_L2_CACHE=${DISABLE_L2_CACHE:-1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}

if [[ -n "${MEGATRON_LM_PATH:-}" ]]; then
    export PYTHONPATH="${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi

python3 -c 'import megatron.training' || {
    echo "Megatron-LM training modules are required. Set MEGATRON_LM_PATH to the Megatron-LM source root." >&2
    exit 1
}

########################### user-adjustable ###########################

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}

LOSS_MODE=${LOSS_MODE:-dppo_tv}
case ${LOSS_MODE} in
    dppo_tv) CLIP_DEFAULT=0.15 ;;
    dppo_kl) CLIP_DEFAULT=0.05 ;;
    *) echo "LOSS_MODE must be dppo_tv or dppo_kl, got: ${LOSS_MODE}" >&2; exit 1 ;;
esac
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-${CLIP_DEFAULT}}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-${CLIP_DEFAULT}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 4))}

ACTOR_LR=${ACTOR_LR:-1e-5}
OPTIMIZER_DTYPE=${OPTIMIZER_DTYPE:-fp32} # fp32 | bf16; fp32 is recommended.
OPTIMIZER_CPU_OFFLOAD_FRACTION=${OPTIMIZER_CPU_OFFLOAD_FRACTION:-0.40}
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-4}
ACTOR_EP=${ACTOR_EP:-$((NDEVICES_PER_NODE / ACTOR_PP))}
ACTOR_ETP=${ACTOR_ETP:-1}

ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}
ROLLOUT_NUM_GPU_BLOCKS_OVERRIDE=${ROLLOUT_NUM_GPU_BLOCKS_OVERRIDE:-256}
UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-640}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16,32]'}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
RESUME_MODE=${RESUME_MODE:-disable}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PWD/checkpoints/qwen3_30b_a3b_dppo}

PROJECT_NAME=${PROJECT_NAME:-verl_dppo_qwen3_moe}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_dppo_megatron_vllm_ascend}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.dataloader_num_workers=0
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE}
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10000.0
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
    actor_rollout_ref.actor.optim.main_grads_dtype=${OPTIMIZER_DTYPE}
    actor_rollout_ref.actor.optim.exp_avg_dtype=${OPTIMIZER_DTYPE}
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=${OPTIMIZER_DTYPE}
    +actor_rollout_ref.actor.optim.override_optimizer_config.main_params_dtype=${OPTIMIZER_DTYPE}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_CPU_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=${ROLLOUT_NUM_GPU_BLOCKS_OVERRIDE}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.skip_tokenizer_init=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=${CUDAGRAPH_CAPTURE_SIZES}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.balance_batch=False
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="$CHECKPOINT_DIR"
    trainer.resume_mode=${RESUME_MODE}
)

MODEL_ENGINE=(
    model_engine=megatron
)

########################### launch ###########################

LOG_DIR=${LOG_DIR:-$PWD/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${LOG_FILE}")"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${MODEL_ENGINE[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
