#!/usr/bin/env bash
# GRPO + LoRA with merge | Qwen3-8B | gsm8k | vLLM-Ascend rollout | FSDP2 training | Ascend A2 (910B) NPUs
#
# Ported from verl `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` (GPU, 8 cards). Algorithm and LoRA
# hyper-parameters are kept identical; only the NPU launch items differ (trainer.device=npu, Ascend env vars,
# eager vLLM, no uv wrapper). With `model.lora.merge=True` the adapters are merged into the base weights before
# every rollout weight sync, so vLLM-Ascend receives plain full weights and no inference-side LoRA support is needed.
#
# Validated on 4 x 910B1 (64 GB HBM), see README.md: 10-step smoke ~277 s/step, ~825 tokens/s (4-card global),
# reward 0.23 -> 0.60, actor peak HBM 32.6 GB. All knobs are env-overridable.
set -xeuo pipefail

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}      # 4 x 910B1 validated; rollout_tp must divide it

# Ascend: container-logical NPU ids are 0..N-1. PHYS_CARDS (e.g. "4,5,6,7") only feeds the fallback path of the
# unpatched verl get_npu_versions() when the container mounts a subset of cards (see patches/ and README.md).
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE-1)))}
export ASCEND_VISIBLE_DEVICES=${PHYS_CARDS:-$ASCEND_RT_VISIBLE_DEVICES}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-5400}
export TOKENIZERS_PARALLELISM=false
export USE_OPTIMIZED_MODEL=0             # inference-only fast paths interfere with RL log-prob / online weight update
export VLLM_USE_V1=1

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1.0e-05}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-64}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-8}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}

total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
save_freq=${SAVE_FREQ:-10}               # checkpoint every 10 steps so a shared host can pause/resume
test_freq=${TEST_FREQ:-20}
val_before_train=${VAL_BEFORE_TRAIN:-True}
resume_mode=${RESUME_MODE:-auto}

project_name=${PROJECT_NAME:-verl_ascend_lora_merge}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_lora_merge_vllm_fsdp2_npu}
default_local_dir=${DEFAULT_LOCAL_DIR:-$HOME/ckpts/${project_name}/${experiment_name}}
logger=${LOGGER:-'["console"]'}
ray_temp_dir=${RAY_TEMP_DIR:-}           # set when /tmp is small or full on the host
# ---- end user-adjustable ----
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${DATA_DIR}/train.parquet"
    data.val_files="${DATA_DIR}/test.parquet"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    ++actor_rollout_ref.model.lora.merge=True
    actor_rollout_ref.model.lora_rank=${lora_rank}
    actor_rollout_ref.model.lora_alpha=${lora_alpha}
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_torch_compile=False
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
)

# With lora_rank > 0 verl computes the reference log-prob from the actor with adapters disabled (ref_in_actor);
# these settings are kept for parity with the GPU reference script.
REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.val_before_train=${val_before_train}
    trainer.logger="${logger}"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NPROC_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.device=npu
    trainer.default_local_dir="${default_local_dir}"
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    trainer.resume_mode=${resume_mode}
    critic.enable=False
)

EXTRA=()
if [ -n "${ray_temp_dir}" ]; then
    EXTRA+=(++ray_kwargs.ray_init._temp_dir="${ray_temp_dir}")
fi

########################### launch ###########################
# Run from the verl repo root (the Ascend image ships verl at /verl).
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
