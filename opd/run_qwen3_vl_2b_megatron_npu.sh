#!/usr/bin/env bash
# Qwen3-VL-2B <- Qwen3-VL-4B OPD | Megatron training | vLLM Ascend inference

set -xeuo pipefail

PYTHON=${PYTHON:-python3}
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/geo3k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/geo3k/test.parquet}

NNODES=${NNODES:-1}
STUDENT_WORLD_SIZE=${STUDENT_WORLD_SIZE:-2}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}
ACTOR_TP=${ACTOR_TP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHER_TP=${TEACHER_TP:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}
ACTOR_LR=${ACTOR_LR:-1e-6}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
PROJECT_NAME=${PROJECT_NAME:-verl_opd_qwen3_vl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_2b_from_4b_npu_megatron_vllm}

ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.20}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-true}

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

max_num_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

"$PYTHON" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.image_key=images \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$ACTOR_TP" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="$OPTIMIZER_OFFLOAD_FRACTION" \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer="$USE_PRECISION_AWARE_OPTIMIZER" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.seed="$ROLLOUT_SEED" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len="$max_num_tokens" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_num_seqs="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_num_tokens" \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$STUDENT_WORLD_SIZE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    model_engine=megatron \
    distillation.enabled=True \
    distillation.n_gpus_per_node="$TEACHER_WORLD_SIZE" \
    distillation.nnodes="$NNODES" \
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEM_UTIL" \
    distillation.teacher_models.teacher_model.inference.enforce_eager=False \
    distillation.teacher_models.teacher_model.inference.max_model_len="$max_num_tokens" \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=1 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="$max_num_tokens" \
    distillation.distillation_loss.loss_mode=k1 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    "$@"
