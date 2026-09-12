#!/usr/bin/env bash
# Multi-teacher on-policy distillation | Qwen3-0.6B | GSM8K + MATH | FSDP | vLLM-Ascend

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-0.6B}
GSM8K_TEACHER_MODEL=${GSM8K_TEACHER_MODEL:-Qwen/Qwen3-4B}
MATH_TEACHER_MODEL=${MATH_TEACHER_MODEL:-Qwen/Qwen3-8B}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}  # Student pool; teachers need additional NPUs.

# Per-teacher replicas; total teacher NPUs = sum(num_replicas) * teacher_tp
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NUM_REPLICAS_GSM8K=${TEACHER_NUM_REPLICAS_GSM8K:-1}
TEACHER_NUM_REPLICAS_MATH=${TEACHER_NUM_REPLICAS_MATH:-1}
teacher_tp=${TEACHER_TP:-1}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_training_steps=${TOTAL_TRAINING_STEPS:-100}
total_epochs=${TOTAL_EPOCHS:-$total_training_steps}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-20}

project_name=${PROJECT_NAME:-verl_distill_mopd_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_06b_from_qwen3_4b_and_qwen3_8b_mopd_npu}
# ---- end user-adjustable ----

require_positive_integer() {
    local name=$1
    local value=$2
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer; got $value." >&2
        exit 2
    fi
}

require_positive_integer NNODES "$NNODES"
require_positive_integer NGPUS_PER_NODE "$NGPUS_PER_NODE"
require_positive_integer TEACHER_NNODES "$TEACHER_NNODES"
require_positive_integer TEACHER_NUM_REPLICAS_GSM8K "$TEACHER_NUM_REPLICAS_GSM8K"
require_positive_integer TEACHER_NUM_REPLICAS_MATH "$TEACHER_NUM_REPLICAS_MATH"
require_positive_integer TEACHER_TP "$teacher_tp"
require_positive_integer ROLLOUT_TP "$rollout_tp"

student_world_size=$((NNODES * NGPUS_PER_NODE))
if ((student_world_size % rollout_tp != 0)); then
    echo "ROLLOUT_TP must divide NNODES * NGPUS_PER_NODE." >&2
    exit 2
fi
TEACHER_WORLD_SIZE=$(((TEACHER_NUM_REPLICAS_GSM8K + TEACHER_NUM_REPLICAS_MATH) * teacher_tp))

DATA_DIR=${DATA_DIR:-$HOME/data_mopd_nonthinking}
gsm8k_train=$DATA_DIR/gsm8k/train.parquet
gsm8k_test=$DATA_DIR/gsm8k/test.parquet
math_train=$DATA_DIR/math/train.parquet
math_test=$DATA_DIR/math/test.parquet
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$HOME/ckpts/$project_name/$experiment_name}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-5400}
export TOKENIZERS_PARALLELISM=false
export USE_OPTIMIZED_MODEL=0
export VLLM_USE_V1=1
# Device IDs must come from the container's actual device mapping, including teacher resources.
TOTAL_NPUS=$((student_world_size + TEACHER_NNODES * TEACHER_WORLD_SIZE))
echo "Student NPUs=$student_world_size; teacher NPUs=$((TEACHER_NNODES * TEACHER_WORLD_SIZE)); total=$TOTAL_NPUS" >&2

train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    +data.apply_chat_template_kwargs.enable_thinking=False
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=False
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR"
    trainer.resume_mode=${RESUME_MODE:-auto}
    trainer.device=npu
    critic.enable=False
)

# Multi-teacher: one teacher per dataset, routed by the sample's `data_source` value.
# Use `+distillation.teacher_models.<name>.*` to add named teachers; the default `teacher_model`
# entry is silently popped when other teacher entries are added.
EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    # --- gsm8k teacher (text) ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$GSM8K_TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=${TEACHER_NUM_REPLICAS_GSM8K}
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.gsm8k.inference.max_model_len=${max_num_tokens}
    +distillation.teacher_models.gsm8k.inference.enforce_eager=True
    +distillation.teacher_models.gsm8k.inference.load_format=safetensors
    # --- math teacher (text) ---
    +distillation.teacher_models.math.key="DigitalLearningGmbH/MATH-lighteval"
    +distillation.teacher_models.math.model_path="$MATH_TEACHER_MODEL"
    +distillation.teacher_models.math.num_replicas=${TEACHER_NUM_REPLICAS_MATH}
    +distillation.teacher_models.math.inference.name=vllm
    +distillation.teacher_models.math.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.math.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.math.inference.max_model_len=${max_num_tokens}
    +distillation.teacher_models.math.inference.enforce_eager=True
    +distillation.teacher_models.math.inference.load_format=safetensors
    # --- loss ---
    distillation.distillation_loss.loss_mode=k1
    # k1 does not consume top-k logits; keep the measured resolved-config value explicit.
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
    EXTRA+=(++ray_kwargs.ray_init._temp_dir="$RAY_TEMP_DIR")
fi

########################### launch ###########################
command=(python3 -m verl.trainer.main_ppo
    "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${ROLLOUT[@]}"
    "${TRAINER[@]}" "${EXTRA[@]}" "$@")
# Hydra inspection does not train and must keep stdout suitable for config export.
for argument in "$@"; do
    case "$argument" in
        --cfg|--cfg=*|--help|-h|--info|--info=*) exec "${command[@]}" ;;
    esac
done

expected_steps=$total_training_steps
log_dir=$DEFAULT_LOCAL_DIR
console_logger=console
for override in "$@"; do
    case "$override" in
        trainer.total_training_steps=*) expected_steps=${override#*=} ;;
        trainer.default_local_dir=*) log_dir=${override#*=} ;;
        trainer.logger=*) console_logger=${override#*=} ;;
    esac
done
if ! [[ "$expected_steps" =~ ^[1-9][0-9]*$ ]] || [[ "$console_logger" != *console* ]]; then
    echo "A positive trainer.total_training_steps and console logger are required for completion verification." >&2
    exit 2
fi
mkdir -p "$log_dir"
training_log=$(mktemp "$log_dir/training.XXXXXXXX.log")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "Training log: $training_log"
"${command[@]}" 2>&1 | tee "$training_log"
python3 "$script_dir/tools/check_training_completion.py" "$training_log" --expected-step "$expected_steps"
