#!/usr/bin/env bash
# Qwen3-8B / GSM8K GRPO, Megatron actor and native Ascend prefix attention.
set -euo pipefail

model_path=${MODEL_PATH:-/models/Qwen3-8B}
data_root=${DATA_ROOT:-/datasets/gsm8k}
npus=${NPUS_PER_NODE:-8}
rollout_n=${ROLLOUT_N:-2}
positive_integer() { [[ $1 =~ ^[1-9][0-9]*$ ]]; }
if ! positive_integer "$npus" || ! positive_integer "$rollout_n" || (( rollout_n < 2 )); then
    echo "NPUS_PER_NODE must be positive and ROLLOUT_N must be at least 2." >&2
    exit 2
fi
default_tp=2
if (( npus == 1 )); then
    default_tp=1
fi
actor_tp=${ACTOR_TP:-$default_tp}
rollout_tp=${ROLLOUT_TP:-$default_tp}
if [[ $actor_tp != 1 && $actor_tp != 2 ]] || (( npus % actor_tp )); then
    echo "ACTOR_TP must be 1 or 2 and divide NPUS_PER_NODE." >&2
    exit 2
fi
if ! positive_integer "$rollout_tp" || (( npus % rollout_tp )); then
    echo "ROLLOUT_TP must be a positive divisor of NPUS_PER_NODE." >&2
    exit 2
fi
data_parallel_size=$((npus / actor_tp))
rollout_replicas=$((npus / rollout_tp))
train_batch=${TRAIN_BATCH_SIZE:-$((8 * npus))}
mini_batch=${PPO_MINI_BATCH_SIZE:-$train_batch}
prompt_len=${MAX_PROMPT_LENGTH:-512}
response_len=${MAX_RESPONSE_LENGTH:-384}
steps=${TOTAL_TRAINING_STEPS:-100}
for count in "$train_batch" "$mini_batch" "$prompt_len" "$response_len" "$steps"; do
    positive_integer "$count" || { echo "Batch sizes, sequence lengths and steps must be positive." >&2; exit 2; }
done
if (( train_batch % mini_batch || mini_batch % data_parallel_size )); then
    echo "TRAIN_BATCH_SIZE must be a multiple of PPO_MINI_BATCH_SIZE, itself a multiple of NPUS_PER_NODE / ACTOR_TP." >&2
    exit 2
fi
for split in train test; do
    [[ -r "$data_root/$split.parquet" ]] || { echo "Cannot read $data_root/$split.parquet" >&2; exit 2; }
done

if [[ -n ${RUN_DIR:-} ]]; then
    mkdir -p -- "$RUN_DIR"
    run_dir=$(cd -- "$RUN_DIR" && pwd -P)
else
    run_dir=$(mktemp -d "$PWD/prefix_grouper_run.XXXXXX")
fi
export VERL_FILE_LOGGER_PATH="$run_dir/metrics.jsonl"
if [[ -e "$VERL_FILE_LOGGER_PATH" ]] || ! ( set -o noclobber; : > "$run_dir/console.log" ); then
    echo "Run directory already has logs: $run_dir" >&2
    exit 2
fi
echo "Run directory: $run_dir"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
offload=${OFFLOAD:-True}
sequence_len=$((prompt_len + response_len))
default_batched_tokens=4096
if (( sequence_len > default_batched_tokens )); then
    default_batched_tokens=$sequence_len
fi
sequences_per_replica=$(((train_batch * rollout_n + rollout_replicas - 1) / rollout_replicas))

args=(
    model_engine=megatron
    trainer.use_v1=False
    trainer.device=npu
    trainer.nnodes=1
    "trainer.n_gpus_per_node=$npus"
    'trainer.logger=["console","file"]'
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=$VERL_FILE_LOGGER_PATH"
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=$VERL_LOGGING_LEVEL"
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NPU_ALLOC_CONF='$PYTORCH_NPU_ALLOC_CONF'"
    trainer.project_name=verl_prefix_grouper_gsm8k
    trainer.experiment_name=qwen3_8b_megatron_ascend
    "trainer.default_local_dir=$run_dir/checkpoints"
    trainer.resume_mode=disable
    "trainer.save_freq=${SAVE_FREQ:--1}"
    "trainer.test_freq=${TEST_FREQ:-25}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}"
    "trainer.total_training_steps=$steps"
    "trainer.total_epochs=${TOTAL_EPOCHS:-100}"
    trainer.balance_batch=True
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=$data_root/train.parquet"
    "data.val_files=$data_root/test.parquet"
    "data.train_batch_size=$train_batch"
    "data.val_max_samples=${VAL_MAX_SAMPLES:-128}"
    "data.max_prompt_length=$prompt_len"
    "data.max_response_length=$response_len"
    data.dataloader_num_workers=0
    data.filter_overlong_prompts=True
    data.truncation=error
    data.seed=42
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-False}"
    "actor_rollout_ref.model.path=$model_path"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.actor.use_prefix_grouper=True
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_torch_compile=False
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$rollout_n"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$actor_tp"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.use_mbridge=True
    "actor_rollout_ref.actor.megatron.param_offload=$offload"
    "actor_rollout_ref.actor.megatron.optimizer_offload=$offload"
    +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$actor_tp"
    actor_rollout_ref.ref.megatron.sequence_parallel=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$rollout_n"
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp"
    "actor_rollout_ref.rollout.n=$rollout_n"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_n"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEMORY_UTILIZATION:-0.5}"
    "actor_rollout_ref.rollout.max_model_len=$sequence_len"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$default_batched_tokens}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-$sequences_per_replica}"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
)
if [[ ${CPU_OPTIMIZER_OFFLOAD:-True} == True ]]; then
    args+=(
        actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
        actor_rollout_ref.actor.optim.main_grads_dtype=bf16
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    )
fi

# The patch must be importable by the same interpreter that launches verl.
run_training() {
    python3 - "$npus" <<'PY' || return $?
import sys
from importlib.metadata import version

import torch
import torch_npu
import vllm_ascend
from verl.models.mcore.prefix_grouper_forward import grouped_forward
from verl.workers.config import McoreEngineConfig

if "use_prefix_grouper" not in McoreEngineConfig.__dataclass_fields__:
    sys.exit("Apply patches/verl.patch before starting the trainer.")
if torch.npu.device_count() < int(sys.argv[1]):
    sys.exit(f"Requested {sys.argv[1]} NPUs; visible devices: {torch.npu.device_count()}")
print("Ascend device:", torch.npu.get_device_name(0), flush=True)
for package in ("torch", "torch-npu", "vllm", "vllm-ascend"):
    print(f"{package}={version(package)}", flush=True)
PY
    python3 -m verl.trainer.main_ppo "${args[@]}" "$@" || return $?
    if [[ ! -s "$VERL_FILE_LOGGER_PATH" ]]; then
        echo "Missing verl metrics file." >&2
        return 1
    fi
}

run_training "$@" 2>&1 | tee -a "$run_dir/console.log" || {
    pipeline_status=("${PIPESTATUS[@]}")
    # If both processes failed, preserve the training or preflight exit code.
    if (( pipeline_status[0] != 0 )); then
        exit "${pipeline_status[0]}"
    fi
    exit "${pipeline_status[1]}"
}
finished="prefix_grouper_run_finished steps_requested=$steps"
printf '%s\n' "$finished" >> "$run_dir/console.log" || exit $?
printf '%s\n' "$finished"
