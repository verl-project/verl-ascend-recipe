#!/usr/bin/env bash
# Run from the pinned verl checkout after installing both recipe overlays.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a model checkpoint directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a prepared training parquet file}"
: "${TEST_FILE:?Set TEST_FILE to a prepared validation parquet file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a run-specific checkpoint directory}"

python_bin="${PYTHON:-python3}"
backend="${BACKEND:-censored_lognormal}"
npu_count="${NPU_COUNT:-2}"
tp_size="${TP_SIZE:-1}"
batch_size="${BATCH_SIZE:-8}"
response_length="${RESPONSE_LENGTH:-1024}"
prompt_length="${PROMPT_LENGTH:-512}"

if [[ ! -f recipe/dapo/config/dapo_trainer.yaml ]]; then
    printf '%s\n' 'Run this script from the pinned verl source checkout.' >&2
    exit 2
fi
if [[ "$backend" != linear_listmle && "$backend" != censored_lognormal ]]; then
    printf '%s\n' 'BACKEND must be linear_listmle or censored_lognormal.' >&2
    exit 2
fi
if ! [[ "$npu_count" =~ ^[1-9][0-9]*$ && "$tp_size" =~ ^[1-9][0-9]*$ ]] ||
    (( npu_count % tp_size != 0 )); then
    printf '%s\n' 'NPU_COUNT and TP_SIZE must be positive; TP_SIZE must divide NPU_COUNT.' >&2
    exit 2
fi

export WANDB_MODE=disabled
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

command=(
    "$python_bin" -m recipe.dapo_predictor.main_dapo_predictor_reorder
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=false
    algorithm.filter_groups.enable=false
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$TEST_FILE"
    "data.train_batch_size=$batch_size"
    "data.gen_batch_size=$batch_size"
    "data.max_prompt_length=$prompt_length"
    "data.max_response_length=$response_length"
    data.filter_overlong_prompts=true
    data.truncation=error
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=false
    ++actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    "actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=false
    actor_rollout_ref.actor.use_torch_compile=false
    actor_rollout_ref.actor.use_kl_loss=false
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
    actor_rollout_ref.ref.use_torch_compile=false
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N:-2}"
    "actor_rollout_ref.rollout.max_num_seqs=${SLOTS_PER_SERVER:-8}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    "reward.reward_kwargs.max_resp_len=$response_length"
    "trainer.n_gpus_per_node=$npu_count"
    trainer.nnodes=1
    trainer.device=npu
    trainer.use_legacy_worker_impl=enable
    'trainer.logger=["console"]'
    trainer.project_name=dapo-ews
    trainer.experiment_name=ews
    trainer.val_before_train=false
    trainer.test_freq=-1
    "trainer.save_freq=${SAVE_FREQ:-10}"
    "trainer.total_epochs=${TOTAL_EPOCHS:-20}"
    "trainer.default_local_dir=$OUTPUT_DIR"
    trainer.resume_mode=disable
    "+trainer.predictor_reorder.enable=${PREDICTOR_ENABLE:-true}"
    +trainer.predictor_reorder.scheduler=ews
    "+trainer.predictor_reorder.backend=$backend"
    "+trainer.predictor_reorder.activation.min_epoch=${MIN_EPOCH:-10}"
    "+trainer.predictor_reorder.activation.min_samples=${MIN_SAMPLES:-2560}"
    +trainer.predictor_reorder.epochs=10
    +trainer.predictor_reorder.batch_size=32
    +trainer.predictor_reorder.lr=3e-5
    +trainer.predictor_reorder.weight_decay=1e-4
    +trainer.predictor_reorder.seed=1
    +trainer.predictor_reorder.calibration_max_samples=4096
    "+trainer.predictor_reorder.ews.slots_per_server=${SLOTS_PER_SERVER:-8}"
    +trainer.predictor_reorder.ews.max_concurrent_requests=null
)
if [[ "$backend" == censored_lognormal ]]; then
    if [[ -n "${D2_CANDIDATE_REPORT:-}" ]]; then
        command+=(+trainer.predictor_reorder.d2_online.enable=false
            "+trainer.predictor_reorder.d2_lifecycle.candidate_report=$D2_CANDIDATE_REPORT")
    else
        command+=(+trainer.predictor_reorder.d2_online.enable=true)
    fi
fi
if [[ "${1:-}" == --dry-run ]]; then
    shift
    printf '%q ' "${command[@]}" "$@"
    printf '\n'
    exit 0
fi
exec "${command[@]}" "$@"
