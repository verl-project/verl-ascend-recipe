#!/usr/bin/env bash
# GPG | Qwen3-4B | vLLM-Ascend rollout | FSDP actor | Ascend NPU

set -xeuo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
DEVICE=${DEVICE:-npu}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-4B checkout}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to an attached-storage directory}
VERL_ROOT=${VERL_ROOT:?Set VERL_ROOT to the current upstream verl checkout}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

GSM8K_DIR=${GSM8K_DIR:-${DATA_ROOT}/gsm8k}
MATH_DIR=${MATH_DIR:-${DATA_ROOT}/math}
TRAIN_FILE_GSM8K=${TRAIN_FILE_GSM8K:-${GSM8K_DIR}/train.parquet}
VAL_FILE_GSM8K=${VAL_FILE_GSM8K:-${GSM8K_DIR}/test.parquet}
TRAIN_FILE_MATH=${TRAIN_FILE_MATH:-${MATH_DIR}/train.parquet}
VAL_FILE_MATH=${VAL_FILE_MATH:-${MATH_DIR}/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-${DATA_ROOT}/checkpoints/qwen3-4b-gpg}
LOG_FILE=${LOG_FILE:-${DATA_ROOT}/logs/qwen3-4b-gpg/train.log}
RAY_TMPDIR=${RAY_TMPDIR:-${DATA_ROOT}/ray-tmp}

if [[ ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]]; then
    echo "VERL_ROOT does not contain the upstream verl trainer: ${VERL_ROOT}" >&2
    exit 1
fi
if [[ "${DEVICE}" != npu ]]; then
    echo "This recipe is intended for DEVICE=npu" >&2
    exit 1
fi

export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_DATA_HOME=${DATA_ROOT}
export TMPDIR=${TMPDIR:-${DATA_ROOT}/tmp}
mkdir -p "${TMPDIR}"
export RAY_TMPDIR
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
unset RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "${GSM8K_DIR}" "${MATH_DIR}" "${CKPTS_DIR}" "$(dirname "${LOG_FILE}")" "${RAY_TMPDIR}"
if [[ ! -f "${TRAIN_FILE_GSM8K}" || ! -f "${VAL_FILE_GSM8K}" ]]; then
    "${PYTHON_BIN}" "${VERL_ROOT}/examples/data_preprocess/gsm8k.py" \
        --local_save_dir "${GSM8K_DIR}"
fi
if [[ ! -f "${TRAIN_FILE_MATH}" || ! -f "${VAL_FILE_MATH}" ]]; then
    "${PYTHON_BIN}" "${VERL_ROOT}/examples/data_preprocess/math_dataset.py" \
        --local_save_dir "${MATH_DIR}"
fi

DATA=(
    algorithm.adv_estimator=gpg
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE_GSM8K}','${TRAIN_FILE_MATH}']"
    data.val_files="['${VAL_FILE_GSM8K}','${VAL_FILE_MATH}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=gpg
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NDEVICES_PER_NODE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,16,32,64,128,256]"
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name=qwen3-4b-gpg
    trainer.experiment_name=qwen3-4b-gpg-fsdp-npu
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=disable
    trainer.resume_from_path=null
    trainer.val_before_train=False
    trainer.max_actor_ckpt_to_keep=1
    trainer.save_freq=${TOTAL_TRAINING_STEPS}
    trainer.test_freq=10
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

cd "${VERL_ROOT}"
set +e
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" \
    "${ROLLOUT[@]}" "${TRAINER[@]}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
