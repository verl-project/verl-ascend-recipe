#!/usr/bin/env bash
# GRPO | Qwen3.5-2B | FSDP2 training | Open-R1 multimodal math dataset
#
# This recipe follows the upstream
# examples/grpo_trainer/run_qwen3_5_2b_openr1_fsdp.sh example.  The NPU
# additions are limited to the official vLLM-Ascend graph and distributed
# settings used by the upstream Qwen3.5-2B NPU examples.

set -xeuo pipefail

########################### user-adjustable ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
INFER_BACKEND=${INFER_BACKEND:-vllm}
DEVICE=${DEVICE:-npu}
PROJECT_NAME=${PROJECT_NAME:-GRPO-OpenR1MM}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3.5-2B-GRPO-OpenR1MM-FSDP2-NPU}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}

GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-${NDEVICES_PER_NODE}}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_N=${ROLLOUT_N:-5}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

# Set VERL_ROOT to a clean checkout of the current upstream verl main branch.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=${VERL_ROOT:-${PWD}}
if [[ ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" && -f "${SCRIPT_DIR}/../../verl/verl/trainer/main_ppo.py" ]]; then
    VERL_ROOT=$(cd -- "${SCRIPT_DIR}/../../verl" && pwd)
fi

RAY_DATA_HOME=${RAY_DATA_HOME:-/data/verl}
RAY_TMPDIR=${RAY_TMPDIR:-/data/tmp/ray}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-2B"}
DATA_DIR=${DATA_DIR:-"${RAY_DATA_HOME}/data/openr1mm"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${RAY_DATA_HOME}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
RUNTIME_DIR=${RUNTIME_DIR:-"${RAY_DATA_HOME}/runtime/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
########################### end user-adjustable ###########################

if [[ ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]]; then
    echo "VERL_ROOT does not contain the latest verl checkout: ${VERL_ROOT}" >&2
    exit 1
fi

export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_DATA_HOME
export TMPDIR=${TMPDIR:-${RAY_DATA_HOME}/tmp}
mkdir -p "${TMPDIR}"
export RAY_TMPDIR
export TOKENIZERS_PARALLELISM=false

########################### device and official NPU settings ###########################
case "${DEVICE}" in
    npu)
        # These values match the current upstream Ascend Qwen3.5-2B examples.
        export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
        export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
        export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
        export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
unset RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
        n_devices_per_node=${NDEVICES_PER_NODE}
        fsdp_size=${FSDP_SIZE}
        export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
        ;;
    gpu)
        n_devices_per_node=${NDEVICES_PER_NODE:-4}
        fsdp_size=${FSDP_SIZE}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'npu' or 'gpu'." >&2
        exit 1
        ;;
esac

mkdir -p "${DATA_DIR}" "${CKPTS_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}" "${RUNTIME_DIR}"

########################### data and reward preparation ###########################
# The upstream preprocessing program is used unchanged.  It is invoked only
# when the expected parquet files are absent, so reruns reuse the same data.
if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" ]]; then
    "${PYTHON_BIN}" "${VERL_ROOT}/examples/data_preprocess/openr1mm.py" \
        --local_save_dir "${DATA_DIR}"
fi

# The upstream training example refers to openr1mm.py, but that reward module
# is not present in the current verl main checkout.  Keep the compatibility
# adapter self-contained in the recipe runtime and leave the installed
# dependencies and the upstream checkout untouched.
REWARD_PATH="${RUNTIME_DIR}/openr1mm_reward.py"
cat > "${REWARD_PATH}" <<'PY'
import re


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _extract_answer(text):
    text = "" if text is None else str(text)
    matches = _ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\s*\{([^{}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    return text.strip()


def _normalise(text):
    text = _extract_answer(text).strip().strip("`$")
    text = re.sub(r"^final\s*answer\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[.!。！？?]+$", "", text).strip()
    return re.sub(r"\s+", "", text).casefold()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score the answer enclosed by the Open-R1 <answer> tags.

    Open-R1 stores the complete reference solution in ``ground_truth``.  The
    answer tag is the stable target for both multiple-choice and numeric
    geometry questions.  The upstream math reward is used as a second pass for
    equivalent LaTeX forms when it is available.
    """
    prediction = _extract_answer(solution_str)
    target = _extract_answer(ground_truth)
    if _normalise(prediction) == _normalise(target):
        return 1.0

    try:
        from verl.utils.reward_score import math_reward

        candidate = "\\boxed{" + prediction + "}"
        if math_reward.compute_score(candidate, target) > 0:
            return 1.0
    except Exception:
        pass
    return 0.0
PY

start_time=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/qwen3_5-2b-openr1mm-${start_time}.log"}

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
    data.truncation=error
    data.image_patch_size=16
    data.image_key=images
    data.shuffle=False
)

REWARD=(
    reward.custom_reward_function.path="${REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=5e-7
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.02
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,16,32,64,128,256]"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=disable
    trainer.resume_from_path=null
    trainer.val_before_train=False
    trainer.max_actor_ckpt_to_keep=1
    trainer.save_freq=${TOTAL_TRAINING_STEPS}
    trainer.test_freq=10
    trainer.total_epochs=15
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

########################### launch ###########################
cd "${VERL_ROOT}"
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
set +e
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${REWARD[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
