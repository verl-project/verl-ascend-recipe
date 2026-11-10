#!/usr/bin/env bash
# SFT | DeepSeek-Coder-6.7B-Instruct | FSDP | Ascend NPU

set -xeuo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local DeepSeek-Coder checkout}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to data directory}
VERL_ROOT=${VERL_ROOT:?Set VERL_ROOT to the current upstream verl checkout}

GSM8K_DIR=${GSM8K_DIR:-${DATA_ROOT}/gsm8k-sft}
TRAIN_FILE=${TRAIN_FILE:-${GSM8K_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${GSM8K_DIR}/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-${DATA_ROOT}/checkpoints/deepseek-coder-6.7b-sft}
LOG_FILE=${LOG_FILE:-${DATA_ROOT}/logs/deepseek-coder-6.7b-sft/train.log}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-1024}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
LR=${LR:-1e-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SP_SIZE=${SP_SIZE:-2}

if [[ ! -f "${VERL_ROOT}/verl/trainer/sft_trainer.py" ]]; then
    echo "VERL_ROOT does not contain the upstream verl SFT trainer: ${VERL_ROOT}" >&2
    exit 1
fi

export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

mkdir -p "${GSM8K_DIR}" "${CKPTS_DIR}" "$(dirname "${LOG_FILE}")"
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    "${PYTHON_BIN}" "${VERL_ROOT}/examples/data_preprocess/gsm8k_multiturn_sft.py" \
        --local_dir "${GSM8K_DIR}"
fi

cd "${VERL_ROOT}"
set +e
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NDEVICES_PER_NODE}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.use_dynamic_bsz=True \
    data.ignore_input_ids_mismatch=True \
    optim.lr="${LR}" \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.project_name=deepseek-coder-6.7b-sft \
    trainer.experiment_name=deepseek-coder-6.7b-sft-fsdp-npu \
    trainer.logger=['console'] \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}" \
    trainer.nnodes=1 "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
