#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROOT
cd "$ROOT"
if [[ -n "${SANDBOX_ENV_FILE:-}" ]]; then
    [[ -r "$SANDBOX_ENV_FILE" ]] || { echo 'Cannot read SANDBOX_ENV_FILE' >&2; exit 2; }
    set -a; source "$SANDBOX_ENV_FILE"; set +a
fi
export UNI_AGENT_ROOT="${UNI_AGENT_ROOT:-$ROOT/.runtime/uni-agent}"
: "${VERL_SOURCE:?Set VERL_SOURCE to the compatible Ascend verl checkout}"
export PYTHONPATH="$UNI_AGENT_ROOT:$ROOT:$VERL_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_PROFILE="$ROOT/configs/models/qwen3_5_9b.sh"
export EXPERIMENT_CONFIG="$ROOT/examples/e2b_swe_grpo/experiment.sh"
export TRAINING_ENV_CONFIG="$ROOT/scripts/training/ascend.sh"
export GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
export EXP_NAME="${EXP_NAME:-kasandbox_qwen35_9b}"
export VLLM_USE_V2_MODEL_RUNNER=1 HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto WANDB_MODE=disabled
[[ -d "$UNI_AGENT_ROOT/uni_agent" ]] || { echo 'Run setup.sh first' >&2; exit 2; }
if [[ "${1:-}" == --dry-run ]]; then
    shift
    export RUNTIME_ENV="${RUNTIME_ENV:-$ROOT/.runtime/created-at-submit.json}"
    exec bash "$ROOT/examples/e2b_swe_grpo/launch.sh" --dry-run "$@"
fi
: "${VC_WORKER_HOSTS:?Comma-separated head and worker hosts}" "${VC_TASK_INDEX:?Node index: 0 or 1}"
IFS=, read -ra hosts <<< "$VC_WORKER_HOSTS"
[[ ${#hosts[@]} == 2 && "$VC_TASK_INDEX" =~ ^[01]$ && -n ${hosts[0]} && -n ${hosts[1]} ]] || { echo 'Expected two workers and index 0 or 1' >&2; exit 2; }
set -a
source "$ROOT/examples/e2b_swe_grpo/zmj.sh"
source "$ROOT/configs/sandbox/kasandbox.sh"
set +a
: "${E2B_API_KEY:?Inject E2B_API_KEY}"
python3 "$ROOT/preflight.py"
export NO_PROXY="127.0.0.1,localhost,$VC_WORKER_HOSTS" no_proxy="127.0.0.1,localhost,$VC_WORKER_HOSTS"
mkdir -p "$ROOT/logs_reproduce"
bash "$ROOT/launch_uniagent_kasandbox.sh" "$@" 2>&1 | tee "$ROOT/logs_reproduce/${EXP_NAME}.node${VC_TASK_INDEX}.log"
