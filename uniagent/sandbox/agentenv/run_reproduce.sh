#!/usr/bin/env bash
# Run on each of the two workers. Dry-run never starts SSH, Ray or training.
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
export REPO_ROOT="$UNI_AGENT_ROOT"
: "${VERL_SOURCE:?Set VERL_SOURCE to the compatible Ascend verl checkout}"
source "$ROOT/config.sh"
source "$ROOT/training_overrides.sh"
export PYTHONPATH="$ROOT:$UNI_AGENT_ROOT:$VERL_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export RAY_SUBMIT_MODE=local RAY_INIT_ADDRESS=auto
[[ -d "$UNI_AGENT_ROOT/uni_agent" ]] || { echo 'Run setup.sh first' >&2; exit 2; }
if [[ "${1:-}" == --dry-run ]]; then
    shift
    export AGENTENV_DRY_RUN=1
    exec bash "$ROOT/train.sh" "${TRAINING_OVERRIDES[@]}" "$@"
fi
: "${MODEL_PATH:?Set MODEL_PATH}" "${TRAIN_DATA:?Set TRAIN_DATA}" "${VAL_DATA:?Set VAL_DATA}"
: "${VC_WORKER_HOSTS:?Comma-separated head and worker hosts}" "${VC_TASK_INDEX:?Node index: 0 or 1}"
: "${E2B_API_KEY:?Inject E2B_API_KEY}" "${AGENTENV_RELAY_TARGET:?Set user@relay-host}"
: "${AGENTENV_HOST:?Sandbox-reachable callback host}" "${AGENTENV_ASSET_URL:?Sandbox-reachable asset URL}"
: "${CLAUDE_CODE_SHA256:?SHA256 of the ARM64 Claude binary served as /claude}"
IFS=, read -ra hosts <<< "$VC_WORKER_HOSTS"
[[ ${#hosts[@]} == 2 && "$VC_TASK_INDEX" =~ ^[01]$ && -n ${hosts[0]} && -n ${hosts[1]} ]] || { echo 'Expected two workers and index 0 or 1' >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" && -r "$TRAIN_DATA" && -r "$VAL_DATA" && -r "$TASK_CONFIG" ]] || { echo 'Check model, train/validation data and task config paths' >&2; exit 2; }
python3 "$ROOT/preflight.py"
source "$ROOT/ascend.sh"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"
job_id=$(printf '%s' "$VC_WORKER_HOSTS" | sha256sum | cut -c1-10)
export AGENTENV_LOCAL_RUNTIME="${AGENTENV_LOCAL_RUNTIME:-/tmp/agentenv-$UID-$job_id}"
mkdir -p "$AGENTENV_LOCAL_RUNTIME"
case "$(stat -f -c %T "$AGENTENV_LOCAL_RUNTIME")" in nfs*) echo 'AGENTENV_LOCAL_RUNTIME must use node-local storage' >&2; exit 2;; esac
export HF_HOME="$AGENTENV_LOCAL_RUNTIME/huggingface" TMPDIR="$AGENTENV_LOCAL_RUNTIME/tmp"
export TRITON_CACHE_DIR="$AGENTENV_LOCAL_RUNTIME/triton" XDG_CACHE_HOME="$AGENTENV_LOCAL_RUNTIME/cache"
export AGENTENV_CONTROL_DIR="$AGENTENV_LOCAL_RUNTIME/ssh"
mkdir -p "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$AGENTENV_CONTROL_DIR"
export E2B_API_URL="http://127.0.0.1:${AGENTENV_API_PORT:-40190}"
export E2B_SANDBOX_URL="$E2B_API_URL" E2B_DEBUG=false AGENTENV_CALLBACK_ENABLED=1
export AGENTENV_RELAY_SSH_PORT="${AGENTENV_RELAY_SSH_PORT:-22}"
export AGENTENV_CALLBACK_PORT="${AGENTENV_CALLBACK_PORT:-40193}"
export AGENTENV_RELAY_CALLBACK_PORT="${AGENTENV_RELAY_CALLBACK_PORT:-30193}"
export NO_PROXY="127.0.0.1,localhost,$VC_WORKER_HOSTS" no_proxy="127.0.0.1,localhost,$VC_WORKER_HOSTS"
source "$ROOT/tunnel.sh"
trap stop_api_tunnel EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_api_tunnel
export RAY_ADDRESS="${hosts[0]}:6767"
if [[ "$VC_TASK_INDEX" == 0 ]]; then
    ray start --head --port=6767 --dashboard-port=8261 --dashboard-host=0.0.0.0 \
        --node-ip-address="$LOCAL_IP" --resources='{"NPU":16}' --temp-dir="$AGENTENV_LOCAL_RUNTIME/ray"
    python3 "$ROOT/wait_cluster.py"
    bash "$ROOT/train.sh" "${TRAINING_OVERRIDES[@]}" "$@"
else
    for ((attempt=0; attempt<120; attempt++)); do
        if ray start --address="$RAY_ADDRESS" --node-ip-address="$LOCAL_IP" --resources='{"NPU":16}'; then
            while ssh -S "$API_CONTROL_DIR/s" -O check "$AGENTENV_RELAY_TARGET" 2>/dev/null; do sleep 15; done
            exit 1
        fi
        sleep 5
    done
    echo 'Timed out joining Ray head' >&2; exit 1
fi
