#!/usr/bin/env bash
# Platform entry point: run on every node in each fresh training container.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export MODEL_PROFILE="${MODEL_PROFILE:-$ROOT/configs/models/qwen3_5_9b.sh}"
source "$MODEL_PROFILE"
source "${TRAINING_ENV_CONFIG:-$ROOT/scripts/training/ascend.sh}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-$GLOO_SOCKET_IFNAME}"
source "${EXPERIMENT_CONFIG:-$ROOT/examples/e2b_swe_grpo/experiment.sh}"
source "$ROOT/scripts/sandbox/setup.sh"
trap stop_sandbox EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

: "${VC_WORKER_HOSTS:?Platform must supply VC_WORKER_HOSTS}"
: "${VC_TASK_INDEX:?Platform must supply VC_TASK_INDEX}"
HEAD_HOST="${VC_WORKER_HOSTS%%,*}"
setup_sandbox

if [[ "$VC_TASK_INDEX" == 0 ]]; then
    prepare_sandbox_callback
    # Private per-launch output avoids concurrent writers on the shared mount.
    if [[ -z "${RUNTIME_ENV:-}" ]]; then
        mkdir -p "$ROOT/.runtime"
        RUNTIME_ENV=$(mktemp "$ROOT/.runtime/ray.XXXXXX.json")
    fi
    export RUNTIME_ENV
    python3 "$ROOT/scripts/training/prepare_runtime.py" \
        --verl-source "${VERL_SOURCE:?Set VERL_SOURCE}" --output "$RUNTIME_ENV"
    ray start --head --port=6766 --dashboard-port=8260 --dashboard-host=0.0.0.0 \
        --node-ip-address="$HEAD_HOST" --resources="{\"NPU\":$NGPUS_PER_NODE}"
    # The submission requests all 32 NPUs; wait for both platform workers.
    export RAY_JOB_ADDRESS="http://$HEAD_HOST:8260"
    ready=false
    for ((attempt=0; attempt<120; attempt++)); do
        if python3 - "$HEAD_HOST" "$NNODES" "$NGPUS_PER_NODE" <<'PYREADY'
import sys
import ray
ray.init(address=f"{sys.argv[1]}:6766", logging_level="ERROR")
nodes = [node for node in ray.nodes() if node["Alive"]]
ready = len(nodes) == int(sys.argv[2]) and all(
    node["Resources"].get("NPU", 0) >= int(sys.argv[3]) for node in nodes
)
ray.shutdown()
raise SystemExit(0 if ready else 1)
PYREADY
        then ready=true; break; fi
        sleep 5
    done
    [[ "$ready" == true ]] || { echo "Timed out waiting for 2 x 16 NPU Ray workers" >&2; exit 1; }
    # Keep this shell alive to own tunnels until the blocking submission ends.
    bash "${EXPERIMENT_LAUNCHER:-$ROOT/examples/e2b_swe_grpo/launch.sh}" --submit \
        "${EXPERIMENT_ARGS[@]}" "$@"
else
    joined=false
    for ((attempt=0; attempt<120; attempt++)); do
        if ray start --address="$HEAD_HOST:6766" --node-ip-address="$LOCAL_IP" \
            --resources="{\"NPU\":$NGPUS_PER_NODE}"; then joined=true; break; fi
        sleep 5
    done
    [[ "$joined" == true ]] || { echo 'Timed out joining Ray head' >&2; exit 1; }
    while [[ "$KASANDBOX_TUNNEL_MODE" == direct ]] || ssh -S "$SANDBOX_CONTROL_SOCKET" -O check "$KASANDBOX_SSH_TARGET" 2>/dev/null; do sleep 15; done
    exit 1
fi
