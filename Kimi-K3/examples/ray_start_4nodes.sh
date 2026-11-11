#!/usr/bin/env bash
set -euo pipefail
ROOT=/project/root/path
MODEL_PATH_DIR=/model/path
DATASET_DIR=/dataset/path
# Runtime settings for the validated four-node container environment.
export KIMI_BACKEND_ROOT="$ROOT"
export IBV_DRIVERS="$KIMI_BACKEND_ROOT/rdma-libs/libhrn5"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export XDG_CACHE_HOME=/root/.cache
export HF_HOME=/root/.cache/huggingface
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export WANDB_MODE=disabled
export TRITON_CACHE_DIR=/root/.cache/triton
export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
export ASCEND_CACHE_PATH=/root/.cache/ascend
export ASCEND_PROCESS_LOG_PATH=/root/ascend
export ASCEND_WORK_PATH=/root/ascend
export ACL_OP_COMPILER_CACHE_DIR=/root/.cache/acl
mkdir -p "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$ASCEND_CACHE_PATH" "$ACL_OP_COMPILER_CACHE_DIR"

# CANN setup scripts reference optional, unset variables.
set +u
if [[ -f /opt/kimi-k3/scripts/cann_env.sh ]]; then
    source /opt/kimi-k3/scripts/cann_env.sh
else
for setup in /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    [[ ! -f "$setup" ]] || source "$setup"
done
fi
if [[ -f /opt/profile1102-venv/bin/activate ]]; then
    source /opt/profile1102-venv/bin/activate
fi
set -u
JEMALLOC="/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2"
if [[ -f "$JEMALLOC" && ":${LD_PRELOAD:-}:" != *":$JEMALLOC:"* ]]; then
    export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$KIMI_BACKEND_ROOT/verl:$KIMI_BACKEND_ROOT/vllm-ascend:$KIMI_BACKEND_ROOT/FSDPTurbo:${PYTHONPATH:-}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_DEDUP_LOGS=0
export RAY_USAGE_STATS_ENABLED=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=enp35s0f2
export GLOO_SOCKET_IFNAME=data0.3001
export HCCL_HOST_SOCKET_PORT_RANGE=30000-30999
export HCCL_NPU_SOCKET_PORT_RANGE=31000-31999
export HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_EXEC_TIMEOUT=17340
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=256
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
export VLLM_USE_V1=1
export VLLM_USE_V2_MODEL_RUNNER=1
export TASK_QUEUE_ENABLE=1
export STREAMS_PER_DEVICE=32
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHDYNAMO_DISABLE=1
ulimit -n 65536
export VLLM_KV_CACHE_LAYOUT=BLNHC
cd "$ROOT"

MASTER_ADDR=141.61.54.135
RAY_PORT=6788
export NNODES=4 NPUS_PER_NODE=8 KIMI_RUN_MODE=multinode
export RAY_ADDRESS="$MASTER_ADDR:$RAY_PORT"
export VERL_RUN_NAME="${VERL_RUN_NAME:-kimi_k3_200steps_20261109}"
RUN_DIR="$ROOT/runs/$VERL_RUN_NAME"
export VERL_CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR"
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
    DRY_RUN=true
    shift
fi
NODE_IP=$(python3 - "$HCCL_SOCKET_IFNAME" <<'PY'
import fcntl
import socket
import struct
import sys
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
    address = fcntl.ioctl(connection.fileno(), 0x8915, struct.pack('256s', sys.argv[1].encode()[:15]))
    print(socket.inet_ntoa(address[20:24]))
PY
)
case "$NODE_IP" in
    141.61.54.135|141.61.54.87|141.61.54.95|141.61.54.99) ;;
    *) echo "Unexpected training node IP: $NODE_IP" >&2; exit 1 ;;
esac

node_registered() {
    timeout 30 python3 - "$MASTER_ADDR" "$RAY_PORT" "$NODE_IP" <<'PY'
import socket
import sys
import ray
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2):
        pass
except OSError:
    raise SystemExit(1)
ray.init(address=f'{sys.argv[1]}:{sys.argv[2]}', logging_level='ERROR')
registered = any(n['Alive'] and n['NodeManagerAddress'] == sys.argv[3]
                 and n['Resources'].get('NPU') == 8 for n in ray.nodes())
ray.shutdown()
raise SystemExit(0 if registered else 1)
PY
}

if [[ "$NODE_IP" == "$MASTER_ADDR" ]] || "$DRY_RUN"; then
    if ! "$DRY_RUN"; then
    # Only the dedicated container is controlled. No host-wide ray stop/pkill.
    exec 9>"$RUN_DIR/head.lock"
    flock -n 9 || { echo 'A release head launcher is already running'; exit 1; }
    if node_registered; then
        echo 'Reusing the registered release Ray head'
    else
        ray start --head --port "$RAY_PORT" --node-ip-address "$NODE_IP" \
        --dashboard-host 127.0.0.1 --dashboard-port 8260 \
        --num-cpus 64 --resources '{"NPU":8}' \
        --object-store-memory 17179869184 --temp-dir /tmp/ray_k3
    fi
    python3 - <<'PY'
import os
import time
import ray
ray.init(address=os.environ['RAY_ADDRESS'])
expected = {'141.61.54.135', '141.61.54.87', '141.61.54.95', '141.61.54.99'}
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    nodes = {n['NodeManagerAddress']: n for n in ray.nodes() if n['Alive']}
    counts = {ip: n['Resources'].get('NPU', 0) for ip, n in nodes.items()}
    print(f'Ray resources: {counts}', flush=True)
    if set(nodes) == expected and all(counts[ip] == 8 for ip in expected):
        break
    time.sleep(5)
else:
    raise SystemExit('Four-node/32-NPU Ray cluster did not become ready')
ray.shutdown()
PY
    fi
    export VERL_PROJECT_NAME=Kimi-K3-release
    export VERL_EXPERIMENT_NAME="${VERL_RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
    export MODEL_PATH="$MODEL_PATH_DIR"
    export TRAIN_FILE="$DATASET_DIR"
    export TEST_FILE="$TRAIN_FILE"
    export VERL_TOTAL_TRAINING_STEPS=200 VERL_SAVE_FREQ=25 VERL_RESUME_MODE="${VERL_RESUME_MODE:-disable}"
    # The framework retains the previous actor checkpoint until the new save succeeds.
    export VERL_MAX_ACTOR_CKPT_TO_KEEP=1
    export TRAIN_BATCH_SIZE=16 ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=16
    export VERL_MAX_RESPONSE_LENGTH=2048 VERL_VLLM_MAX_NUM_SEQS=64
    export VERL_VLLM_MAX_NUM_BATCHED_TOKENS=8192 VERL_VLLM_ENFORCE_EAGER=False
    export ROLLOUT_TP=8 ROLLOUT_DP=1 TURBO_EP_SIZE=8 TURBO_CP_SIZE=1
    export VLLM_ASCEND_ENABLE_MC2_HIERARCHY_COMM=False
    export VERL_ROLLOUT_DATA_DIR="$RUN_DIR/rollout_dump"
    # Saving rollout text is opt-in; null disables the trainer dump.
    ROLLOUT_DATA_DIR=null
    case "${VERL_SAVE_ROLLOUT:-false}" in
        true) ROLLOUT_DATA_DIR="$VERL_ROLLOUT_DATA_DIR" ;;
        false) ;;
        *) echo 'VERL_SAVE_ROLLOUT must be true or false' >&2; exit 1 ;;
    esac
    if "$DRY_RUN"; then
        set -- --cfg job --resolve "$@"
    else
        rm -f "$RUN_DIR/training_exit_code.txt"
    fi
    set +e
    bash "$ROOT/verl/examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh" \
        "trainer.default_local_dir=$VERL_CHECKPOINT_DIR" \
        "trainer.rollout_data_dir=$ROLLOUT_DATA_DIR" \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
        "++ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
        "++ray_kwargs.ray_init.runtime_env.env_vars.IBV_DRIVERS=$IBV_DRIVERS" \
        ++ray_kwargs.ray_init.runtime_env.py_executable=/opt/profile1102-venv/bin/python3 \
        ++actor_rollout_ref.actor.fsdp_config.turbo_config.model.attn_implementation=flash_attention_2 \
        ++actor_rollout_ref.ref.fsdp_config.turbo_config.model.attn_implementation=flash_attention_2 \
        '+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2' \
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16,32,64]' \
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=64 \
        "hydra.run.dir=$RUN_DIR/hydra/$VERL_EXPERIMENT_NAME" \
        "$@"
    TRAIN_EXIT=$?
    set -e
    if "$DRY_RUN"; then
        exit "$TRAIN_EXIT"
    fi
    printf '%s\n' "$TRAIN_EXIT" > "$RUN_DIR/training_exit_code.txt"
    exit "$TRAIN_EXIT"
else
    # Joining does not mutate the head node or any other containers.
    for attempt in $(seq 1 60); do
        if node_registered; then
            echo 'Release Ray worker is already registered'
            exit 0
        fi
        if ray start --address "$RAY_ADDRESS" --node-ip-address "$NODE_IP" \
            --num-cpus 64 --resources '{"NPU":8}' --object-store-memory 17179869184; then
            ray status --address "$RAY_ADDRESS"
            exit 0
        fi
        sleep 5
    done
    echo "Unable to join release Ray head after 60 attempts" >&2
    exit 1
fi