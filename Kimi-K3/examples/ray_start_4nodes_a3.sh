#!/usr/bin/env bash
# A3 release run: 16 NPUs per node, rollout TP/EP=16, no checkpoint save/resume.
set -euo pipefail

# ==================== 只修改这三个参数 ====================
MASTER_ADDR=xxx.xxx.xxx.xxx
RAY_PORT=xxxx
NNODES=4

# ==================== 路径配置 ====================
ROOT=/path/to/root
MODEL_PATH_DIR=/path/to/model/weight
DATASET_DIR=/path/to/dataset

export VERL_RUN_NAME="kimi_k3_200steps_20261109"
export RAY_ADDRESS="$MASTER_ADDR:$RAY_PORT"

export MASTER_ADDR RAY_PORT NNODES
export KIMI_RELEASE_ROOT="$ROOT" MODEL_PATH="$MODEL_PATH_DIR"
export TRAIN_FILE="$DATASET_DIR"
if [[ -d "$TRAIN_FILE" ]]; then TRAIN_FILE="${TRAIN_FILE%/}/train_448.parquet"; fi
export TEST_FILE="$TRAIN_FILE"

# ==================== 训练配置 ====================
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp48s3u1u1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}"
export VERL_EXPERIMENT_NAME="${VERL_EXPERIMENT_NAME:-$VERL_RUN_NAME}"
export VERL_TOTAL_TRAINING_STEPS="${VERL_TOTAL_TRAINING_STEPS:-200}"
export KIMI_EXPECTED_MODEL_LAYERS="${KIMI_EXPECTED_MODEL_LAYERS:-93}"
export VERL_SAVE_ROLLOUT="${VERL_SAVE_ROLLOUT:-true}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

[[ "$NNODES" =~ ^[1-9][0-9]*$ ]] || { echo 'NNODES must be a positive integer' >&2; exit 1; }
[[ "$RAY_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( RAY_PORT <= 65535 )) || { echo 'Invalid RAY_PORT' >&2; exit 1; }
[[ "$VERL_RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid VERL_RUN_NAME' >&2; exit 1; }
cd "$ROOT"
RUN_DIR="$ROOT/runs/$VERL_RUN_NAME"
mkdir -p "$RUN_DIR"
if [[ "${1:-}" == --log ]]; then
    shift
    exec >>"$RUN_DIR/launcher.$(hostname -s).log" 2>&1
fi
TRAINER="$ROOT/verl/examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh"
[[ -f "$TRAINER" ]] || { echo "Missing trainer: $TRAINER" >&2; exit 1; }
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi

# Only environment needed before Ray starts belongs here. The source trainer
# sets PYTHONPATH, FSDP_TURBO_ROOT and VLLM_ASCEND_PATH for training workers.
export KIMI_BACKEND_ROOT="$ROOT"
# Resolve CANN from the selected image instead of pinning a toolkit version.
export ASCEND_CANN_HOME="$(readlink -f "${ASCEND_CANN_HOME:-${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}}")"
[[ -f "$ASCEND_CANN_HOME/set_env.sh" ]] || { echo "Missing CANN environment: $ASCEND_CANN_HOME" >&2; exit 1; }
export ASCEND_TOOLKIT_HOME="$ASCEND_CANN_HOME"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR=/tmp XDG_CACHE_HOME=/root/.cache
export HF_HOME="$XDG_CACHE_HOME/huggingface" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_MODULES_CACHE="$HF_HOME/modules" HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 WANDB_MODE=disabled
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor" ASCEND_CACHE_PATH="$XDG_CACHE_HOME/ascend"
export ASCEND_PROCESS_LOG_PATH=/root/ascend ASCEND_WORK_PATH=/root/ascend
export ACL_OP_COMPILER_CACHE_DIR="$XDG_CACHE_HOME/acl"
mkdir -p "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$ASCEND_CACHE_PATH" "$ACL_OP_COMPILER_CACHE_DIR"
set +u
for setup in "$ASCEND_CANN_HOME/set_env.sh" /usr/local/Ascend/nnal/atb/set_env.sh \
    "$ASCEND_CANN_HOME/opp/vendors/custom_transformer/bin/set_env.bash"; do
    [[ ! -f "$setup" ]] || source "$setup"
done
set -u
PYTHON_BIN="$(command -v python3)"
JEMALLOC="/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2"
if [[ -f "$JEMALLOC" && ":${LD_PRELOAD:-}:" != *":$JEMALLOC:"* ]]; then
    export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0
export HCCL_HOST_SOCKET_PORT_RANGE=30000-30999 HCCL_NPU_SOCKET_PORT_RANGE=31000-31999 HCCL_IF_BASE_PORT=30000
export HCCL_CONNECT_TIMEOUT=1500 HCCL_EXEC_TIMEOUT=17340 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256
export OMP_NUM_THREADS=10 OMP_PROC_BIND=false VLLM_USE_V1=1 VLLM_USE_V2_MODEL_RUNNER=1
export TASK_QUEUE_ENABLE=1 STREAMS_PER_DEVICE=32 CUDA_DEVICE_MAX_CONNECTIONS=1 TORCHDYNAMO_DISABLE=1
unset VLLM_KV_CACHE_LAYOUT
ulimit -n 65536

export NPUS_PER_NODE=16 KIMI_RUN_MODE=multinode
if (( NNODES == 1 )); then export KIMI_RUN_MODE=single; fi
export ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 $((NPUS_PER_NODE - 1)))"
export VERL_PROJECT_NAME=Kimi-K3-release
export VERL_SAVE_FREQ=-1 VERL_RESUME_MODE=disable
unset VERL_RESUME_FROM_PATH VERL_CHECKPOINT_DIR VERL_MAX_ACTOR_CKPT_TO_KEEP KIMI_RESUME_STEP
export TRAIN_BATCH_SIZE=16 ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=16
export ROLLOUT_TP=16 ROLLOUT_DP=1 TURBO_EP_SIZE=16 TURBO_CP_SIZE=1
export VLLM_ASCEND_ENABLE_MC2_HIERARCHY_COMM=False
export VERL_MAX_RESPONSE_LENGTH=2048 VERL_VLLM_MAX_NUM_SEQS=64 VERL_VLLM_MAX_NUM_BATCHED_TOKENS=8192
export VERL_VLLM_ENFORCE_EAGER=False VERL_VLLM_MULTIMODAL_TOKEN_MARGIN=16384 VERL_VLLM_MAX_MODEL_LEN=19456
export VERL_ROLLOUT_DATA_DIR="$RUN_DIR/rollout_dump"
case "$VERL_SAVE_ROLLOUT" in
    true) ROLLOUT_DATA_DIR="$VERL_ROLLOUT_DATA_DIR" ;;
    false) ROLLOUT_DATA_DIR=null ;;
    *) echo 'VERL_SAVE_ROLLOUT must be true or false' >&2; exit 1 ;;
esac
NODE_IP=$("$PYTHON_BIN" - "$HCCL_SOCKET_IFNAME" "$MASTER_ADDR" <<'PY'
import fcntl, ipaddress, socket, struct, sys
try:
    ipaddress.IPv4Address(sys.argv[2])
except ValueError:
    raise SystemExit(f'MASTER_ADDR must be the IPv4 address of the Ray head: {sys.argv[2]}')
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
    try:
        address = fcntl.ioctl(connection.fileno(), 0x8915, struct.pack('256s', sys.argv[1].encode()[:15]))
    except OSError as exc:
        raise SystemExit(f'Cannot read IPv4 on {sys.argv[1]}: {exc}. Check HCCL_SOCKET_IFNAME.')
    print(socket.inet_ntoa(address[20:24]))
PY
)
export NODE_IP
printf 'Node IP: %s (%s), Ray head: %s, expected nodes: %s\n' "$NODE_IP" "$HCCL_SOCKET_IFNAME" "$RAY_ADDRESS" "$NNODES"
[[ -f "$TRAIN_FILE" ]] || { echo "Missing training dataset: $TRAIN_FILE" >&2; exit 1; }
"$PYTHON_BIN" - "$MODEL_PATH/config.json" <<'PY'
import json, os, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = int(os.environ['KIMI_EXPECTED_MODEL_LAYERS'])
actual = config.get('text_config', config).get('num_hidden_layers')
if expected < 1 or actual != expected:
    raise SystemExit(f'Expected {expected} model layers, found {actual}')
PY

node_registered() {
    timeout 30 "$PYTHON_BIN" - "$MASTER_ADDR" "$RAY_PORT" <<'PY'
import os, socket, sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2): pass
except OSError:
    raise SystemExit(1)
import ray
ray.init(address=os.environ['RAY_ADDRESS'], logging_level='ERROR')
registered = any(n['Alive'] and n['NodeManagerAddress'] == os.environ['NODE_IP']
                 and n['Resources'].get('NPU') == int(os.environ['NPUS_PER_NODE'])
                 and n['Resources'].get('kimi_release64') == 1 for n in ray.nodes())
ray.shutdown()
raise SystemExit(0 if registered else 1)
PY
}
RAY_ARGS=(--node-ip-address "$NODE_IP" --num-cpus 64
    --resources "{\"NPU\":$NPUS_PER_NODE,\"kimi_release64\":1}" --object-store-memory 17179869184)
if ! "$DRY_RUN" && [[ "$NODE_IP" != "$MASTER_ADDR" ]]; then
    for attempt in $(seq 1 60); do
        if node_registered; then echo 'Release64 Ray worker is registered'; exit 0; fi
        if ray start --address "$RAY_ADDRESS" "${RAY_ARGS[@]}"; then
            ray status --address "$RAY_ADDRESS"; exit 0
        fi
        sleep 5
    done
    echo 'Unable to join release Ray head after 60 attempts' >&2; exit 1
fi
if "$DRY_RUN"; then
    set -- "$@" --cfg job --resolve
else
    exec 9>"$RUN_DIR/head.lock"
    flock -n 9 || { echo 'A release launcher is already running'; exit 1; }
    if (( NNODES > 1 )); then
        if ! node_registered; then
            ray start --head --port "$RAY_PORT" "${RAY_ARGS[@]}" \
                --dashboard-host 127.0.0.1 --dashboard-port 8260 --temp-dir /tmp/ray_k3_release64
        fi
        "$PYTHON_BIN" - <<'PY'
import os, time, ray
ray.init(address=os.environ['RAY_ADDRESS'])
expected, per_node = int(os.environ['NNODES']), int(os.environ['NPUS_PER_NODE'])
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    nodes = {n['NodeManagerAddress']: n for n in ray.nodes() if n['Alive']}
    counts = {ip: n['Resources'].get('NPU', 0) for ip, n in nodes.items()}
    print(f'Ray resources: {counts}', flush=True)
    if len(nodes) > expected:
        raise SystemExit(f'Expected {expected} Ray nodes, found {sorted(nodes)}')
    if len(nodes) == expected and os.environ['MASTER_ADDR'] in nodes and all(n == per_node for n in counts.values()):
        allowed = os.environ.get('KIMI_NODE_IPS', '').split()
        if allowed and set(nodes) != set(allowed):
            raise SystemExit(f'Registered nodes differ from the supplied KIMI_NODE_IPS: {sorted(nodes)}')
        if not all(n['Resources'].get('kimi_release64') == 1 for n in nodes.values()):
            raise SystemExit('Unexpected Ray cluster owner')
        if ray.available_resources().get('NPU', 0) != expected * per_node:
            raise SystemExit('The release cluster already has allocated NPUs')
        break
    time.sleep(5)
else:
    raise SystemExit(f'Ray cluster did not become ready: expected {expected} nodes with {per_node} NPUs each')
ray.shutdown()
PY
    else
        [[ "$NODE_IP" == "$MASTER_ADDR" ]] || { echo "Single-node MASTER_ADDR must match local node" >&2; exit 1; }
        echo "Single-node mode: source trainer starts the local Ray runtime"
    fi
    rm -f "$RUN_DIR/training_exit_code.txt"
fi
EXTRA_ENV=("++ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    "++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
    "++ray_kwargs.ray_init.runtime_env.py_executable=$PYTHON_BIN")
[[ -z "${IBV_DRIVERS:-}" ]] || EXTRA_ENV+=("++ray_kwargs.ray_init.runtime_env.env_vars.IBV_DRIVERS=$IBV_DRIVERS")
printf 'Source trainer: %s\nCheckpoint save/resume: disabled\n' "$TRAINER"
set +e
bash "$TRAINER" \
    "trainer.default_local_dir=$RUN_DIR" trainer.checkpoint_callback_class=null \
    "trainer.rollout_data_dir=$ROLLOUT_DATA_DIR" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
    "${EXTRA_ENV[@]}" \
    ++actor_rollout_ref.actor.fsdp_config.turbo_config.model.attn_implementation=flash_attention_2 \
    ++actor_rollout_ref.ref.fsdp_config.turbo_config.model.attn_implementation=flash_attention_2 \
    '+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2' \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY \
    '++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16,32,64]' \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=64 \
    "hydra.run.dir=$RUN_DIR/hydra/$VERL_EXPERIMENT_NAME" "$@"
TRAIN_EXIT=$?
set -e
if ! "$DRY_RUN"; then printf '%s\n' "$TRAIN_EXIT" > "$RUN_DIR/training_exit_code.txt"; fi
exit "$TRAIN_EXIT"
