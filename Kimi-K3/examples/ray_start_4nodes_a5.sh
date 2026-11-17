#!/usr/bin/env bash
set -euo pipefail

ray stop

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

# ==================== 环境配置 ====================
export KIMI_BACKEND_ROOT="$ROOT"
export IBV_DRIVERS=/usr/lib64/libhrn5

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

mkdir -p "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$ASCEND_CACHE_PATH" "$ACL_OP_COMPILER_CACHE_DIR"

# ==================== CANN ====================
set +u

if [[ -f /opt/kimi-k3/scripts/cann_env.sh ]]; then
    source /opt/kimi-k3/scripts/cann_env.sh
else
    for setup in \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/nnal/atb/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
    do
        [[ ! -f "$setup" ]] || source "$setup"
    done
fi

source /opt/profile1102-venv/bin/activate

set -u

# ==================== 动态库 ====================
JEMALLOC="/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2"

if [[ -f "$JEMALLOC" && ":${LD_PRELOAD:-}:" != *":$JEMALLOC:"* ]]; then
    export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$ROOT/verl:$ROOT/vllm-ascend:$ROOT/FSDPTurbo:${PYTHONPATH:-}"

# ==================== Ray / Ascend ====================
export NNODES
export NPUS_PER_NODE=8
export KIMI_RUN_MODE=multinode

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
export VLLM_KV_CACHE_LAYOUT=BLNHC

ulimit -n 65536

# ==================== 获取本机 IP ====================
NODE_IP=$(ip -4 -o addr show dev "$HCCL_SOCKET_IFNAME" scope global |
    awk 'NR==1 {sub(/\/.*/, "", $4); print $4}')

cd "$ROOT"

RUN_DIR="$ROOT/runs/$VERL_RUN_NAME"
export VERL_CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$RUN_DIR"

# =====================================================
# Head：本机 IP 等于 MASTER_ADDR
# Worker：其他节点
# =====================================================

if [[ "$NODE_IP" == "$MASTER_ADDR" ]]; then

    echo "Starting Ray Head: $MASTER_ADDR:$RAY_PORT"

    ray start --head \
        --port "$RAY_PORT" \
        --node-ip-address "$NODE_IP" \
        --dashboard-host 127.0.0.1 \
        --dashboard-port 8260 \
        --num-cpus 64 \
        --resources '{"NPU":8}' \
        --object-store-memory 17179869184 \
        --temp-dir /tmp/ray_k3

    # 只检查 Ray 存活节点数量
    echo "Waiting for $NNODES Ray nodes..."

    for i in $(seq 1 360); do
        NODE_COUNT=$(
            ray list nodes \
                --address=http://127.0.0.1:8260 \
                --format=json 2>/dev/null |
            grep -oE '"state"[[:space:]]*:[[:space:]]*"ALIVE"' |
            wc -l || true
        )

        NODE_COUNT=$(echo "$NODE_COUNT" | tr -d '[:space:]')

        echo "Ray nodes: $NODE_COUNT/$NNODES"

        if (( NODE_COUNT >= NNODES )); then
            break
        fi

        sleep 5
    done

    if (( NODE_COUNT < NNODES )); then
        echo "Ray cluster startup timeout"
        exit 1
    fi

    echo "Ray cluster ready. Starting training..."

    # ==================== 训练配置 ====================
    export VERL_PROJECT_NAME=Kimi-K3-release
    export VERL_EXPERIMENT_NAME="${VERL_RUN_NAME}_$(date +%Y%m%d_%H%M%S)"

    export MODEL_PATH="$MODEL_PATH_DIR"
    export TRAIN_FILE="$DATASET_DIR"
    export TEST_FILE="$TRAIN_FILE"

    export VERL_TOTAL_TRAINING_STEPS=200
    export VERL_SAVE_FREQ=-1
    export VERL_RESUME_MODE="${VERL_RESUME_MODE:-disable}"
    export VERL_MAX_ACTOR_CKPT_TO_KEEP=1

    export TRAIN_BATCH_SIZE=16
    export ROLLOUT_N=4
    export PPO_MINI_BATCH_SIZE=16

    export VERL_MAX_RESPONSE_LENGTH=2048
    export VERL_VLLM_MAX_NUM_SEQS=64
    export VERL_VLLM_MAX_NUM_BATCHED_TOKENS=8192
    export VERL_VLLM_ENFORCE_EAGER=False

    export ROLLOUT_TP=8
    export ROLLOUT_DP=1
    export TURBO_EP_SIZE=8
    export TURBO_CP_SIZE=1

    export VLLM_ASCEND_ENABLE_MC2_HIERARCHY_COMM=False

    export VERL_ROLLOUT_DATA_DIR="$RUN_DIR/rollout_dump"

    ROLLOUT_DATA_DIR=null
    if [[ "${VERL_SAVE_ROLLOUT:-false}" == "true" ]]; then
        ROLLOUT_DATA_DIR="$VERL_ROLLOUT_DATA_DIR"
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
        "hydra.run.dir=$RUN_DIR/hydra/$VERL_EXPERIMENT_NAME"

    TRAIN_EXIT=$?
    set -e

    printf '%s\n' "$TRAIN_EXIT" > "$RUN_DIR/training_exit_code.txt"
    exit "$TRAIN_EXIT"

else

    echo "Starting Ray Worker: $NODE_IP"

    ray start \
        --address "$RAY_ADDRESS" \
        --node-ip-address "$NODE_IP" \
        --num-cpus 64 \
        --resources '{"NPU":8}' \
        --object-store-memory 17179869184 \
        --temp-dir /tmp/ray_k3

    echo "Ray Worker started."

fi