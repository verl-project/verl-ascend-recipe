pkill -9 python
pkill -9 VLLM
ray stop --force
rm -rf /tmp/ray
rm -rf kernel_meta
rm -rf /root/.triton/cache/
ulimit -n 65536
set -x 

echo "===================================================="
# 修改为当前需要跑的用例路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DEFAULT_SH=$SCRIPT_DIR/train_deepseek_v4_grpo_mindspeed_vllm_A5.sh
echo "Use $DEFAULT_SH"

python3 - <<'PY'
import os
import torch
import cann_ops_transformer

print("OPP:", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
print("attention:", hasattr(torch.ops.cann_ops_transformer, "sparse_flash_mla"))
print("metadata:", hasattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata"))
PY

RAY_PORT=6379

export CPU_AFFINITY_CONF=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES='true'
export TRANSFORMERS_VERBOSITY=error
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ASCEND_DSV4_BF16_DEBUG_DEFERRED=1
export VLLM_ASCEND_A5_FULL_CAPTURE_MC2=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_SLEEP_LEVEL=2
export USE_MULTI_BLOCK_POOL=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=128
export ACL_OP_INIT_MODE=0
export TRITON_ALLWAYS_COMPILE=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export TASK_QUEUE_ENABLE=2
export VLLM_ASCEND_TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=auto
# EI0020 NPU socket bind 冲突（vLLM DP/EP 单卡多进程下 auto 会撞端口），固定端口段
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export STREAMS_PER_DEVICE=32
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1

# Project Configuration
project_name='dsv4'
exp_name='dsv4-a5'

# 按需修改节点数
NNODES=8
NPUS_PER_NODE=8
MASTER_ADDR="IP FOR MASTER NODE"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"

# 所有节点统一获取 IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
    # 启动 Ray head
    ray start --head --port 6766 \
      --dashboard-host=$MASTER_ADDR \
      --node-ip-address=$CURRENT_IP \
      --dashboard-port=8260 \
      --resources='{"NPU": '$NPUS_PER_NODE'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | awk '$0 ~ /NPU/ { split($1, a, "/"); print a[2]; exit }')
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / $NPUS_PER_NODE))

        if [ "$device_count" -eq "$NNODES" ]; then
            
            ray status
            bash $DEFAULT_SH
            break
        else
            echo "Waiting for Ray cluster... Current: $device_count / $NNODES nodes"
            sleep 5
        fi
    done
else
    # 子节点注册到 Ray
    while true; do
        ray start --address="$MASTER_ADDR:6766" \
          --resources='{"NPU": '$NPUS_PER_NODE'}' \
          --node-ip-address=$CURRENT_IP

        if [ $? -eq 0 ]; then
            echo "Successfully connected to Ray cluster"
            break
        else
            echo "Failed to connect. Retrying in 5 seconds..."
            sleep 5
        fi
    done
fi
