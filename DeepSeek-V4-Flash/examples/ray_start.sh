
pkill -9 python
ray stop --force

# CANN env
CANN_DIR=/usr/local/Ascend
source $CANN_DIR/ascend-toolkit/set_env.sh
source $CANN_DIR/nnal/atb/set_env.sh

# vLLM 使用开头的环境配置及当前 Python 安装，避免旧路径抢占优先级。
export PYTHONPATH="/workspace-verl/vllm:$PYTHONPATH"
export PYTHONPATH="/workspace-verl/vllm-ascend:$PYTHONPATH"

#!/usr/bin/env bash
export CPU_AFFINITY_CONF=1
export HCCL_BUFFSIZE=150
export HCCL_CONNECT_TIMEOUT=900
export HCCL_EXEC_TIMEOUT=900
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_IF_BASE_PORT=48890
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_OP_EXPANSION_MODE=AIV
export HYDRA_FULL_ERROR=1
export MERLIN_ROBUST_ENFORCE=1
export MULTI_STREAM_MEMORY_REUSE=2
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=1
export TORCH_NCCL_USE_HIGH_PRIORITY_STREAM=0
export TRANSFORMERS_VERBOSITY=error
export VLLM_USE_V1=1
export VLLM_VERSION=0.30.0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V2_MODEL_RUNNER=1

# 修改为当前需要跑的用例路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DEFAULT_SH=$SCRIPT_DIR/train_deepseek_v4_grpo_mindspeed_vllm.sh
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=4
NPUS_PER_NODE=16
# 修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
export GLOO_SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"
# 获取当前IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
# 主节点启动
ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

while true; do
    ray_status_output=$(ray status)
    npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
    npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
    device_count=$((npu_count_int / $NPUS_PER_NODE))

    # 判断device_count 是否与 NNODES 相等
    if [ "$device_count" -eq "$NNODES" ]; then
        echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
        ray status
        bash $DEFAULT_SH
        break
    else
        echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
        sleep 5
    fi
done
else
# 子节点尝试往主节点注册 ray 直到成功
while true; do
    # 尝试连接 ray 集群
    ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

    # 检查连接是否成功
    ray status
    if [ $? -eq 0 ]; then
        echo "Successfully connected to the Ray cluster!"
        break
    else
        echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
        sleep 5
    fi
done
fi

sleep 600
