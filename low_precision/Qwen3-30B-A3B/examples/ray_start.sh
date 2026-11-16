# set -x pipefail

# pkill -9 python
ray stop --force
pkill -9 ray
pkill -9 VLLM

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
ulimit -n 32768
umask 0027

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1
export RAY_DEBUG_POST_MORTEM=0
export RAY_DEBUG=0
export VLLM_ASCEND_TASK_QUEUE_ENABLE=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export HCCL_IF_BASE_PORT=24703
export LCAL_COMM_ID=127.0.0.1:27001
export ASCEND_LAUNCH_BLOCKING=0
export MULTI_STREAM_MEMORY_REUSE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1


# 机器配置：A5=4机x8卡，A3=2机x16卡
CLUSTER_PROFILE=${CLUSTER_PROFILE:-A5}

# 是否启用 MXFP8 低精度：True 或 False
export ENABLE_MXFP8="${ENABLE_MXFP8:-True}"

# 是否忽略 EOS：性能测试 True，精度测试 False
export IGNORE_EOS="${IGNORE_EOS:-True}"

case "${CLUSTER_PROFILE}" in
  A5)
    export NNODES=4
    export NPUS_PER_NODE=8
    ;;
  A3)
    export NNODES=2
    export NPUS_PER_NODE=16
    ;;
  *)
    echo "Unsupported CLUSTER_PROFILE: ${CLUSTER_PROFILE}. Expected A5 or A3." >&2
    exit 1
    ;;
esac

if [ "${CLUSTER_PROFILE}" = "A3" ] && [ "${ENABLE_MXFP8}" = "True" ]; then
  echo "MXFP8 is not supported on A3. Set ENABLE_MXFP8=False or use CLUSTER_PROFILE=A5." >&2
  exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 "$((NPUS_PER_NODE - 1))")"
echo "Cluster profile: ${CLUSTER_PROFILE}, nodes: ${NNODES}, NPUs per node: ${NPUS_PER_NODE}"

# 修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto

# 修改为当前节点的通信网卡
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000


# 获取当前节点IP
export CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
echo $MASTER_ADDR
echo $CURRENT_IP

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_SH="${SCRIPT_DIR}/run_grpo_qwen3_moe_30b_megatron_32k.sh"
echo "Use $DEFAULT_SH"

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=0.0.0.0 --node-ip-address=$CURRENT_IP --dashboard-port=4919 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash ${DEFAULT_SH}
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
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
