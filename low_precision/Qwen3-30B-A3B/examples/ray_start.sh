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

export HCCL_OP_EXPANSION_MODE="AIV" 
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


NNODES=2
NPUS_PER_NODE=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#修改为对应主节点IP
MASTER_ADDR="IP FOR MASTER NODE"

#修改为当前节点的通信网卡
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
SOCKET_IFNAME="Your SOCKET IFNAME"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export HCCL_CONNECT_TIMEOUT=3000
export HCCL_EXEC_TIMEOUT=3000


# 获取当前节点IP
export CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
echo $MASTER_ADDR
echo $CURRENT_IP

DEFAULT_SH="run_grpo_qwen3_moe_30b_megatron.sh"
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
