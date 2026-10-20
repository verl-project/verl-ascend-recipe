# set -x
pkill -9 python
ray stop --force
rm -rf /tmp/ray
export RAY_DEDUP_LOGS=1
export RAY_DEBUG=0
export RAY_DEBUG_POST_MORTEM=0
export HYDRA_FULL_ERROR=1
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
#TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
export TASK_QUEUE_ENABLE=2
export VLLM_ASCEND_TASK_QUEUE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export GLOO_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=50000
export HCCL_HOST_SOCKET_PORT_RANGE="60000-60050"
export HCCL_NPU_SOCKET_PORT_RANGE="61000-61050"
export HCCL_BUFFSIZE=400
export LD_PRELOAD="/usr/local/lib/libjemalloc.so.2"
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export PYTHONUNBUFFERED=1

## VLLM AND CUSTOM
export VLLM_VERSION="0.18.0"
export VLLM_ASCEND_ENABLE_NZ=0

#修改为当前需要跑的用例路径
DEFAULT_SH="./train_glm5_grpo_megatron_vllm_32nodes.sh"
echo "Use $DEFAULT_SH"
ulimit -n 32768
mkdir -p logs


# 精度用
export CLOSE_MATMUL_K_SHIFT=1   # 关闭矩阵乘法（MatMul）的 K 维度优化移位。
export ATB_MATMUL_SHUFFLE_K_ENABLE=0    # 关闭 ATB 库的矩阵乘法 K 维度重排优化。

# 预编译防止超时
python -c "import mindspeed; from mindspeed.op_builder.swiglu_builder import SwigluOpBuilder; SwigluOpBuilder().load()"
python -c "import mindspeed; from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()"

export NNODES=32
NPUS_PER_NODE=16

#修改为当前节点的通信网卡
export SOCKET_IFNAME="Your IFNAME"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME

#获取当前节点IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
#修改为对应主节点IP
MASTER_ADDR="Your IP_ADDRESS"


if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 8260 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8265 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
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
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:8260" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

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


