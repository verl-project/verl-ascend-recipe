

pkill -9 python
ray stop --force
rm -rf /tmp/ray
rm -rf /root/.triton/cache/
rm -rf /root/.triton/dump/
rm -rf /tmp/torchinductor_root/*
rm -rf /root/.cache/torch_extensions

# CANN env
CANN_DIR=/usr/local/Ascend
source $CANN_DIR/ascend-toolkit/set_env.sh
source $CANN_DIR/nnal/atb/set_env.sh
source $CANN_DIR/cann-9.0.0-beta.2/opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=$CANN_DIR/cann-9.0.0-beta.2/opp/vendors/custom_transformer
export PATH=$CANN_DIR/ascend-toolkit/latest/tools/bishengir/bin/:$PATH
export LD_LIBRARY_PATH=$CANN_DIR/cann-9.0.0-beta.2/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
python -c "import mindspeed; from mindspeed.op_builder.npu_sparse_attn_shared_kv_builder import NPUSparseAttnSharedKVOpBuilder; NPUSparseAttnSharedKVOpBuilder().load()"

# Fix:关闭训练图模式，待修复
export TORCHDYNAMO_VERBOSE=1
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_DISABLE=1

# vllm路径
export PYTHONPATH="/workspace-verl/vllm:$PYTHONPATH"
export PYTHONPATH="/workspace-verl/vllm-ascend:$PYTHONPATH"

# TASK_QUEUE_ENABLE，下发优化，图模式设置为1，非图模式设置为2
export TASK_QUEUE_ENABLE=1
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export STREAMS_PER_DEVICE=32
export CUDA_DEVICE_MAX_CONNECTIONS=1

# VLLM env 
export CPU_AFFINITY_CONF=1
export USE_MULTI_BLOCK_POOL=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=500
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_VERSION=0.13.0
export TRITON_ALLWAYS_COMPILE=1

# 规避8.5.0 CANN mbridge 卡死
export HCCL_OP_EXPANSION_MODE="AIV" 
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

# 修改为当前需要跑的用例路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DEFAULT_SH=$SCRIPT_DIR/train_deepseek_v4_grpo_mindspeed_vllm.sh
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir logs

NNODES=8
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

