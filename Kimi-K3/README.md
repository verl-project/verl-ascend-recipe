# Kimi K3 FSDPTurbo NPU 使用指南

本文用于指导在 Ascend A3 上使用 verl + FSDPTurbo + vLLM-Ascend 运行 Kimi K3 多模态 MoE GRPO 训练。适配支持动态批量、训练权重同步和 rollout decode 图模式。

## 1. 配套软件版本

| A3 配套软件 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.12.16 | |
| CANN | 9.2.1 | |
| torch | 2.10.0+cpu | |
| torch-npu | 2.10.0.post7 | |
| vLLM | 0.30.0+empty | |
| transformers | 5.14.1 | |
| triton | 3.6.0 | |
| triton-ascend | 3.6.0.post2 | |
| flash-linear-attention | 0.6.0 | |
| triton-ascend-kernels | 0.1.dev167+g3365d7053 | |
| verl | main | 基于3c6d9b43b67c74ec343d77874cd8fa6e841523da基线应用 Patch |
| FSDPTurbo | main | 基于1fa40ad350eb681ed413ca64bffedc166c986e0f基线应用 Patch |
| vllm-ascend | v0.30.0 | 基于cb629ea1925a280d57b41b48416a85097f6cdb6b基线应用 Patch |

| A5 配套软件 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.12.15 | |
| CANN | 9.2.1 | |
| torch | 2.10.0 | |
| torch-npu | 2.10.0.post6 | |
| vLLM | 0.30.0+empty | |
| transformers | 5.16.1 | |
| triton | 3.5.0 | |
| triton-ascend | 3.2.2 | |
| flash-linear-attention | 0.5.2 | |
| triton-ascend-kernels | 0.1.dev1+gd4bb38e3e.d20261027 | |
| verl | main | 基于3c6d9b43b67c74ec343d77874cd8fa6e841523da基线应用 Patch |
| FSDPTurbo | main | 基于1fa40ad350eb681ed413ca64bffedc166c986e0f基线应用 Patch |
| vllm-ascend | v0.30.0 | 基于cb629ea1925a280d57b41b48416a85097f6cdb6b基线应用 Patch |

请使用满足上述版本要求的 Ascend NPU 环境，并同时准备配套源码、Python 依赖及编译产物。源码版本与算子编译产物需要匹配。

## 2. A3 四节点环境部署与启动
```bash
# A3 创建容器（每台宿主机执行）
IMAGE=quay.io/albert54/verl:kimi-k3-fsdpturbo-vllm-cann921-a3-1109
CONTAINER=rl_test
WORK=/mnt/share/kimi-k3-a3
LOCAL=/var/tmp/kimi-k3-a3
mkdir -p "$WORK" "$LOCAL/tmp" "$LOCAL/cache" "$LOCAL/ascend"

docker pull "$IMAGE"
docker run -dit --init --name "$CONTAINER" --hostname "$(hostname -s)" \
  --network host --shm-size=32g --privileged \
  --entrypoint /bin/bash \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /mnt/share:/mnt/share \
  -v "$LOCAL/tmp:/tmp" \
  -v "$LOCAL/cache:/root/.cache" \
  -v "$LOCAL/ascend:/root/ascend" \
  "$IMAGE" -c 'sleep infinity'
docker exec -it "$CONTAINER" bash
```

```bash
# 准备工作目录
WORK=/mnt/share/kimi-k3-a3
mkdir -p "$WORK"
cd "$WORK"

# 下载源码
git clone https://github.com/verl-project/verl-ascend-recipe.git "$WORK/verl-ascend-recipe"
git clone https://github.com/verl-project/verl.git "$WORK/verl"
git clone https://gitcode.com/Ascend/FSDPTurbo.git "$WORK/FSDPTurbo"
git clone https://github.com/vllm-project/vllm.git "$WORK/vllm"
git clone https://github.com/vllm-project/vllm-ascend.git "$WORK/vllm-ascend"

# check到指定commit
git -C "$WORK/verl" checkout --detach 3c6d9b43b67c74ec343d77874cd8fa6e841523da
git -C "$WORK/FSDPTurbo" checkout --detach 1fa40ad350eb681ed413ca64bffedc166c986e0f
git -C "$WORK/vllm" checkout --detach ced6857afa0ea7b2e3f0846a62e1394e90f15607
git -C "$WORK/vllm-ascend" checkout --detach cb629ea1925a280d57b41b48416a85097f6cdb6b

# 应用Patch
cd verl
git am --keep ../verl-ascend-recipe/Kimi-K3/patch/verl.patch
cd ../FSDPTurbo
git am --keep ../verl-ascend-recipe/Kimi-K3/patch/fsdpturbo.patch
cd ../vllm-ascend
git am --keep ../verl-ascend-recipe/Kimi-K3/patch/vllm-ascend.patch
git submodule update --init --recursive

cp "$WORK/verl-ascend-recipe/Kimi-K3/examples/ray_start_4nodes_a3.sh" "$WORK/"
# 脚本若经 Windows 下载或复制，统一为 Unix LF 换行。
sed -i 's/\r$//' "$WORK/ray_start_4nodes_a3.sh"
```

源码准备完成后，在四个容器中分别执行以下安装命令。先加载 CANN，再加载 ATB，然后安装四个框架仓库。

```bash
# 在固定 commit 并应用 patch 后，逐个安装工作目录中的框架源码。
WORK=/mnt/share/kimi-k3-a3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd "$WORK/vllm"
VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0 \
  python3 -m pip install --no-build-isolation --no-deps -v -e .
cd "$WORK/FSDPTurbo"
python3 -m pip install --no-build-isolation --no-deps -v -e .
cd "$WORK/verl"
python3 -m pip install --no-build-isolation --no-deps -v -e .
cd "$WORK/vllm-ascend"
COMPILE_CUSTOM_KERNELS=1 SOC_VERSION=ascend910_9392 MAX_JOBS=32 \
  python3 -m pip install --no-build-isolation --no-deps -v -e .

cd "$WORK"
```

在 `ray_start_4nodes_a3.sh` 顶部配置 `MASTER_ADDR`（Head 通信网卡的 IPv4 地址）、`RAY_PORT` 和 `NNODES`（含 Head，默认 4）。路径区的 `ROOT` 指向框架工作目录（如/mnt/share/kimi-k3-a3），`MODEL_PATH_DIR` 指向模型目录，`DATASET_DIR` 指向包含 `train_448.parquet` 的目录。

每个容器执行同一脚本。脚本从 `HCCL_SOCKET_IFNAME`（默认 `enp48s3u1u1`）读取本机 IPv4，与 `MASTER_ADDR` 相同的节点启动 Head，其余节点加入 Ray。网卡名不同时可在对应容器覆盖此变量。Head 等到指定数量的节点就绪、每节点 16 张 NPU 可用后启动训练。训练步数由 `VERL_TOTAL_TRAINING_STEPS` 控制，默认训练 200 步并保存 rollout。

```bash
# 四个容器分别执行（按实际共享目录调整脚本路径）
bash /mnt/share/kimi-k3-a3/ray_start_4nodes_a3.sh
```

## 3. A5 四节点环境部署与启动
```
# A5镜像
docker pull quay.io/albert54/verl:kimi-k3-fsdpturbo-vllm-cann921-a5-1106

# A5创建容器
docker run -dit --ipc=host --network host --name 'rl_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data 镜像名:标签 /bin/bash

# 进入容器
docker exec -it rl_test bash
mkdir /workspace-verl 
cd /workspace-verl

# 下载源码
git clone https://github.com/verl-project/verl-ascend-recipe.git
git clone https://github.com/vllm-project/vllm-ascend.git && cd vllm-ascend && git switch releases/v0.30.0 && cd ..
git clone https://gitcode.com/Ascend/FSDPTurbo.git
git clone https://github.com/verl-project/verl.git

# check到指定commit
cd verl && git checkout 3c6d9b43b67c74ec343d77874cd8fa6e841523da && cd ..
cd FSDPTurbo && git checkout 1fa40ad350eb681ed413ca64bffedc166c986e0f && cd ..
cd vllm-ascend && git checkout cb629ea1925a280d57b41b48416a85097f6cdb6b && cd ..

# 应用Patch
cd verl && git am --keep ../verl-ascend-recipe/Kimi-K3/patch/verl.patch && cd ..
cd FSDPTurbo && git am --keep ../verl-ascend-recipe/Kimi-K3/patch/fsdpturbo.patch && cd ..
cd vllm-ascend && git am --keep ../verl-ascend-recipe/Kimi-K3/patch/vllm-ascend.patch && cd ..

# 准备多机脚本并修改ROOT、MODEL_PATH_DIR、DATASET_DIR后启动训练
cp verl-ascend-recipe/Kimi-K3/examples/ray_start_4nodes_a5.sh .
bash ray_start_4nodes_a5.sh
```

## 4. 可选: 异步训练功能修复

默认流程只应用 `Kimi-K3/patch/verl.patch`。若需使用异步（separate async）训练，需在应用 `verl.patch` 之后追加应用 `Kimi-K3/patch/verl-separate-async.patch`，该 patch 修复了以下问题：

- **权重同步**：为 FSDPTurbo（`fsdp_turbo`）补齐 CPU sharded 权重的保存与恢复路径，使异步模式下 Actor 与 Rollout 的权重同步可用。
- **Kimi 专家权重流式传输**：新增 `kimi_expert_stream.py`，按有界 tile 分块传输专家权重，避免一次性聚合整份 packed 参数带来的显存与内存峰值。
- **分片布局校验**：`fsdp2_sharded_save_to_cpu` / `fsdp2_sharded_load_from_cpu` 改为按参数各自的 DTensorSpec 校验，支持 FSDP 与 EP/eFSDP 混合布局。
- **HCCL 建链端口竞争**：预留监听 socket 并交给 TCPStore 复用，修复异步初始化阶段端口被抢占导致的建链失败。

```bash
# A3：在已应用 verl.patch 的 verl 目录中执行
git -C "$WORK/verl" -c user.name='A3 setup' -c user.email='a3-setup@localhost' am --keep "$WORK/verl-ascend-recipe/Kimi-K3/patch/verl-separate-async.patch"

# A5：在已应用 verl.patch 的 verl 目录中执行
cd verl && git am --keep ../verl-ascend-recipe/Kimi-K3/patch/verl-separate-async.patch && cd ..
```

## 5. 训练效果

以下展示 Kimi K3 16 专家裁剪模型在 Ascend A3 和 A5 上的 GRPO 训练结果。

### 5.1 Ascend A3：4 机、16 专家训练结果

| 配置项 | 配置说明 |
| --- | --- |
| 计算资源 | 4 机，每机 16 个 NPU，共 64 个 NPU |
| 模型 | Kimi K3 16 专家裁剪模型（`K3-top16-pruned_bf16_countbench_128_n4`） |
| 训练步数 | 200 steps|
| 批量与采样 | `TRAIN_BATCH_SIZE=16`，`ROLLOUT_N=4`，`PPO_MINI_BATCH_SIZE=16` |
| 训练并行 | 专家并行 `TURBO_EP_SIZE=16`，上下文并行 `TURBO_CP_SIZE=1` |
| Rollout 并行 | 张量并行 `ROLLOUT_TP=16`，数据并行 `ROLLOUT_DP=1` |
| 最大回答长度 | 2048 tokens |

![Kimi K3 A3训练结果](src/kimi_k3_200steps_a3.png)

### 5.2 Ascend A5：4 机、16 专家训练结果

| 配置项 | 配置说明 |
| --- | --- |
| 计算资源 | 4 机，每机 8 个 NPU，共 32 个 NPU |
| 模型 | Kimi K3 16 专家裁剪模型 |
| 训练步数 | 200 steps |
| 批量与采样 | `TRAIN_BATCH_SIZE=16`，`ROLLOUT_N=4`，`PPO_MINI_BATCH_SIZE=16` |
| 训练并行 | 专家并行 `TURBO_EP_SIZE=8`，上下文并行 `TURBO_CP_SIZE=1` |
| Rollout 并行 | 张量并行 `ROLLOUT_TP=8`，数据并行 `ROLLOUT_DP=1` |
| 最大回答长度 | 2048 tokens |

![Kimi K3 A5训练结果](src/kimi_k3_200steps_a5.png)
