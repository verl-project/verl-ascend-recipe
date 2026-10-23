# DeepSeek-V4 on Ascend NPU
基于DeepSeek-V4-Flash模型在NPU上进行RLHF后训练的样例。

本用例基于8 x Atlas A3 实现， 开发者可以参照调整。

## 环境版本
由于当前部分组件依赖尚未发布正式版本，以下基础镜像及版本用于快速复现；升级镜像时请同步核对下方镜像摘要和依赖版本。主要依赖版本如下

| 依赖组件                 | 版本            | 
| :--------------------- | :------         | 
| CANN                   | 9.1.0           | 
| PyTorch                | 2.10.0          | 
| torch\_npu             | 2.10.0.post4    |
| Python                 | 3.12             |
| verl                   | 809f2d8         | 
| vLLM                   | v0.23.0         | 
| vLLM-Ascend            | releases/0.23.0 | 
| MindSpeed-LLM          | repository default branch (not commit-pinned by install.sh) |
| MindSpeed              | repository default branch (not commit-pinned by install.sh) |
| Megatron               | core_v0.12.1    | 


`install.sh` 对 MindSpeed 与 MindSpeed-LLM 执行默认分支克隆、不固定 commit；表中的 default branch 即克隆时的仓库默认分支。若需复现同一份代码，请记录安装后各仓库的 `git rev-parse HEAD`。

### 环境部署
我们基于VLLM+MindSpeed-LLM后端支持DeepSeekV4的强化学习。使用与A3、CANN 9.1.0和Python 3.12匹配的Quay镜像作为基础环境；以下固定到2026-10-22发布的镜像摘要以便复现。升级镜像时应同时核对依赖版本。

```bash

# Quay A3 / CANN 9.1.0 / torch_npu 2.10.0.post4 / Python 3.12 镜像（2026-10-22 摘要）
IMAGE='quay.io/ascend/verl:v0.9.1-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12-vllm@sha256:975e7a8a38d693ce97330c047b341add8fdb20c9e3b968d918618062fe110984'
docker pull "$IMAGE"

# 创建容器
docker run -dit --ipc=host --network host --name 'rl_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data "$IMAGE" /bin/bash

# 进入容器
docker exec -it rl_test bash
mkdir /workspace-verl 
cd /workspace-verl

# 下载并安装新版cann环境
./Ascend-cann-toolkit_9.1.0_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./Ascend-cann-nnal_9.1.0_linux-aarch64.run --install
./Ascend-cann-A3-ops_9.1.0_linux-aarch64.run --install

# 安装环境依赖
git clone https://github.com/verl-project/verl-ascend-recipe.git
bash verl-ascend-recipe/DeepSeek-V4-Flash/install.sh

# 创建软链接
cd verl
ln -s ../MindSpeed/mindspeed mindspeed
ln -s ../MindSpeed-LLM/mindspeed_llm mindspeed_llm
ln -s ../Megatron-LM/megatron megatron
ln -s ../mbridge/mbridge mbridge
```

### 权重下载与反量化

1. 权重下载

    从 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) 下载权重和配置文件

2. 权重转换

    开源DeepSeekV4-Flash权重为FP8 mixed数据格式，使用A3训练前需要对原始权重做反量化后获得bf16格式的权重，反量化方法请参考下述脚本
    ```bash
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_fp8_to_bf16.sh
    ```

3. 减层（4层）配置生成
    
    仅需修改模型权重路径下的 config.json 的 num_hidden_layers 对应数值改成 4

### 启动训练
请根据实际数据/权重等路径修改ray_start.sh 以及 train_deepseek_v4_grpo_mindspeed_vllm.sh的中相应路径
```bash
cd verl
bash ../verl-ascend-recipe/DeepSeek-V4-Flash/examples/ray_start.sh
```

单机减层可参考[train_deepseek_v4_4layer_grpo_mindspeed_vllm_single_node.sh](examples/train_deepseek_v4_4layer_grpo_mindspeed_vllm_single_node.sh)

### 训练效果

长跑250步训练效果曲线示例：

![deepseekv4](./src/run_deepseek_v4_npu.jpg)