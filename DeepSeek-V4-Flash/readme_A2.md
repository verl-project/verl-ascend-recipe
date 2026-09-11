# DeepSeek-V4 on Ascend NPU (A2)
基于DeepSeek-V4-Flash模型在NPU上进行RLHF后训练的样例。

本用例基于 8 x 16 Atlas A2 (910B) 实现， 开发者可以参照调整。

## 环境版本
由于当前部分组件依赖尚未发布正式版本，我们将提供用于快速复现的基础镜像及部署方法，获取参照环境部署章节，主要依赖版本如下
后续会更新正式版本

| 依赖组件                 | 版本              | 
| :--------------------- | :------           | 
| CANN                   | 9.1.0             | 
| PyTorch                | 2.10.0            | 
| torch\_npu             | 2.10.0.post2      | 
| verl                   | 809f2d8           | 
| vLLM                   | v0.23.0           | 
| vLLM-Ascend            | releases/0.23.0   | 
| MindSpeed-LLM          | 4afa7391          | 
| MindSpeed              | a15490cd          | 
| Megatron               | core_v0.12.1      |


### 环境部署
我们基于VLLM+MindSpeed-LLM后端在A2(910B)上支持DeepSeekV4的强化学习, 使用verl日构建开源镜像作为基础镜像，请使用此镜像作为基础镜像安装环境

```bash

# verl日构建开源镜像
docker pull quay.io/ascend/verl:verl-9.0.0-a3-ubuntu22.04-py3.11-latest

# 创建容器
docker run -dit --ipc=host --network host --name 'rl_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data 镜像名:标签 /bin/bash

# 进入容器
docker exec -it rl_test bash
mkdir /workspace-verl 
cd /workspace-verl

# 下载并安装新版cann环境（按机器架构选择 linux-aarch64 或 linux-x86_64 安装包）
./Ascend-cann-toolkit_9.1.0_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./Ascend-cann-nnal_9.1.0_linux-aarch64.run --install

# 安装环境依赖（A2 使用 A2_patch/，与 patch/ 为替代关系，禁止与 install.sh 混用叠加）
# 注意：install_A2.sh 会把各仓库克隆到当前工作目录，请固定在一个统一目录下执行（示例为 /workspace-verl），
# 后续 ray_start.sh 的 VERL_WORKSPACE 与 runtime_env_A2.yaml 中的路径都需与该目录保持一致
git clone https://github.com/verl-project/verl-ascend-recipe.git
bash verl-ascend-recipe/DeepSeek-V4-Flash/install_A2.sh

# 创建软链接
cd verl
ln -s ../MindSpeed/mindspeed mindspeed
ln -s ../MindSpeed-LLM/mindspeed_llm mindspeed_llm
ln -s ../Megatron-LM/megatron megatron
ln -s ../mbridge/mbridge mbridge
```

A2_patch 说明：

| patch | 目标仓库 | 作用 |
| :---- | :------ | :-- |
| megatron.patch | Megatron-LM | 与 patch/ 中 A3 版本相同（p2p 通信 / config） |
| megatron-expert-bias-keep-fp32.patch | Megatron-LM | expert_bias buffer 保持 fp32，避免 mbridge 加载 gate.bias 精度损失 |
| mbridge_deepseek_v4_optimized.patch | mbridge | 独立 DeepseekV4Bridge + memory efficient 加载 + GEMM 优化 |
| vllm-ascend-rope-original-seq-len.patch | vllm-ascend | rope 按 original_seq_len 计算 inv_freq |
| vllm-ascend-camem-sleep-sync.git.patch | vllm-ascend | sleep unmap 前 torch.npu.synchronize，规避 SMMU/MTE 507011 |
| vllm-ascend-dsv4-gate-fp32-inplace.patch | vllm-ascend | RL 更新权重时 gate fp32 权重原地 copy |
| verl.patch | verl | seqlen 2048 对齐 / NPU sleep level=1 / bucket 传输对齐等 |

注意：A2_patch 与 patch/ 为替代关系，禁止与 install.sh(A3) / install_A5.sh 的 patch 叠加混用。

### 权重下载与反量化

1. 权重下载

    从 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) 下载权重和配置文件

2. 权重转换

    开源DeepSeekV4-Flash权重为FP8 mixed数据格式，训练前需要对原始权重做反量化后获得bf16格式的权重，反量化方法请参考下述脚本
    ```bash
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_fp8_to_bf16.sh
    ```

### 启动训练

多机样例 [train_deepseek_v4_grpo_mindspeed_vllm_8node_A2.sh](examples/train_deepseek_v4_grpo_mindspeed_vllm_8node_A2.sh)：8节点 x 16 NPU、全层 43 层、2K prompt + 20K response。

请根据实际数据/权重等路径修改ray_start.sh 以及 train_deepseek_v4_grpo_mindspeed_vllm_8node_A2.sh的中相应路径（ray_start.sh 中需修改主节点 IP 与各节点通信网卡）
```bash
cd verl
bash ../verl-ascend-recipe/DeepSeek-V4-Flash/examples/ray_start.sh ../verl-ascend-recipe/DeepSeek-V4-Flash/examples/train_deepseek_v4_grpo_mindspeed_vllm_8node_A2.sh
```

ray_start.sh 在各节点分别执行一次：主节点启动 ray head 并等待 8 x 16 NPU 全部注册后自动拉起训练脚本，其余节点注册进集群后保持待命。

训练脚本通过 `ray job submit` 提交，默认使用同目录 [runtime_env_A2.yaml](examples/runtime_env_A2.yaml) 注入 CANN / HCCL 等环境变量。yaml 与 ray_start.sh 中的仓库路径默认按 `/workspace-verl`（readme 环境部署章节的安装目录）编写：若 install_A2.sh 在其他目录执行，请同步修改 yaml 的 PYTHONPATH / LD_LIBRARY_PATH / ASCEND_CUSTOM_OPP_PATH，并在启动前 `export VERL_WORKSPACE=<实际安装目录>`。非 root 用户请把 yaml 中 /root 替换为实际 HOME。如需完全自定义 env，通过 RUNTIME_ENV 指定其他 runtime env yaml：

```bash
export RUNTIME_ENV=/path/to/your/runtime_env.yaml
```

并行度默认配置（可通过环境变量覆盖）：训练 tp2 x pp4 x cp4 x ep32（kvallgather_cp_algo）；rollout gen_tp8 x gen_dp8 x gen_ep64，2 replica。其余可调项见脚本头部注释。
