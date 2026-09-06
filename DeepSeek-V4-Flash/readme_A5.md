# DeepSeek-V4 on Ascend NPU
基于DeepSeek-V4-Flash减层模型在NPU上进行RLHF后训练的样例。

本用例基于8 x Atlas A5 实现， 开发者可以参照调整。

## 环境版本
由于当前部分组件依赖尚未发布正式版本，我们将提供用于快速复现的基础镜像及部署方法，获取参照环境部署章节，主要依赖版本如下
后续会更新正式版本

| 依赖组件                 | 版本            | 
| :--------------------- | :------         | 
| CANN                   | 9.2.0           | 
| PyTorch                | 2.10.0          | 
| torch\_npu             | 2.10.0.post4    | 
| verl                   | 809f2d8         | 
| vLLM                   | v0.27.1         | 
| vLLM-Ascend            | 343743a (master)| 
| MindSpeed-LLM          | a83d51e (master)| 
| MindSpeed              | 4fb7dc0 (master)| 
| Megatron               | core_v0.12.1    | 


### 环境部署
我们基于VLLM+MindSpeed-LLM后端在A5上支持DeepSeekV4的强化学习

```bash

conda create -n verl-npu python=3.11 -y
conda activate verl-npu

# 获取环境依赖
git clone https://github.com/verl-project/verl-ascend-recipe.git

# 首先根据实际cann的安装路径source cann
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/cann/set_env.sh

# 然后执行安装步骤
bash verl-ascend-recipe/DeepSeek-V4-Flash/install_A5.sh

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

目前提供减层脚本train_deepseek_v4_4layer_grpo_mindspeed_vllm_single_node_A5.sh，可用单机A5进行体验，请根据实际数据/权重等路径修改其中相应路径

```bash
cd verl
bash ../verl-ascend-recipe/DeepSeek-V4-Flash/examples/train_deepseek_v4_4layer_grpo_mindspeed_vllm_single_node_A5.sh
```

### 性能效果

![deepseekv4-4layer-a5](./src/run_deepseek_v4_4layer-a5.PNG)
