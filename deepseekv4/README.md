# DeepSeek-V4 on Ascend NPU
本recipe是基于DeepSeek-V4-Flash-Base模型在NPU上进行RLHF后训练的样例，基于GRPO与规则奖励，使用gsm8k数据集。

本用例基于8 x Atlas A3 实现， 开发者可以参照调整。

## 环境版本
由于当前部分组件依赖尚未发布正式版本，我们将提供用于快速复现的基础镜像及部署方法，获取参照环境部署章节，主要依赖版本如下
后续会更新正式版本

| 组件                     | 版本      | 
| :--------------------- | :------ | 
| PyTorch                | 2.8.0   | 
| torch\_npu             | 2.8.0.post2   | 
| verl                   | 809f2d8 | 
| vLLM           | 0.13.0   | 
| vLLM-Ascend            | 0.13.0rc3   | 
| MindSpeed-LLM          |62c42653  | 
| MindSpeed          |6ce32f57  | 
| Megatron          |core_v0.12.1  | 


### 环境部署
我们基于VLLM+MindSpeed-LLM后端支持DeepSeekV4的强化学习, 使用MindSpeed-LLM的开源镜像作为基础镜像，请使用此镜像作为基础镜像安装环境

镜像链接 https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/deepseek_train/dsv4_train_mindspeed_v1.0.tar.gz

```bash

# 创建容器
docker run -dit --ipc=host --network host --name 'rl_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ -v /data/:/data 镜像名:标签 /bin/bash

# 进入容器
docker exec -it rl_test bash
mkdir /workspace-verl 
cd /workspace-verl

# 安装vllm推理及verl补丁环境
# 注：当前该镜像已在默认目录/usr/local/Ascend/cann安装CANN与自定义融合算子包, 但vllm-ascend依赖CANN 8.5.0版本进行编译安装，您需要首先在其他自定义路径安装CANN 8.5.0编译vllm-ascend，以下安装脚本默认CANN_INSTALL_PATH=/usr/local/Ascend-8.5.0，您可自行更改为自定义路径, 后续推理版本迭代将解除此限制
git clone https://github.com/verl-project/verl-ascend-recipe.git
bash verl-ascend-recipe/deepseekv4/scripts/install.sh

# 创建软链接
cd verl
ln -s ../MindSpeed/mindspeed mindspeed
ln -s ../MindSpeed-LLM/mindspeed_llm mindspeed_llm
ln -s ../Megatron-LM/megatron megatron
```

### 下载数据集
```bash
#GSM8k:
python3 examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
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

### 启动训练
请根据实际数据/权重等路径修改ray_start.sh 以及 train_deepseek_v4_grpo_mindspeed_vllm.sh的中相应路径
```bash
cd verl
bash ../verl-ascend-recipe/deepseekv4/scripts/ray_start.sh
```
