# Qwen3-30B-A3B mxfp8 on Ascend NPU
基于Qwen3-30B-A3B模型在NPU上进行RLHF mxfp8后训练的样例。

本用例基于 4 x Atlas A5 实现， 开发者可以参照调整。

## 环境版本
由于当前部分组件依赖尚未发布正式版本，我们将提供用于快速复现的基础镜像及部署方法，获取参照环境部署章节，主要依赖版本如下
后续会更新正式版本

| 依赖组件                | 版本            |
| :--------------------- | :------         |
| CANN                   | 9.2.1           |
| Python                 | 3.12            | 
| torch                  | 2.10.0          | 
| torch\_npu             | 2.10.0.post7    |
| torchvision            | 0.25.0          |
| torchaudio             | 2.10.0          |
| triton-ascend          | 3.6.0.post2     |
| transformers           | 5.14.1          |
| xgrammar               | >=0.2.1,<1.0.0  |
| vLLM                   | 0.30.0          | 
| Megatron-LM            | core_v0.18.0    |
| MindSpeed              | core_r0.18.0    |
| verl                   | 07ba1a85(main)  |
| vLLM-Ascend            | 59ebc438(0.30.0)|


### 环境部署
我们基于vLLM+Megatron后端在A5上支持Qwen3-30B-A3B mxfp8的强化学习
采用的镜像为
```bash
docker pull quay.io/ascend/verl:latest-vllm-950-ubuntu
```
若要与A3进行比较，用2台A3机器
则采用的镜像为
```bash
docker pull quay.io/ascend/verl:latest-vllm-a3-ubuntu
```

```bash
# 获取环境依赖
git clone https://github.com/verl-project/verl-ascend-recipe.git

# 然后执行安装步骤
bash verl-ascend-recipe/low_precision/Qwen3-30B-A3B/install.sh
```

### 权重下载与quant_model_description.json文件生成

1. 权重下载

    从 [huggingface](https://huggingface.co/Qwen/Qwen3-30B-A3B) 下载权重和配置文件

2. quant_model_description.json文件生成

```bash
python /verl-ascend-recipe/low_precision/tool/smart_create_quant_weight_dir.py /model/Qwen3-30B-A3B /model/Qwen3-30B-A3B-MXFP8
```

就会生成一个Qwen3-30B-A3B-MXFP8文件夹以及需要的quant_model_description.json


### 启动训练

请根据实际数据/权重等路径修改 ray_start.sh 以及 run_grpo_qwen3_moe_30b_megatron_32k.sh 的中相应路径

```bash
cd /verl
bash ../verl-ascend-recipe/low_precision/Qwen3-30B-A3B/examples/ray_start.sh
```
