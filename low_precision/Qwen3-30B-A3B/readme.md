# Qwen3-30B-A3B mxfp8 on Ascend NPU
基于Qwen3-30B-A3B模型在NPU上进行RLHF mxfp8后训练的样例。

本用例基于 2 x Atlas A5 实现， 开发者可以参照调整。

## 环境版本
由于当前部分组件依赖尚未发布正式版本，我们将提供用于快速复现的基础镜像及部署方法，获取参照环境部署章节，主要依赖版本如下
后续会更新正式版本

| 依赖组件                | 版本            | 
| :--------------------- | :------         | 
| CANN                   | 9.2.0           | 
| PyTorch                | 2.10.0          | 
| torch\_npu             | 2.10.0.post4    | 
| verl                   | main            | 
| vLLM                   | 0.23.0          | 
| vLLM-Ascend            | 0.23.0          | 
| MindSpeed              | master          | 
| Megatron               | core_v0.12.1    | 
| triton\_ascend         | 3.2.2           |
| transformers           | 4.57.6          |


### 环境部署
我们基于vLLM+Megatron后端在A5上支持Qwen3-30B-A3B mxfp8的强化学习

```bash
conda create -n verl-npu python=3.12 -y
conda activate verl-npu

# 获取环境依赖
git clone https://github.com/verl-project/verl-ascend-recipe.git

# 首先根据实际cann的安装路径source cann
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/cann/set_env.sh

# 然后执行安装步骤
bash verl-ascend-recipe/low_precision/Qwen3-30B-A3B/install.sh

# 创建软链接
cd verl
ln -s ../MindSpeed/mindspeed mindspeed
ln -s ../Megatron-LM/megatron megatron
```

### 权重下载与quant_model_description.json文件生成

1. 权重下载

    从 [huggingface](https://huggingface.co/Qwen/Qwen3-30B-A3B) 下载权重和配置文件

2. quant_model_description.json文件生成

```bash
python /verl-ascend-recipe/low_precision/Qwen3-30B-A3B/tool/smart_create_mxfp8_weight_dir.py /model/Qwen3-30B-A3B /model/Qwen3-30B-A3B-MXFP8
```

就会生成一个Qwen3-30B-A3B-MXFP8文件夹以及需要的quant_model_description.json


### 启动训练

请根据实际数据/权重等路径修改 ray_start.sh 以及 run_grpo_qwen3_moe_30b_megatron.sh 的中相应路径

```bash
cd verl
bash ../verl-ascend-recipe/low_precision/Qwen3-30B-A3B/examples/ray_start.sh
```

并在上述脚本训练开始第一步时进行绑核操作，可提高性能
```bash
cd /msboost_1088/affinity-sched
export PYTHONPATH=$PWD

# 1. 执行 -d 观察调度策略是否正确
python examples/affinity_verl.py -tp-size 2 -dp-size 4 -d

# 2. 执行 -r 使能自适应亲和调度
python examples/affinity_verl.py -tp-size 2 -dp-size 4 -r

# 3. 若不符合预期，可执行 -restore 重置绑核操作，恢复亲和性到初始状态
python examples/affinity_verl.py -tp-size 2 -dp-size 4 -restore
```
