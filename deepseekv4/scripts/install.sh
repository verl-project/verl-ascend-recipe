#!/bin/bash
set -ex
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend-8.5.0"}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install vllm v0.13.0 from source"
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git
cd vllm && pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e. && cd ..

echo "2. install vllm-ascend from source"
git clone -b releases/v0.13.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -r requirements.txt    
export COMPILE_CUSTOM_KERNELS=1 && pip install -v -e . && cd ..

echo "3.install mbridge"
git clone -b v0.15.1 https://github.com/ISEEKYAN/mbridge.git 
cd mbridge 
pip install -e . && cd ..

echo "4.install transformers"
git clone -b v4.57.6 https://github.com/huggingface/transformers.git 
cd transformers 
pip install -e . && cd ..

echo "5.install verl"
git clone https://github.com/verl-project/verl.git
cd verl && git checkout 809f2d8f
pip install -r requirements-npu.txt && pip install -v -e . && cd ..

echo "6.install MindSpeed & MindSpeed-LLM & Megatron"
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 6ce32f57  # checkout commit from MindSpeed 
pip3 install -r requirements.txt 
cd ..
git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
cd ..
git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
git checkout 62c42653
cp pretrain_deepseek4.py mindspeed_llm
pip3 install -r requirements.txt  # 安装其余依赖库
cd ..

echo "6.apply patch"
cd Megatron-LM
git apply ../verl-ascend-recipe/deepseekv4/patch/megatron.patch && cd ..
cd mbridge
git apply ../verl-ascend-recipe/deepseekv4/patch/mbridge.patch && cd ..
cd transformers
git apply ../verl-ascend-recipe/deepseekv4/patch/transformers.patch && cd ..
cd verl
git apply ../verl-ascend-recipe/deepseekv4/patch/verl.patch && cd ..
cd vllm-ascend
git apply ../verl-ascend-recipe/deepseekv4/patch/vllm-ascend.patch && cd ..

