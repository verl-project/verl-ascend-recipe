#!/bin/bash
set -ex
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install vllm v0.23.0 from source"
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git
cd vllm && python use_existing_torch.py --prefix && pip install -r requirements/build/cuda.txt
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e .
cd ..

echo "2. install vllm-ascend from source"
git clone -b releases/v0.23.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn 
cd ..

echo "3.install mbridge"
git clone -b v0.15.1 https://github.com/ISEEKYAN/mbridge.git 
cd mbridge 
pip install -e . 
cd ..

echo "4.install verl"
git clone https://github.com/verl-project/verl.git
cd verl && git checkout 809f2d8f
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e .
cd ..

echo "5.install MindSpeed & MindSpeed-LLM & Megatron"
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
pip3 install -r requirements.txt 
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
cd ..

git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
cd MindSpeed-LLM
cp pretrain_deepseek4.py mindspeed_llm
pip3 install -r requirements.txt
cd ..

echo "6.update triton-ascend && transformers"
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install transformers==5.8.1

echo "7.apply patch"
cd Megatron-LM
git apply --whitespace=nowarn ../verl-ascend-recipe/DeepSeek-V4-Flash/patch/megatron.patch && cd ..

cd mbridge
git apply --whitespace=nowarn ../verl-ascend-recipe/DeepSeek-V4-Flash/patch/mbridge.patch && cd ..

cd vllm-ascend
git apply --whitespace=nowarn ../verl-ascend-recipe/DeepSeek-V4-Flash/patch/vllm-ascend.patch && cd ..

cd verl
git apply --whitespace=nowarn ../verl-ascend-recipe/DeepSeek-V4-Flash/patch/verl.patch && cd ..

cd vllm
git apply --whitespace=nowarn ../verl-ascend-recipe/DeepSeek-V4-Flash/patch/vllm.patch && cd ..

cd MindSpeed-LLM
git apply --whitespace=nowarn ../verl-ascend-recipe/DeepSeek-V4-Flash/patch/mindspeed-llm.patch && cd ..


