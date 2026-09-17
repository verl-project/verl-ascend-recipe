#!/bin/bash
set -ex

echo "1. install vllm"
pip install vllm==0.23.0

echo "2. install vllm-ascend"
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout releases/v0.23.0
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
cd ..

echo "3. install Megatron-LM"
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
pip install -v -e .
cd ..

echo "4. install MindSpeed"
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
pip install -r requirements.txt 
pip install -e .
cd ..

echo "5. install verl"
git clone https://github.com/verl-project/verl
cd verl
sed -i '/^triton-ascend==/ s/^/# /' requirements-npu.txt
pip install -r requirements-npu.txt
cd ..

echo "6. install mbridge && triton-ascend && transformers"
pip install mbridge
pip install transformers==4.57.6
pip install triton-ascend==3.2.2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn

echo "7. apply patch"
cd Megatron-LM
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/Qwen3-30B-A3B/patch/megatron.patch
cd ..

echo "8. install MindStudio-Boost"
pip install psutil
dnf install numactl
git clone -b verl https://gitcode.com/xnm-s83u/msboost_1088.git
