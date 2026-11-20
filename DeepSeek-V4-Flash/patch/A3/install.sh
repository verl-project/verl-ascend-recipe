#!/bin/bash
set -ex
PATCH_DIR=$(cd "$(dirname "$0")" && pwd)
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install vllm v0.30.0 from source"
git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm.git
cd vllm
git checkout ced6857afa0ea7b2e3f0846a62e1394e90f15607
python use_existing_torch.py --prefix
python -m pip install -r requirements/build/cuda.txt --user
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e . --user
cd ..

echo "2. install vllm-ascend from source"
git clone -b releases/v0.30.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 2df8721f4ed5d581136d7a5e3dd2487ef5036899
pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..

echo "3.install mbridge"
git clone -b v0.15.1 https://github.com/ISEEKYAN/mbridge.git 
cd mbridge 
git checkout 0cd4ae23f2425da77a80cb3f517828452fa8e984
pip install -e . 
cd ..

echo "4.install verl"
git clone https://github.com/verl-project/verl.git
cd verl
git checkout release/v0.9.0
git checkout 91c297dc8d9991b79c90cad5b50683a5d6d8c02b
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e .
cd ..

echo "5.install MindSpeed & MindSpeed-LLM & Megatron"
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 26.2.0_core_r0.12.1
git checkout 15023c2d0fed38d17f01b5fa7baf4ede63e0dd4e
pip3 install -r requirements.txt 
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
git checkout a845aa7e12b3a117e24c2352b9e3e60bad2e3a17
cd ..

git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
cd MindSpeed-LLM
git checkout 26.2.0
git checkout e4a75c1faa05637496133e1e8ff7c78451fe92e2
cp pretrain_deepseek4.py mindspeed_llm
pip3 install -r requirements.txt
cd ..

echo "6.update transformers"
pip install transformers==5.14.1

echo "7.apply patch"
cd mbridge
for patch in "$PATCH_DIR/mbridge"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..

cd vllm
for patch in "$PATCH_DIR/vllm"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..

cd vllm-ascend
for patch in "$PATCH_DIR/vllm-ascend"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..

cd verl
for patch in "$PATCH_DIR/verl"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..

cd Megatron-LM
for patch in "$PATCH_DIR/megatron"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..

cd MindSpeed
for patch in "$PATCH_DIR/mindspeed"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..

cd MindSpeed-LLM
for patch in "$PATCH_DIR/mindspeed-llm"/*.patch; do
    git apply --check "$patch"
    git apply --whitespace=nowarn "$patch"
done
cd ..
