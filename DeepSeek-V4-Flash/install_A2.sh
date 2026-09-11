#!/bin/bash
# DeepSeek-V4-Flash GRPO on Ascend A2 (910B) 环境安装脚本
# 与 install.sh(A3) 的差异：
#   1. patch 使用 A2_patch/ 目录（与 patch/ 为替代关系，禁止叠加）
#   2. MindSpeed / MindSpeed-LLM 锁定 commit（对齐 A2 验证镜像，禁止漂 master）
#   3. mbridge 在打 patch 后再 pip install（A2 patch 会改 pyproject.toml 依赖）
set -ex
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

# A2_patch 目录（按脚本自身路径解析，与执行 cwd 无关）
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/A2_patch"

# A2 验证镜像锁定的 commit（对齐 cann9.1.0 npu deepseekv4 rl 镜像构建时点）
MINDSPEED_COMMIT=${MINDSPEED_COMMIT:-"a15490cd75453201ec02fdb18787ba33ae773ce3"}
MINDSPEED_LLM_COMMIT=${MINDSPEED_LLM_COMMIT:-"4afa739149dc3a275e5516df044f0a63b9076cf0"}

echo "1. install vllm v0.23.0 from source"
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git
cd vllm && python use_existing_torch.py --prefix && pip install -r requirements/build/cuda.txt
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e .
cd ..

echo "2. install vllm-ascend releases/v0.23.0 from source + A2 patch"
git clone -b releases/v0.23.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git apply --whitespace=nowarn ${PATCH_DIR}/vllm-ascend-rope-original-seq-len.patch
git apply --whitespace=nowarn ${PATCH_DIR}/vllm-ascend-camem-sleep-sync.git.patch
git apply --whitespace=nowarn ${PATCH_DIR}/vllm-ascend-dsv4-gate-fp32-inplace.patch
pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..

echo "3. install mbridge v0.15.1 + A2 patch（先打 patch 再安装，pyproject 依赖有变化）"
git clone -b v0.15.1 https://github.com/ISEEKYAN/mbridge.git
cd mbridge
git apply --whitespace=nowarn ${PATCH_DIR}/mbridge_deepseek_v4_optimized.patch
pip install -e .
cd ..

echo "4. install verl @809f2d8f + A2 patch"
git clone https://github.com/verl-project/verl.git
cd verl && git checkout 809f2d8f
git apply --whitespace=nowarn ${PATCH_DIR}/verl.patch
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e .
cd ..

echo "5. install MindSpeed & MindSpeed-LLM & Megatron（锁定 commit）"
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout ${MINDSPEED_COMMIT}
pip3 install -r requirements.txt
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
git apply --whitespace=nowarn ${PATCH_DIR}/megatron.patch
git apply --whitespace=nowarn ${PATCH_DIR}/megatron-expert-bias-keep-fp32.patch
cd ..

git clone https://gitcode.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout ${MINDSPEED_LLM_COMMIT}
cp pretrain_deepseek4.py mindspeed_llm
pip3 install -r requirements.txt
cd ..

echo "6. update triton-ascend && transformers"
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install transformers==5.8.1

echo ""
echo "================================================================"
echo "安装完成。各仓库克隆在当前工作目录: $(pwd)"
echo "启动训练前，请确保 ray_start.sh 的 VERL_WORKSPACE 及"
echo "runtime_env yaml 中的路径与此目录一致，例如："
echo "  export VERL_WORKSPACE=$(pwd)"
echo "================================================================"
