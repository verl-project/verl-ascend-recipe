#!/bin/bash
# Qwen3.5-35B-A3B MXFP8 GRPO（FSDPTurbo 长跑版）环境安装脚本
# 用法: bash install.sh（在目标工作目录执行，依赖仓会克隆到当前目录）
set -ex

RECIPE_ROOT="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="${RECIPE_ROOT}/patch"

echo "1. install vllm"
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout ced6857
VLLM_TARGET_DEVICE=empty pip install -e . --no-deps --no-build-isolation
cd ..

echo "2. install vllm-ascend"
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout c09d3e2
git apply "${PATCH_DIR}/vllm-ascend.patch"
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
pip install -e . --no-deps --no-build-isolation
cd ..

echo "3. install FSDPTurbo"
git clone https://gitcode.com/Ascend/FSDPTurbo.git
cd FSDPTurbo
git checkout 759a348
git apply "${PATCH_DIR}/fsdpturbo.patch"
pip install -e .
cd ..

echo "4. install verl"
git clone https://github.com/verl-project/verl.git
cd verl
git checkout 7454326
git apply "${PATCH_DIR}/verl.patch"
sed -i '/^triton-ascend==/ s/^/# /' requirements-npu.txt
pip install -r requirements-npu.txt
cd ..

echo "5. install triton-ascend and transformers"
pip install transformers==5.10.4
pip install triton-ascend==3.2.2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn

echo "6. check required features (hard fail)"
fail=0
if ! grep -rq --exclude-dir=.git "AscendMxfp8WeightQuantizer" verl/; then
    echo "ERROR: 未在 verl 中找到 AscendMxfp8WeightQuantizer。"
    fail=1
fi
if ! grep -rq --exclude-dir=.git "mx_fused_swiglu_group_gemm" FSDPTurbo/; then
    echo "ERROR: 未在 FSDPTurbo 中找到 mx_fused_swiglu_group_gemm。"
    fail=1
fi
if ! grep -rq --exclude-dir=.git "VLLM_ASCEND_ONLINE_WEIGHT_QUANT" vllm-ascend/; then
    echo "ERROR: 未在 vllm-ascend 中找到 VLLM_ASCEND_ONLINE_WEIGHT_QUANT。"
    fail=1
fi
if [ "$fail" -ne 0 ]; then
    echo "install 自检失败，已中止。"
    exit 1
fi
echo "自检通过：verl / FSDPTurbo / vllm-ascend 长跑补丁均在位。"

echo "7. env"
echo "请确保 PYTHONPATH 包含 FSDPTurbo: export PYTHONPATH=\$PWD/FSDPTurbo:\$PYTHONPATH"
