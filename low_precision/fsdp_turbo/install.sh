#!/bin/bash
# Qwen3.5-35B-A3B MXFP8 GRPO（FSDPTurbo 链路）环境安装脚本
# 用法：在 recipe 仓的父目录执行（依赖仓克隆到当前目录；目录布局与成因说明见 install-guide.md）
set -ex

echo "0. mirrors (opt-in: USE_CN_MIRROR=1)"
if [ "${USE_CN_MIRROR:-0}" = "1" ]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.https://ghfast.top/https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple/}"
  export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://triton-ascend.osinfra.cn/pypi/simple/}"
fi
FSDPTURBO_URL="${FSDPTURBO_URL:-https://gitcode.com/Ascend/FSDPTurbo.git}"

echo "1. vllm-ascend requirements"
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 59ebc4384b340065127e9b1e1bb73a257c51895e
pip install -r requirements.txt
pip install "setuptools-rust>=1.9.0"
cd ..

echo "2. vllm editable + runtime deps (conflict-driven)"
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout ced6857
VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation

pip install --no-deps "fastapi==0.123.10" "starlette==0.50.0" "prometheus-fastapi-instrumentator==8.1.0" "numpy==1.26.4"
python -c "import numpy; assert numpy.__version__ == '1.26.4', 'numpy face drifted: ' + numpy.__version__; print('numpy face OK:', numpy.__version__)"
cd ..

echo "3. vllm-ascend (patch + editable)"
cd vllm-ascend
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/patch/vllm-ascend-attention.patch
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/fsdp_turbo/patch/vllm-ascend-scalefix.patch
export COMPILE_CUSTOM_KERNELS=1
pip install -e . --no-deps --no-build-isolation
cd ..

echo "4. FSDPTurbo"
git clone "$FSDPTURBO_URL"
cd FSDPTurbo
# db33a840 = 官方 main（2026-11-20 tip）；NPU 走 AscendC 真核路径，head_first 由 chunk_gdn_ascendc 原生接参处理，不依赖未合入的 GDN MR #367
git checkout db33a84024bc263bb95f06e1353caf992060d4fe
pip install -e .
cd ..

echo "5. verl"
git clone https://github.com/verl-project/verl.git
cd verl
git checkout 07ba1a854
sed -i '/^triton-ascend==/ s/^/# /' requirements-npu.txt
pip install -r requirements-npu.txt
pip install -e . --no-deps
cd ..

echo "6. verl patches"
cd verl
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/patch/verl.patch
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/fsdp_turbo/patch/verl-fsdp-turbo.patch
cd ..

echo "7. triton-ascend && transformers"
pip install transformers==5.22.2
pip install triton-ascend==3.6.0.post2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn

echo "8. NPU kernel-layer deps (fla / fla_npu / torchao / torchao_npu / mindspeed_ops)"
pip install "fla-core==0.5.2" "flash-linear-attention==0.5.2"
pip install "flash-linear-attention-npu-a5==26.9.1" --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
pip install "torchao==0.18.0" --no-deps
git clone https://gitcode.com/Ascend/torchao-ascend.git
cd torchao-ascend
git checkout a92753dee6e6cc625eca1f5943f0ec07459bfab0
pip install -e . --no-deps
cd ..
git clone https://gitcode.com/ascend/MindSpeed-Ops.git
cd MindSpeed-Ops
git checkout b1ef3adcfc2f0a2932d6b2eb831f275fe1daacfb
pip install -e . --no-deps --no-build-isolation
cd ..

echo "9. check required features (hard fail)"
fail=0
if ! grep -rq --exclude-dir=.git "translate_weight_names_for_vllm" verl/; then
    echo "ERROR: 未在 verl 中找到 translate_weight_names_for_vllm（量化权重名映射失效）"
    fail=1
fi
if ! grep -rq --exclude-dir=.git "quant_ignored_modules" FSDPTurbo/; then
    echo "ERROR: 未在 FSDPTurbo 中找到 quant_ignored_modules（linear_attn 排除不生效）"
    fail=1
fi
if [ "$fail" -ne 0 ]; then
    echo "install 自检失败，已中止（版本核对见 readme.md「版本依赖」节）。"
    exit 1
fi
echo "自检通过：translate_weight_names_for_vllm / quant_ignored_modules 均在位。"

SMOKE_DIR=$(mktemp -d)
trap 'rm -rf "$SMOKE_DIR"' EXIT
(cd "$SMOKE_DIR" && python -c "import vllm; print('vllm import smoke OK:', vllm.__version__)")

echo "10. done"
echo "训练启动前：export PYTHONPATH=\$PWD/FSDPTurbo:\$PYTHONPATH（追加式）"
