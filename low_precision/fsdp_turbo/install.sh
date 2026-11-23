#!/usr/bin/env bash
# Install the FSDPTurbo MXFP8 recipe on Ascend A5.
set -Eeuo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work_dir=${WORK_DIR:-"$recipe_dir/../../.."}
phase=preflight
trap 'printf "Installation failed during %s (line %s).\n" "$phase" "$LINENO" >&2' ERR

fail() { printf '%s\n' "$*" >&2; exit 1; }
step() { phase=$1; printf '\n== %s ==\n' "$phase"; }

cann_env=${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
atb_env=${ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}
python_bin=${PYTHON_BIN:-python3.12}
[[ -r $cann_env ]] || fail "CANN setup script is missing: $cann_env. Set CANN_ENV to its full path."
for tool in git gcc g++; do
    command -v "$tool" >/dev/null || fail "Required build tool is missing: $tool"
done
command -v "$python_bin" >/dev/null || fail "Python 3.12 is required. Set PYTHON_BIN to its full path."
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
mkdir -p -- "$work_dir"
cd -- "$work_dir"
work_dir=$PWD
venv_dir=${VENV_DIR:-${VIRTUAL_ENV:-"$work_dir/.venv"}}
[[ $venv_dir = /* ]] || fail "VENV_DIR must be an absolute path."
if [[ ! -x $venv_dir/bin/python ]]; then
    "$python_bin" -m venv "$venv_dir"
fi
export PYTHONNOUSERSITE=1
set +u
source "$cann_env"
if [[ -n ${ATB_ENV:-} || -r $atb_env ]]; then
    source "$atb_env"
fi
source "$venv_dir/bin/activate"
set -u
cd -- "$work_dir"
python -c 'import sys; assert sys.version_info[:2] == (3, 12) and sys.prefix != sys.base_prefix, sys.executable'

triton_index=https://triton-ascend.osinfra.cn/pypi/simple/
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:+$PIP_EXTRA_INDEX_URL }$triton_index"

clone_at() {
    local name=$1 url=$2 rev=$3
    if [[ ! -d $name/.git ]]; then
        git clone "$url" "$name"
        git -C "$name" checkout --detach "$rev"
    else
        [[ $(git -C "$name" config --get remote.origin.url) == "$url" &&
           $(git -C "$name" rev-parse HEAD) == "$rev" ]] ||
            fail "$work_dir/$name already exists with a different origin or revision. Choose a fresh WORK_DIR."
    fi
}

apply_recipe_patches() {
    local repo=$1 patch
    shift
    # The last patch can modify hunks from preceding patches.
    if git -C "$repo" apply --reverse --check "${!#}" 2>/dev/null; then
        return
    fi
    for patch in "$@"; do
        if ! git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
            git -C "$repo" apply --whitespace=nowarn "$patch"
        fi
    done
}

step "Prepare Python build dependencies"
python -m pip install --upgrade pip packaging 'setuptools==80.10.2'
python -m pip install 'cmake>=3.26.1' ninja 'setuptools-scm>=8' 'setuptools-rust>=1.9' wheel jinja2 'torch==2.10.0'

step "Install vLLM"
clone_at vllm https://github.com/vllm-project/vllm.git ced6857afa0ea7b2e3f0846a62e1394e90f15607
# The recipe uses torch 2.10. The upstream isolated build requests torch 2.13.
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e "$work_dir/vllm"

step "Install vLLM-Ascend requirements and build the extension"
clone_at vllm-ascend https://github.com/vllm-project/vllm-ascend.git 59ebc4384b340065127e9b1e1bb73a257c51895e
python -m pip install -r vllm-ascend/requirements.txt
apply_recipe_patches vllm-ascend "$recipe_dir/../patch/vllm-ascend-attention.patch" "$recipe_dir/patch/vllm-ascend-scalefix.patch"
COMPILE_CUSTOM_KERNELS=1 python -m pip install --no-build-isolation --no-deps -e "$work_dir/vllm-ascend"

step "Install FSDPTurbo, verl and training extensions"
clone_at FSDPTurbo "${FSDPTURBO_URL:-https://gitcode.com/Ascend/FSDPTurbo.git}" db33a84024bc263bb95f06e1353caf992060d4fe
python -m pip install -e "$work_dir/FSDPTurbo"

clone_at verl https://github.com/verl-project/verl.git 07ba1a854e57356ccf1c9d604898607a88e3da3e
apply_recipe_patches verl "$recipe_dir/../patch/verl.patch" "$recipe_dir/patch/verl-fsdp-turbo.patch"
# Use the vLLM-Ascend Triton version instead of verl's shared NPU pin.
python -m pip install -r <(sed '/^triton-ascend==/d' verl/requirements-npu.txt)
python -m pip install --no-deps -e "$work_dir/verl"

clone_at torchao-ascend https://gitcode.com/Ascend/torchao-ascend.git a92753dee6e6cc625eca1f5943f0ec07459bfab0
python -m pip install --no-deps 'torchao==0.18.0' -e "$work_dir/torchao-ascend"
clone_at MindSpeed-Ops https://gitcode.com/ascend/MindSpeed-Ops.git b1ef3adcfc2f0a2932d6b2eb831f275fe1daacfb
python -m pip install --no-build-isolation --no-deps -e "$work_dir/MindSpeed-Ops"

step "Restore the recipe runtime versions"
python -m pip install 'torch-npu==2.10.0.post7' 'transformers==5.22.2' 'triton-ascend==3.6.0.post2' 'xgrammar>=0.2.1,<1.0.0'
python -m pip install 'fla-core==0.5.2' 'flash-linear-attention==0.5.2'
PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL https://mirrors.huaweicloud.com/ascend/repos/pypi" \
    python -m pip install --no-deps 'flash-linear-attention-npu-a5==26.9.1'
# These runtime pins deliberately span incompatible metadata from the two vLLM repositories.
python -m pip install --no-deps 'fastapi==0.123.10' 'starlette==0.50.0' \
    'prometheus-fastapi-instrumentator==8.1.0' 'numpy==1.26.4'

step "Verify patches, installed versions and imports"
grep -rq --include="*.py" "def translate_weight_names_for_vllm" "$work_dir/verl/verl"
git -C FSDPTurbo grep -q quant_ignored_modules -- fsdp_turbo/
# Starting from the dependency root can shadow the installed vLLM module.
cd -- "$VIRTUAL_ENV"
export PYTHONPATH="$work_dir/FSDPTurbo${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PYTHON'
from importlib.metadata import version

expected = {
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post7",
    "transformers": "5.22.2",
    "triton-ascend": "3.6.0.post2",
    "numpy": "1.26.4",
    "fla-core": "0.5.2",
    "flash-linear-attention": "0.5.2",
    "flash-linear-attention-npu-a5": "26.9.1",
    "torchao": "0.18.0",
    "fastapi": "0.123.10",
    "starlette": "0.50.0",
    "prometheus-fastapi-instrumentator": "8.1.0",
}
for package, expected_version in expected.items():
    installed = version(package)
    assert installed.split("+")[0] == expected_version, (package, installed, expected_version)

import acl
import cachetools
import fastapi
import fla
import fla_npu
import mindspeed_ops
import prometheus_client
import starlette
import torchao
import torchao_npu
import verl
import vllm
from fsdp_turbo.models.utils import get_model_by_name

assert vllm.__version__.split("+")[0] == "0.30.0", vllm.__version__
print(f"FSDPTurbo recipe imports OK (vLLM {vllm.__version__}).")
PYTHON
