#!/usr/bin/env bash
# Run on every node inside an already provisioned Ascend / vLLM environment.
set -euo pipefail

RECIPE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=${WORKSPACE_DIR:-"$PWD"}
mkdir -p "$WORKSPACE_DIR"
WORKSPACE_DIR=$(cd -- "$WORKSPACE_DIR" && pwd)
VERL_DIR="$WORKSPACE_DIR/verl"
VEOMNI_DIR="$WORKSPACE_DIR/VeOmni"

read_version() {
    local value
    value=$(sed -n "s/^$1=//p" "$RECIPE_DIR/REQUIRED_VERL.txt")
    if [[ -z "$value" || "$value" == *$'\n'* ]]; then
        echo "Expected exactly one $1 in REQUIRED_VERL.txt" >&2
        exit 1
    fi
    printf '%s' "$value"
}
VERL_COMMIT=$(read_version COMMIT)
VEOMNI_COMMIT=$(read_version VEOMNI_COMMIT)
TRANSFORMERS_VERSION=$(read_version TRANSFORMERS_VERSION)

# Never overwrite an existing checkout.
for repo_dir in "$VERL_DIR" "$VEOMNI_DIR"; do
    if [[ -e "$repo_dir" ]]; then
        echo "$repo_dir already exists; choose a fresh WORKSPACE_DIR." >&2
        exit 1
    fi
done

# CANN/ATB must already be sourced by the caller.
python - <<'PY'
import sys
import torch
import torch_npu
import vllm
import vllm_ascend

if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError("The pinned VeOmni requires Python 3.11 or 3.12.")
if not torch.npu.is_available():
    raise RuntimeError("No Ascend NPU is available; check CANN and device mounts.")
print(f"torch={torch.__version__}, torch_npu={torch_npu.__version__}, vllm={vllm.__version__}")
PY

echo "1. Fetch the pinned verl base and apply the recipe patch"
git clone https://github.com/verl-project/verl.git "$VERL_DIR"
git -C "$VERL_DIR" checkout --detach "$VERL_COMMIT"
git -C "$VERL_DIR" apply --check "$RECIPE_DIR/patch/verl.patch"
git -C "$VERL_DIR" apply "$RECIPE_DIR/patch/verl.patch"

echo "2. Install verl NPU dependencies"
python -m pip install -r "$VERL_DIR/requirements-npu.txt" \
    --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/

echo "3. Install the pinned VeOmni and Transformers"
git clone https://github.com/ByteDance-Seed/VeOmni.git "$VEOMNI_DIR"
git -C "$VEOMNI_DIR" checkout --detach "$VEOMNI_COMMIT"
# Use the prepared NPU environment, as in the source PR's Ascend installation.
python -m pip install --no-deps -e "$VEOMNI_DIR"
python -m pip install "transformers==$TRANSFORMERS_VERSION"
python -m pip install --no-deps -e "$VERL_DIR"

echo "4. Verify the checkpoint API required by this patch"
python -c 'from veomni.models.checkpoint import get_checkpoint_tensor_converter'
echo "Installed. Start the Ray cluster, then run the example from $VERL_DIR."
