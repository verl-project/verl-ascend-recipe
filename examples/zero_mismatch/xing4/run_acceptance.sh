#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$recipe_dir/../../.." && pwd)"
run_name="${1:?usage: run_acceptance.sh RUN_NAME [STEPS]}"
steps="${2:-30}"
[[ "$run_name" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]] || exit 2
export XING_MODEL_PATH="${XING_MODEL_PATH:-$repo_root/models/Xing4.0-29B-A4B}"
export XING_WORK_DIR="${XING_WORK_DIR:-$repo_root/outputs/zero_mismatch/xing4}"
run_root="$XING_WORK_DIR/runs/$run_name"
[[ ! -e "$run_root" ]] || { echo "Run already exists: $run_root" >&2; exit 2; }
mkdir -p "$run_root" "$XING_WORK_DIR/cache" "$XING_WORK_DIR/tmp"
export PYTHONPATH="$repo_root:$recipe_dir:${PYTHONPATH:-}"
cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$XING_WORK_DIR/tmp"
export XDG_CACHE_HOME="$XING_WORK_DIR/cache"
export ASCEND_WORK_PATH="$XDG_CACHE_HOME/ascend"
export ASCEND_CACHE_PATH="$XDG_CACHE_HOME/ascend"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_RPC_BASE_PATH="$TMPDIR"
export VLLM_MLA_DISABLE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_BATCH_INVARIANT=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HCCL_CONNECT_TIMEOUT=180 HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto
python "$recipe_dir/stamp_run.py" "$run_root"
python "$recipe_dir/acceptance.py" --run-name "$run_name" --steps "$steps" > "$run_root/driver.log" 2>&1
python "$recipe_dir/audit_geometry.py" "$run_root"
python "$recipe_dir/audit_continuous.py" "$run_root"
