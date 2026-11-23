#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$recipe_dir/../../.." && pwd)"
ascend_source="${1:?usage: run_unit_tests.sh VLLM_ASCEND_SOURCE [PYTEST_ARGS...]}"
ascend_source="$(cd -- "$ascend_source" && pwd)"
shift
export PYTHONPATH="$repo_root:$ascend_source:${PYTHONPATH:-}"
cd "$repo_root"
python -m pytest -q \
  "$ascend_source/tests/e2e/nightly/single_node/ops/singlecard_ops/test_xing4_training_math.py" \
  tests/utils/kernel/test_npu_training_matmul.py \
  tests/utils/npu/test_vocab_reduction.py "$@"
