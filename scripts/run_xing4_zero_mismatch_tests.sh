#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
cd "$repo_root"
python -m pytest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_xing4_training_math.py "$@"
