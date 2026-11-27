#!/usr/bin/env bash
# Colleague defaults; deployment paths remain in our user directory.
set -a
source "$ROOT/examples/e2b_swe_grpo/zmj.sh"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$ROOT/.artifacts/rollouts/$EXP_NAME}"
set +a
EXPERIMENT_ARGS=()
