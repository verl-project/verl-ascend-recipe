#!/usr/bin/env bash
# Render the actual Ray command by default; --submit contacts the Ray cluster.
set -euo pipefail
mode=${1:---dry-run}
case "$mode" in
    --dry-run|--submit) shift "$(( $# > 0 ? 1 : 0 ))";;
    *) echo "Use --dry-run or --submit" >&2; exit 2;;
esac
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
source "${MODEL_PROFILE:-$ROOT/configs/models/qwen3_5_9b.sh}"
source "${EXPERIMENT_CONFIG:-$ROOT/examples/e2b_swe_grpo/experiment.sh}"
if [[ "$mode" == --dry-run ]]; then
    export TRAINING_DRY_RUN=true
    export RUNTIME_ENV="${RUNTIME_ENV:-$ROOT/.runtime/created-at-submit.json}"
    ray() { printf '%q ' ray "$@"; printf '\n'; }
    export -f ray
fi
exec bash "${TRAIN_SCRIPT:-$ROOT/scripts/training/train_megatron_npu.sh}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.session_timeout_seconds=7500 \
    "$@"
