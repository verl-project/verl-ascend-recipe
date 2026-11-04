#!/usr/bin/env bash
# Launch ATRL on a verl checkout that contains recipe/atrl.
set -euo pipefail

: "${VERL_ROOT:?Set VERL_ROOT to the verl checkout}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${TRAIN_FILES:?Set TRAIN_FILES}"
: "${VAL_FILES:?Set VAL_FILES}"

if [[ ! -d "${VERL_ROOT}/recipe/atrl" ]]; then
  echo "Copy this directory to ${VERL_ROOT}/recipe/atrl before launching." >&2
  exit 1
fi

cd "${VERL_ROOT}"
python -m recipe.atrl.main \
  --config-path=recipe/atrl/config \
  --config-name=atrl_ppo_trainer \
  algorithm.adv_estimator=grpo_segment \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  "$@"
