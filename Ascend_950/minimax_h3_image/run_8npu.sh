#!/usr/bin/env bash
# MiniMax H3 Image DiffusionNFT LoRA on one eight-NPU Ascend 950 node.
set -euo pipefail

: "${VERL_OMNI_PATH:?Set VERL_OMNI_PATH to the pinned verl-omni checkout documented in README.md}"
verl_omni_root=$(cd "$VERL_OMNI_PATH" && pwd)
launcher="$verl_omni_root/examples/diffusionnft_trainer/minimax_h3_image/run_minimax_h3_image_ocr_lora_npu.sh"
if [[ ! -f "$launcher" ]]; then
    echo "H3 Image NPU launcher missing at $launcher. See README.md for the required revision." >&2
    exit 1
fi

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-minimax_h3_image_ascend950_100steps}
export OUTPUT_DIR=${OUTPUT_DIR:-"$PWD/outputs/$EXPERIMENT_NAME"}

# Keep imports and relative paths consistent with the upstream recipe.
cd "$verl_omni_root"
exec bash "$launcher" \
    trainer.val_before_train=True \
    "$@"
