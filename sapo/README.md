# SAPO Qwen3-30B-A3B on Ascend

The launcher uses Megatron training with colocated vLLM-Ascend rollout.
All experimental reports, logs, curves, compatibility notes, and validation
results are consolidated in
[`verl_ascend_practice/reports/sapo`](../verl_ascend_practice/reports/sapo/README.md).

## Before launching

Read the report's **Runtime prerequisites** first. `REQUIRED_VERL.txt` pins
the tested source revision, but the successful 910C run also applied the
documented patches; installing that revision alone is not equivalent to the
validated environment. The script's historical defaults use 2 nodes x 8 NPUs.

The following is the 1 node x 16 NPU invocation used for the new report,
after preparing the patched runtime and initializing the Ascend environment:

```bash
export NNODES=1 NDEVICES_PER_NODE=16
export MODEL_PATH=/path/to/Qwen3-30B-A3B-Base
export TRAIN_FILE=/path/to/dapo-math-17k/train.parquet
export VAL_FILE=/path/to/aime-2024/test.parquet
export TOTAL_TRAINING_STEPS=100 FILTER_WORKERS=16
export EXPERIMENT_NAME=sapo_qwen3_30b_910c_unique_run

# Keep disposable compiler and Ray temporaries off a shared filesystem.
scratch=$(mktemp -d /tmp/sapo.XXXXXXXX)
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export TRITON_CACHE_DIR="$scratch/triton" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
export TORCH_EXTENSIONS_DIR="$scratch/extensions" ASCEND_CACHE_PATH="$scratch/ascend"
export RAY_TMPDIR="$scratch/ray" XDG_CACHE_HOME="$scratch/cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$ASCEND_CACHE_PATH" "$RAY_TMPDIR" "$XDG_CACHE_HOME"

bash sapo/run_qwen3_30b_a3b_megatron_npu.sh \
    trainer.resume_mode=disable \
    trainer.default_local_dir=/persistent/path/checkpoints \
    +ray_kwargs.ray_init.include_dashboard=False \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True
```

This invocation loads HF weights through legacy mbridge. An existing converted
Megatron checkpoint can instead be selected with `MCORE_MODEL_PATH`; that route
belongs to the historical 910B3 report and was not revalidated on 910C.
Keep model/data/log paths on persistent storage. Checkpoint saving and validation
remain disabled by default; neither save/resume nor downstream evaluation was
validated by the attached 100-step run.
