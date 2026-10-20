# FAPO on Ascend

Last updated: 10/20/2026

Qwen2.5-Math-7B + FAPO-GenRM-4B using Megatron and vLLM-Ascend.
Related task: verl-ascend-recipe issue #16.
All contribution files are delivered here, including the core compatibility
changes as a patch against the pinned verl checkout. No separate core PR
or replacement of verl's `recipe` submodule is required.

## Change inventory

| Previous location in verl | Delivery here | Purpose |
| --- | --- | --- |
| `examples/ascend_extras/fapo_trainer/run_qwen2_5_math_7b_megatron.sh` | `run_qwen2_5_math_7b_megatron.sh` | Megatron launcher, memory controls, warmup, reward resource pool, 4096-token Actor context |
| External 16-card job overrides | `run_16card_100steps.sh` | 12 Actor/Ref devices + 4 GenRM devices, 100 steps |
| `recipe/fapo/reward_fn_reasoning.py` | `reward_fn_reasoning.py` | vLLM completions endpoint, Transformers token IDs, generation cap and GenRM logs |
| `recipe/fapo` helpers | `prepare_fapo_data.py`, `reward_fn_reasoning_remote.py` | Original data preparation and experimental external reward helper |
| `scripts/converter_hf_to_mcore.py` | `patches/verl.patch` | Device-specific distributed backend |
| `scripts/verify_hf_to_mcore.py` | `patches/verl.patch` | HF/Megatron weight verification |
| `verl/utils/vllm/npu_vllm_patch.py` | `patches/verl.patch` | NPU vLLM version detection and compatibility |
| Two `tests/utils/test_*_on_cpu.py` files | `patches/verl.patch` | QKV conversion and vLLM version regression tests |
| `.pre-commit-config.yaml`, device API sanity checker | `patches/verl.patch` | Device API checks also cover scripts |
| Two Markdown experiment reports | `reports/` | Historical 8-card and 16-card results |
| Core `recipe` gitlink update | Removed from delivery | The reward implementation is resolved here by absolute path |

## Setup

Use a separate verl checkout, with an Ascend training environment already
installed. The recorded 910C run used Python 3.12.13, torch 2.10.0,
torch-npu 2.10.0.post4, vLLM/vLLM-Ascend 0.23.0, Transformers 5.5.4
and Ray 2.48.0. This is an experiment environment record, not a claim that
installing these packages alone reproduces the container and its runtime fixes.

```bash
git clone https://github.com/verl-project/verl.git verl-fapo
git -C verl-fapo checkout fa856c3d97fbf93c059322298cc24bec3a944bfe
export VERL_ROOT="$(cd verl-fapo && pwd)"
export FAPO_DIR=/absolute/path/to/verl-ascend-recipe/fapo
bash "$FAPO_DIR/apply_verl_patch.sh"
```

Prepare `Qwen/Qwen2.5-Math-7B`, `dyyyyyyyy/FAPO-GenRM-4B`, and the
native **train and test splits** of `dyyyyyyyy/FAPO-Reasoning-Dataset`.
Keep `extra_info.question` and `extra_info.split` in the parquet files.
Use the test split for evaluation; do not derive evaluation data from training.
Convert the dense Actor without `--use_cpu_initialization`:

```bash
cd "$VERL_ROOT"
python scripts/converter_hf_to_mcore.py \
  --hf_model_path /models/Qwen2.5-Math-7B \
  --output_path /models/Qwen2.5-Math-7B-mcore

MODEL_PATH=/models/Qwen2.5-Math-7B \
MCORE_MODEL_PATH=/models/Qwen2.5-Math-7B-mcore \
GRM_PATH=/models/FAPO-GenRM-4B \
TRAIN_FILE=/data/fapo/train.parquet \
TEST_FILE=/data/fapo/test.parquet \
CKPTS_DIR=/checkpoints/fapo \
bash "$FAPO_DIR/run_16card_100steps.sh"
```

Source the container's CANN environment before running. Use node-local
temporary directories for Ray and compilation caches; keep models and logs
on persistent storage. The 100-step preset disables validation and checkpoint
saving, matching the recorded run; it does not produce a saved trained model.

## Verification and scope

The previous deployment completed smoke job `1279516` (4 steps) and job
`1280184` (100 steps, 16 x 910C). The training loop took 2:02:05;
step 100 throughput was 98.3 tokens/s normalized over all 16 devices.
See [the 16-card report](reports/FAPO_Ascend_16C_100STEP_REPORT.md).
These are results from the pre-migration deployment; the newly packaged
entry points have not been rerun for 100 steps.

The launcher's `colocate` mode means same-node, **separate RM resource pool**,
not Actor and RM sharing the same cards. External `remote` mode remains
experimental: its helper still requires editing the service address and was
not validated by these runs. The weight verifier was developed for the dense
BF16 Qwen2.5-Math model; it is not a general MoE verifier.
Checkpoint/resume and downstream evaluation were not exercised in the
16-card experiment. This PR therefore references #16 without claiming all
acceptance criteria are complete.

Packaging checks run on October 20, 2026:

| Check | Result |
| --- | --- |
| `python3 fapo/test_launcher.py` | 2 tests passed: absolute reward path and 16-card overrides; invalid reward mode rejected |
| `bash -n` on the three entry scripts | Passed |
| Target-repository pre-commit on changed files | Ruff and Ruff format passed |
| Apply patch to clean pinned verl checkout | Passed; second application correctly detected as already applied |
| `git diff --check` in patched verl checkout | Passed |

AI assistance was used for implementation and packaging. Human review of
every changed line and reproduction of relevant checks remain required.
