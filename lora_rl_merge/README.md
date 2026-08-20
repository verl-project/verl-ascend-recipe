# Recipe: LoRA-RL with merge (GRPO + LoRA, `model.lora.merge=True`) on Ascend NPUs

Qwen3-8B + GRPO + LoRA on Atlas 800T A2 (910B), with the LoRA adapters **merged into the base model before every
rollout weight sync** (`actor_rollout_ref.model.lora.merge=True`). Training uses the FSDP2 backend, rollout uses
vLLM-Ascend. This recipe answers verl-ascend-recipe issue [#78](https://github.com/verl-project/verl-ascend-recipe/issues/78).

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, the validated commit, the container image and
the copy-pastable install line. The validated commit is the `/verl` checkout shipped in
`quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm`.

## Why "merge" needs no inference-side LoRA support

With `model.lora.merge=True` (`peft_merge` in `verl/workers/engine_workers.py`), the FSDP2 engine merges the adapters into
the base weights inside `merged_lora_context` and streams **full HF-keyed bf16 tensors** to the rollout engine
(`_merged_lora_per_tensor_param()` in `verl/workers/engine/fsdp/transformer_impl.py`); the vLLM server sets
`lora_rank = 0` when merge is on (`vllm_async_server.py`). vLLM-Ascend therefore receives a plain weight update and its
(experimental) LoRA adapter path is never exercised. Because `lora_rank > 0`, verl also computes the reference log-prob
from the actor with adapters disabled (`ref_in_actor`), so no separate reference model is built.

## Environment (validated)

| Component | Version |
| --- | --- |
| Hardware | Atlas 800T A2, 4 × 910B1 (64 GB HBM each) — 8 cards also fit |
| Image | `quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm` |
| verl | 0.9.0.dev0, commit `bc72e38e` (see `REQUIRED_VERL.txt`) |
| vllm / vllm-ascend | 0.18.0 / 0.18.1.dev44 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |
| CANN / driver | 9.0.0 / 26.0.rc1 |
| transformers / peft | 5.10.4 / 0.20.0 |

## Quickstart

1. Prepare gsm8k parquet files (same preprocessing as verl):

   ```bash
   cd /verl
   python3 examples/data_preprocess/gsm8k.py --local_save_dir $HOME/data/gsm8k
   ```

2. (Only if your container mounts a **subset** of the node's NPUs) apply the `get_npu_versions` patch, see
   [NPU adaptation notes](#npu-adaptation-notes):

   ```bash
   cd /verl && git apply /path/to/lora_rl_merge/patches/0001-get_npu_versions-first-visible-npu-id.patch
   ```

3. Launch (from the verl repo root; all knobs are env-overridable):

   ```bash
   # 10-step smoke on 4 NPUs
   NPROC_PER_NODE=4 TOTAL_TRAINING_STEPS=10 SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
     MODEL_PATH=/path/to/Qwen3-8B DATA_DIR=$HOME/data/gsm8k \
     bash run_qwen3_8b_lora_merge_fsdp2_npu.sh

   # 100-step run (checkpoint every 10 steps, validation every 20 steps, resume_mode=auto)
   NPROC_PER_NODE=4 MODEL_PATH=/path/to/Qwen3-8B DATA_DIR=$HOME/data/gsm8k \
     bash run_qwen3_8b_lora_merge_fsdp2_npu.sh
   ```

   `tools/parse_step_metrics.py <console.log>` prints the per-step timing / throughput / reward table used below.

## Validated results

### 10-step smoke, 4 × 910B1 (2026-08-20)

Configuration = the GPU reference script `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` (batch 128 / mini 64 /
`rollout.n` 8, prompt 1024 + response 1024, LoRA rank 32 / alpha 64, lr 1e-5, FSDP2 bf16 without offload, vLLM TP 2,
`gpu_memory_utilization` 0.6), only NPU launch items changed.

| step | step time (s) | gen (s) | update_actor (s) | merge + weight sync (s) | throughput (tokens/s, 4 cards) | reward mean |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 360.6 | 166.4 | 81.6 | 11.7 | 681 | 0.229 |
| 2 | 280.5 | 144.5 | 71.4 | 12.1 | 861 | 0.291 |
| 3 | 281.7 | 144.3 | 70.1 | 12.5 | 854 | 0.314 |
| 4 | 276.4 | 146.4 | 66.5 | 12.5 | 846 | 0.382 |
| 5 | 279.5 | 144.4 | 67.9 | 12.8 | 833 | 0.404 |
| 6 | 282.2 | 146.3 | 69.7 | 12.7 | 834 | 0.387 |
| 7 | 281.6 | 146.4 | 68.5 | 12.9 | 808 | 0.447 |
| 8 | 275.0 | 148.4 | 63.8 | 12.9 | 813 | 0.490 |
| 9 | 273.4 | 148.4 | 64.1 | 12.7 | 803 | 0.530 |
| 10 | 267.0 | 146.4 | 62.0 | 12.6 | 770 | 0.596 |

- Mean over steps 2–10: **277.5 s/step**, **824.7 tokens/s** global (206 tokens/s per card), MFU (actor) 0.44–0.51.
- Actor peak HBM 32.6 GB allocated / 40.2 GB reserved per card; `npu-smi` shows ≈50 GB per card during training
  (vLLM keeps 0.6 × HBM while awake and sleeps during training).
- Reward (`critic/score/mean`, gsm8k exact-match) rises monotonically from 0.229 to 0.596; response length drops from
  875 to 725 tokens and the 1024-token clip ratio from 0.53 to 0.25; `response/aborted_ratio` = 0.
- Exit code 0; the only tracebacks are `multiprocessing.resource_tracker` `KeyError('/psm_*')` printed by the vLLM
  server at shutdown (shared-memory cleanup noise).

### 100-step run

_To be filled after the scheduled 100-step run (≈7.7 h on 4 cards at the measured step time; checkpoint every 10 steps)._

### Unit test `tests/utils/test_fsdp_lora_merge.py` on NPU

`.github/workflows/npu_unit_tests.yml` currently excludes this file with `--ignore-glob="*test_fsdp_lora_merge*"`.
On 2 × 910B1 (container mounting only those two cards, patched `get_npu_versions`) the fsdp2 subset passes:

```
pytest -v -s tests/utils/test_fsdp_lora_merge.py -k fsdp2
test_merged_lora_context_qwen2[True-fsdp2-2]      PASSED
test_merged_lora_context_qwen2[False-fsdp2-2]     PASSED
test_merged_lora_context_gptoss[True-fsdp2-2]     PASSED
test_merged_lora_context_gptoss[False-fsdp2-2]    PASSED
test_collect_merged_lora_params[all-linear-fsdp2-2]   PASSED
test_collect_merged_lora_params[lora_targets0-fsdp2-2] PASSED
6 passed, 6 deselected in 199.34s
```

The FSDP1 (`strategy=fsdp`) half of the parametrization was not run here; the exclusion can at least be narrowed to it.

## NPU adaptation notes

1. **`get_npu_versions()` hard-codes `npu-smi info -t board -i 1`** (`verl/utils/device.py`). In a container that mounts
   only a subset of the node's NPUs (e.g. `/dev/davinci4-7`) the call fails with exit 215 and
   `actor_rollout_ref_init_model` aborts. `patches/0001-get_npu_versions-first-visible-npu-id.patch` adds
   `get_first_visible_npu_id()` (parses `npu-smi info -m`, skips Mcu rows) and uses it as the probe id; it also ships unit
   tests. Setting `ASCEND_VISIBLE_DEVICES` to the physical ids (`PHYS_CARDS=4,5,6,7`) is the unpatched fallback.
2. **Container namespaces**: do not start the container with `--network host` / `--ipc host` on a shared host — the
   Ascend driver then treats the container as a host-namespace accessor and device init fails with `aclInit 507899`
   whenever any other container has a card open. Bridge network + private IPC with `--shm-size 128g` works.
   The driver also keeps a card occupied by a container namespace until that container is **stopped**, not merely
   until its processes exit.
3. **Ray temp dir**: `RAY_TEMP_DIR=/path/with/space` adds `++ray_kwargs.ray_init._temp_dir` (the default `/tmp` is
   often full on shared hosts).
4. vLLM-Ascend runs with `enforce_eager=True`, `max_num_batched_tokens=8192`, `USE_OPTIMIZED_MODEL=0`, `VLLM_USE_V1=1`;
   actor/ref use `entropy_from_logits_with_chunking=True` and `use_torch_compile=False` — the same settings as the
   validated `verl_ascend_practice/run_rl_qwen3_8b_npu.sh`.
5. Optimizer state for LoRA (≈87 M trainable parameters at rank 32, all-linear) is tiny, so actor parameter /
   optimizer offload is unnecessary on 4 × 64 GB; avoid it — FSDP CPU offload on this host hit the pinned-host-memory
   limit (`aclrtMallocHostWithCfg` error 207001).

## Files

| File | Purpose |
| --- | --- |
| `run_qwen3_8b_lora_merge_fsdp2_npu.sh` | training script (GPU reference hyper-parameters, NPU launch items) |
| `REQUIRED_VERL.txt` | validated verl commit, image and install line |
| `patches/0001-get_npu_versions-first-visible-npu-id.patch` | verl fix for containers that mount a subset of NPUs |
| `tools/parse_step_metrics.py` | per-step metrics table from the console log |
