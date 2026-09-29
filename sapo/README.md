# Recipe: SAPO on Ascend NPU — Qwen3-30B-A3B (MoE) with Megatron

[SAPO](https://arxiv.org/abs/2511.20347) replaces PPO's ratio clipping with a
smooth tau-parameterized surrogate. This recipe runs it on **Qwen3-30B-A3B-Base**
(MoE, 128 experts) with the **Megatron** backend and a colocated **vLLM-Ascend**
rollout, on **16 x Ascend 910B3** (2 nodes x 8, 60.96 GiB HBM per device).

`run_qwen3_30b_a3b_megatron_npu.sh` carries the exact configuration that passed
acceptance — every default in it is a measured choice, not a guess.

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). Install it with the repository
root helper:

```bash
./install_verl.sh --recipe sapo --show   # dry-run, prints the pip command
./install_verl.sh --recipe sapo
```

## Quickstart

Convert the HF weights to a Megatron dist checkpoint once (from the verl tree):

```bash
python3 scripts/converter_hf_to_mcore.py \
    --hf_model_path /path/to/Qwen3-30B-A3B-Base \
    --output_path   /path/to/Qwen3-30B-A3B-Base-mcore \
    --use_cpu_initialization
```

Then launch on the Ray head node:

```bash
NNODES=2 NDEVICES_PER_NODE=8 \
MODEL_PATH=/path/to/Qwen3-30B-A3B-Base \
MCORE_MODEL_PATH=/path/to/Qwen3-30B-A3B-Base-mcore \
TRAIN_FILE=/path/to/dapo-math-17k/train.parquet \
VAL_FILE=/path/to/aime-2024/test.parquet \
EXPERIMENT_NAME=my_unique_run_name \
bash sapo/run_qwen3_30b_a3b_megatron_npu.sh
```

`EXPERIMENT_NAME` matters: `resume_mode` defaults to `auto`, so reusing a name
silently resumes from whatever checkpoint the previous run of that name left
behind.

## Acceptance run

100 steps, ~12.8 hours, both nodes exit 0. No OOM, no `AssertionError`, no gloo
timeout.

| Metric | Result |
| --- | --- |
| `perf/throughput` | mean **123.94** tokens/s/device, min 110.98, max 147.16 — all 50 readings above 100 |
| throughput trend | first 25 steps 119.06 -> last 25 steps 128.83 (+8.2%) |
| `critic/rewards/mean` | -0.1474 -> -0.0496 (+66% toward zero), peaking at +0.148 on step 100 |
| `max_memory_allocated_gb` | flat at 29.22 / 60.96 GiB (48%) for the whole run, no drift |
| step time | ~450 s (with profiling off) |

After rebasing onto a later `main` that restructured
`verl/utils/vllm/npu_vllm_patch.py` (verl#7190, verl#7147), a 4-step smoke run
on the same 16 x 910B3 configuration reproduced the numbers: throughput 113.82
vs 113.2 baseline (+0.5%), `max_memory_allocated_gb` byte-identical at 29.2208,
`timing_s/step` 446.7 s vs 449.1 s (-0.5%), `response/aborted_ratio` 0.0 on
every step.

## Why these defaults

**Token-budget packing is the throughput lever.** The best static configuration
reached only 80.9. Enabling `use_dynamic_bsz` on the actor plus
`log_prob_use_dynamic_bsz` on rollout and ref took it to 113.2 (+40%) and over
the acceptance line. `rearrange_micro_batches` packs by token budget instead of
splitting by sequence count, which equalises the tokens each rank actually
computes; the bulk of the gain lands in `old_log_prob` (173.9 s -> 30.5 s,
-82.5%). All three flags must agree — verl asserts actor against rollout and
refuses to start when they disagree, while ref is not validated and will
silently run a different batching scheme.

**EP=8 / ETP=1 compresses expert communication.** Against EP=4 / ETP=4, operator
level profiling shows AlltoAllV absolute duration down 70% (16.1 s -> 4.9 s) and
total device time down 66% (42.1 s -> 14.4 s). Communication remains the
non-compute bottleneck, but its absolute cost drops sharply.

**The HybridDeviceOptimizer is load-bearing; verl's `optimizer_offload` is not.**
Disabling all three offloads OOMs in step 1. Keeping Adam state resident on the
host via `optimizer_cpu_offload=True` with `optimizer_offload_fraction=1` is what
prevents that. By contrast, toggling verl's own
`megatron.optimizer_offload` moved `update_actor` by 1.5% — noise — so this
recipe leaves it `False` and relies on `param_offload` + `grad_offload` + HDO.

**`RECOMPUTE=full` beats `selective` here.** `selective` cut step time by only
10.4% while costing 7.6 GB more device memory, which the colocated rollout
needs.

**`PPO_MICRO_BATCH_SIZE_PER_GPU=4` is a packing ceiling, not a split.** micro2 ->
micro4 bought -28.7% on `update_actor` at zero memory cost. micro8 bought a
further -26.1% but still could not clear the throughput target without dynamic
batching.

**Keep `PROFILE=0` for any run you intend to quote.** Discrete per-role tracing
inflated step time by roughly 3.3x.

## Known limitation: checkpoint saving

`SAVE_FREQ` defaults to `-1` (never save) because saving is not reliable on this
configuration, and the acceptance run avoided it entirely.

An earlier run completed 99 steps and died in the step-100 save: one node wrote
its shards in about 2 minutes while the other produced zero shards for 76
minutes, and every rank then timed out on the `torch.distributed.barrier()` in
`verl/utils/checkpoint/megatron_checkpoint_manager.py`. This is not the
filesystem — 16 plain writer processes on the same nodes, same mount, same
3.8 GB per writer sustain ~60 MiB/s each, roughly 38x what the same ranks
achieved from inside the training process.

`async_save` does not appear to be a substitute on the pinned commit: a static
review suggests `async_calls_finalize_fn_exec` was not carried over during the
workers -> engines migration (verl#6067) and the trainer's call site is
`hasattr`-guarded, which would make it a silent no-op producing an unloadable
checkpoint.

Neither finding has been reproduced on a clean release yet, so no fix is
proposed here. If you need checkpoints on this configuration, validate one save
end to end before trusting a long run to it.

## Relation to the other SAPO recipes in this directory

This recipe targets a **sparse MoE** model at 16 devices and leans on expert
parallelism and token-budget packing. A SAPO recipe for a **dense** model with
linear attention has different constraints — notably that packed sequences are
unavailable, which rules out the `use_dynamic_bsz` lever this recipe depends on.
Read the per-script header before copying flags between them.
