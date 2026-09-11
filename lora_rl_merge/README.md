# Recipe: LoRA-RL with merge (GRPO + LoRA, `model.lora.merge=True`) on Ascend NPUs

Qwen3-8B + GRPO + LoRA on Atlas 800T A2 (910B), with the LoRA adapters **merged into the base model before every
rollout weight sync** (`actor_rollout_ref.model.lora.merge=True`). Training uses the FSDP2 backend, rollout uses
vLLM-Ascend. This recipe answers verl-ascend-recipe issue [#78](https://github.com/verl-project/verl-ascend-recipe/issues/78).

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, the validated commit, the container image and
the copy-pastable install line. The validated commit is the `/verl` checkout shipped in
`quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm`.

### 固定镜像的 vLLM-Ascend sampler 补丁

本次固定镜像 ID 为 `sha256:b25fae72319d0c16451fb27afd6391f97e8734452ce369901082dd5b8956ae91`，
其中 vLLM-Ascend 提交为 `a43c8cc8057f490ed1df2c6ed66253e2d7817da4`。训练前在 **vLLM-Ascend**
源码目录应用补丁；`install_verl.sh` 只安装 verl，不处理此依赖：

```bash
cd /vllm-ascend
test "$(git rev-parse HEAD)" = a43c8cc8057f490ed1df2c6ed66253e2d7817da4
git apply --check /path/to/lora_rl_merge/patches/vllm-ascend/0001-record-sampler-stream.patch
git apply /path/to/lora_rl_merge/patches/vllm-ascend/0001-record-sampler-stream.patch
cd /verl
```

补丁回移上游已合入的 [PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394)，提交为
`fc0ce85b58f019e5a1988dbd3f39d014052fb44b`。`wait_stream` 建立执行依赖，新增的
`q.record_stream(torch.npu.current_stream())` 防止随机采样张量在消费者 stream 完成前被分配器重用。
2026-09-10 同卡 Qwen3-8B、TP=1 对照中，原版 224621 个 token 出现 4 个越界 token 及对应的 `-inf`
log-prob；补丁组 225828 个 token 中两类异常均为 0。两组各 256 条序列，此结果只支持该样本，
修复后的目标训练完整验证仍在准备。包含上游修复的其他版本无需重复应用，但不能直接继承本镜像的验证结论。

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
| Hardware | Atlas 800T A2, 4 × 910B1 (64 GB HBM each); 8-card configuration untested |
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
     bash /path/to/lora_rl_merge/run_qwen3_8b_lora_merge_fsdp2_npu.sh

   # 100-step training, or continuation from the latest checkpoint in DEFAULT_LOCAL_DIR
   TOTAL_TRAINING_STEPS=100 MODEL_PATH=/path/to/Qwen3-8B DATA_DIR=$HOME/data/gsm8k \
     DEFAULT_LOCAL_DIR=$HOME/ckpts/lora_merge_100 \
     bash /path/to/lora_rl_merge/run_qwen3_8b_lora_merge_fsdp2_npu.sh
   ```

   The default epoch limit equals `TOTAL_TRAINING_STEPS`, so even a filtered dataset with only one full batch
   per epoch has enough capacity. The trainer still stops at the requested step. `TOTAL_EPOCHS` can override
   that limit. Each invocation retains a unique `training.*.log` under `DEFAULT_LOCAL_DIR` and checks that its
   consecutive training metrics reach the requested final step. A successful trainer exit before that step
   is an error. This checks training length; reward, throughput and numerical diagnostics require separate review.
   Keep the `console` logger enabled. Checkpoints are saved every 10 steps and `RESUME_MODE=auto` is the default.

   `tools/parse_step_metrics.py <console.log>` prints the per-step timing / throughput / reward table used below.

## Validated results

### 10-step smoke, 4 × 910B1 (2026-08-20)

Configuration = the GPU reference script `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` (batch 128 / mini 64 /
`rollout.n` 8, prompt 1024 + response 1024, LoRA rank 32 / alpha 64, lr 1e-5, FSDP2 bf16 without offload, vLLM TP 2,
`gpu_memory_utilization` 0.6), only NPU launch items changed.

| step | step time (s) | gen (s) | update_actor (s) | merge + weight sync (s) | throughput (tokens/s/NPU, 4 NPUs) | reward mean |
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

- Mean over steps 2–10: **277.5 s/step**, **824.7 tokens/s/NPU** (approximately 3299 tokens/s across 4 NPUs), MFU (actor) 0.44–0.51.
- Actor peak HBM 32.6 GiB allocated / 40.2 GiB reserved per card; `npu-smi` shows ≈50 GB per card during training
  (vLLM keeps 0.6 × HBM while awake and sleeps during training).
- Reward (`critic/score/mean`, gsm8k exact-match) increases overall from 0.229 to 0.596, with a decrease at step 6; response length drops from
  875 to 725 tokens and the 1024-token clip ratio from 0.53 to 0.25; `response/aborted_ratio` = 0.
- Exit code 0; the only tracebacks are `multiprocessing.resource_tracker` `KeyError('/psm_*')` printed by the vLLM
  server at shutdown (shared-memory cleanup noise).

### Attempted 100-step run: 58 steps completed (2026-08-28)

The retained log `lora_merge_100step_4npu_0828T0414Z.log` contains steps 1–58 and a progress duration of 4:34:03.
The process exited with code 0, but the default `TOTAL_EPOCHS=1` limits this dataset to `7473 // 128 = 58` steps
in the validated trainer. The current script corrects the epoch limit and checks completed training steps; the continuation result is recorded below.

- Mean reward over the first/last 10 steps: 0.3990234375 / 0.90390625.
- Arithmetic mean of the 58 per-step `perf/throughput` values: 673.807293 tokens/s/NPU, or 2695.229173 tokens/s
  across four NPUs. This metric divides total training-step tokens by step duration and NPU count; it does not
  include all job startup/validation time and is not pure rollout generation throughput.
- The latest checkpoint is step 50. It was successfully restored in the subsequent continuation.
- All 58 steps contain non-finite `rollout_corr` diagnostics. Actor losses, gradient norms and rewards are finite;
  the sampler investigation and its limits are recorded above.

### 原版 sampler 的 100 步续训结果（2026-09-10）

从 checkpoint50 恢复后实际完成第 51–100 步，保存 checkpoint100，并完成最终 GSM8K 验证。
统计只采用原始第 1–50 步和续训第 51–100 步，不重复计入原先第 51–58 步。首末各 10 步 reward 均值为
0.3990234375 与 0.94755859375；GSM8K greedy 准确率从 321/1319 上升到 1216/1319。
总训练步 token 数 66803533 除以累计步时 25897.94468626156 秒和 4 张卡，得到
644.8729214739403 tokens/s/NPU；该口径不包含所有启动、恢复、停机与验证时间。

100 步 actor loss 和梯度均有限、梯度均非零，但每步都存在非有限 `rollout_corr` 诊断。
此轨迹使用未修复的 sampler，不能作为当前补丁版的完整验证。checkpoint50 的恢复已实测；
checkpoint100 仅核查文件和额外状态，尚未实际加载模型及优化器继续训练。

### 修复版 sampler 的完整 100 步结果（2026-09-11，910B1 + 910B3）

应用 vLLM-Ascend `record_stream` 回移补丁后的独立轨迹：第 1–90 步在 4 × 910B1 上完成，
第 91–100 步从 `global_step_90` 自动恢复后在 4 × 910B3 上完成（原主机可用的四张卡已被其他
使用者的容器声明，迁移过程与镜像内容等价性核对见管理仓证据目录）。两段的 verl 提交、
vLLM-Ascend 提交、镜像内容、两个 patch 和全部训练超参一致，容器内逐字节比对了
`runtime.patch`、`vllm-ascend-runtime.patch` 与 `packages.txt`。统计只取原运行的第 1–90 步
与本次的第 91–100 步，排除被替代的旧第 91–99 步。容器以退出码 0 结束。

| 项目 | 第 1–90 步（4 × 910B1） | 第 91–100 步（4 × 910B3） |
| --- | --- | --- |
| 非有限 `rollout_corr` 指标 | 0 | 0 |
| 零梯度步 | 0 | 0 |
| 平均步时 | 259.88 s | 261.51 s |
| 加权吞吐（tokens/s/NPU，4 卡分母） | 702.64 | 574.36 |
| 最低逐步吞吐 | 593.42 | 551.23 |

- 首 10 步 reward 均值 0.39462890625，末 10 步（第 91–100 步）0.926953125，上升。
- GSM8K greedy 准确率：step 0 为 322/1319（24.41%）、step 20 为 929（70.43%）、
  step 40 为 1045（79.23%）、step 60 为 1119（84.84%）、step 80 为 1180（89.46%）、
  step 100 为 1148（87.04%）。恢复时对同一 `global_step_90` 的验证在原主机为 1198（90.90%）、
  在新主机为 1205（91.36%），相差 7 条样本。**第 100 步低于第 80 步与第 90 步，
  末 10 步 reward 也在 0.8867–0.9717 之间波动；本次未做多随机种子重复，不声称第 100 步是精度上限。**
- **同一 recipe 在单台 4 × 910B3 上从第 1 步重跑的独立轨迹**（消除跨硬件拼接）同样满足全部判据：
  100 步连续、非有限 `rollout_corr` 为 0、无零梯度步、首末十步 reward 0.39160 → 0.83555、
  四卡分母加权吞吐 655.51 tokens/s/NPU（最低 565.17）、checkpoint100 与最终验证齐备、退出码 0。
  两条轨迹的 GSM8K 验证在第 80 步前始终吻合在 2 个百分点以内
  （step 20 为 70.43% 与 70.13%，step 40 为 79.23% 与 80.21%，step 60 为 84.84% 与 86.66%，
  step 80 为 89.46% 与 89.08%）。
- **精度峰值在第 80 步附近，不是第 100 步。** 两条轨迹的 step 100 验证分别回落到 87.04% 与 80.67%。
  单机轨迹在第 96 步起出现训练不稳定：`actor/grad_norm` 由第 81–95 步的 0.0435–0.8906 跳升到
  2.1289–2.7148，reward 由 0.8174 降到 0.6113，`response_length/mean` 由 543 增至 668；
  拼接轨迹未出现同样跳变（第 91–100 步 `grad_norm` 仅 0.0846–0.7689）。
  两条轨迹在第 80 步前吻合、之后分岔，说明这是 run-to-run 随机性而非硬件差异。
  **以精度为目标时建议取 checkpoint 80 而不是 100，或在第 80 步之后加强梯度裁剪与早停监控。**
- 对照：原版 sampler 的 100 步轨迹在全部 100 步上都有 7 项非有限 `rollout_corr` 指标
  （`kl`、`k3_kl`、`rollout_ppl`、`rollout_log_ppl`、`log_ppl_diff`、`log_ppl_abs_diff`、
  `log_ppl_diff_min`），本轨迹为 0。
- 两段硬件不同，吞吐按段披露，不合并成单一 TPS 结论。两段平均步时相近（259.88 与 261.51 s），
  吞吐差异主要来自 `response_length/mean` 由第 1 步的 895 降到第 100 步的 517 后每步 token 总量下降；
  910B1 自身也呈同一趋势（第 1–10 步 810.54、第 81–90 步 593–651）。

These are partial validation results. The issue does not explicitly require eight NPUs; neither four-card results
nor a successful process exit establish final acceptance.

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
6 passed, 6 deselected, 1 warning in 199.34s
```

The FSDP1 (`strategy=fsdp`) half of the parametrization was not run here. These historical results do not establish
that the current upstream CI exclusion can be changed without further validation.

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
| `ALGORITHM_TUNING_REPORT.md` | 算法适配调优文档（中文）：机制、环境、配置对照、实测、调参建议、排障 |
