# Qwen3-8B LoRA-RL（merge）在 Atlas 800T A2 上的算法适配调优文档

对应 verl-ascend-recipe issue [#78](https://github.com/verl-project/verl-ascend-recipe/issues/78)：基于 verl 的
vllm_ascend 推理后端，补齐 Qwen3-8B + GRPO + LoRA（`model.lora.merge=True`，训练中 merge 进 base 模型用于 rollout）
对 FSDP2 后端的支持。配套脚本与文件见本目录 [`README.md`](README.md)。

> 状态说明：第 6.1、6.2 节为已完成的实测；第 6.3 节（100 步训练）待排期完成后补全。

## 1. 结论概览

| 项目 | 结果 |
| --- | --- |
| 路径可用性 | 参考脚本 `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` 的算法与 LoRA 配置不改，只改 NPU 启动项，即可在 4 × 910B1 上跑通 |
| 10 步冒烟 | 步时均值 277.5 s，全局吞吐 824.7 tokens/s（每卡 206），reward 均值 0.229 → 0.596 单调上升，rc=0 |
| merge 权重同步 | 每步 11.7–12.9 s（占步时 ≈4.5%），vLLM-Ascend 收到全量 bf16 权重，无需推理侧 LoRA |
| 显存 | actor 侧峰值 32.6 GB allocated / 40.2 GB reserved（每卡 64 GB），训练中 `npu-smi` ≈50 GB/卡 |
| NPU 单测 | `tests/utils/test_fsdp_lora_merge.py -k fsdp2` 在 2 卡上 6/6 通过（199 s） |
| 需要的适配 patch | 1 个：`get_npu_versions()` 不再硬编码 `npu-smi -i 1`（容器只挂部分卡时必需） |
| 100 步 / 长跑 | 待补（按实测步时 4 卡约 7.7 h） |

## 2. 机制：为什么 merge 路径不依赖 vllm-ascend 的 LoRA 能力

- `verl/workers/engine_workers.py:674` 读取 `model_config.lora.get("merge")` 为 `peft_merge`；`:735` 的说明指出 merge 模式下
  adapter 在权重同步前并入 base，engine 返回全量 HF 键名参数、`peft_config=None`，rollout 收到的是普通权重更新。
- `verl/workers/engine/fsdp/transformer_impl.py:973` 起：merge 模式走 `_merged_lora_per_tensor_param()`，在
  `merged_lora_context(backup_adapters=True)` 内逐张量 `full_tensor()` 并转 bf16 后流式发送；上下文退出时恢复未合并的
  base 权重与 adapter。
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py:401-402`：`lora.merge=True` 时把 `lora_rank` 置 0，vLLM 不启用 LoRA。
- `verl/trainer/ppo/v1/trainer_base.py:316-323`：`lora_rank > 0` 时 `ref_in_actor=True`，参考模型 log-prob 由 actor 在
  `disable_adapter()` 下计算，不构建独立 ref worker（`engine_workers.py:424`）。

因此本任务在昇腾侧的工作是"NPU 上验证 + 修复 + 交付"，而非从零实现。

## 3. 环境与版本

| 组件 | 版本 |
| --- | --- |
| 硬件 | Atlas 800T A2，4 × 910B1（64 GB HBM），8 卡 HCCS 全互联 |
| 镜像 | `quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm` |
| verl | 0.9.0.dev0，commit `bc72e38e`（`REQUIRED_VERL.txt`） |
| vllm / vllm-ascend | 0.18.0 / 0.18.1.dev44 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |
| CANN / 驱动 | 9.0.0 / 26.0.rc1 |
| transformers / peft / ray | 5.10.4 / 0.20.0 / 2.56.1 |

## 4. 数据与模型

- 模型：`Qwen/Qwen3-8B`（hidden 4096、intermediate 12288、36 层、GQA 32/8 头、head_dim 128），bf16 权重 16.4 GB。
- 数据：`openai/gsm8k`，用 verl 自带 `examples/data_preprocess/gsm8k.py` 生成 `train.parquet`（7473 条）/`test.parquet`（1319 条）。

## 5. 训练配置与 GPU 参考脚本的对照

与 `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` **完全相同**的项：GRPO、`use_kl_in_reward=False`、`kl_loss_coef 0.001`
（`low_var_kl`）、`entropy_coeff 0`、lr 1e-5、`train_batch_size 128`、`ppo_mini_batch_size 64`、`rollout.n 8`、
prompt 1024 + response 1024、`use_dynamic_bsz` + `ppo_max_token_len_per_gpu 24576`、`lora_rank 32` / `lora_alpha 64`
（`target_modules` 默认 all-linear）、`actor.strategy=fsdp2`、bf16、actor 不 offload、ref `param_offload=True`、vLLM
TP 2、`gpu_memory_utilization 0.6`、`layered_summon=True`、`load_format=safetensors`。

**只改的 NPU 启动项**：

| 项 | 取值 | 原因 |
| --- | --- | --- |
| `trainer.device` | `npu` | 设备选择 |
| `trainer.n_gpus_per_node` | 4（可 8） | 共享主机实际可用卡数；LoRA 优化器状态极小，4 卡显存充裕 |
| `rollout.enforce_eager` | `True` | 与已验证的 `verl_ascend_practice/run_rl_qwen3_8b_npu.sh` 一致；ACL graph 模式未评估 |
| `rollout.max_num_batched_tokens` | 8192 | 同上 |
| `actor/ref.entropy_from_logits_with_chunking` | `True` | 降低 logits 熵计算峰值显存 |
| `actor/ref.use_torch_compile` | `False` | NPU 上 torch.compile 路径未验证 |
| 环境变量 | `USE_OPTIMIZED_MODEL=0`、`VLLM_USE_V1=1`、`HCCL_CONNECT_TIMEOUT=5400`、`TOKENIZERS_PARALLELISM=false` | 纯推理优化路径干扰 RL 的 log-prob 与在线权重更新；HCCL 首次建链较慢 |
| 启动方式 | 直接 `python3 -m verl.trainer.main_ppo` | 参考脚本的 uv 封装仅对 GPU 生效 |
| `++ray_kwargs.ray_init._temp_dir` | 可写目录 | 共享主机 `/tmp` 常满 |

## 6. 实测结果

### 6.1 10 步冒烟（4 × 910B1，2026-08-20）

| step | 步时 (s) | gen (s) | old_log_prob (s) | ref (s) | update_actor (s) | merge + 权重同步 (s) | 吞吐 (tokens/s, 4 卡) | reward 均值 | 响应长度均值 | 截断比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 360.6 | 166.4 | 32.9 | 22.6 | 81.6 | 11.7 | 681 | 0.229 | 875 | 0.53 |
| 2 | 280.5 | 144.5 | 28.5 | 21.5 | 71.4 | 12.1 | 861 | 0.291 | 857 | 0.51 |
| 3 | 281.7 | 144.3 | 29.6 | 22.6 | 70.1 | 12.5 | 854 | 0.314 | 855 | 0.49 |
| 4 | 276.4 | 146.4 | 27.7 | 20.8 | 66.5 | 12.5 | 846 | 0.382 | 831 | 0.48 |
| 5 | 279.5 | 144.4 | 29.5 | 22.3 | 67.9 | 12.8 | 833 | 0.404 | 827 | 0.45 |
| 6 | 282.2 | 146.3 | 29.0 | 22.0 | 69.7 | 12.7 | 834 | 0.387 | 836 | 0.44 |
| 7 | 281.6 | 146.4 | 29.2 | 21.5 | 68.5 | 12.9 | 808 | 0.447 | 807 | 0.44 |
| 8 | 275.0 | 148.4 | 25.7 | 20.3 | 63.8 | 12.9 | 813 | 0.490 | 788 | 0.39 |
| 9 | 273.4 | 148.4 | 25.8 | 20.1 | 64.1 | 12.7 | 803 | 0.530 | 774 | 0.33 |
| 10 | 267.0 | 146.4 | 24.3 | 19.2 | 62.0 | 12.6 | 770 | 0.596 | 725 | 0.25 |

- steps 2–10 均值：步时 277.5 s，吞吐 824.7 tokens/s（每卡 206.2），actor MFU 0.44–0.51；10 步共 8.24 M tokens。
- 显存：`actor/perf/max_memory_allocated_gb` 32.6、`max_memory_reserved_gb` 40.2；主机内存 `cpu_memory_used_gb` 208–256。
- reward（gsm8k 精确匹配，`critic/score/mean`）0.229 → 0.596 单调上升；响应长度均值 875 → 725、1024 截断比例 0.53 → 0.25；
  `actor/kl_loss` 3.4e-4 → 8e-3，`grad_norm` 0.018–0.033，`response/aborted_ratio` 0。
- 退出码 0；结束时 vLLM 服务打印 `multiprocessing.resource_tracker` 的 `KeyError('/psm_*')`，为关闭期共享内存清理噪音。

### 6.2 NPU 单测

`.github/workflows/npu_unit_tests.yml` 以 `--ignore-glob="*test_fsdp_lora_merge*"` 排除了该文件。2 × 910B1（容器只挂
这两张卡，已打 `get_npu_versions` 补丁）上 `pytest -v -s tests/utils/test_fsdp_lora_merge.py -k fsdp2`：
`test_merged_lora_context_qwen2[True/False-fsdp2-2]`、`test_merged_lora_context_gptoss[True/False-fsdp2-2]`、
`test_collect_merged_lora_params[all-linear-fsdp2-2]`、`[lora_targets0-fsdp2-2]` 全部 PASSED，6 passed in 199 s。
FSDP1（`strategy=fsdp`）子集未在本任务范围内运行；排除项至少可缩小到 FSDP1。

### 6.3 100 步训练（待补）

计划：同配置 100 步，`save_freq 10`、`test_freq 20`、`val_before_train True`、`resume_mode auto`（共享主机可按 checkpoint
分段续跑）。按 277.5 s/步估计 4 卡约 7.7 h。待完成后补充：训练 reward 曲线、验证集 `val-core/openai/gsm8k/reward/mean@1`
起点/终点、吞吐均值、显存峰值、分段续跑记录。

## 7. 显存与性能分析、调参建议

**显存账（每卡 64 GB，估算 vs 实测）**

| 项 | 估算 | 实测 |
| --- | --- | --- |
| 基座 bf16 16.4 GB，FSDP2 4 卡分片 | ≈4.1 GB/卡常驻，前反向逐层 all-gather | 含在 32.6 GB 峰值内 |
| LoRA 参数（rank 32，all-linear）≈87.3 M | fp32 主权重 + Adam + 梯度 ≈1.2 GB 总量，分片后 <0.4 GB/卡 | 无需 optimizer offload |
| 独立 ref 模型 | 无（`ref_in_actor`） | — |
| vLLM 唤醒时 | 0.6 × 61 GiB ≈ 37 GB | `npu-smi` 训练期 ≈50 GB/卡（含 actor） |
| merge 峰值 | 最大张量（embed/lm_head 151936 × 4096 bf16 ≈ 1.2 GB） | 未观察到额外峰值 |

**时间分解（steps 2–10 均值）**：gen 146 s（53%）> update_actor 67 s（24%）> old_log_prob 28 s（10%）> ref 21 s（8%）>
merge + 权重同步 12.6 s（4.5%）。瓶颈在 rollout 生成。

**可尝试的调参（未在本文档验证，标注为建议）**：

1. rollout：关闭 `enforce_eager` 试 ACL graph 模式；提高 `max_num_batched_tokens`（8192 → 16384/32768）；
   `gpu_memory_utilization` 0.6 → 0.7 以增大 KV cache（4 卡训练期 HBM 仍有约 10 GB 余量）。
2. 训练：`ppo_max_token_len_per_gpu` 24576 可按显存余量上调以减少 micro-batch 数；8 卡时 `rollout_tp` 可保持 2
   （4 个 DP 副本）或试 1。
3. 算法：响应 1024 截断比例初期高达 0.53（Qwen3 思考模式输出长），若关注精度可放宽 `max_response_length` 到 2048 并同步
   调大 `ppo_max_token_len_per_gpu`；本文档为与 GPU 参考脚本对照保持 1024。
4. LoRA：rank 32 / alpha 64 下 `actor/kl_loss` 10 步内升至 8e-3，学习平稳；rank 16 或 lr 5e-6 可作为更保守的对照。

## 8. 昇腾适配点与排障

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `actor_rollout_ref_init_model` 报 `npu-smi info -t board -i 1` exit 215 | `verl/utils/device.py` `get_npu_versions()` 硬编码卡 1；容器只挂部分卡时不存在 | 打 `patches/0001-get_npu_versions-first-visible-npu-id.patch`（按 `npu-smi info -m` 取首个可见 NPU，含单测）；或设 `ASCEND_VISIBLE_DEVICES=<物理卡号>` 走回退路径 |
| 容器内 `torch.npu.device_count()=0`、`aclInit 507899`、`dcmi -8020 device is used` | 共享主机上昇腾驱动按容器命名空间占用设备：`--network host`/`--ipc host` 的容器与任何他人已打开的卡冲突；某张卡被另一容器 open 后到该容器 stop 前都不可用 | 容器用桥接网络 + 私有 IPC（`--shm-size 128g`）；启动前逐卡做一次真实 open 探针（`torch.npu.set_device` + 计算），释放时 `docker stop` 而不只是杀进程 |
| `aclrtMallocHostWithCfg` 207001（主机锁页内存） | FSDP actor param/optimizer CPU offload 触发驱动侧锁页内存上限 | LoRA 路径不需要 offload；保持 `param_offload=False`、`optimizer_offload=False` |
| Ray 告警 `/tmp/ray ... over 95% full` | 主机根分区满 | `++ray_kwargs.ray_init._temp_dir=<可写目录>` |
| 结束时 `resource_tracker KeyError('/psm_*')` | vLLM 服务关闭期共享内存清理 | 忽略，不影响退出码 |
| 镜像 entrypoint `import torch` 触发对首张映射卡的 open | `/usr/local/Ascend/nnal/atb/set_env.sh` 启动时探测 torch | 只在拿到卡后再 `docker run/start`；预建容器须立即 stop |

## 9. 复现步骤

```bash
# 1) 容器（桥接网络 + 私有 IPC，只映射分配到的卡）
docker run -d --name verl-lora --shm-size 128g \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info -v /etc/hccn.conf:/etc/hccn.conf \
  -v /data/work:/workspace/work -w /workspace/work \
  quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm sleep infinity
# 2) 补丁（容器只挂部分卡时）
docker exec verl-lora bash -c 'cd /verl && git apply /workspace/work/lora_rl_merge/patches/0001-get_npu_versions-first-visible-npu-id.patch'
# 3) 数据
docker exec verl-lora bash -c 'cd /verl && python3 examples/data_preprocess/gsm8k.py --local_save_dir /workspace/work/data/gsm8k'
# 4) 冒烟 10 步 / 正式 100 步
docker exec verl-lora bash -c 'cd /verl && NPROC_PER_NODE=4 TOTAL_TRAINING_STEPS=10 SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
  MODEL_PATH=/workspace/work/models/Qwen3-8B DATA_DIR=/workspace/work/data/gsm8k RAY_TEMP_DIR=/workspace/work/ray_tmp \
  bash /workspace/work/lora_rl_merge/run_qwen3_8b_lora_merge_fsdp2_npu.sh'
# 5) 指标表
python3 lora_rl_merge/tools/parse_step_metrics.py <console.log>
```

## 10. 限制与后续

- 仅 10 步冒烟 + 单测已完成；100 步精度/性能证据待排期（第 6.3 节）。
- 吞吐在 `enforce_eager=True`、TP 2、4 卡下测得，未做性能调优；无同配置 GPU 对照数据，性能判据按 issue 的"无 GPU 标杆时
  TPS > 100"兜底。
- 未测：8 卡配置、FSDP1 子集、ACL graph 模式、`max_response_length 2048`。
