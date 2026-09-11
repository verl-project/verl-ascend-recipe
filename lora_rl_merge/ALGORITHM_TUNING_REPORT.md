# Qwen3-8B LoRA-RL（merge）在 Atlas 800T A2 上的算法适配调优文档

对应 verl-ascend-recipe issue [#78](https://github.com/verl-project/verl-ascend-recipe/issues/78)：基于 verl 的
vllm_ascend 推理后端，补齐 Qwen3-8B + GRPO + LoRA（`model.lora.merge=True`，训练中 merge 进 base 模型用于 rollout）
对 FSDP2 后端的支持。配套脚本与文件见本目录 [`README.md`](README.md)。

> 状态说明（2026-09-11）：补丁后已有两条独立轨迹各自连续完成 1–100 步——一条由 4 × 910B1 的第 1–90 步与 4 × 910B3 的第 91–100 步拼接（6.6），一条全程在 4 × 910B3 上从第 1 步跑完（6.7）。两条都没有非有限 rollout 诊断、没有零梯度步，首末十步 reward 分别由 0.3946 升到 0.9270、由 0.3916 升到 0.8356，四卡分母吞吐分别为 689.74（分段 702.64 / 574.36）与 655.51 tokens/s/NPU，checkpoint100 与最终验证齐备，退出码均为 0。原版 sampler 的轨迹每步仍有 7 项非有限 rollout 诊断，保留作对照。两条轨迹的精度峰值都在第 80 步附近（89.46% 与 89.08%），第 100 步分别回落到 87.04% 与 80.67%，单机轨迹第 96 步起出现梯度范数跳升的训练不稳定，详见 6.7。

## 1. 结论概览

| 项目 | 结果 |
| --- | --- |
| 路径可用性 | 参考脚本 `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` 的算法与 LoRA 配置不改，只改 NPU 启动项，即可在 4 × 910B1 上跑通 |
| 10 步冒烟 | 第 2–10 步均值为 277.5 s/步、824.7 tokens/s/NPU；reward 首末 0.229 → 0.596，总体上升但非单调，rc=0 |
| merge 权重同步 | 每步 11.7–12.9 s（占步时 ≈4.5%），vLLM-Ascend 收到全量 bf16 权重，无需推理侧 LoRA |
| 显存 | actor 侧峰值 32.6 GiB allocated / 40.2 GiB reserved（每卡 64 GB），训练中 `npu-smi` ≈50 GB/卡 |
| NPU 单测 | `tests/utils/test_fsdp_lora_merge.py -k fsdp2` 在 2 卡上 6/6 通过（199 s） |
| 需要的适配 patch | verl 的 `get_npu_versions()` 设备探测补丁，以及 vLLM-Ascend 的 sampler `record_stream` 回移补丁分别应用到各自源码目录。 |
| 100 步 / 长跑 | 补丁版两条独立轨迹各自完成 1–100 步：非有限 rollout 诊断均为 0，首末十步 reward 0.3946 → 0.9270（拼接）与 0.3916 → 0.8356（单机），四卡分母吞吐 689.74 与 655.51 tokens/s/NPU，checkpoint100 与最终验证齐备，退出码均为 0。精度峰值在第 80 步附近，跑满 100 步反而更差。原版轨迹保留作对照。 |

## 2. 机制：为什么 merge 路径不依赖 vllm-ascend 的 LoRA 能力

- `verl/workers/engine_workers.py:674` 读取 `model_config.lora.get("merge")` 为 `peft_merge`；`:735` 的说明指出 merge 模式下
  adapter 在权重同步前并入 base，engine 返回全量 HF 键名参数、`peft_config=None`，rollout 收到的是普通权重更新。
- `verl/workers/engine/fsdp/transformer_impl.py:973` 起：merge 模式走 `_merged_lora_per_tensor_param()`，在
  `merged_lora_context(backup_adapters=True)` 内逐张量 `full_tensor()` 并转 bf16 后流式发送；上下文退出时恢复未合并的
  base 权重与 adapter。
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py:401-402`：`lora.merge=True` 时把 `lora_rank` 置 0，vLLM 不启用 LoRA。
- `verl/trainer/ppo/v1/trainer_base.py:316-323`：`lora_rank > 0` 时 `ref_in_actor=True`，参考模型 log-prob 由 actor 在
  `disable_adapter()` 下计算，不构建独立 ref worker（`engine_workers.py:424`）。

本任务基于上述已有实现，补齐固定昇腾环境下的依赖修复、真实训练验证和可复现材料。

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
| `trainer.n_gpus_per_node` | 4（8 卡未验证） | 共享主机实际可用卡数；LoRA 优化器状态极小，4 卡显存充裕 |
| `rollout.enforce_eager` | `True` | 与已验证的 `verl_ascend_practice/run_rl_qwen3_8b_npu.sh` 一致；ACL graph 模式未评估 |
| `rollout.max_num_batched_tokens` | 8192 | 同上 |
| `actor/ref.entropy_from_logits_with_chunking` | `True` | 降低 logits 熵计算峰值显存 |
| `actor/ref.use_torch_compile` | `False` | NPU 上 torch.compile 路径未验证 |
| 环境变量 | `USE_OPTIMIZED_MODEL=0`、`VLLM_USE_V1=1`、`HCCL_CONNECT_TIMEOUT=5400`、`TOKENIZERS_PARALLELISM=false` | 纯推理优化路径干扰 RL 的 log-prob 与在线权重更新；HCCL 首次建链较慢 |
| 启动方式 | 直接 `python3 -m verl.trainer.main_ppo` | 参考脚本的 uv 封装仅对 GPU 生效 |
| `++ray_kwargs.ray_init._temp_dir` | 可写目录 | 共享主机 `/tmp` 常满 |

## 6. 实测结果

### 6.1 10 步冒烟（4 × 910B1，2026-08-20）

| step | 步时 (s) | gen (s) | old_log_prob (s) | ref (s) | update_actor (s) | merge + 权重同步 (s) | 吞吐 (tokens/s/NPU，4 卡) | reward 均值 | 响应长度均值 | 截断比例 |
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

- steps 2–10 均值：步时 277.5 s，吞吐 824.7 tokens/s/NPU（四卡合计约 3299 tokens/s），actor MFU 0.44–0.51。
- 显存：`actor/perf/max_memory_allocated_gb` 为32.6 GiB，`actor/perf/max_memory_reserved_gb` 为40.2 GiB；
  `actor/perf/cpu_memory_used_gb` 为208–256 GiB。
- reward（gsm8k 精确匹配，`critic/score/mean`）首末值 0.229 → 0.596，总体上升但非单调（step 5 → 6 从约 0.404 降至 0.387）；响应长度均值 875 → 725、1024 截断比例 0.53 → 0.25；
  `actor/kl_loss` 3.4e-4 → 8e-3，`grad_norm` 0.018–0.033，`response/aborted_ratio` 0。
- 退出码 0；结束时 vLLM 服务打印 `multiprocessing.resource_tracker` 的 `KeyError('/psm_*')`，为关闭期共享内存清理噪音。

固定 verl 提交的 `verl/workers/engine_workers.py` 将上述内存字节数除以 `1024**3`，因此字段虽以
`_gb` 命名，单位实际为GiB。CPU字段使用 `psutil.virtual_memory().used`，表示进程可见的系统已用内存，
不能解释为actor进程RSS或主机锁页内存。固定 torch_npu 版本提供 `torch_npu.npu.host_memory_stats()`，
其 `allocated_bytes` 和 `reserved_bytes` 分别统计锁页分配器的活动字节与保留字节；`peak` 是按分配桶
汇总的近似峰值。当前训练没有采集该接口，主机锁页内存占用仍未测得。

### 6.2 NPU 单测

`.github/workflows/npu_unit_tests.yml` 以 `--ignore-glob="*test_fsdp_lora_merge*"` 排除了该文件。2 × 910B1（容器只挂
这两张卡，已打 `get_npu_versions` 补丁）上 `pytest -v -s tests/utils/test_fsdp_lora_merge.py -k fsdp2`：
`test_merged_lora_context_qwen2[True/False-fsdp2-2]`、`test_merged_lora_context_gptoss[True/False-fsdp2-2]`、
`test_collect_merged_lora_params[all-linear-fsdp2-2]`、`[lora_targets0-fsdp2-2]` 全部 PASSED，原始总计为 `6 passed, 6 deselected, 1 warning in 199.34s`。
FSDP1（`strategy=fsdp`）子集未运行；这一历史结果不能直接证明当前上游 CI 的排除项可以解除。

### 6.3 正式运行实际完成 58 步（2026-08-28）

原始日志 `lora_merge_100step_4npu_0828T0414Z.log` 连续记录 step 1–58，进度条为 58/100、4:34:03，
随后 `LAUNCH_WRAP_EXIT rc=0`。默认 `TOTAL_EPOCHS=1`，该版本按 `7473 // 128 = 58` 计算每 epoch 步数，
达到 epoch 上限后正常退出。设置 `total_training_steps=100` 不会自动扩大 epoch 上限。
当前脚本将默认 epoch 上限设置为目标步数，并检查日志中的实际训练步数；这两项修正已在随后恢复至100步的作业中执行。

首 10 步 reward 均值为 0.3990234375，末 10 步为 0.90390625。58 步 `perf/throughput` 算术均值为
673.807293 tokens/s/NPU，四卡合计为 2695.229173 tokens/s。验证版本 `metric_utils.py` 的定义为
`total_num_tokens / (time_per_step * n_gpus)`；它不等于含启动、完整验证等开销的全作业吞吐，也不是纯生成速度。
issue 没有明确要求八卡，也未定义 TPS 是整机或每卡；这里披露实际卡数和源码口径，不能据此声称已获维护者接受。

该次最新checkpoint为step50，随后已成功加载并继续训练。58步均存在非有限 `rollout_corr` 指标，
actor loss、grad_norm和reward有限；原版轨迹的数值问题不能由正常退出消除。

### 6.4 原版轨迹恢复到100步（2026-09-10）

固定recipe `4130a43` 从checkpoint50继续完成第51–100步，训练启动器退出0。
统计保留原始第1–50步和续训第51–100步，排除被重新计算的原始第51–58步。首末各10步reward均值为
0.3990234375与0.94755859375，GSM8K greedy准确率由321/1319上升至1216/1319。
66803533个训练步token除以25897.94468626156秒和4张卡，得到644.8729214739403 tokens/s/NPU。
该分母不含停机间隔、启动、恢复以及部分验证时间，不能表述为全作业吞吐。

100步actor loss和梯度均有限、梯度均非零，但每步均有非有限rollout诊断。原始第51步有效token中有
366个rollout `-inf`，其training log-prob有限，K3中的 `exp(log_ratio)-log_ratio-1` 因此产生366个NaN。
关闭IS/RS时，原函数重放确认不生成importance weights且不改变response mask；这不证明生成的token正确。
checkpoint100文件和额外状态已检查，但该checkpoint尚未实际加载模型和优化器继续训练。

### 6.5 sampler生命周期对照与修复

固定镜像的 vLLM-Ascend `a43c8cc8057f490ed1df2c6ed66253e2d7817da4` 在辅助stream生成随机张量 `q`，
等待生产stream完成后由当前stream消费。`wait_stream` 建立执行顺序，却不能单独阻止分配器提前重用该内存。
回移的[上游PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394)增加
`q.record_stream(torch.npu.current_stream())`，其主分支合入提交为
`fc0ce85b58f019e5a1988dbd3f39d014052fb44b`。

同一物理卡0、相同镜像及脚本，以Qwen3-8B base模型、TP1、32个GSM8K prompt各采样8条序列进行对照。
原版224621个token出现4个越界ID151669，对应log-prob均为 `-inf`；补丁组225828个token中两类异常均为0。
tokenizer长度为151669，因此该ID位于有效范围之外。两组启动器均退出0，实际sampler差异仅新增上述一行。
原始训练捕获没有保存response token IDs，不能把此对照逐token对应到第51步的366个异常位置。

本次样本支持生命周期修复，但不排除其他低概率问题，也不代替LoRA merge、TP2完整训练。
recipe `84142b1` 已用独立checkpoint目录启动修复后的四卡100步作业，保留save10与resumeauto；结果仍待完成。

### 6.6 修复版 sampler 的完整 100 步验证（2026-09-11）

固定 recipe `84142b1` 与 vLLM-Ascend sampler `record_stream` 回移补丁的独立轨迹。
第 1–90 步在原主机的 4 × 910B1 上完成；该主机其余可用卡已被其他使用者的容器显式声明，
四次续训尝试中有三次被外部 Docker `stop` 请求终止（`OOMKilled=false`，本方启动器不含任何
`docker stop`），因此把 `global_step_90` 直连迁移到另一台的 4 × 910B3 完成第 91–100 步。
两段的 verl 提交 `bc72e38e`、vLLM-Ascend 提交 `a43c8cc8`、镜像内容（12 层 RootFS diff ID
与运行时配置字段哈希一致）、两个 patch 与全部训练超参相同，容器内逐字节比对了
`runtime.patch`、`vllm-ascend-runtime.patch` 和 `packages.txt`，并做了真实四卡计算。
统计只取原运行的第 1–90 步与本次的第 91–100 步，排除被替代的旧第 91–99 步。

| 指标 | 原版 sampler（1–100 步，4 × 910B1） | 修复版（1–90 步，4 × 910B1） | 修复版（91–100 步，4 × 910B3） |
| --- | --- | --- | --- |
| 非有限 `rollout_corr` | 7 项 × 100 步 | 0 | 0 |
| `actor/loss`、`actor/grad_norm` 有限 | 是 | 是 | 是 |
| 零梯度步 | 无 | 无 | 无 |
| 平均步时 | — | 259.88 s | 261.51 s |
| 加权吞吐（tokens/s/NPU，4 卡分母） | 644.87 | 702.64 | 574.36 |

首 10 步 reward 均值 0.39462890625，末 10 步 0.926953125，上升；合并轨迹的加权吞吐为
689.74 tokens/s/NPU。GSM8K greedy 准确率：step 0 为 322/1319、step 20 为 929、step 40 为 1045、
step 60 为 1119、step 80 为 1180、step 100 为 1148（87.04%）。恢复时对同一 `global_step_90`
的验证在原主机为 1198（90.90%）、在新主机为 1205（91.36%），相差 7 条样本，说明 FSDP2 分片
checkpoint 跨 910B1 → 910B3 恢复正确。

**第 100 步的验证低于第 80 步（89.46%）与第 90 步（91.36%）**，末 10 步 reward 也在
0.8867–0.9717 之间波动（第 96 与第 100 步均为 0.8867）。该配置在第 80–100 步已进入平台并有波动，
不是单调上升；本次没有做多随机种子重复，不能区分随机波动与后期过拟合，也不声称第 100 步是精度上限。

**吞吐口径说明**：两段平均步时相近（259.88 s 与 261.51 s），吞吐差异主要来自每步 token 总量下降——
`response_length/mean` 由第 1 步的 895 降到第 90 步的 525、第 100 步的 517。910B1 自身的第 1–10 步为
810.54 tokens/s/NPU、第 81–90 步已降到 593–651，同一趋势在 910B3 段延续。因此本节按硬件分段披露，
不把两段合并成单一 TPS 结论，也不据此比较两代硬件的性能。可直接归因于 sampler 补丁的只有
「非有限 rollout 诊断消失」这一项——两条轨迹的 recipe 提交分别为 `4130a43` 与 `84142b1`，
差异不只有该补丁，吞吐差异不能单独归因。

### 6.7 单机 910B3 的独立 100 步轨迹与后期不稳定（2026-09-11）

为消除 6.6 的跨硬件拼接这一可质疑点，在同一台 4 × 910B3 上以完全相同的 recipe `84142b1`、
镜像内容、两个 patch 和全部训练超参从第 1 步重跑了一条完整轨迹（仅运行目录、`default_local_dir`
与 Ray 临时目录不同）。历时 8.04 小时，退出码 0。

| 指标 | 拼接轨迹（1–90 在 910B1） | 单机轨迹（全程 910B3） |
| --- | --- | --- |
| 步数连续 | 1–100 | 1–100 |
| 非有限 `rollout_corr` | 0 | 0 |
| 零梯度步 | 无 | 无 |
| 首 10 步 → 末 10 步 reward | 0.39463 → 0.92695 | 0.39160 → 0.83555 |
| 加权吞吐（tokens/s/NPU，4 卡分母） | 689.74（分段 702.64 / 574.36） | 655.51 |
| 最低逐步吞吐 | 551.23 | 565.17 |
| step 80 GSM8K | 89.46% | 89.08% |
| step 100 GSM8K | 87.04% | 80.67% |

**两代硬件在同等负载下性能相当**：两条轨迹的第 1 步为 813 与 786.4 tokens/s/NPU（步时 308.4 与
318.7 s），第 10/11 步为 731 与 725.9（步时 288.7 与 290.9 s），差别在 3% 左右。
6.6 中 91–100 段吞吐偏低是因为响应长度已降到约 500 token，每步 token 总量下降，不是硬件差异。

**后期训练不稳定**：单机轨迹第 81–95 步的 reward 稳定在 0.8594–0.9443、`actor/grad_norm`
在 0.0435–0.8906、`response_length/mean` 在 482–547；第 96 步起 reward 依次为 0.8174、0.8145、
0.7881、0.6982、0.6113，`grad_norm` 依次为 0.6250、2.1797、2.1289、2.0723、2.7148，
响应长度增至 543、549、566、608、668。梯度范数跳升约 5–10 倍的同时 reward 塌落、响应重新变长。
拼接轨迹的第 91–100 步没有出现（`grad_norm` 仅 0.0846–0.7689）。两条轨迹在第 80 步前逐点吻合、
之后分岔，因此这是 run-to-run 随机性，不是硬件差异，本次只观察到一次，没有量化发生概率，
也没有做学习率、梯度裁剪或 KL 系数的消融来定位触发原因。

**实践建议**：该配置的精度峰值在第 80 步附近。以精度为目标时取 checkpoint 80，
或在第 80 步之后监控 `actor/grad_norm`，出现数量级跳升时早停；
不要默认把第 100 步当作最优点。

## 7. 显存与性能分析、调参建议

**显存占用（每卡64 GB，估算与实测对照）**

| 项 | 估算 | 实测 |
| --- | --- | --- |
| 基座 bf16 16.4 GB，FSDP2 4 卡分片 | ≈4.1 GB/卡常驻，前反向逐层 all-gather | 含在 32.6 GiB 峰值内 |
| LoRA 参数（rank 32，all-linear）≈87.3 M | fp32 主权重 + Adam + 梯度 ≈1.2 GB 总量，分片后 <0.4 GB/卡 | 无需 optimizer offload |
| 独立 ref 模型 | 无（`ref_in_actor`） | — |
| vLLM 唤醒时 | 0.6 × 61 GiB ≈ 36.6 GiB | `npu-smi` 训练期 ≈50 GB/卡（含 actor） |
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
| 结束时 `resource_tracker KeyError('/psm_*')` | 历史日志出现在vLLM服务关闭期 | 保留告警，结合训练启动器退出码、完成步数和checkpoint判断；不能据此忽略运行期错误。 |
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
# 2) verl补丁（容器只挂部分卡时），以及固定vLLM-Ascend版本必需的sampler补丁
docker exec verl-lora bash -c 'cd /verl && git apply /workspace/work/lora_rl_merge/patches/0001-get_npu_versions-first-visible-npu-id.patch'
docker exec verl-lora bash -c 'cd /vllm-ascend && test "$(git rev-parse HEAD)" = a43c8cc8057f490ed1df2c6ed66253e2d7817da4 && git apply --check /workspace/work/lora_rl_merge/patches/vllm-ascend/0001-record-sampler-stream.patch && git apply /workspace/work/lora_rl_merge/patches/vllm-ascend/0001-record-sampler-stream.patch'
# 3) 数据
docker exec verl-lora bash -c 'cd /verl && python3 examples/data_preprocess/gsm8k.py --local_save_dir /workspace/work/data/gsm8k'
# 4) 冒烟 10 步；正式训练设置 TOTAL_TRAINING_STEPS=100，并保留默认 checkpoint 保存与自动恢复配置
docker exec verl-lora bash -c 'cd /verl && NPROC_PER_NODE=4 TOTAL_TRAINING_STEPS=10 SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
  MODEL_PATH=/workspace/work/models/Qwen3-8B DATA_DIR=/workspace/work/data/gsm8k RAY_TEMP_DIR=/workspace/work/ray_tmp \
  bash /workspace/work/lora_rl_merge/run_qwen3_8b_lora_merge_fsdp2_npu.sh'
# 5) 指标表
python3 lora_rl_merge/tools/parse_step_metrics.py <console.log>
```

## 10. 限制与后续

- 已有历史10步冒烟、FSDP2单测和原版恢复后的100步训练；sampler补丁后的完整训练仍未完成（第6.4–6.5节）。
- 吞吐在 `enforce_eager=True`、TP 2、4 卡下测得，未做性能调优；无同配置 GPU 对照数据，性能判据按 issue 的"无 GPU 标杆时
  TPS > 100"兜底。
- 未测：8 卡配置、FSDP1 子集、ACL graph 模式、`max_response_length 2048`。
