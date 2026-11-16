# Qwen3.5-35B MXFP8 R2 长跑代码差异说明

本文记录 `GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2` 使用的低精代码面，并解释相对 PR #8511 与旧 recipe 的差异。目标是交付这次长跑实际依赖的代码，而不是旧探索分支。

## 1. 结论

R2 是训练与 rollout 同为低精的语义：

- 训练侧：actor/ref 使用 FSDPTurbo 动态 MXFP8/E4M3，并排除 `linear_attn`。
- rollout 侧：`actor_rollout_ref.rollout.quantization=ascend` 走 vLLM-Ascend 原生 ModelSlim MXFP8。
- RL 权重同步：verl 将高精 master 权重导出为 FP8 权重和 per-group scale，再异步 refit 到 vLLM。
- checkpoint：保存 BF16 master 权重，不保存 rollout 侧临时 FP8 权重。

因此 checkpoint 中出现 BF16 权重不代表低精未生效；低精生效点在训练前向和 rollout 同步路径。

## 2. 代码固定方式

R2 实际运行信息：

| 项 | 值 |
| :--- | :--- |
| 运行名 | `GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2` |
| 机器 | `177.5.28.1` |
| 启动脚本 | `/mnt/share/j00949742/tmp/20261104_pr8511/run_exp1102_08_pr8511_dualmxfp8_lpoff_20261110.sh` |
| 日志 | `/mnt/share/j00949742/tmp/20261104_pr8511/nohup_exp08_MXFP8LR200_R2_20261114_2200.out` |
| 模型 | `/mnt/share/j00949742/models/Qwen3.5-35B-A3B-MXFP8-nolin` |
| verl 工作树 | `/mnt/share/j00949742/verl_turbo_1102/verl_pr8511` |

recipe 使用固定基线 commit + patch 复现：

| 依赖 | 基线 | patch | 规模 |
| :--- | :--- | :--- | :--- |
| verl | `3d050639`（PR #8511 head） | `patch/verl.patch` | 4 个文件，76 行新增、5 行删除 |
| FSDPTurbo | `759a348` | `patch/fsdpturbo.patch` | 2 个文件，136 行新增 |
| vLLM-Ascend | `c09d3e2` | 无 patch | 0 个文件 |
| vLLM | `ced6857` | 无 patch | - |

patch 校验：

| patch | SHA256 |
| :--- | :--- |
| `patch/verl.patch` | `C9A5055AC605CD374BD1F878EA260EA6C212EA2F9A30D0F1BBE39B6D2688BFCB` |
| `patch/fsdpturbo.patch` | `B539BC13ACD1708E9F004B9919C5D42404949729858EF808DD669046F59E4E30` |

- `verl.patch` 由 `git diff 3d050639..eb485584` 生成，只包含 PR #8511 之后的 R2 修正。
- `verl.patch` 已通过 `git apply --check --reverse` 校验，可在 `eb485584` 工作树上完全逆向还原。
- FSDPTurbo patch 来自 R2 工作树的实际未提交源码改动。
- vLLM-Ascend 不使用旧在线量化 patch，也不设置 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT`。

## 3. R2 日志证据

日志可直接确认：

1. `EXPERIMENT_NAME=GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2`。
2. `MODEL_PATH=/mnt/share/j00949742/models/Qwen3.5-35B-A3B-MXFP8-nolin`。
3. `QUANT_MODE=ascend`、`MXFP8=1`。
4. actor/ref 均展开 7 项 FSDPTurbo MXFP8 训练侧配置。
5. 命令行包含 `actor_rollout_ref.rollout.quantization=ascend`。
6. 命令行包含 `trainer.total_training_steps=200`、`trainer.save_freq=100`。
7. worker 日志出现 `[P2a] fused swiglu MX MoE no-grad path engaged`。
8. vLLM 日志出现 `FP8 weights loaded (async), loaded_params: ...`，说明 FP8 权重与 scale 进入 rollout。

console 日志未保留 `[GDN-REPACK]` INFO 行，因此不能仅凭日志证明该分支打印过；本交付以源码存在该修复、权重加载成功和 CPU 单测为准，不把未保留的日志写成证据。

## 4. 启动脚本相对 BF16 对照的差异

### 4.1 训练侧 7 项 gate

`MXFP8=1` 时 actor/ref 各追加：

| 配置 | 值 | 作用 |
| :--- | :--- | :--- |
| `quant_recipe` | `mxfp8` | 启用训练侧动态 MXFP8。 |
| `quant_format` | `E4M3` | 使用 FP8 E4M3。 |
| `converters` | `quantize.linear.mx`, `quantize.moe.mx` | 覆盖 Linear 与 MoE 专家路径。 |
| `quant_apply_modules` | `model.language_model.layers.{*}` | 限定语言模型层。 |
| `quant_ignored_modules` | `model.language_model.layers.{*}.linear_attn.*` | 排除对量化敏感的 GDN/linear attention。 |
| `enable_fsdp_low_precision_all_gather` | `False` | R2 实际值，不引入低精 all-gather 变量。 |
| `fsdp_low_precision_all_gather_mode` | `on-demand` | 保持 FSDPTurbo 按需模式。 |

`linear_attn` 的 `in_proj_qkv`、`in_proj_z`、`out_proj` 会被量化误差沿递归状态放大。两侧排除后，step0 验证从全量 MXFP8 的均值约 380 恢复到 BF16 簇约 400 左右。

### 4.2 rollout 侧固定 ascend 量化

```bash
QUANT_MODE=${QUANT_MODE:-ascend}
actor_rollout_ref.rollout.quantization=${QUANT_MODE}
```

`ascend` 是 vLLM-Ascend 原生字符串路由，必须直接传入 engine kwargs；不能当成 verl 结构化 compressed-tensors 配置解析。

### 4.3 不使用 vLLM 在线量化开关

R2 不设置 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT`，也不应用旧 vLLM-Ascend patch。

原因是 R2 的 RL refit 路径由 verl 在每次权重同步前导出 FP8 权重和 scale，vLLM-Ascend 直接加载这些已量化张量。旧在线量化 patch 解决的是“初始 BF16 checkpoint + ModelSlim 描述如何在线量化”的问题，不是 R2 结果的必要条件。

### 4.4 模型目录

R2 使用：

```text
/mnt/share/j00949742/models/Qwen3.5-35B-A3B-MXFP8-nolin
```

含义：

- 权重本体是 BF16。
- `quant_model_description.json` 描述哪些层按 `W8A8_MXFP8` 处理。
- `nolin` 表示 `linear_attn` 为 `FLOAT`。

### 4.5 长跑配置

recipe launcher 固化了 R2 的 200 step、100 step 保存、EP `all2all + use_fused_ops=True`、fused causal conv1d/GDN、`weight_nz_mode=0` 等实际配置。

## 5. 相对 PR #8511 的 verl 新增

PR #8511 已包含 Ascend ModelSlim 探测与路由、`ModelSlimMxfp8WeightExporter` 基础实现、异步 vLLM 字符串 quantization 传递、refit 前后的恢复/处理调用和 CPU 单测基础。这些内容不计入本次 recipe patch。

本次新增基于：

```text
3d050639 PR #8511 head
  + 65177e1d fix(ascend): correct ModelSlim MXFP8 rollout refit
  + eb485584 fix(ascend): preserve packed expert scale shape
```

净新增 4 个文件：

| 文件 | 修改 | 修复 |
| :--- | :--- | :--- |
| `verl/workers/engine/fsdp/transformer_impl.py` | FSDP engine 增加 rollout 权重量化器接入；checkpoint 导出时临时禁用 | rollout 侧低精同步生效，同时 checkpoint 保持 BF16 master 权重 |
| `verl/utils/qat/ascend_modelslim.py` | expert index 折回 `experts.0` 模板；量化前统一 BF16 contiguous；scale 保留 flatten 结果 | 修复只量化 expert 0、输入 dtype/布局不一致和 packed expert scale 形状错误 |
| `verl/utils/vllm/vllm_quant_utils.py` | HF→vLLM 权重名映射；每次 refit 后重打包 GDN conv | 修复权重名查不到、GDN conv 派生 buffer 残留 dummy 值 |
| `tests/utils/test_ascend_mxfp8_on_cpu.py` | 增加 expert 模板键单测 | 防止后续合入时回退为只处理 expert 0 |

## 6. verl.patch 逐项说明

### 6.1 FSDP rollout 量化接线

`FSDPEngine` 新增 `_rollout_weight_quantizer` 和 `set_rollout_weight_quantizer()`。`get_per_tensor_param()` 在 LoRA/base 同步语义允许时：

1. 调用量化器的 `refresh_input_global_scales()`（若存在）。
2. 将高精 per-tensor 权重交给 `ModelSlimMxfp8WeightExporter`。
3. 输出 FP8 权重和 `_scale`。

checkpoint 导出使用独立的 `get_hf_checkpoint_param()` 语义，并临时把量化器置空，避免把 rollout 临时 FP8 权重写进磁盘 checkpoint。

### 6.2 expert 模板键映射

ModelSlim 描述通常只保存 `experts.0` 模板键，而实际权重包含所有 expert。新增 `_modelslim_template_key()` 将任意 `.experts.<index>.` 折回 `.experts.0.`，使所有 expert 都命中同一层的 `W8A8_MXFP8` 规则。

若不修复，只有 expert 0 被量化，其他 expert 可能保留错误 dtype 或未初始化 scale。

### 6.3 scale 数值和形状

导出前统一 `weight.to(torch.bfloat16).contiguous()`，与 R2 训练侧语义和 `npu_dynamic_mx_quant` 输入契约一致；scale 使用 `flatten(-2, -1)` 保留算子的完整 per-group grid。

`eb485584` 特别修复 packed expert `down_proj` scale：不再额外 `squeeze(-1)`，使其保持 `[expert_count, 1]`，匹配 vLLM/ModelSlim 对 packed expert 的 scale 布局。该问题来自 vLLM PR review，CPU 单测已覆盖。

### 6.4 HF→vLLM 权重名映射

新增 `translate_weight_names_for_vllm()`，加载前用模型上的 `hf_to_vllm_mapper` 将 HF checkpoint 权重名映射到 vLLM 模块名。

修复的问题：直接按 HF 名查 vLLM 参数会得到 `None`，权重更新被跳过，导致 refit 后仍保留 dummy 或旧权重。

### 6.5 GDN conv 重打包

新增 `refresh_ascend_gdn_conv_weights()`，每次 `process_quanted_weights_after_loading()` 后遍历模块并调用 `_pack_conv_weights()`。

修复的问题：RL refit 会改回原始参数布局，但 GDN conv 的 kernel-layout buffer 不重建时，推理继续使用旧布局或 dummy 值，导致输出错误。

## 7. FSDPTurbo.patch 逐项说明

### 7.1 no-grad 分支

`fsdp_turbo/distributed/expert_parallel/dispatcher.py` 在满足以下条件时进入 P2a：

- `torch.is_grad_enabled()` 为 false。
- quant config 是 `MXFP8LinearConfig`。
- 非 W4A16/W4A8。
- 无 ep balance、clipping/SiTU、其他激活函数配置。

该分支不影响带梯度 backward、W4 QAT 和带附加策略的 MoE。

### 7.2 fused SwiGLU group GEMM

`fsdp_turbo/quantization/mx_formats/mx_gmm.py` 新增 `mx_fused_swiglu_group_gemm()`：

- 用 `npu_grouped_matmul_swiglu_quant_v2` 融合 gate/up GMM、SiLU*up 和输出量化。
- 用 `npu_grouped_matmul` 执行 down GMM。
- 按 kernel 契约 pack activation scale。
- 对 weight scale 形状做显式检查。

修复的问题：训练侧 no-grad scoring 与 rollout 路径原先存在“先回 BF16、再重复量化”的不对称，每层引入约 1.2%~1.4% cross-feed 误差。P2a 使用与 rollout 相同的融合 kernel 消除这段额外误差。

## 8. vLLM-Ascend 边界

vLLM-Ascend 固定 `c09d3e2`，不应用 patch：

- 初始模型加载使用原生 ModelSlim MXFP8 权重加载。
- RL refit 由 verl 发送 FP8 权重和 scale。
- 不需要 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT`。

旧 vLLM 在线量化 patch 曾用于探索“BF16 checkpoint + ModelSlim 描述在 vLLM 内在线量化”的方案，但 review 发现 repeated BF16 reload 存在 staging buffer dtype 问题，且该路径不是 R2 成功结果的必要依赖。因此不作为交付内容，避免把非必要且未完全验证的开关混入 recipe。

## 9. checkpoint 为什么是 BF16

这是有意设计：

1. FSDP trainer master 权重保持高精。
2. rollout 同步时，`get_per_tensor_param()` 临时经过 `ModelSlimMxfp8WeightExporter`。
3. checkpoint 导出时临时禁用该量化器。
4. 磁盘 checkpoint 保存 BF16 master 权重。

下游评测：

- 按普通 BF16 权重目录加载。
- 不需要 `quant_model_description.json`。
- 不开启 `--quantization ascend`。

## 10. 交付边界

- 本文档固定 2026-11-16 整理的 R2 实际代码面。
- verl patch 只包含 PR #8511 head 之后的 R2 新增，不重复提交 PR #8511 内容。
- FSDPTurbo patch 是本次长跑必要新增。
- vLLM-Ascend 无 patch，且不依赖在线量化开关。
- 远端 `.bak_*` 调查备份不进入交付。
