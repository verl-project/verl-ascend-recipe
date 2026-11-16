# Qwen3.5-35B MXFP8 R2 长跑代码差异说明

本文档记录 `GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2` 这次低精长跑使用的代码面，并解释相对原先代码的改动原因。目标是保证 recipe 交付的是这次 R2 运行的低精代码，而不是旧实验分支或探索性代码。

## 1. 结论

这次 R2 是明确的“双低精”语义：

- **训练侧**：actor 和 ref 的 FSDPTurbo 动态量化计划启用 MXFP8/E4M3。
- **rollout 侧**：vLLM-Ascend 使用 `actor_rollout_ref.rollout.quantization=ascend`，并配合 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT=1`。
- **checkpoint**：仍保存高精 BF16 master 权重；MXFP8 只发生在 rollout 权重同步/加载路径。

因此，不能把 checkpoint 目录中出现 BF16 权重理解为“低精没有生效”。低精生效点在训练前向和 rollout 引擎同步，checkpoint 保存时刻意绕过 rollout 量化器。

## 2. 这次代码的来源和固定方式

R2 实际运行信息：

| 项 | 值 |
| :--- | :--- |
| 运行名 | `GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2` |
| 机器 | `177.5.28.1` |
| 启动脚本 | `/mnt/share/j00949742/tmp/20261104_pr8511/run_exp1102_08_pr8511_dualmxfp8_lpoff_20261110.sh` |
| 日志 | `/mnt/share/j00949742/tmp/20261104_pr8511/nohup_exp08_MXFP8LR200_R2_20261114_2200.out` |
| 模型 | `/mnt/share/j00949742/models/Qwen3.5-35B-A3B-MXFP8-nolin` |
| verl 工作树 | `/mnt/share/j00949742/verl_turbo_1102/verl_pr8511` |

recipe 采用“固定基线 commit + 精确 patch”的方式复现该代码面：

| 依赖 | 基线 | patch | 规模 |
| :--- | :--- | :--- | :--- |
| verl | `7454326` | [`patch/verl.patch`](patch/verl.patch) | 7 个文件，508 行新增、6 行删除 |
| FSDPTurbo | `759a348` | [`patch/fsdpturbo.patch`](patch/fsdpturbo.patch) | 2 个文件，136 行新增 |
| vLLM-Ascend | `c09d3e2` | [`patch/vllm-ascend.patch`](patch/vllm-ascend.patch) | 1 个文件，78 行新增、3 行删除 |
| vLLM | `ced6857` | 不需要 patch | - |

patch 的 SHA256：

| patch | SHA256 |
| :--- | :--- |
| `patch/verl.patch` | `6E7E6E7A149323CCA2FA04E9C4A29C543EED5BD8697D02F08AE92A69688263C8` |
| `patch/fsdpturbo.patch` | `B539BC13ACD1708E9F004B9919C5D42404949729858EF808DD669046F59E4E30` |
| `patch/vllm-ascend.patch` | `7FC433C4114E4CA7BEF4A40F161E8D0809293798B4DC9D91C8134EF24B0A1A48` |

导出与校验方式：

- `verl.patch` 来自 `git diff origin/main`，包含 R2 工作树相对 `origin/main@7454326` 的全部提交和未提交源码改动。
- `fsdpturbo.patch` 和 `vllm-ascend.patch` 来自对应仓 `git diff`，只包含 R2 工作树上实际存在的未提交源码改动。
- 三份 patch 在远端均做过 `git apply --check --reverse`，确认能完全逆向还原为实际工作树状态。
- 远端工作树中的 `.bak_*` 备份文件不进入 patch，也不属于交付内容。

## 3. R2 日志证据

R2 日志开头能直接确认以下事实：

1. `EXPERIMENT_NAME=GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2`。
2. `MODEL_PATH=/mnt/share/j00949742/models/Qwen3.5-35B-A3B-MXFP8-nolin`。
3. `QUANT_MODE=ascend`、`MXFP8=1`。
4. actor 和 ref 均展开 7 项 MXFP8 训练侧量化配置。
5. 命令行包含 `actor_rollout_ref.rollout.quantization=ascend`。
6. 命令行包含 `trainer.total_training_steps=200`、`trainer.save_freq=100`。
7. worker 日志出现 `[P2a] fused swiglu MX MoE no-grad path engaged`，确认 FSDPTurbo 的 P2a fused 路径生效。
8. vLLM 子进程日志出现 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT` 环境变量，确认该开关传入引擎进程。

说明：当前 console 日志没有保留 `[SCALEFIX] online-quantized` 或 `[GDN-REPACK]` INFO 行。因此不能仅凭 console 输出证明这两个日志点曾打印；但 R2 使用的源码树确实包含对应代码，且环境开关已传入 vLLM。本文档只把日志能直接证明的内容写成证据。

## 4. 启动脚本相对 BF16 对照的差异

R2 的 MXFP8 脚本由 BF16 对照脚本派生。真正构成低精实验变量的差异如下。

### 4.1 训练侧 7 项 gate

`MXFP8=1` 时，actor 和 ref 各追加以下配置：

| 配置 | 值 | 作用 |
| :--- | :--- | :--- |
| `quant_recipe` | `mxfp8` | 启用 MXFP8 训练侧动态量化。 |
| `quant_format` | `E4M3` | 使用 FP8 E4M3 数据格式。 |
| `converters` | `quantize.linear.mx`, `quantize.moe.mx` | 同时覆盖普通 Linear 和 MoE 专家路径。 |
| `quant_apply_modules` | `model.language_model.layers.{*}` | 量化作用域限定语言模型层。 |
| `quant_ignored_modules` | `model.language_model.layers.{*}.linear_attn.*` | 排除对 MXFP8 极敏感的 GDN/linear attention。 |
| `enable_fsdp_low_precision_all_gather` | `False` | R2 实际值。低精 all-gather 不作为本次实验变量。 |
| `fsdp_low_precision_all_gather_mode` | `on-demand` | 保持 FSDPTurbo 的按需模式。 |

修复的问题：

- 旧实验曾把 `linear_attn` 一起量化，导致递归线性注意力误差沿序列累积，step0 验证集明显掉点。
- 该配置把 `in_proj_qkv`、`in_proj_z`、`out_proj` 等 `linear_attn` 投影留在高精，实验结果回到 BF16 簇。
- `enable_fsdp_low_precision_all_gather=False` 是脚本实际展开值；文件注释里提到的 True 是历史讨论，不代表 R2 实际配置。

### 4.2 rollout 侧固定 ascend 量化

新增：

```bash
QUANT_MODE=${QUANT_MODE:-ascend}
actor_rollout_ref.rollout.quantization=${QUANT_MODE}
```

修复的问题：

- 旧脚本存在 `MXFP8_ROLLOUT` 条件分支，容易让使用者以为训练侧和 rollout 侧可以独立选择。
- R2 的实际语义是双低精：rollout 必须传给 vLLM-Ascend 原生 `ascend` 量化路由。
- 去掉条件分支可以避免“训练侧低精、rollout 实际仍 BF16”或相反的错误复现。

### 4.3 在线权重量化开关

新增：

```bash
VLLM_ASCEND_ONLINE_WEIGHT_QUANT=1
export VLLM_ASCEND_ONLINE_WEIGHT_QUANT
```

原因：

- R2 的模型目录是 BF16 权重 + `quant_model_description.json`，不是磁盘上已经保存 FP8 权重和 scale 的目录。
- vLLM-Ascend 需要在加载后用 BF16 权重在线生成 MXFP8 权重和 per-group scale。

### 4.4 模型目录

R2 使用：

```text
/mnt/share/j00949742/models/Qwen3.5-35B-A3B-MXFP8-nolin
```

含义：

- 权重本体是 BF16。
- `quant_model_description.json` 描述哪些层按 `W8A8_MXFP8` 处理。
- `nolin` 表示排除 `linear_attn` 的量化描述。

### 4.5 外层 wrapper 追加的长跑配置

R2 实际命令由外层点火 wrapper 在基础脚本后追加，包括：

- actor/ref 的 `use_fused_causal_conv1d=True`、`use_fused_gdn=True`、`model_name_or_path=qwen3_5_moe`。
- `actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0`。
- `actor_rollout_ref.rollout.engine_kwargs.vllm.mm_device_do_normalize=false`。
- `trainer.total_training_steps=200`。
- `trainer.save_freq=100`。
- `trainer.default_local_dir=.../GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2`。

这些不是训练侧低精开关本身，但属于 R2 长跑实际配置，recipe 已固化到 launcher。

## 5. 相对旧 recipe 依赖面的差异

旧 recipe 分支使用的是另一套过渡依赖：

| 依赖 | 旧 recipe | R2 实际长跑 |
| :--- | :--- | :--- |
| verl | `JingSiyu1/verl@fsdpturbo-ascend-fixes` | `verl@7454326 + verl.patch` |
| vLLM | `0fc695f` | `ced6857` |
| vLLM-Ascend | `f16cfd1`，无 patch | `c09d3e2 + vllm-ascend.patch` |
| FSDPTurbo | `b1793de`，无 patch | `759a348 + fsdpturbo.patch` |

旧 recipe 的 `fsdp_turbo` 目录原本不携带 patch，依赖“修正将合入上游”的假设。R2 的关键修正尚未以完整形态存在于这些旧 pin 中，因此必须改用固定 commit + patch。

## 6. 相对 PR #8511 的新增边界

verl 代码确实以 [PR #8511](https://github.com/verl-project/verl/pull/8511) 为基础。R2 使用的关系是：

```text
origin/main@7454326
  + PR #8511 head f995fdd
  -> merge 80caa31
  + 602df04 FSDP rollout weight quantizer wiring
  + 2df4b43 set_current_vllm_config guard
  + c2a1e5a translate_weight_names_for_vllm
  + f961788 experts template key fix
  + 当前工作树的 SCALEFIX-B / GDN-REPACK 未提交修正
```

PR #8511 已经包含的 verl 能力，不算本次新增：

- `AscendMxfp8WeightQuantizer` 的基础实现。
- `rollout.quantization=ascend` 到 Ascend ModelSlim quantizer 的 dispatch。
- vLLM-Ascend MXFP8 模型识别。
- refit 前后的 `restore_weights_for_rl_loading` / `process_weights_after_loading` 调用。
- 异步 vLLM server 对字符串 quantization 的原生传递。
- Ascend MXFP8 CPU 单测基础。

PR merge 点 `80caa31` 之后，我们在 verl 中新增/修改的只有 4 个文件：

| 文件 | 新增内容 | 修复问题 |
| :--- | :--- | :--- |
| `verl/workers/engine/fsdp/transformer_impl.py` | FSDP engine 增加 `_rollout_weight_quantizer`、`set_rollout_weight_quantizer()`，rollout 流过量化器，checkpoint 流临时禁用量化器 | 把 rollout 同步和 checkpoint 保存解耦；checkpoint 保持 BF16 master 权重 |
| `verl/workers/rollout/vllm_rollout/utils.py` | 在 post-process 外包 `set_current_vllm_config()` guard | 修复 multiproc RPC context 中 vLLM config 缺失导致的 assertion |
| `verl/utils/vllm/vllm_quant_utils.py` | `translate_weight_names_for_vllm()` 和 `refresh_ascend_gdn_conv_weights()` | 修复 HF/vLLM 权重名映射失败；修复 GDN conv 派生权重残留 dummy 值 |
| `verl/utils/qat/ascend.py` | expert index 折回 `experts.0` 模板键；导出前转 BF16；scale 用 flatten+squeeze | 修复只量化 expert 0；修复 scale 形状和数值链路 |

注意：`f961788` 曾加入 L0 计数日志；最终 R2 工作树为了减少噪声移除了这部分日志。因此“相对 PR #8511 的净新增”不包含 L0 计数。

`verl.patch` 之所以同时包含 PR #8511 内容和后继修正，是因为 recipe 的基线是 `origin/main@7454326`，而 PR 当时尚未进入该 main pin。若后续 PR #8511 已合入上游，再向上游提交时，应以 `80caa31..R2工作树` 的差异作为我们真正的新增 patch。

PR #8511 之外的另外两处新增：

- **FSDPTurbo**：`fsdpturbo.patch` 全部是本次新增，核心是 P2a no-grad fused SwiGLU MX MoE 路径。
- **vLLM-Ascend**：`vllm-ascend.patch` 全部是本次新增，核心是 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT=1` 在线权重量化。

## 7. verl.patch 逐项说明

### 7.1 `verl/utils/qat/ascend.py`

新增 `AscendMxfp8WeightQuantizer`。

**修改内容**

- 从 `quant_model_description.json` 读取 ModelSlim 描述。
- 将训练侧权重导出为 vLLM-Ascend 原生 `W8A8_MXFP8` 权重和 `_scale`。
- 对 packed experts 逐个展开导出。
- 命中 `FLOAT` 或不满足条件的权重直接 passthrough。

**修复的问题**

1. **只量化 expert 0 的问题**
   - ModelSlim 描述通常每层只保存 `experts.0` 模板键，而实际权重有所有 expert。
   - 旧逻辑按字面名字查表，只有 expert 0 命中，其他 expert 的 scale 保持 `torch.empty` 初始化。
   - 新逻辑把任意 expert index 折回 `experts.0` 模板键，使所有 expert 使用同一层的量化规则。

2. **scale 导出错误**
   - R2 采用旧栈 R1 链路语义：先 `to(torch.bfloat16).contiguous()`，再 `npu_dynamic_mx_quant`。
   - scale 使用 `flatten(-2, -1).squeeze(-1)`，保留算子的 per-group grid。
   - 这避免了 raw dtype 输入和截断 slice 导致的 scale 形状/数值不一致。

### 7.2 `verl/utils/qat/compressed_tensors.py`

**修改内容**

- `rollout.quantization=ascend` 时构造 `{"quant_method": "ascend"}`。
- 支持从模型目录自动探测 ModelSlim。
- `quant_method=ascend` 时实例化 `AscendMxfp8WeightQuantizer`。

**修复的问题**

- 上游原来没有 Ascend ModelSlim 分支，`ascend` 字符串无法路由到权重导出器。
- 如果不接这个分支，训练侧权重会以高精形式同步给推理引擎，rollout 实际无法按 MXFP8 消费。

### 7.3 `verl/utils/vllm/vllm_quant_utils.py`

**修改内容**

- 识别 `AscendModelSlimConfig` 且描述中包含 `W8A8_MXFP8` 的模型为量化模型。
- refit 前调用 vLLM-Ascend 原生 `restore_weights_for_rl_loading`。
- refit 后重新调用 `process_weights_after_loading`。
- 每次同步后重打包 GDN conv 权重。
- 加载前执行 HF 权重名到 vLLM 模块名的映射。

**修复的问题**

1. **权重名映射失败**
   - Qwen3.5 这类 VLM wrapper 中，HF state dict 名与 vLLM 模块树名不同。
   - 不映射时，`get_module_from_param_name` / `model.load_weights` 大量查名失败，引擎静默保留 dummy 初始权重。
   - 历史现象包括约 2056 个权重映射失败、rollout 全乱码、step0 validation 为 0。

2. **refit 前后布局不匹配**
   - vLLM-Ascend 的推理布局和 checkpoint loader 期望布局不同。
   - refit 前必须恢复到 loader 可写布局，加载后再恢复推理布局。
   - 原逻辑没有对 MXFP8 模块执行这对 restore/process，导致同步后的布局不可用。

3. **GDN conv 派生权重残留 dummy 值**
   - `load_format="dummy"` 会填充所有 persistent tensor，包括 `ascend_conv1d_weight`。
   - GDN conv 是非量化层，普通量化模块的 process/refit 不会触发它的 packing hook。
   - 不重打包时，prefill/decode 每次都读取 dummy/random packed weight。
   - 新增 `refresh_ascend_gdn_conv_weights()` 从已加载 source weight 重新 pack。

### 7.4 `verl/workers/engine/fsdp/transformer_impl.py`

**修改内容**

- FSDP engine 持有 `_rollout_weight_quantizer`。
- `get_per_tensor_param()` 输出前调用该量化器。
- LoRA adapter-only 同步跳过量化。
- checkpoint 导出时临时将量化器置为 `None`，导出后恢复。

**修复的问题**

1. **rollout 同步与 checkpoint 保存共用错误路径**
   - 原先没有区分 rollout 流和 checkpoint 流。
   - 现在只有 rollout sync 消费量化流，checkpoint 始终导出高精 master 权重。

2. **低精训练后无法得到可通用 checkpoint**
   - 如果 checkpoint 直接保存 FP8/scale 推理布局，下游评测必须理解 vLLM-Ascend 私有布局。
   - 保存 BF16 master 权重后，下游可以按普通高精模型评测，不需要 `quant_model_description.json`。

### 7.5 `verl/workers/rollout/vllm_rollout/utils.py`

**修改内容**

- 在调用 `process_quanted_weights_after_loading` 前，用 `set_current_vllm_config()` 绑定当前 engine config。

**修复的问题**

- RL 权重同步发生在 multiproc worker RPC context 中，此时线程本地 `VllmConfig` 可能为空。
- vLLM-Ascend 的 `w8a8_mxfp8` process hooks 会读取当前 config，缺失时触发 `AssertionError`。

### 7.6 `verl/workers/rollout/vllm_rollout/vllm_async_server.py`

**修改内容**

- 如果 `rollout.quantization` 是字符串，直接传给 vLLM engine kwargs。
- 结构化 quantization override 才走 `_apply_quantization()`。
- 增加 vLLM quant config 类型日志。

**修复的问题**

- `ascend` 是 vLLM-Ascend 原生字符串路由，不是 verl 的结构化 compressed-tensors override。
- 旧逻辑会把字符串当结构化配置处理，导致 native quantization 无法传到异步 vLLM 服务。

### 7.7 `tests/utils/test_ascend_mxfp8_on_cpu.py`

新增 CPU 单测，覆盖：

- `rollout.quantization=ascend` 路由。
- ModelSlim 自动探测。
- FP8 权重与 scale 导出。
- packed experts 展开。
- expert 模板键映射。
- scale 形状和 passthrough 行为。

目的：防止后续 verl 合并时重新引入“只量化 expert 0”或 scale 布局回退。

## 8. FSDPTurbo.patch 逐项说明

### 8.1 `fsdp_turbo/distributed/expert_parallel/dispatcher.py`

在 `experts_computation()` 中新增 P2a no-grad 分支。

**触发条件**

- `torch.is_grad_enabled()` 为 false。
- quant config 是 `MXFP8LinearConfig`。
- 非 W4A16、非 W4A8。
- 无 ep balance、无 clipping/SiTU、无其他激活函数配置。

**不影响的路径**

- backward/训练带梯度路径。
- W4A16 QAT。
- W4A8。
- 带 balance/clipping/SiTU 的 MoE。

### 8.2 `fsdp_turbo/quantization/mx_formats/mx_gmm.py`

新增 `mx_fused_swiglu_group_gemm()`。

**修改内容**

- 使用 `torch_npu.npu_grouped_matmul_swiglu_quant_v2` 融合 gate/up GMM、SiLU*up 和输出量化。
- 使用 `npu_grouped_matmul` 执行 down GMM。
- activation scale 从 2D per-block layout pack 成 kernel 需要的 `[T, nb//2, 2]`。
- weight scale 按 axis=-2 的 kernel-native layout 输出。
- 对 weight scale 形状做显式契约检查。

**修复的问题**

- 原训练侧 no-grad scoring 路径和 rollout 路径存在数值不对称：先回到 BF16，再做额外 activation re-quantization。
- 每层会引入约 1.2%–1.4% 的 cross-feed 误差。
- P2a 使用与 rollout 相同的 fused SwiGLU kernel，消除这段不必要的 BF16 round trip 和重复量化。
- R2 日志中的 `[P2a] fused swiglu MX MoE no-grad path engaged` 证明该路径实际进入。

## 9. vLLM-Ascend.patch 逐项说明

### 9.1 `vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py`

新增环境开关：

```text
VLLM_ASCEND_ONLINE_WEIGHT_QUANT=1
```

**Linear 路径**

- 开关打开时，参数分配为 BF16。
- checkpoint BF16 权重先无损加载。
- `process_weights_after_loading()` 中在线执行 `npu_dynamic_mx_quant`。
- 同步写入 FP8 weight 和 per-group scale。

**FusedMoE 路径**

- `w13_weight`、`w2_weight` 在开关打开时分配为 BF16。
- 加载后逐 expert 在线量化。
- 对每个 expert 生成 FP8 权重和 scale，再 stack 回 expert 维度。
- stack FP8 tensor 时先用 uint8 view，再 view 回 FP8，绕过 `aclnnStack` 不接受 `DT_FLOAT8_E4M3FN` 的限制。

**修复的问题**

1. **BF16 checkpoint + ModelSlim 描述无法正确进入 FP8 engine**
   - 原实现直接把参数声明为 FP8，但 checkpoint 里没有预量化 FP8 权重和 scale。
   - 结果是权重被错误 cast，scale 未初始化。

2. **MoE expert scale 缺失**
   - 逐 expert 在线量化并填充 scale，避免专家层 scale 保持未初始化。

3. **FP8 stack 算子限制**
   - `aclnnStack` 拒绝 `DT_FLOAT8_E4M3FN`。
   - uint8 view 是 metadata-only，不搬运数据，再 view 回 FP8 可保持数值不变。

## 10. checkpoint 为什么是 BF16

这是有意设计，不是低精失效。

1. FSDP trainer 的 master 权重保持高精。
2. rollout 同步时，`get_per_tensor_param()` 临时经过 `AscendMxfp8WeightQuantizer`。
3. checkpoint 导出时，FSDP engine 临时禁用该量化器。
4. 因此磁盘 checkpoint 保存 BF16 master 权重，MXFP8 只存在于 rollout 同步和推理引擎内部。

下游评测：

- 直接按 BF16 权重加载。
- 不需要 `quant_model_description.json`。
- 不需要 `--quantization ascend`。

## 11. 交付边界

- 本文档和 patch 固化的是 2026-11-16 从 R2 使用目录导出的实际工作树状态。
- FSDPTurbo 和 vLLM-Ascend 的 patch 是未提交改动；verl patch 是相对 `origin/main@7454326` 的完整代码面。
- 远端 `.bak_*` 文件只是调查过程中的备份，不属于代码差异，也不交付。
- 旧 recipe 的 fork/commit 描述已被替换，不能与 R2 代码面混用。
