# Qwen3.5-35B-A3B MXFP8 GRPO（FSDPTurbo 链路）on Ascend NPU

> ⚠️ **依赖合入状态提示**：本 recipe 使用实际长跑 `GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2` 的代码面；verl 与 FSDPTurbo 以固定 commit + patch 复现，vLLM-Ascend 使用固定基线：
>
> | 依赖 | 基线 commit | patch |
> | :--- | :--- | :--- |
> | verl | `3d050639`（PR #8511 head） | `patch/verl.patch`（R2-only：FSDP rollout 量化接线 + ModelSlim 导出/GDN refit 修正） |
> | FSDPTurbo | `759a348` | `patch/fsdpturbo.patch`（P2a no-grad fused SwiGLU MX MoE 路径） |
> | vLLM-Ascend | `c09d3e2` | 无 patch（原生 ModelSlim MXFP8 加载；RL refit 由 verl 发送 FP8+scale） |
>
> `install.sh` 会先 checkout 上表 commit，再应用对应 patch，并在末尾做硬性自检。

基于 Qwen3.5-35B-A3B 模型在 NPU 上进行 RLHF（GRPO）MXFP8 后训练的样例，训练/推理后端为 **FSDPTurbo（训练侧动态 MXFP8）+ vLLM-Ascend（推理侧 MXFP8 权重加载）**。

checkpoint 仍以训练后的高精权重保存：下游评测时按普通 BF16 权重目录加载，不需要 `quant_model_description.json`，也不要开启 `--quantization ascend`。

本次交付对应的 R2 长跑代码面、启动差异、每个 patch 的修复原因见 [MXFP8-R2-code-differences.md](MXFP8-R2-code-differences.md)。

本用例基于 1 x Atlas A5（8 x Ascend950DT 96G）实现，开发者可以参照调整。

## 环境版本

| 依赖组件                | 版本                          |
| :--------------------- | :------                      |
| CANN                   | 9.2.0                        |
| PyTorch                | 2.10.0+cpu                   |
| torch\_npu             | 2.10.0.post4                 |
| verl                   | 3d050639 + patch/verl.patch  |
| vLLM                   | ced6857                      |
| vLLM-Ascend            | c09d3e2                     |
| FSDPTurbo              | 759a348 + patch              |
| triton\_ascend         | 3.2.2                        |
| transformers           | 5.10.4                       |

## 环境部署

```bash
conda create -n verl-npu python=3.12 -y
conda activate verl-npu

# 获取环境依赖
git clone https://github.com/verl-project/verl-ascend-recipe.git

# 首先根据实际 cann 的安装路径 source cann
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/cann/set_env.sh

# 然后执行安装步骤（会克隆并安装 vllm / vllm-ascend / FSDPTurbo / verl）
bash verl-ascend-recipe/low_precision/fsdp_turbo/install.sh

# 使 FSDPTurbo 可被 import
export PYTHONPATH=$PWD/FSDPTurbo:$PYTHONPATH
```

## 版本依赖

本链路附带两个 patch，`install.sh` 会按固定基线 commit 应用：

- **verl**：`patch/verl.patch`，相对 PR #8511 head `3d050639` 共 4 个文件（76 行新增、5 行删除），只包含 FSDP rollout 权重量化接线、checkpoint 高精保存、GDN conv refit 重打包、HF→vLLM 权重名映射和 ModelSlim scale/专家模板修正。
- **FSDPTurbo**：`patch/fsdpturbo.patch`，相对 `759a348` 共 2 个文件，包含 no-grad fused SwiGLU MX MoE 路径，消除 rollout 样本打分时的训练/推理数值不对称。
- **vLLM-Ascend**：不附加 patch，使用 `c09d3e2` 原生 ModelSlim MXFP8 加载；RL 权重同步由 verl 导出 FP8 权重与 scale。

`install.sh` 末尾会对 `ModelSlimMxfp8WeightExporter`、`mx_fused_swiglu_group_gemm` 做硬性自检，缺失即退出。

## 权重下载与 quant_model_description.json 生成

1. 权重下载：从 [huggingface](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 下载权重和配置文件。

2. 生成 MXFP8 权重目录（**必须**用共享工具的结构感知引擎，并排除 linear_attn）：

```bash
python verl-ascend-recipe/low_precision/tool/smart_create_mxfp8_weight_dir.py \
    /model/Qwen3.5-35B-A3B /model/Qwen3.5-35B-A3B-MXFP8 \
    --profile qwen3_5_moe --exclude_pattern '.*linear_attn\..*'
```

生成的 `quant_model_description.json` 应为 **360 项 `W8A8_MXFP8`**（experts 200 + shared\_expert 120 + self\_attn 40），**linear\_attn 的 90 项全部 `FLOAT`**（原因见下节）。

> 为什么不能省掉 `--profile qwen3_5_moe`：该模型的目标 json 含有索引文件中不存在的键（vLLM FusedMoE 查表用的 `mlp.experts.0.{gate,up,down}_proj.weight` 共 120 项，由结构感知引擎按层合成）与无 `.weight` 后缀的融合专家键（80 项），纯索引正则（默认引擎）生成不出；同时默认正则会把 768 个 MTP 专家键误纳入量化。结构感知引擎按模型结构语义（量化作用域 `model.language_model.layers.{*}`、nn.Linear 且 min(in,out)>=64 才量化、融合专家合成查表键）生成，生成后自动模拟 vllm-ascend 查表校验无缺键/分片冲突。同架构新模型可直接复用该 profile；新架构需在工具中新增 profile。

## 启动训练

脚本已固化实际长跑语义：

- 训练侧 actor/ref 各 7 项 FSDPTurbo MXFP8 配置。
- rollout 侧固定 `actor_rollout_ref.rollout.quantization=ascend`。
- 不设置 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT`；RL refit 使用 verl 导出的 FP8 权重与 scale。
- EP 使用 `all2all + use_fused_ops=True`，并启用 fused causal conv1d / fused GDN。
- `trainer.total_training_steps=200`、`trainer.save_freq=100`，与实际长跑一致。

请根据实际数据/权重等路径修改 `examples/Qwen3.5-35B-A3B/run_qwen3_5_35b_fsdp_turbo_mxfp8.sh` 中相应路径（`MODEL_PATH` 默认已指向上面生成的 MXFP8 目录名）。

```bash
cd verl
bash ../verl-ascend-recipe/low_precision/fsdp_turbo/examples/Qwen3.5-35B-A3B/run_qwen3_5_35b_fsdp_turbo_mxfp8.sh
```

复现实际 R2 长跑时可按下面方式覆盖路径：

```bash
RAY_DATA_HOME=/path/to/verl \
EXPERIMENT_NAME=GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2 \
MODEL_PATH=/path/to/Qwen3.5-35B-A3B-MXFP8-nolin \
CKPTS_DIR=/path/to/ckpts/GRPO-Qwen3_5/GRPO-Qwen3_5-35B-exp1102-08-pr8511-MXFP8LR200-R2 \
WORKING_DIR=/path/to/verl \
bash ../verl-ascend-recipe/low_precision/fsdp_turbo/examples/Qwen3.5-35B-A3B/run_qwen3_5_35b_fsdp_turbo_mxfp8.sh
```

checkpoint 仍为 BF16 高精格式；下游评测时按普通 BF16 权重目录加载，不需要 `quant_model_description.json`。

### 为什么必须排除 linear_attn（重要）

Qwen3.5-35B-A3B 的 30 层 gated delta net（`linear_attn`）中，`in_proj_qkv` / `in_proj_z` / `out_proj` 三个投影（共 90 项）对 MXFP8（E4M3 + 块 scale）量化**异常敏感**：递归式线性注意力的投影误差会沿序列递归累积放大，这是 softmax 注意力没有的特性（量化白名单已经把它的递归状态项 `A_log`/`conv1d`/`dt_bias`/`in_proj_a/b`/`norm` 留为 FLOAT，却量化喂给门控/递归动力学的三个投影）。

实测（GRPO、geo3k 数据集、601 题 step0 验证集、温度 0 贪心）：

| rollout 配置 | step0 val（答对题数/601） |
| :---------- | :------------------------ |
| bf16（3 次独立运行） | 403 / 407 / 409（均值 406） |
| MXFP8 全量（3 次独立运行） | 373 / 380 / 387（均值 380） |
| **MXFP8 排除 linear_attn**（本 recipe） | **402 / 396**（两次复现，回到 bf16 簇） |

即量化 linear_attn 使 val 下跌 **26 题（相对 −6.5%）**；两侧（训练+推理）同时排除后恢复。同时训推一致性同向改善：`rollout_probs_diff_mean` 由 0.0111 降至 0.0097（step1 对 step1，−13%），`rollout_corr/kl` 由 0.0037 降至 0.0028（−26%），训练动态（entropy/grad\_norm/response\_length）无伤。

**训练侧同步排除已固化在 launcher**：FSDPTurbo 的 `quant_ignored_modules` 配置项对 MoE 专家**静默不生效**（`fsdp_turbo/quantization/utils.py` 中 `moe_filter_fn` 不接收 ignored 列表），但对普通线性层生效。mxfp8 launcher 中已固化（actor 与 ref 各一条）：

```bash
QUANT_IGNORED_MODULES="['model.language_model.layers.{*}.linear_attn.*']"
```

**两侧必须同时排除**：只修推理侧时训练前向仍在量化 linear_attn，训推不一致问题依旧。

## 验证基线（验收参考）

按本 recipe 配置跑 1~5 步（`val_before_train=True`）应观察到：

- `step0 val` ∈ [390, 410]（bf16 簇）
- `training/rollout_probs_diff_mean` ≈ 0.0090~0.0105
- `training/rollout_actor_probs_pearson_corr` ≈ 0.995~0.997
- `rollout_corr/kl` ≈ 0.0024~0.0028

## 已知注意事项

- 训练侧 `quant_ignored_modules` 对 MoE experts 不生效（见上），如需排除专家请改走 json/权重侧手段。
- 推理侧 json 中 `FLOAT` 即跳过该 shard 量化（`vllm_ascend/quantization/modelslim_config.py`），是唯一生效的推理侧排除机制。
- 共享机器上若 `/etc/hccl_rootinfo.json` 被 docker bind-mount 创建为**空目录**，HCCL 新建通信域会报 `RootInfoDetect failed`（error code 4），将该目录改名即可恢复。
