# Qwen3.5-35B-A3B MXFP8 GRPO（FSDPTurbo 链路）on Ascend NPU

基于 Qwen3.5-35B-A3B 在昇腾 NPU 上进行 GRPO MXFP8 后训练的样例：**FSDPTurbo（训练侧动态 MXFP8）＋ vLLM-Ascend（推理侧 MXFP8 权重加载）**。实测于 1 × Atlas A5（8 × Ascend950DT 96G）。

> 📌 **代码源口径**：全部依赖基于官方仓 commit——verl @ `07ba1a854`、vLLM @ `ced6857`、vLLM-Ascend @ `59ebc4384`、FSDPTurbo（gitcode）@ `db33a840`；未合入上游的其余修正以 patch 随本仓携带（见文末「patch 清单」），本地 patch 不推送到任何仓、进官方的唯一路径是上游 PR。缺失这些 patch 时链路**不报错而是静默产出错误结果**（权重名映射失败 → rollout 输出乱码），因此 `install.sh` 末尾自检为硬性失败。

## 环境版本

| 依赖组件 | 版本 |
| :--- | :--- |
| CANN | 9.2.0 |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| verl | 官方 `07ba1a854` ＋ patch 层 |
| vLLM | 0.30.0+empty（官方 `ced6857`） |
| vLLM-Ascend | 官方 `59ebc4384`（v0.30.0）＋ patch 层 |
| FSDPTurbo | `db33a840` |
| triton_ascend | 3.6.0.post2 |
| transformers | 5.22.2 |
| flash-linear-attention（＋fla-core） | 0.5.2 |
| flash-linear-attention-npu-a5 | 26.9.1 |
| torchao / torchao_npu | 0.18.0（gitcode `Ascend/torchao-ascend`） |
| mindspeed_ops | 26.2.0（gitcode `Ascend/MindSpeed-Ops`） |

## 快速开始

```bash
git clone https://github.com/verl-project/verl-ascend-recipe.git
cd ..
source <CANN 安装路径>/ascend-toolkit/set_env.sh
bash verl-ascend-recipe/low_precision/fsdp_turbo/install.sh
```

安装的完整步骤（虚拟环境、目录布局、装后验证与注意事项）见 [install-guide.md](install-guide.md)。

## 权重下载与 quant_model_description.json 生成

1. 从 [huggingface](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) 下载权重和配置文件。

2. 生成 MXFP8 权重目录（必须用共享工具的结构感知引擎，并排除 linear_attn）：

```bash
python verl-ascend-recipe/low_precision/tool/smart_create_quant_weight_dir.py \
    /model/Qwen3.5-35B-A3B /model/Qwen3.5-35B-A3B-MXFP8 \
    --profile qwen3_5_moe --exclude_pattern '.*linear_attn\..*'
```

生成的 `quant_model_description.json` 应为 **360 项 `W8A8_MXFP8`**（experts 200 + shared_expert 120 + self_attn 40），**linear_attn 的 90 项全部 `FLOAT`**（原因见下节）。

> `--profile qwen3_5_moe` 不可省：目标 json 含索引文件中不存在的键（vLLM FusedMoE 查表用的融合专家键，由结构感知引擎按层合成），纯索引正则生成不出，且默认正则会把 768 个 MTP 专家键误纳入量化。同架构模型可直接复用该 profile；新架构需在工具中新增 profile。

## 启动训练

按实际数据/权重路径修改 `examples/Qwen3.5-35B-A3B/run_qwen3_5_35b_fsdp_turbo_mxfp8.sh`（`MODEL_PATH` 默认已指向上一步生成的 MXFP8 目录名）。训练启动前设置（**追加式**，覆盖式会丢 CANN 注入路径、致 vLLM 子进程 `No module named 'acl'`）：

```bash
export PYTHONPATH=$PWD/FSDPTurbo:$PYTHONPATH
cd verl
bash ../verl-ascend-recipe/low_precision/fsdp_turbo/examples/Qwen3.5-35B-A3B/run_qwen3_5_35b_fsdp_turbo_mxfp8.sh
```








## patch 清单（fsdp_turbo）

`install.sh` 按序应用：

| patch | 侧 | 作用 |
| :--- | :--- | :--- |
| `low_precision/patch/verl.patch` | verl（公共层） | 上游 #169 原件：R1 发送侧预量化链（PR 8511）；fsdp_turbo 与 megatron 两链路共用，先应用 |
| `low_precision/patch/vllm-ascend-attention.patch` | vllm-ascend（公共层） | attention 连续 KV（上游 #169 原件）；两链路共用 |
| `verl-fsdp-turbo.patch` | verl | fsdp 链路专属四段：接线／权重名映射／GDN repack／ModelSlim（前置公共层 verl.patch，两步 apply） |
| `vllm-ascend-scalefix.patch` | vllm-ascend | w8a8_mxfp8 SCALEFIX（在线 MXFP8 重载；env 门 `VLLM_ASCEND_ONLINE_WEIGHT_QUANT=1`，低精 launcher 已内置默认开，详见「启动训练」段） |
