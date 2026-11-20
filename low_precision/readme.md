# low_precision — Ascend NPU MXFP8 GRPO 低精度训练

基于 verl 在 Ascend NPU 上进行 MXFP8 低精度 RLHF（GRPO）后训练。本目录**按训练后端分治**，每个后端目录自成一个安装单元（readme / install / examples / patch 齐全）；`tool/` 为两条链路公用的权重目录生成工具。

```
low_precision/
├── readme.md                 # 本文件（总览 + 公用工具用法）
├── tool/                     # 公用：MXFP8 权重目录 + quant_model_description.json 生成（双引擎）
├── patch/                    # 公用补丁层：上游 #169 原件（verl.patch、vllm-ascend-attention.patch）
├── megatron/                 # Megatron + MindSpeed 链路（上游 #169 Qwen3-30B-A3B 配方原件）
│   ├── readme.md / install.sh                    # 补丁引用上层 patch/（上游 #169 原件）
│   └── examples/Qwen3-30B-A3B/                   # Qwen3-30B-A3B（4 x Atlas A5）
└── fsdp_turbo/               # FSDPTurbo 链路（训练侧动态 MXFP8）
    ├── readme.md / install.sh / patch/           # verl 合并件＋SCALEFIX＋GDN fallback
    └── examples/Qwen3.5-35B-A3B/                 # Qwen3.5-35B-A3B（1 x Atlas A5，8 x Ascend950DT）
```

各链路的版本矩阵、安装步骤、模型使用方式见对应后端目录的 readme.md。**新增模型**：确认所属后端后，只在该后端 `examples/<模型>/` 下添加 launcher（权重生成按下方工具用法）；**新增后端**：新增同级目录（readme + install + examples + patch，按需）。

## 公用工具：MXFP8 权重目录生成

`tool/smart_create_quant_weight_dir.py` 把 bf16 权重目录装配为 MXFP8 rollout 目录（小文件复制、大分片软链），并生成 `quant_model_description.json`（vllm-ascend `rollout.quantization=ascend` 必需；`FLOAT` 即跳过该权重量化）。**单文件双引擎**：

- **正则引擎（默认）**：按 `--quant_pattern` 正向正则（可多次，缺省 7 条投影正则 down/gate/up/q/k/v/o\_proj）从 `model.safetensors.index.json` 生成量化名单。适用于 Qwen3-30B-A3B 等标准 MoE 架构。
- **结构感知引擎（`--profile qwen3_5_moe`）**：按模型结构语义生成——量化作用域限 `model.language_model.layers.{*}`、nn.Linear 且 min(in,out)>=64 才量化（对齐 FSDPTurbo `quantize.linear.mx`）、融合 3D 专家量化并**按层合成** vLLM FusedMoE 查表键、MTP/visual/norm 等一律 FLOAT，生成后自动模拟 vllm-ascend 查表校验。适用于 Qwen3.5-35B-A3B 这类 hybrid（GatedDeltaNet + full attention）+ 融合 MoE 架构；同架构新模型直接复用，**新架构需在工具中新增 profile**。

通用开关：

- `--exclude_pattern`（两引擎通用，可多次）：负向正则，**优先级最高**，命中一律 `FLOAT`。用于 include 侧宽正则误伤敏感子结构的场景（如 `.*.out_proj.weight` 同时命中 self\_attn.out\_proj（应量化）与 linear\_attn.out\_proj（不能量化））。
- `--group_size`：可选，写入 json 的 MXFP8 块大小元信息（vllm-ascend 内部默认 32）。正则引擎缺省不写入；`--profile qwen3_5_moe` 缺省按 32 写入。
- `--quant_config` / `--quant_method`：量化取值与方法标识（默认 `W8A8_MXFP8` / `ascend`）。

用法示例：

```bash
# Qwen3-30B-A3B（正则引擎，默认）
python tool/smart_create_quant_weight_dir.py /model/Qwen3-30B-A3B /model/Qwen3-30B-A3B-MXFP8

# Qwen3.5-35B-A3B（结构感知引擎 + 排除 linear_attn，原因见 fsdp_turbo/readme.md）
python tool/smart_create_quant_weight_dir.py \
    /model/Qwen3.5-35B-A3B /model/Qwen3.5-35B-A3B-MXFP8 \
    --profile qwen3_5_moe --exclude_pattern '.*linear_attn\..*'
```

## 链路指路

| 链路 | 模型 | 后端组合 | 入口 |
| :--- | :--- | :--- | :--- |
| megatron | Qwen3-30B-A3B | vLLM 0.30.0 + Megatron core_v0.18.0 + MindSpeed | [megatron/readme.md](megatron/readme.md) |
| fsdp\_turbo | Qwen3.5-35B-A3B | vLLM-Ascend + FSDPTurbo（训练侧动态 MXFP8） | [fsdp_turbo/readme.md](fsdp_turbo/readme.md) |
