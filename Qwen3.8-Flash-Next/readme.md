# Qwen3.8 Flash Next on Ascend NPU

基于 VeOmni 训练后端和 vLLM rollout 后端，在 Ascend NPU 上对 Qwen3.8 Flash Next（`model_type=qwen4_exp`）进行 GRPO 后训练。

目录沿用 [DeepSeek-V4-Flash](../DeepSeek-V4-Flash) 的 `install.sh`、`examples/`、`patch/verl.patch` 和 `readme.md` 组织方式。补丁来自 [verl PR #8583](https://github.com/verl-project/verl/pull/8583)，固定到 head `34bf6219a86a87e9224d584de072adcecaf0850c`，移除 CI workflow 和测试文件的改动，并将训练示例的 `async_scheduling` 设为 `true`、`max_num_seqs` 设为 `128`；其余功能改动保持一致。

## Required `verl` version

版本以 [REQUIRED_VERL.txt](REQUIRED_VERL.txt) 为准。必须在基线 `592cba13531a32b69eccd490a04b795a89587854` 上应用 [patch/verl.patch](patch/verl.patch) 后安装；该基线是源 PR 的 merge base。仓库根目录的 `install_verl.sh` 仅安装基线，不会自动应用此补丁，请使用本目录的安装脚本。

## 环境版本

| 依赖组件 | 版本 / 要求 |
| --- | --- |
| 硬件 | 源 PR 使用 4 台 Ascend 910C；示例默认每台 16 个 NPU |
| Python | 3.11 或 3.12（固定版本 VeOmni 的要求） |
| verl | `592cba13531a32b69eccd490a04b795a89587854` + 本目录补丁 |
| VeOmni | `a8d357cec86643ab8fec12acc6c60a26bc7d1d29` |
| Transformers | `5.16.1` |
| vLLM | 源 PR 记录为 `0.32.0.empty` |
| vLLM-Ascend | `fd4be4dbcf751fcff0dd99d18389caf40680e9e7` |
| CANN / ATB / PyTorch / torch_npu | 使用与上述 rollout 后端匹配、已配置好的 Ascend 环境 |

源 PR 未给出完整基础镜像摘要和 CANN / PyTorch 版本，本文不额外指定未经验证的组合。训练记录中的 VeOmni 为当时的 `main`；此 recipe 使用源 PR 代码明确固定的 VeOmni commit。多机必须使用相同依赖、模型权重和数据。

### 环境部署

先在每台机器准备上述 vLLM / vLLM-Ascend 环境及 VeOmni 所需的 NPU 运行依赖，并加载 CANN / ATB 环境变量。本安装脚本安装 verl 的 NPU 依赖、固定版本的 VeOmni 和 Transformers；不会构建 vLLM 或重装 NPU 驱动。VeOmni 使用 `--no-deps` 安装以沿用已配置的 NPU 环境。

```bash
git clone https://github.com/verl-project/verl-ascend-recipe.git
export RECIPE_DIR="$(pwd)/verl-ascend-recipe/Qwen3.8-Flash-Next"
# 选择可写的工作目录，其中不能已有 verl/ 或 VeOmni/。
export WORKSPACE_DIR="$HOME/workspace-qwen38"
bash "$RECIPE_DIR/install.sh"
cd "$WORKSPACE_DIR/verl"
```

如已有匹配环境，也可以手动应用补丁：

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git checkout 592cba13531a32b69eccd490a04b795a89587854
git apply --check "$RECIPE_DIR/patch/verl.patch"
git apply "$RECIPE_DIR/patch/verl.patch"
python -m pip install --no-deps -e .
```

### 权重与数据

准备可供所有节点访问的 Qwen3.8 Flash Next checkpoint（包含配置、多模态 processor 和完整 PLE 表），将其路径传给 `model_path`。Actor 与 rollout 必须读取同一份初始权重。

在已安装的 verl 目录中预处理 Geo3K：

```bash
python examples/data_preprocess/geo3k.py --local_dir "$HOME/data/geo3k"
```

将生成的 `train.parquet`、`test.parquet` 及模型放在各节点相同的路径下，或使用共享存储。

### 启动训练

默认拓扑为 4 × 16 NPU：actor FSDP=64、EP=32、PLE=8、SP=1；rollout TP=8、DP=8、EP=64。调整机器数时需同步调整并行参数。EP 和 PLE 大小必须整除参数分片组大小，且不能跨 HSDP replica。

在 head 节点启动 Ray（将 IP 替换为实际地址）：

```bash
export HEAD_IP=<head-node-ip>
ray start --head --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=16
```

在其余三个节点分别执行：

```bash
export HEAD_IP=<head-node-ip>
ray start --address="$HEAD_IP:6379" --num-gpus=16
```

在 head 节点的 verl 目录启动训练：

```bash
cd "$WORKSPACE_DIR/verl"
export RAY_ADDRESS="$HEAD_IP:6379"
model_path=/path/to/Qwen3.8-Flash-Next \
data_path="$HOME/data/geo3k" \
output_path="$HOME/output" \
bash "$RECIPE_DIR/examples/run_qwen3_8_flash_next_veomni.sh"
```

[启动脚本](examples/run_qwen3_8_flash_next_veomni.sh) 基于源 PR 示例，将 `++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true` 和 `actor_rollout_ref.rollout.max_num_seqs=128` 作为默认值；支持环境变量和追加 Hydra 参数，例如：

```bash
bash "$RECIPE_DIR/examples/run_qwen3_8_flash_next_veomni.sh" \
  actor_rollout_ref.actor.veomni.frozen_ple_dtype=bfloat16 \
  trainer.total_training_steps=250
```

### 关键配置

- `ple_parallel_size=8`，`ep_sharded_stream_load=true`，`broadcast_model_weights_from_rank0=false`：按 PLE 分片加载权重。
- `freeze_ple_embeddings=true`：冻结 PLE lookup table，不将它们加入 optimizer 或 actor-to-rollout 同步；其余参数保持可训练。
- `rollout.load_format=safetensors`：从 checkpoint 初始化 rollout 的冻结表，不能改用 dummy 加载。
- `frozen_ple_dtype`：可选 `float32` 或 `bfloat16`，仅在冻结 PLE 时使用；示例默认不额外转换。
- `enable_autocast=false`：关闭 VeOmni 外层及继承的 autocast，使原始 forward 与 checkpoint 重计算的 activation dtype 一致；独立于 PLE 存储 dtype。
- `qsa_attention_implementation=torch_npu`：使用 VeOmni 的 NPU QSA 实现。
- 示例启用 R3 router replay，需要训练和 rollout 两端都有对应实现。

## 训练效果与验证范围

源 PR 报告：在 4 台 Ascend 910C 上训练 Geo3K，reward 上升，`rollout_probs_diff_mean < 0.01`。曲线见 [源 PR](https://github.com/verl-project/verl/pull/8583)。

本次 recipe 移植验证补丁可应用，并核对除上述两项启动参数调整外，应用后文件与指定 PR head 一致；不代表在当前环境重新完成了 NPU 多机训练。CI workflow 和测试代码不包含在本 recipe 的补丁中。
