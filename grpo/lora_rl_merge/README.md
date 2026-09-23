# Qwen3-8B 的 FSDP2 LoRA merge 训练

本实践对应 [issue #78](https://github.com/verl-project/verl-ascend-recipe/issues/78)，
使用 FSDP2 训练 Qwen3-8B 的 GRPO + LoRA，使用 vLLM-Ascend 生成响应。
每次同步推理权重前，将 LoRA 合并到基座模型；推理端接收完整权重。

固定配置在单台四张 910B3 上连续完成 100 步，首末十步平均 reward 为 0.39160 → 0.83555，
每卡训练吞吐量为 655.51 tokens/s。第 80 步后验证准确率下降，原因尚未确定。

## 实现与配置

复用固定 verl 版本的
[`run_qwen3_8b_merge_fsdp.sh`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh)
及 `merged_lora_context`，没有重写权重合并或 GRPO。
`model.lora.merge=True` 使 FSDP2 同步合并权重，参考 log-prob 由禁用 LoRA 的 actor 计算。
NPU 适配采用 eager 模式、分块熵计算并关闭 torch.compile。

| 参数 | 实测值 |
| --- | --- |
| 模型 / 数据 | Qwen3-8B / GSM8K，训练 7473 条、验证 1319 条 |
| train batch / PPO mini-batch | 128 / 64 |
| prompt / response 上限 | 1024 / 1024 token |
| 每提示响应数 / rollout TP | 8 / 2 |
| 学习率 / KL 系数 | 1e-5 / 0.001 |
| LoRA rank / alpha / target modules | 32 / 64 / `all-linear` |
| 训练参数 | BF16，actor 不卸载 |
| rollout 显存比例 | 0.6 |
| 保存 / 验证 | 每 10 步保存，每 20 步验证，训练前验证 |

原配置只有一个 epoch，`7473 // 128 = 58` 步就结束。
当前 epoch 上限默认等于目标步数，由 `total_training_steps=100` 控制结束。
模型、数据、补丁或参数发生变化后，需要单独验证新配置。

## 环境与依赖

固定版本见 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。本次实测软件如下。

| 组件 | 版本或提交 |
| --- | --- |
| 硬件 | 单台 4 × Ascend 910B3，64 GB/卡 |
| Python / 系统 | 3.11 / Ubuntu 22.04 |
| CANN / 驱动 | 9.0.0 / 26.0.rc1 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |
| verl | `bc72e38edba78e778bfbd462638f9634b9140a76` |
| vLLM | 0.18.0，`bcf2be96120005e9aea171927f85055a6a5c0cf6` |
| vLLM-Ascend | 0.18.1.dev44，`a43c8cc8057f490ed1df2c6ed66253e2d7817da4` |
| transformers / peft / ray | 5.10.4 / 0.20.0 / 2.56.1 |

实测容器来自历史镜像
`quay.io/ascend/verl:latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm`，
加载后的 image ID 为 `sha256:25cdc80fb496290267a07dd270023caf9a473c4167959b90b8ffe2bc3fbda27c`。
这是运行身份记录，不是当前可下载镜像的承诺；历史拉取曾返回 404。
复现按软件版本、源码和补丁核对，可使用已有兼容环境；本次未验证从空环境重新构建这套依赖。

仓库的 `install_verl.sh` 只安装 verl，不能替代 CANN 和 vLLM-Ascend 环境准备。
在 recipe 仓库根目录设置实际路径，三个源码目录须对应当前 Python 环境安装的代码：

```bash
RECIPE_DIR=$(realpath grpo/lora_rl_merge)
VERL_DIR=/path/to/verl
VLLM_DIR=/path/to/vllm
VLLM_ASCEND_DIR=/path/to/vllm-ascend
python3 -m pip show torch torch-npu verl vllm vllm-ascend transformers peft ray
npu-smi info
```

CANN 版本另查安装目录中的版本文件；`npu-smi` 用于核对设备和驱动。
以下命令应用实测的两份补丁。已应用时用 `git apply --reverse --check` 确认，不重复应用。

```bash
(
  set -e
  test "$(git -C "$VERL_DIR" rev-parse HEAD)" = bc72e38edba78e778bfbd462638f9634b9140a76
  test "$(git -C "$VLLM_DIR" rev-parse HEAD)" = bcf2be96120005e9aea171927f85055a6a5c0cf6
  test "$(git -C "$VLLM_ASCEND_DIR" rev-parse HEAD)" = a43c8cc8057f490ed1df2c6ed66253e2d7817da4
  git -C "$VERL_DIR" apply --check "$RECIPE_DIR/patches/0001-get_npu_versions-first-visible-npu-id.patch"
  git -C "$VERL_DIR" apply "$RECIPE_DIR/patches/0001-get_npu_versions-first-visible-npu-id.patch"
  git -C "$VLLM_ASCEND_DIR" apply --check "$RECIPE_DIR/patches/vllm-ascend/0001-record-sampler-stream.patch"
  git -C "$VLLM_ASCEND_DIR" apply "$RECIPE_DIR/patches/vllm-ascend/0001-record-sampler-stream.patch"
)
```

修改必须在当前 Python 环境中生效。可编辑安装直接使用源码；预构建安装包需要按对应项目流程重新构建或安装。

- 设备探测补丁从 `npu-smi info -m` 取得首个可见物理卡，保留上游已有的环境变量回退。
  上游在正确设置 `ASCEND_VISIBLE_DEVICES` 时也可处理部分设备映射；保留此补丁是为了复现实测源码，不能称其为所有环境的必需修复。
- 采样器补丁回移上游 [PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394) 的 `q.record_stream(...)`。
  `wait_stream` 保证执行顺序，`record_stream` 防止辅助流创建的张量在消费者使用完成前被分配器重用。
  历史 TP=1 对照中，原版 224621 个 token 出现 4 个越界 ID，补丁组 225828 个 token 中为 0；这不能单独解释 reward 收益。

## 模型、数据与运行

将 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) 的权重、配置和 tokenizer 下载到模型目录。
在上述环境中生成 GSM8K 数据：

```bash
WORK_DIR=/path/to/issue78
mkdir -p "$WORK_DIR"
python3 "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$WORK_DIR/data/gsm8k"
```

下面假设获准使用的物理卡 4–7 映射为容器逻辑卡 0–3。
`ASCEND_RT_VISIBLE_DEVICES` 控制运行设备，`PHYS_CARDS` 提供物理编号。
直接使用宿主机设备时，逻辑编号也应按实际映射修改。
首次训练使用新的 checkpoint 目录；中断后使用同一目录和 `RESUME_MODE=auto`。

```bash
(
  set -euo pipefail
  cd "$VERL_DIR"
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 PHYS_CARDS=4,5,6,7 \
    MODEL_PATH="$WORK_DIR/models/Qwen3-8B" DATA_DIR="$WORK_DIR/data/gsm8k" \
    DEFAULT_LOCAL_DIR="$WORK_DIR/checkpoints/clean100" RAY_TEMP_DIR=/tmp/ray-l78 \
    TOTAL_TRAINING_STEPS=100 SAVE_FREQ=10 TEST_FREQ=20 VAL_BEFORE_TRAIN=True RESUME_MODE=auto \
    bash "$RECIPE_DIR/run_qwen3_8b_lora_merge_fsdp2_npu.sh" 2>&1 | tee "$WORK_DIR/training.log"
)
```

脚本直接调用 `verl.trainer.main_ppo`，支持末尾追加 Hydra 参数，也支持 `--cfg job --resolve` 导出配置。
上面的 `pipefail` 保留训练失败的退出状态；退出码 0 本身不能证明训练达到目标。
核对日志中的更新步数为 1–100、最终验证，以及 checkpoint 100 的模型、优化器和额外状态文件。
恢复运行应另核对实际加载记录和后续更新，不能只看 checkpoint 文件存在。

## 实测结果与证据

| 项目 | 结果 |
| --- | --- |
| 完成步数 | 连续 100 步，没有拼接其他运行 |
| 首末十步平均 reward | 0.3916015625 → 0.835546875 |
| 每卡训练吞吐量 | 655.5106 tokens/s，最低单步 565.1711 |
| GSM8K 验证准确率 | 326/1319（24.7157%）→ 1064/1319（80.6672%） |
| 数值检查 | 已记录训练指标和七项 rollout_corr 均有限，各步梯度范数非零 |
| 保存与退出 | checkpoint 100 文件存在，退出码 0，未发生 OOM |

吞吐量按 `sum(perf/total_num_tokens) / sum(perf/time_per_step) / 4` 计算，
本次为 `71621758 / 27315.25477820309 / 4`，token 包含 prompt 与 response。
这不是单步吞吐量算术平均、纯生成速度或整个作业的墙钟吞吐量。

| 验证步 | 0 | 20 | 40 | 60 | 80 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K 准确率 | 24.7157% | 70.1289% | 80.2123% | 86.6566% | 89.0826% | 80.6672% |

![奖励、验证准确率、吞吐量、梯度范数和训练步时间](assets/training-curves.png)

实测 recipe 提交为 `84142b1934539b1d14600cbe87a8ce92e709260f`；后续交付整理不改变训练参数和依赖补丁。
完整日志和元数据保存在贡献者仓库的固定历史提交，作为外部证据引用，不随当前 recipe 重复提交：

- [完整训练日志](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/training_100step.log)保留配置、初始化、100 步更新和最终验证。
- [指标摘要](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/summary.json)、[输入哈希](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/inputs.sha256)及[运行版本](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/provenance.json)支持逐项核对。
- [checkpoint 记录](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/checkpoint100.json)和[进程状态](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/process.json)保存终态信息。

日志仅脱敏两处内部 IP，101 个指标行保持原样，见[脱敏记录](https://github.com/xchencehn/verl-ascend-recipe/blob/0fbc67b7bc66b925c8c1d72ed06de8e28ce107b4/lora_rl_merge/evidence/910b3-100step/log_sanitization.json)。
第 100 步比第 80 步低 8.42 个百分点，不能把第 100 步称为最佳 checkpoint，后期退化原因仍未确定。
本次没有从 checkpoint 100 实际恢复，也没有 GPU 对照、八卡或 A3 验证。
现有结果满足 issue 无 GPU 标杆时的训练指标要求；正式完成仍需 PR 合入及实践文档提交到 issue。
