# Qwen3-4B GPG Ascend Recipe

对应任务：[#71](https://github.com/verl-project/verl-ascend-recipe/issues/71)。

4 张 Ascend 910B3 从初始权重训练 100 步，训练进度记录时长为 2:17:01。reward 首末 10 步均值为 0.362109 → 0.831641，单 NPU TPS 均值为 1023.88 tokens/s。

## 训练链路

GSM8K / MATH prompt → vLLM-Ascend TP1（每 prompt 4 条 response）→ 数学规则 reward → GPG advantage / policy loss → FSDP2 actor 更新 → rollout 权重同步。

沿用官方 GPG 示例的 advantage estimator 与 policy loss，调整为 Qwen3-4B、FSDP2 四卡训练和 vLLM-Ascend rollout。 参考入口为 [官方训练示例](https://github.com/verl-project/verl/blob/main/examples/gpg_trainer/run_qwen3_8b_fsdp.sh)。

## 文件

| 路径 | 内容 |
| --- | --- |
| `run_qwen3_4b_gpg_fsdp_npu.sh` | 训练配置、数据准备与启动入口 |
| `README.md` | 环境、复现命令、关键配置与结果 |
| `assets/reward_curve.png` | reward 逐步值及 10-step moving average |
| `assets/loss_curve.png` | actor loss 逐步值及 10-step moving average |

## 环境

| 组件 | 已验证版本 |
| --- | --- |
| NPU / OS | 4 × Ascend 910B3 / Ubuntu 22.04 |
| HDK / CANN | 26.0.rc1 / 9.1.0 |
| Python | 3.12.13 |
| PyTorch / torch-npu | 2.10.0 / 2.10.0.post4 |
| vLLM / vLLM-Ascend | 0.23.0 / 0.23.0 |
| Ray | 2.58.0 |
| Triton / Triton-Ascend | 3.5.0 / 3.2.2 |
| transformers | 5.3.0 |
| verl | `0.10.0.dev` |

安装流程见 [verl Ascend 安装指南](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/zh/get_start/install_guidance.rst)。启动前加载 CANN 环境并激活安装上述组件的 Python 环境：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
```



## 模型与数据

```bash
hf download Qwen/Qwen3-4B --local-dir /path/to/Qwen3-4B
```

数据为 [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)（7473 / 1319 条）和 [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)（7500 / 5000 条），使用对应训练及验证 parquet，过长 prompt 按配置过滤。

```bash
python /path/to/verl/examples/data_preprocess/gsm8k.py \
  --local_save_dir /path/to/data/gsm8k
python /path/to/verl/examples/data_preprocess/math_dataset.py \
  --local_save_dir /path/to/data/math
```

脚本在 parquet 缺失时调用上述预处理；存在时直接使用。

## 四卡启动

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
NDEVICES_PER_NODE=4 \
PYTHON_BIN=python \
VERL_ROOT=/path/to/verl \
DATA_ROOT=/path/to/data \
MODEL_PATH=/path/to/Qwen3-4B \
CKPTS_DIR=/path/to/checkpoints/qwen3-4b-gpg \
LOG_FILE=/path/to/logs/qwen3-4b-gpg/train.log \
RAY_TMPDIR=/data/ray71 \
TMPDIR=/data/tmp71 \
TOTAL_TRAINING_STEPS=100 \
bash /path/to/verl-ascend-recipe/grpo/qwen3-4b-gpg/run_qwen3_4b_gpg_fsdp_npu.sh \
  ray_kwargs.ray_init.num_cpus=48 \
  +ray_kwargs.ray_init.object_store_memory=20000000000 \
  +ray_kwargs.ray_init.include_dashboard=False
```

脚本接收追加的 Hydra overrides。`RAY_TMPDIR` 与 `TMPDIR` 使用短路径。

## 关键配置

| 参数 | 验证值 |
| --- | --- |
| actor / rollout topology | FSDP2 size 4 / TP1，4 replicas |
| train batch / responses per prompt | 64 / 4 |
| PPO mini / micro batch per NPU | 16 / 1 |
| max prompt / response | 1024 / 1024 |
| dynamic batch / token cap per NPU | true / 8192 |
| actor LR / KL loss | 1e-6 / false |
| gradient checkpointing / remove padding | true / true |
| rollout memory utilization | 0.45 |
| dtype | BF16 |
| parameter / optimizer offload | false / false |
| chunked prefill / max batched tokens | true / 4096 |
| rollout max sequences | 256 |
| graph mode | `FULL_DECODE_ONLY` |
| graph capture sizes | 4, 8, 16, 32, 64, 128, 256 |
| training steps / save frequency / test frequency | 100 / 100 / 10 |
| Ray CPUs / object store | 48 / 20,000,000,000 bytes |

## Checkpoint 与日志

默认从初始权重开始，在第 100 步保存最终 actor checkpoint，仅保留一份。每 10 步验证一次，console logger 输出完整训练和验证指标，`LOG_FILE` 保存框架控制台日志。

训练日志：[完整训练日志](https://gist.github.com/SkyTD256/a5bc0be92bb1a0b394a8c7ea328d802e)。

## 指标与结果

reward 使用 `critic/rewards/mean`，比较首 10 步和末 10 步均值；actor loss 使用 `actor/loss`。单 NPU TPS 是全部 100 步 `perf/throughput` 的算术均值，该字段按 `perf/total_num_tokens / perf/time_per_step / 4` 计算。训练 token 包含 prompt 与 response；step time 不包含每 10 步的独立验证时间，包含最终 checkpoint 的保存计时。进度记录时长和启动至退出总时长包含验证。四卡聚合 TPS 为单 NPU 值乘 4。

| 指标 | 结果 |
| --- | ---: |
| 训练进度记录时长 / 完成步数 | 2:17:01 / 100 |
| 启动至退出总时长 / 退出状态 | 02:24:11 / 0 |
| reward 首 10 步 / 末 10 步均值 | 0.362109 / 0.831641 |
| reward 增量 | +0.469531 |
| actor loss 均值 / 范围 | -0.000675 / -0.006606 至 0.000682 |
| 单 NPU TPS 均值 / 首 10 步均值 | 1023.88 / 1105.36 tokens/s |
| 四卡聚合 TPS | 4095.51 tokens/s |
| 平均训练 step 时间 | 48.56 s |
| 训练处理 token 总数 | 19,803,811 |

![Reward curve](assets/reward_curve.png)

![Actor loss curve](assets/loss_curve.png)
