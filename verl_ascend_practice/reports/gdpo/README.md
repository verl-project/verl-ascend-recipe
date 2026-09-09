# Qwen2.5-1.5B-Instruct GDPO Ascend 整体交付报告

对应任务：[verl-ascend-recipe #18](https://github.com/verl-project/verl-ascend-recipe/issues/18)

## 1. 交付概览

本交付提供 Qwen2.5-1.5B-Instruct GDPO 在 Ascend NPU 上的可复现训练 recipe。训练侧
使用 Megatron 管理 actor 和 reference model，rollout 侧使用 vLLM-Ascend，训练入口为
`verl.trainer.main_ppo`。

| 项目 | 配置 |
| --- | --- |
| 模型 | Qwen2.5-1.5B-Instruct |
| 数据集 | RLLA-4K |
| 算法 | GDPO，按奖励维度独立 group normalization |
| 训练后端 | Megatron actor/reference |
| Rollout 后端 | vLLM-Ascend |
| 验证平台 | Atlas 800T A2，4 x Ascend 910B |
| 训练规模 | 100 global steps |
| 运行脚本 | `verl_ascend_practice/run_qwen2_5_1_5b_gdpo_megatron_npu.sh` |

## 2. 适配方案

GDPO 通过以下配置启用：

```text
algorithm.adv_estimator=gdpo
algorithm.gdpo_reward_keys=["accuracy_reward","format_reward"]
reward.reward_manager.name=gdpo
reward.custom_reward_function.name=compute_score
model_engine=megatron
```

RLLA reward function 为每个 response 返回 total、accuracy 和 format 三类信号。GDPO 在
同一 prompt 的 response group 内分别标准化 `accuracy_reward` 与 `format_reward`，再将
聚合后的 advantage 送入 Megatron actor 的统一 policy objective。

```text
RLLA-4K prompts
       |
       v
vLLM-Ascend TP2 rollout (2 replicas)
       |
       v
accuracy + format rewards
       |
       v
GDPO per-component group normalization
       |
       v
Megatron TP2 actor update
       |
       v
weight synchronization
```

### 2.1 关键训练配置

| 配置项 | 值 |
| --- | ---: |
| train / validation batch size | 16 / 16 |
| PPO mini batch size | 8 |
| micro batch size per NPU | 2 |
| rollout responses per prompt | 4 |
| prompt / response length | 2048 / 1024 |
| actor learning rate | `1e-6` |
| actor TP / PP | 2 / 1 |
| rollout TP / replicas | 2 / 2 |
| rollout max sequences per replica | 32 |
| rollout max batched tokens | 8192 |
| rollout memory utilization | 0.35 |
| rollout mode | eager |
| weight synchronization bucket | 512 MB |
| precision | BF16 |

actor/reference 不启用 offload；actor 启用 gradient checkpointing、动态 token batch 和
sequence balancing，rollout 启用 chunked prefill 与 prefix caching。`REWARD_FUNCTION_PATH`
可覆盖默认从已安装 verl 包中定位的 `verl/utils/reward_score/rlla.py`。

## 3. 环境与数据准备

### 3.1 已验证软件环境

| 组件 | 版本 |
| --- | --- |
| CANN | 9.0 |
| Python | 3.11.15 |
| torch / torch-npu | 2.9.0 / 2.9.0.post2 |
| vLLM / vLLM-Ascend | 0.18.0 / 0.18.0 |
| Megatron Core / MindSpeed | 0.16.0 / 0.16.0 |
| MBridge | 0.15.1 |

### 3.2 数据准备

准备 RLLA-4K parquet，并按以下目录放置：

```text
/path/to/data/
└── rlla_4k/
    ├── train.parquet
    └── test.parquet
```

parquet 需要包含 `data_source`、`prompt`、`ability`、`reward_model` 和 `extra_info` 字段，
其中 `data_source` 为 `rlla`。训练会过滤超过 2048 tokens 的 prompt。

## 4. 运行与恢复

在 verl 根目录执行：

```bash
MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct \
DATA_ROOT=/path/to/data \
NPUS_PER_NODE=4 \
TOTAL_TRAINING_STEPS=100 \
bash /path/to/verl-ascend-recipe/verl_ascend_practice/run_qwen2_5_1_5b_gdpo_megatron_npu.sh
```

默认训练日志写入 `LOG_DIR`。需要周期性 checkpoint 时设置 `SAVE_FREQ`；使用相同的
`OUTPUT_DIR` 并保持 `RESUME_MODE=auto` 可恢复最新 checkpoint。额外参数会作为 Hydra
overrides 继续传递给 `verl.trainer.main_ppo`。

## 5. 100-step 长跑结果

本次训练在 Atlas 800T A2 的 4 x Ascend 910B 上连续完成 100/100 global steps，累计
step 时间为 56 分 59.120 秒。每步总 reward、两个 GDPO reward 分量、actor loss、
gradient norm、step time 和 throughput 均存在且为有限值。

| 指标 | 结果 |
| --- | ---: |
| 连续训练步数 | 100 / 100 |
| 总 reward 首 10 步均值 | -2.076495 |
| 总 reward 末 10 步均值 | -0.357547 |
| 总 reward 首尾窗口增量 | +1.718949 |
| 总 reward 线性斜率 | +0.01420827 / step |
| accuracy reward 首 / 末 10 步均值 | -2.079620 / -0.357547 |
| format reward 首 / 末 10 步均值 | 0.003125 / 0.000000 |
| `actor/loss` 样本数 | 100 |
| `actor/loss` 首 10 步均值 | 0.003225 |
| `actor/loss` 末 10 步均值 | -0.002620 |
| 平均 step 时间 | 34.1912 s |
| 平均吞吐 | 427.0988 token/s/NPU |
| 4 NPU 实测总吞吐均值 | 1708.3954 token/s |

### 5.1 长跑曲线

下图覆盖完整 100 steps，依次展示总 reward 和 GDPO actor loss。两项指标均保留原始值和
10-step moving average；reward 使用 `critic/score/mean`，loss 使用 `actor/loss`。

![Qwen2.5-1.5B-Instruct GDPO 100-step reward and loss](training_curves.png)

### 5.2 性能与稳定性

- 总 reward 的末 10 步均值比首 10 步提高 1.718949。
- 4 NPU 实测总吞吐均值为 1708.3954 token/s，高于无 GPU 标杆时的 100 TPS 门槛。
- 训练连续完成 100 steps，所有必需训练指标均有 100 个有限值。

## 6. 复现证据与验收结论

完整训练日志：
[Qwen2.5-1.5B-Instruct GDPO 100-step training log](https://gist.githubusercontent.com/RordChang/026f04feae655f3fca610890e41d9895/raw/d444daa300e63b79624e71f515a2321a7894808b/training_100step_sanitized.log)

| Issue #18 验收项 | 本次结果 |
| --- | --- |
| 完成 100 steps 或运行 12 小时 | 完成连续 100/100 steps |
| reward 上升 | 首 10 步均值 -2.076495，末 10 步均值 -0.357547 |
| 无 GPU 标杆时 TPS > 100 | 4 NPU 实测总吞吐均值 1708.3954 token/s |
| 提供可复现 recipe | 提供模型、数据、环境、启动、checkpoint/resume 和日志配置 |

本次结果覆盖 Issue #18 在无同版本 GPU 标杆时的效果和性能验收项，并提供了
Qwen2.5-1.5B-Instruct GDPO 在 Megatron + vLLM-Ascend 组合上的完整复现入口。
