# Qwen3-8B ReMax FSDP vLLM-Ascend 整体交付报告

对应任务：[verl-ascend-recipe #23](https://github.com/verl-project/verl-ascend-recipe/issues/23)

## 1. 交付概览

本交付提供 Qwen3-8B ReMax 在 Ascend NPU 上的可复现训练 recipe。训练侧使用 FSDP
管理 actor 和 reference model，rollout 侧使用 vLLM-Ascend，训练入口为
`verl.trainer.main_ppo`。

| 项目 | 配置 |
| --- | --- |
| 模型 | Qwen3-8B |
| 数据集 | GSM8K 和 MATH |
| 算法 | ReMax，采样 response 与 greedy baseline 的 reward 差作为 advantage |
| 训练后端 | FSDP actor/reference |
| Rollout 后端 | vLLM-Ascend |
| 验证平台 | Atlas 800T A2，8 x Ascend 910B3 64 GB |
| 验证规模 | Qwen3-8B 连续 60 steps，另含 smoke、稳定性和长序列压力测试 |
| 运行脚本 | `verl_ascend_practice/run_qwen3_8b_remax_fsdp_npu.sh` |

## 2. 适配方案

ReMax 通过以下关键配置启用：

```text
algorithm.adv_estimator=remax
algorithm.use_kl_in_reward=True
algorithm.kl_penalty=kl
algorithm.kl_ctrl.kl_coef=0.001
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.rollout.name=vllm
trainer.device=npu
```

ReMax 对每个 prompt 生成采样 response 和一个 greedy baseline，使用二者的 reward 差降低
policy-gradient 方差。verl 已在 agent-loop/vLLM 请求中实现 combined rollout，因此无需修改
trainer 或 rollout 核心代码，现有流程可以直接使用 vLLM-Ascend。

```text
GSM8K / MATH prompts
          |
          v
vLLM-Ascend sampled rollout + greedy baseline
          |
          v
sampled reward - baseline reward
          |
          v
ReMax advantage + KL-in-reward
          |
          v
FSDP actor update
          |
          v
rollout weight synchronization
```

### 2.1 关键训练配置

| 配置项 | 脚本默认值 | 60-step 验证值 |
| --- | ---: | ---: |
| train batch size | 256 | 128 |
| PPO mini batch size | 128 | 32 |
| rollout responses per prompt | 4 | 4 |
| prompt / response length | 1024 / 2048 | 1024 / 1024 |
| actor learning rate | `1e-6` | `1e-6` |
| rollout tensor parallel size | 2 | 2 |
| rollout memory utilization | 0.6 | 0.6 |
| KL coefficient | 0.001 | 0.001 |
| actor parameter / optimizer offload | 开启 / 开启 | 开启 / 开启 |
| reference parameter offload | 开启 | 开启 |
| precision | BF16 | BF16 |

actor 和 reference model 均关闭 torch compile，避免当前 NPU 软件栈下的不稳定行为。
参数及优化器 offload 为同机部署的 vLLM-Ascend rollout engine 保留 HBM 空间。
rollout 使用 full-decode-only ACL Graph，并开启 cache engine 释放。

## 3. 环境与数据准备

### 3.1 已验证软件环境

| 组件 | 版本 |
| --- | --- |
| 硬件 | Atlas 800T A2，8 x Ascend 910B3 64 GB |
| CANN | 25.5.1 |
| torch-npu | 2.9.0.post2 |
| vLLM | 0.18 |
| vLLM-Ascend | 0.18.1.dev41 |
| transformers | 5.3.0.dev0 |

### 3.2 数据准备

在 verl 根目录执行：

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k"
python3 examples/data_preprocess/math_dataset.py --local_save_dir "$HOME/data/math"
```

默认数据目录结构如下：

```text
$HOME/data/
├── gsm8k/
│   ├── train.parquet
│   └── test.parquet
└── math/
    ├── train.parquet
    └── test.parquet
```

可以通过 `GSM8K_DATA_DIR` 和 `MATH_DATA_DIR` 覆盖两个数据目录。训练会过滤超过
1024 tokens 的 prompt，并将最大 response length 设为 2048。

## 4. 运行方式

在 verl 根目录执行：

```bash
MODEL_PATH=/data/models/Qwen3-8B \
bash /path/to/verl-ascend-recipe/verl_ascend_practice/run_qwen3_8b_remax_fsdp_npu.sh \
    'trainer.logger=["console"]'
```

模型、数据、节点数、batch、序列长度、rollout 并行度、显存利用率、保存频率和测试频率
均可通过环境变量修改；额外参数会作为 Hydra overrides 继续传递给
`verl.trainer.main_ppo`。

例如，复现 60-step 验证所使用的主要 batch 配置：

```bash
TRAIN_BATCH_SIZE=128 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=4 \
ROLLOUT_TP=2 \
ROLLOUT_GPU_MEM_UTIL=0.6 \
MAX_RESPONSE_LENGTH=1024 \
MODEL_PATH=/data/models/Qwen3-8B \
bash /path/to/verl-ascend-recipe/verl_ascend_practice/run_qwen3_8b_remax_fsdp_npu.sh
```

## 5. 验证结果

### 5.1 功能与稳定性验证

| 测试 | 结果 |
| --- | --- |
| Smoke | Qwen3-0.6B 在 8 张 NPU 上完成 2 steps |
| ReMax 链路 | `reward_kl_penalty` 和 `critic/advantages` 确认进入 ReMax advantage 流程 |
| Greedy baseline | server 日志确认 baseline 请求使用 `temperature=0` |
| Qwen3-8B 稳定性 | batch size 128、response length 1024 连续完成 5 steps，无 OOM 或 HCCL timeout |
| 长序列压力测试 | response length 2048 完成 3 steps，无 OOM |

### 5.2 60-step 训练结果

Qwen3-8B 在 GSM8K 上使用 batch size 128、mini-batch size 32、`n=4`、学习率
`1e-6` 和 `kl_coef=1e-3`，在约 2 小时 25 分钟内完成 60 steps。

| 指标 | 结果 |
| --- | ---: |
| 连续训练步数 | 60 |
| 训练耗时 | 约 2 小时 25 分钟 |
| `critic/score/mean` 初始值 | 约 0.26 |
| `critic/score/mean` 后期区间 | 0.60-0.76 |
| `critic/score/mean` 峰值 | 0.764 |
| 平均集群吞吐 | 约 381 tokens/s |
| 60-step 吞吐范围 | 337-425 tokens/s |
| response length 2048 压力测试吞吐 | 343-411 tokens/s |

### 5.3 Reward 曲线

下图展示训练期间的 `critic/rewards/mean` 变化趋势：

![Qwen3-8B ReMax reward curve](https://github.com/user-attachments/assets/45428637-fe4b-45ba-a63a-96cac151e719)

### 5.4 性能与稳定性

- `critic/score/mean` 从约 0.26 上升至 0.60-0.76 区间，最高达到 0.764。
- 60-step 平均集群吞吐约为 381 tokens/s，高于无 GPU 标杆时的 100 TPS 门槛。
- Qwen3-8B 稳定性测试和长序列压力测试均未出现 OOM 或 HCCL timeout。
- offload 会增加 HBM 与 CPU 之间的数据传输开销，但能为 64 GB 设备上的 colocated
  rollout engine 提供安全显存余量。

## 6. 验收结论

| Issue #23 验收项 | 本次结果 |
| --- | --- |
| reward 上升 | `critic/score/mean` 从约 0.26 上升至 0.60-0.76 区间 |
| 无 GPU 标杆时 TPS > 100 | 平均集群吞吐约 381 tokens/s |
| 提供可复现 recipe | 提供环境、数据、关键参数及启动脚本 |
| 完成 100 steps 或运行 12 小时 | 当前已记录 60 steps、约 2 小时 25 分钟，仍需补充更长验证日志 |

现有结果验证了 Qwen3-8B ReMax 在 FSDP + vLLM-Ascend 组合上的功能、reward 趋势、
性能与短期稳定性。
