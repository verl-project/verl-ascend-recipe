# Qwen3-8B RLOO：8×Ascend 910B3 适配与验证

对应 [issue #79](https://github.com/verl-project/verl-ascend-recipe/issues/79)。本报告是 2026-11-19 的独立复现实验，不复用 README 中历史 20 步结果。

本次 PR 更新提交文档与结果摘要。下文描述的新增算法/模型补丁、数据转换、测试和训练脚本改动保存在本地及实验服务器工作树，尚待代码审查与单独提交；不能仅凭本报告认定这些实现已在 PR 中发布。归档结果索引见 [results/issue79/README.md](results/issue79/README.md)。

## 100 步实测结果

| 指标 | 结果 |
| --- | --- |
| 连续训练步数 | 100 |
| 全局训练 TPS | 1588.43 tokens/s（平均每卡 198.55） |
| 总训练 token 数 | 4,390,387（含 prompt 和 response） |
| 日志累计 step time | 2763.97 秒 |
| 前 10 步平均 reward | 0.578812 |
| 后 10 步平均 reward | 0.741040 |
| reward 窗口增量 | +0.162228 |
| 固定验证集准确率 | 51.5625% → 69.53125%（+17.96875 个百分点） |
| 数值检查 | 所有已记录指标有限 |
| 算法单元测试 | CPU/NPU 共 8 项通过 |

验证集为固定的 128 条 GSM8K test 样本，不是完整 1,319 条测试集。greedy、关闭 thinking、最多生成 512 tokens，训练前与第 20/40/60/80/100 步使用相同配置。对应准确率为 51.5625%、64.0625%、65.625%、67.1875%、68.75%、69.53125%。

TPS 定义为 `sum(perf/total_num_tokens) / sum(perf/time_per_step)`，是八卡合计的训练口径，不是单卡输出-only 推理 TPS，也不包含全部环境初始化时间。没有匹配的 GPU 标杆，因此不报告 GPU 精度误差或 A100 性能比。

## 软件、硬件与输入

- 8×Ascend 910B3，每卡 64GB；模型 `/workspace/Qwen3_8b`。
- torch `2.10.0+cpu`，torch-npu `2.10.0.post4`。
- vLLM `0.23.0+empty`，vLLM Ascend `0.23.0`，Transformers `5.5.4`。
- verl 固定到 `1a8a0f5ffd9d3f169ae4432b68526233ba028102`，应用本目录两份 patch。
- 独立环境 `/workspace/rloo/validation/issue79/env`；通过 `.pth` 复用原 SFT 环境的依赖，PYTHONPATH 显式优先使用固定 verl 源码。TransferQueue 下载归档和依赖清单保存在实验目录。
- GSM8K：7,473 条 train、1,319 条 test。原始数据为 `/workspace/val_sft/data/` 的两轮 SFT 对话。

`prepare_gsm8k_from_sft.py` 将 user 问题转换为训练 prompt，要求 `####` 格式答案，从 assistant 解答中提取 ground truth。解答仅保留在 metadata，不放入 prompt。重新转换得到的两个 parquet SHA-256 与训练输入完全一致。

## 模型与算法实现

模型 patch 关闭 Qwen3 thinking，并配置模型/tokenizer 加载选项，使 chat template 与 GSM8K strict reward 口径一致。NPU 启动脚本已经包含相同模型配置。

算法 patch 使用设备端 `scatter_add` 计算各 prompt 分组的 reward 和、样本数，再计算 leave-one-out baseline：`(group_sum - reward) / (group_size - 1)`。保留原目标函数、mask、乱序/不等长分组与 singleton 零基线语义。浮点归约次序可能变化，因此测试使用误差容限。CPU/NPU 共 8 项公式对照测试通过，另与原始函数进行了回归比较。

128 条 response × 512 tokens 的 CPU advantage 微基准：原实现约 7.25ms，向量化约 0.26ms。此结果仅描述 advantage 计算，不代表整次训练加速比。

## 配置与调优

FSDP2 Actor/Reference，BF16，gradient checkpointing；batch 32 prompts、每 prompt 4 条 response、mini-batch 8，学习率 1e-6，KL reward 系数 0.001，无 critic。vLLM TP=1，每卡利用率 0.5，prompt/response 上限均为 512。

原单卡配方默认 CPU offload，首步约 80.73 秒，其中 Actor 更新约 56.42 秒。八卡关闭参数/优化器 offload 后，首步约 33.11 秒，其中 Actor 更新约 9.36 秒，Actor 峰值分配约 11.44GB。两次随机 rollout 不完全相同，这属于配置调优观察，不是严格配对性能基准；最终结论采用完整 100 步统计。

启动脚本增加 `rollout_n >= 2` 检查，将默认 epoch 预算从 1 提高到 10，避免默认 batch 下不足 100 步便结束，并允许通过 `offload` 环境变量设置卸载。100 步实测显式使用 batch 32、epoch 2，最后的总步数限制为 100。

## 复现与证据

```bash
cd /workspace/rloo/validation/issue79
# 不要在已有训练运行时重复启动。
bash run.sh
python summarize.py
```

首次数据转换：

```bash
python /workspace/rloo/rloo/prepare_gsm8k_from_sft.py \
  /workspace/val_sft/data /workspace/rloo/validation/issue79/data
```

加载 CANN、ATB 环境后，参考 `patch/README.md` 应用两个补丁。Ray 还需要 CANN 的 Python site-packages 路径以检测 NPU 类型。实际环境路径、可执行配置见实验目录 `launch.sh`、`frozen_recipe.sh`、`environment.freeze.txt`。

实验目录 `/workspace/rloo/validation/issue79/` 包含：

- `train.log`、`train.exit`：原 100 步日志和包装进程退出状态；
- `summary.json`、`metrics.json`、`summarize.py`：统计结果与提取程序；
- `test_rloo.log`、`pytest.log`：回归与 8 项公式测试；
- `rloo_benchmark.json`：CPU advantage 微基准；
- `checkpoints/global_step_100/`：8 卡 FSDP2 模型、优化器和恢复状态；
- `resume.log`、`resume.exit`：第 100 步 checkpoint 恢复验证。

## 收尾异常与状态

训练器已完成 100 步、最终验证和 checkpoint 保存，但外层脚本返回 127：运行期间修改启动脚本导致 Bash 在训练返回后读到了错误偏移，将配置参数当成命令。原始日志和退出码保留，不能改写成成功。冻结脚本后，已从 step 100 恢复并完成 step 101、保存新 checkpoint，`resume.exit=0`，验证了模型、优化器和训练状态可恢复。

实验指标满足无 GPU 标杆时的 100 步、reward 上升、TPS > 100 条件。完整任务仍需要人工审查代码、更新已有 PR #105、维护者合入以及将实践文档提交 issue；不能将本地实现视作已经合入。
