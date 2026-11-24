# Qwen3-8B RLOO：8 卡 Ascend 910B 验证报告

本报告记录 2026 年 11 月 19 日针对 [Issue #79](https://github.com/verl-project/verl-ascend-recipe/issues/79) 完成的独立实验，与此前的 20 步实验分开统计。PR 中附有[完整的 100 步训练日志](evidence/train.log)和[reward 曲线](reward_curve.png)。

## 实测结果

| 指标 | 结果 |
| --- | ---: |
| 连续训练步数 | 100 |
| prompt 与 response 总 token 数 | 4,390,387 |
| 日志中训练步耗时之和 | 2,763.97 秒 |
| 8 卡合计训练吞吐 | 1,588.43 tokens/s（平均每卡 198.55） |
| 第 1–10 步平均 reward | 0.578812 |
| 第 91–100 步平均 reward | 0.741040 |
| 前后窗口 reward 增量 | +0.162228 |
| 固定 128 道 GSM8K 验证题的准确率 | 51.5625% → 69.53125% |

训练吞吐按所有训练步的 perf/total_num_tokens 之和除以 perf/time_per_step 之和计算，包含 prompt 与 response token，不是仅计算输出 token 的生成速度，也不包含环境初始化时间。平均每卡数值为合计吞吐除以 8。

验证使用 GSM8K 测试集中固定的 128 道题，而非完整的 1,319 道题；采用 greedy 解码、关闭 thinking，最多生成 512 个 token。第 0/20/40/60/80/100 步的准确率依次为 51.5625% / 64.0625% / 65.625% / 67.1875% / 68.75% / 69.53125%。由于没有匹配的 GPU 基线，本报告不推断 GPU 精度误差或 A100 性能比。

## 环境与配置

本次实验使用的运行脚本：[run_qwen3_8b_rloo_fsdp_npu.sh](../../../rloo/run_qwen3_8b_rloo_fsdp_npu.sh)。

- 硬件与模型：8 张 Ascend 910B，Qwen3-8B BF16。
- 训练与推理：FSDP2 Actor/Reference，vLLM Ascend rollout。
- 软件：torch 2.10.0+cpu、torch-npu 2.10.0.post4、vLLM Ascend 0.23.0、Transformers 5.5.4。
- verl 固定提交：1a8a0f5ffd9d3f169ae4432b68526233ba028102。
- 训练参数：每批 32 个 prompt、每个 prompt 采样 4 条 response、mini-batch 8、学习率 1e-6、KL reward 系数 0.001，不启用 critic。
- prompt 和 response 上限各 512 token；rollout 显存利用率 0.5；本次实测关闭参数及优化器 offload。

服务器上的启动脚本在训练期间被修改，因此其运行快照与当前 PR 中的脚本不能视为逐字相同。训练结果与吞吐只代表上述实验环境；该实现改动需要单独审查，不能将本次性能数据归因于未修改的上游实现。

## 复现记录与退出状态

[train.log](evidence/train.log)开头记录了实际执行的 Hydra 命令，逐步记录训练指标；本报告中的统计值由该日志汇总。

训练器完成了连续 100 步、最终验证与 checkpoint 保存。由于训练过程中启动脚本被修改，外层脚本在训练结束后退出 127。冻结脚本后，从第 100 步 checkpoint 恢复并完成第 101 步，退出码为 0。恢复成功不改变原始启动脚本的退出状态；恢复运行的完整日志仍保留在实验服务器。
