# Qwen3-8B 在昇腾 A2 上的 FSDP SFT 验证报告

2026-11-12 在单机 8 张 Ascend 910B4（每卡 32 GiB）上验证了 Qwen3-8B 的两种 FSDP 训练配置。开启 LoRA 和 Ulysses 序列并行的作业完成 100 步训练，关闭两者的全参数作业完成 2 步功能验证。两次作业均完成验证集评估、保存检查点并以状态码 0 退出。[任务 #74](https://github.com/verl-project/verl-ascend-recipe/issues/74)

## 配置与日志

| 配置 | 训练步数 | 全局 batch | 最后一步训练 loss | 验证 loss（样本数） | 峰值已分配显存¹ | 完整日志 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA rank 32、alpha 16，SP=2 | 100 | 32 | 0.2753 | 0.3928（128） | 13.13 GiB | [LoRA 开启](logs/lora_sp2_100steps.log) |
| 全参数，SP=1 | 2 | 8 | 0.8970 | 1.0729（8） | 22.22 GiB | [LoRA 关闭](logs/full_sp1_2steps.log) |

¹ 取训练器记录的 `perf/max_memory_allocated_gb` 最大值。两次运行的 batch、步数及验证集大小不同，全参数结果仅用于验证该分支可以更新权重和保存检查点。

## 数据与环境

训练集为 GSM8K 的 7,473 条训练样本。将每条记录的 question 和 answer 分别作为 user、assistant 消息写入 Parquet 的 `messages` 列，测试集按同样方式处理。两次作业均设置学习率 1e-4、最大序列长度 1024、每卡 micro batch 为 1。

软件版本为 verl 0.9.1、CANN 9.1.0、PyTorch 2.10.0、torch_npu 2.10.0.post4 和 Transformers 5.5.4。首次长跑在完成保存后遇到进程退出异常；在该镜像中预加载系统 `libjemalloc.so.2` 后，表中两次作业均正常退出。运行参数可见各日志开头的 `torchrun` 命令。[训练脚本与复现说明](../../../sft/README.md)

## 精度与性能

LoRA 运行前 10 步平均训练 loss 为 1.0664，最后 10 步为 0.3276。第 11–90 步（含两端）累计处理 487,529 个全局训练 token，从进度日志读取的耗时约为 370 秒，折算八卡合计 **1,318 训练 tokens/s**。计时窗口排除了启动、最终验证和保存检查点的耗时。

按 issue 的无 GPU 标杆口径，本次记录覆盖 100 步 loss 下降，并提供超过 100 tokens/s 的训练吞吐数据。本次没有测量 A100 相对性能、GSM8K 生成准确率或推理输出 TPS。
