# RLOO 算法扩展任务验收标准

## 一、任务范围

本期任务基于 verl 开放仓库进行 RLOO 算法扩展。开发成果需合入目标代码仓库，并保证代码、配置、测试结果和文档可以追溯。

## 二、任务交付件

### 1. 开发代码

- 模型适配 Patch：[`patch/verl_qwen3_8b_rloo_model.patch`](patch/verl_qwen3_8b_rloo_model.patch)
- 算法优化实现

### 2. 验收材料

- 训练日志及关键指标记录
- 下游评测结果
- 性能测试结果
- 算法适配调优实践文档 1 篇

## 三、验收标准

### 1. 精度要求

训练需满足以下要求：

- 连续训练 12 小时，或完成 100 个训练步；二者满足其一即可。
- 训练过程中 Reward 呈上升趋势。
- 有 GPU 标杆时：
  - 平均误差小于 5%。
  - 下游评测结果相对 GPU 标杆的误差在 5% 以内。
- 无 GPU 标杆时：以 Reward 呈上升趋势作为精度验收标准。

验收材料中需说明 GPU 标杆、平均误差和下游评测误差的计算口径，并提供对应日志或结果文件。

### 2. 性能要求

- 有 A100 标杆时：A2 性能不低于 A100 性能的 70%，即：

  ```text
  TPS(A2) / TPS(A100) >= 0.70
  ```

- 无 GPU 标杆时：A2 实测 TPS 绝对值大于 100。

性能测试需记录硬件数量、模型、权重、数据集、输入和输出长度、并行策略、批量大小、软件版本及 TPS 计算口径，确保结果可复现。

### 3. 实践文档要求

提交算法适配调优实践文档 1 篇，至少包含：

- 任务背景与目标
- 环境及软硬件配置
- 模型适配 Patch 说明
- RLOO 算法优化实现说明
- 训练配置与启动方法
- 精度验证方法及结果
- 性能测试方法及结果
- 问题定位、调优过程与结论
- 已知限制及后续建议

## 四、任务完成标准

以下各项全部满足后，任务视为完成：

- [ ] 模型适配 Patch 已提供并通过 `git apply --check`，但尚未合入目标 verl 仓库。
- [ ] RLOO NPU 启动及配置级优化已通过提交 `63a74ca` 合入 `origin/qwe3_8b_rloo`；未发现 verl 算法源码修改，是否满足“算法优化实现”仍需验收方确认。
- [ ] 已连续训练 12 小时或完成 100 个训练步。
- [x] Reward 呈上升趋势，并提供可核验的训练日志。
- [x] 当前无 GPU 精度标杆，Reward 上升，符合无标杆精度验收条件；另有同口径下游评测提升结果。
- [x] 当前无 A100 性能标杆，训练及推理 TPS 均大于 100。
- [ ] 已提交 1 篇算法适配调优实践文档。
- [x] 训练、评测和性能测试结果及其复现配置已归档在实验服务器。

### 当前总体结论

**暂未完成验收。** 当前阻塞项如下：

1. 主实验仅完成 20 步，未达到训练 12 小时或 100 步的门槛。
2. 模型适配 Patch 已新增并验证可应用，但尚未合入目标 verl 仓库。
3. 尚未提交 RLOO 算法适配调优实践文档。
4. 当前已合入内容为启动脚本及配置级优化，未发现 verl RLOO 算法源码修改；需验收方确认其是否满足“算法优化实现”的定义。

## 五、验收结果记录

| 验收项 | 验收指标 | 实测结果 | 证据路径 | 结论 |
| --- | --- | --- | --- | --- |
| 代码交付 | 模型适配 Patch 已合入 | 已新增 Qwen3-8B RLOO 模型适配 Patch；在 verl 提交 `1a8a0f5...` 上通过 `git apply --check`，尚未合入目标仓库 | `rloo/patch/verl_qwen3_8b_rloo_model.patch`；`rloo/patch/README.md` | 待合入 |
| 代码交付 | 算法优化实现已合入 | 提交 `63a74ca` 已在远端分支；包含 RLOO、FSDP2、动态批次、卸载和 vLLM-Ascend 参数，未修改 verl 算法源码 | Git 分支 `origin/qwe3_8b_rloo`；`rloo/run_qwen3_8b_rloo_fsdp_npu.sh` | 部分满足，待确认口径 |
| 训练时长/步数 | 训练 12 小时或 100 步 | 完成 20 步；平均每步 145.52 秒，未达到 12 小时或 100 步 | `/workspace/rloo_2h_budget_20260918.log`；`/workspace/rloo_visualization_step20/training_summary.json` | 不通过 |
| Reward | 呈上升趋势 | 首步 `0.593750`，末步 `0.656004`；前 5 步均值 `0.571879`，后 5 步均值 `0.615449`，提升 `0.043570`；线性斜率 `+0.003911/步` | `/workspace/rloo_2h_budget_20260918.log` | 通过 |
| 平均误差 | 有 GPU 标杆时小于 5% | 未提供 GPU 精度标杆，无法计算平均误差；按无 GPU 标杆规则检查 Reward 趋势 | `/workspace/rloo_2h_budget_20260918.log` | 不适用 |
| 下游评测 | 有 GPU 标杆时误差在 5% 以内 | 无 GPU 标杆；同口径基座准确率 `53.30%`，RLOO 准确率 `62.47%`，提升 `9.17` 个百分点 | `/workspace/eval_qwen3_8b_base_full/summary.json`；`/workspace/eval_rloo_step20_tps_20260922/summary.json` | 不适用；结果有提升 |
| 性能 | 有标杆时 A2/A100 >= 0.70；无标杆时 TPS > 100 | 无 A100 标杆；20 步训练平均吞吐 `153.67 TPS`，训练权重单卡推理复测 `381.31 TPS` | `/workspace/rloo_visualization_step20/training_summary.json`；`/workspace/eval_rloo_step20_tps_20260922/summary.json` | 通过 |
| 实践文档 | 算法适配调优文档 1 篇 | 服务器未发现 RLOO 任务专项适配调优文档 | 无 | 不通过 |

## 六、实验口径

- 训练模型：Qwen3-8B。
- 训练算法：RLOO，`rollout_n=5`，KL 系数 `0.001`。
- 训练权重：`/workspace/models/rloo_qwen3_8b_step20_hf`。
- 下游数据集：GSM8K test，共 1,319 条。
- 下游生成配置：greedy decoding，`temperature=0`、`top_p=1`、`max_tokens=512`、关闭 thinking。
- 性能复测：单张 Ascend910_9382（64 GB HBM）、BF16、TP=1；输出 TPS 按总输出 token 数除以完整生成耗时计算。
- 软件基线：verl `1a8a0f5ffd9d3f169ae4432b68526233ba028102`（`0.10.0.dev0`）、vLLM `0.23.0+empty`、vLLM-Ascend `0.23.0`、Transformers `5.5.4`。
