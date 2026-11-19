# Issue #79 验收记录

本页记录 2026-11-19 在 `/workspace/rloo` 的独立实验；历史 20 步结果见 README。
详细配置、指标口径和异常说明见 [实践文档](PRACTICE_ISSUE79.md)。
本次 PR 更新范围为文档和结果文件；表中的新增补丁与脚本已在本地/服务器准备，尚未随本次更新发布源码。

| 要求 | 本次证据 | 状态 |
| --- | --- | --- |
| Qwen3-8B + RLOO + FSDP + vLLM Ascend | 8×910B3，FSDP2，`adv_estimator=rloo` | 已验证 |
| 模型适配 patch | `patch/verl_qwen3_8b_rloo_model.patch`，通过应用检查 | 已提供，未合入 |
| 算法优化实现 | `patch/verl_rloo_vectorized.patch`，CPU/NPU 8 项测试通过 | 已提供，未合入 |
| 连续 100 步或 12 小时 | 日志包含连续 step 1–100，最终验证、8 卡 checkpoint 已保存 | 已完成训练；原包装脚本收尾异常见下文 |
| 无 GPU 标杆时 reward 上升 | 前/后 10 步均值 0.578812 → 0.741040 | 达标 |
| 无 GPU 标杆时 TPS > 100 | 八卡全局训练 1588.43 tokens/s | 达标 |
| 精度辅助证据 | 固定 128 条 GSM8K test，51.5625% → 69.53125% | 有提升；不是完整测试集 |
| GPU 平均误差及 A100 性能比 | 未提供匹配标杆 | 不适用，不作推断 |
| 实践文档 | `PRACTICE_ISSUE79.md` | 已提供，尚未发布至 issue |
| PR 合入 | 已有 PR #105，当前仍 open | 未完成 |

原训练器完成后，包装脚本因运行中修改导致 Bash 读偏移异常，退出码为 127。原始 `train.log`/`train.exit` 保留；冻结脚本后，已从 step 100 恢复并完成 step 101、保存新 checkpoint，`resume.exit=0`，恢复验证通过。

**完整社区验收尚未完成**：仍需完成代码人工审查、更新现有 PR、维护者合入和 issue 文档提交。本地或服务器修改不等于上游已经合入。
