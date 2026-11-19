# Issue #79 结果归档

实验日期：2026-11-19。完整口径与配置见 [实践文档](../../PRACTICE_ISSUE79.md)。

- [summary.json](summary.json)：100 步指标、验证准确率和恢复状态。
- [pytest.log](pytest.log)：CPU/NPU 8 项公式测试通过的输出。
- [rloo_benchmark.json](rloo_benchmark.json)：CPU advantage 微基准，不代表端到端加速比。
- [train.exit](train.exit)：原包装脚本退出码 127；训练本身已完成 100 步、最终验证和保存。
- [resume.exit](resume.exit)：冻结脚本后恢复至 step 101，退出码 0。
- [artifacts.sha256](artifacts.sha256)：服务器上的源码归档、完整日志和补丁校验和。

`training_complete=false` 沿用原脚本要求退出码为 0 的判定，不代表缺少训练步。`training_steps_complete=true` 表示连续 100 步且记录指标有限；`checkpoint_recovery_passed=true` 表示 checkpoint 恢复成功；`experiment_requirements_passed=true` 只涵盖实验侧指标，不包含 PR 合入或社区验收。

完整训练/恢复日志、模型分片及可执行复现配置保留在实验服务器 `/workspace/rloo/validation/issue79/`，本次文档提交未上传这些大文件或执行脚本。校验和可用于拿到原文件后核验，不能替代原始文件。
