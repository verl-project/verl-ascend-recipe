# Issue #79 实验材料索引

[实践报告](../README.md)说明了实验环境、指标口径和启动脚本的退出状态。

- [train.log](train.log)：原始的完整 100 步训练日志，含实际启动命令和每步指标。
- [reward_curve.png](../reward_curve.png)：从 train.log 中直接提取第 1–100 步 reward 绘制的曲线，未做平滑处理。
- [launch.sh](launch.sh)：服务器在训练结束后更新的启动脚本；它调用 frozen_recipe.sh，不是原始 100 步运行时的脚本快照。

日志和曲线随 PR 提交，评审无需访问实验服务器即可查看。模型 checkpoint 体积较大，仍保留在实验服务器。
