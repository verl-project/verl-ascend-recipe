# 四张 910B3 上的 MOPD 100 步结果

本目录对应非思考模板、明确 GSM8K 答案格式和 batch256 的一次完整运行。

| 文件 | 用途 |
| --- | --- |
| [training_log.txt](training_log.txt) | 完整 TaskRunner 日志，共 1022 行，包含初始化、训练和验证。 |
| [results.json](results.json) | 十个训练 reward 窗口、六次全量验证和累计吞吐量。 |
| [training-curves.png](training-curves.png) | 从完整日志绘制的 reward、验证准确率、吞吐、梯度、步时和损失。 |
| [provenance.json](provenance.json) | 实际运行的源码、软件、镜像、配置、终态及 checkpoint 文件哈希。 |
| [log_sanitization.json](log_sanitization.json) | 两处内部 IP 的替换规则、原始及发布文件哈希；101 个指标行不变。 |
| [delivery_checks.json](delivery_checks.json) | 实测文件与交付脚本之间的等价关系和离线检查。 |

运行 recipe 提交是历史实验身份；复现使用当前交付的训练脚本和必要补丁，并按上级 README 固定依赖。
发布日志的哈希见 `results.json` 的 `raw_log_sha256`，未脱敏原件的哈希见 `original_raw_log_sha256`。
运行信息中的日志哈希仍指向原件，没有改写历史记录。

在仓库根目录执行：

```bash
python3 mopd/tools/summarize_training.py \
  --segment mopd/results/2026-09-11-batch256/training_log.txt 1 100 \
  --expected-steps 100 --devices 4 --output /tmp/mopd-summary.json
```

查看 `training_length_complete`、`nonfinite_metric_step_counts`、`zero_gradient_steps`、`reward` 和 `throughput`。
本次长度完整、非有限指标与零梯度记录均为空，首末十步 reward 上升且四卡加权吞吐大于 100。
汇总器同时支持查看部分轨迹，不能仅由退出码推断完整训练通过。

安装 matplotlib 后，可重绘图表：

```bash
python3 mopd/tools/plot_training.py \
  mopd/results/2026-09-11-batch256/training_log.txt /tmp/mopd-training-curves.png
```

本目录不包含模型权重。checkpoint 文件检查没有执行实际恢复，也不替代同配置完整训练的验证结果。
