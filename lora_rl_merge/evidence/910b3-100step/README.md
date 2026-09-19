# 四张 Ascend 910B3 的100步训练记录

完整 TaskRunner 日志包含第1–100步训练，以及第0、20、40、60、80、100步验证。
只将两处内部 IP 替换为 `[REDACTED_INTERNAL_IP]`，992行全部保留，101个指标行与原始记录一致。
该文件不包含其他分布式进程的单独日志；容器退出状态另行保存。

| 文件 | 内容 |
| --- | --- |
| [training_100step.log](training_100step.log) | 配置、初始化、全部训练步和最终验证。 |
| [log_sanitization.json](log_sanitization.json) | 脱敏规则、原始与公开日志哈希、行数和指标一致性。 |
| [summary.json](summary.json) | 从完整日志计算的指标摘要。 |
| [training-curves.png](training-curves.png) | 奖励、验证准确率、吞吐量、梯度范数和步时曲线。 |
| [provenance.json](provenance.json) | 实测源码、依赖版本及文件哈希。 |
| [inputs.sha256](inputs.sha256) | 模型、分词器和数据哈希，路径相对于复现工作目录。 |
| [process.json](process.json) | 容器退出状态和起止时间。 |
| [checkpoint100.json](checkpoint100.json) | 第100步检查点文件及元数据；不代表恢复训练已验证。 |

在仓库根目录复算并比较摘要，只需要 Python 标准库：

```bash
python3 lora_rl_merge/tools/check_validation.py \
  lora_rl_merge/evidence/910b3-100step/training_100step.log --devices 4 --output /tmp/lora-summary.json
diff -u lora_rl_merge/evidence/910b3-100step/summary.json /tmp/lora-summary.json
```

安装 Matplotlib 后，可重新绘图。绘图不要求奖励上升或吞吐量达标，也可以用于检查未完成的实验：

```bash
python3 lora_rl_merge/tools/plot_validation.py \
  lora_rl_merge/evidence/910b3-100step/training_100step.log /tmp/lora-curves.png \
  --devices 4 --hardware 'Ascend 910B3'
```

实测版本见 `provenance.json` 的 `validated_recipe_commit`。其中的指标摘录哈希保留历史提取记录，
提交材料仅保留完整日志。后续修改涉及文档和日志工具，训练命令、超参数和两份补丁未改变。
后期准确率下降及其他实验限制见[验证报告](../../ALGORITHM_TUNING_REPORT.md)。
