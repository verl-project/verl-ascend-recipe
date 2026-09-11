# 四张 Ascend 910B3 的100步训练记录

`metrics.log` 保留 TaskRunner 日志中以 `step:<整数> -` 开始的完整指标行。
原顺序和数值不变，仅省略非指标文本。

- 训练记录：第1–100步。
- 验证记录：第0、20、40、60、80、100步。
- 原始日志大小：443069字节。
- 原始日志 SHA256：`b4273bb72860b9833fea873783acb833ea09f9f0f003e871adcb5a4d612ac95d`。

| 文件 | 内容 |
| --- | --- |
| `metrics.log` | 原始指标行 |
| `summary.json` | 指标复算结果 |
| `provenance.json` | 源码和依赖版本；日志、脚本和补丁哈希 |
| `inputs.sha256` | 模型、分词器和数据哈希；路径相对于复现工作目录 |
| `process.json` | 容器退出状态、起止时间 |
| `checkpoint100.json` | 第100步检查点的文件及元数据；未测试恢复训练 |
| `image.json` | 实测镜像的历史记录；镜像一致性不作为复现要求 |
| `training-curves.png` / `.svg` | 训练曲线 |

在仓库根目录执行以下命令，即可复算并比较结果。

```bash
python3 lora_rl_merge/tools/check_validation.py \
  lora_rl_merge/evidence/910b3-100step/metrics.log --devices 4 --output /tmp/lora-summary.json
diff -u lora_rl_merge/evidence/910b3-100step/summary.json /tmp/lora-summary.json
```

如需重新绘图，请先安装 Matplotlib，再执行以下命令。数值检查不需要 Matplotlib。

```bash
python3 lora_rl_merge/tools/plot_validation.py \
  lora_rl_merge/evidence/910b3-100step/metrics.log /tmp/lora-curves \
  --devices 4 --hardware 'Ascend 910B3'
```

实测代码版本见 `provenance.json` 的 `validated_recipe_commit`。此后只更正了训练脚本注释，
可执行命令、超参数和两份依赖补丁均未改变。后期准确率下降及实验限制见[验证报告](../../ALGORITHM_TUNING_REPORT.md)。
