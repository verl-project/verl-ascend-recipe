# 四张 Ascend 910B3 的100步训练记录

`metrics.log` 按原顺序保留 TaskRunner 日志中以 `step:<整数> -` 开始的完整指标行，
包含第1–100步训练和第0、20、40、60、80、100步验证。仅省略非指标文本，没有改写指标值。
原始日志为443069字节，SHA256为 `b4273bb72860b9833fea873783acb833ea09f9f0f003e871adcb5a4d612ac95d`。

| 文件 | 说明 |
| --- | --- |
| `metrics.log` / `summary.json` | 前者保存指标行，后者保存从这些指标复算的结果。 |
| `provenance.json` | 记录实测源码版本、依赖提交及日志、脚本和补丁的哈希。 |
| `inputs.sha256` | 记录模型、分词器和数据文件的哈希，路径相对于复现工作目录。 |
| `process.json` | 记录容器退出状态及起止时间。 |
| `checkpoint100.json` | 记录第100步检查点的文件及元数据，未执行加载和恢复训练测试。 |
| `image.json` | 记录镜像归档以及加载前后的 RootFS 层和运行配置校验结果。 |
| `training-curves.png` / `.svg` | 展示从同一份日志生成的训练曲线。 |

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
