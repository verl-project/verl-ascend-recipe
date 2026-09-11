# 单机4 × 910B3的100步训练证据

`metrics.log` 按原顺序保留原始TaskRunner日志中从 `step:<整数> -` 开始的完整指标行，
包含第1–100步训练和第0/20/40/60/80/100步验证。训练后的验证通常与训练指标位于同一行。
没有筛选或改写任何指标值，仅省略非指标日志文本。原始文件为443069字节，SHA256为
`b4273bb72860b9833fea873783acb833ea09f9f0f003e871adcb5a4d612ac95d`。

| 文件 | 用途与边界 |
| --- | --- |
| `metrics.log` | 提供可离线检查的原始指标行；不包含模型权重或训练样本。 |
| `summary.json` | 由交付检查器复算，所有数值均来自上述日志。 |
| `provenance.json` | 记录实测recipe、依赖提交、原始日志及训练脚本/补丁哈希；实测提交不等于整理后的交付提交。 |
| `inputs.sha256` | 记录模型、tokenizer与数据的文件哈希，相对路径基于复现工作目录。 |
| `process.json` | 记录该次容器的实际退出状态和起止时间。 |
| `checkpoint100.json` | 记录模型/优化器分片、额外状态、FSDP2和LoRA元数据；没有加载模型/优化器作恢复测试。 |
| `image.json` | 记录归档加载前后镜像的RootFS层和运行配置校验，不把image ID相等作为唯一依据。 |
| `training-curves.png` / `.svg` | 从同一份指标生成的单机910B3曲线，保留后期reward下降和梯度增大。 |

在recipe仓库根目录执行：

```bash
python3 lora_rl_merge/tools/check_validation.py \
  lora_rl_merge/evidence/910b3-100step/metrics.log --devices 4 --output /tmp/lora-summary.json
diff -u lora_rl_merge/evidence/910b3-100step/summary.json /tmp/lora-summary.json
```

可选重绘依赖Matplotlib，数值检查不需要此依赖：

```bash
python3 lora_rl_merge/tools/plot_validation.py \
  lora_rl_merge/evidence/910b3-100step/metrics.log /tmp/lora-curves \
  --devices 4 --hardware 'Ascend 910B3'
```

实测训练脚本与依赖补丁来自 `provenance.json` 的 `validated_recipe_commit`。
交付整理仅更正训练脚本注释，保留命令及超参；两份依赖补丁字节不变。
离线检查工具和本文档不参与训练更新，不能以工具检查成功代替新配置的实机验证。
后期精度下降与复现环境的镜像取得要求见[实践文档](../../ALGORITHM_TUNING_REPORT.md)。
