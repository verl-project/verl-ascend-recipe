# 专家裁剪 MoE 模型的 SFT 行为恢复（small_for_large 第二阶段）

## 依赖的 `verl` 版本

见 [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt)。本 recipe 的数据构建与评测工具与训练器无关（对接 OpenAI 兼容的 vLLM 服务、产出 HF 格式数据集）；`verl` 只在 RL 一致性阶段（规划中，见路线图）才需要。

## 背景

`small_for_large` recipe 把大 MoE 模型裁剪到 top-k 高频专家，用于快速 RL 精度验证。如其父 README 所记录，裸裁剪是有损的：gsm8k-500 上 DeepSeek-V4-Flash 从 **92.60 掉到 39.58**（256e → 64e）。训练侧适配在那里仍是 TODO。

本 recipe 是对该缺口的回答：**用教师（全模型）输出做一次短 SFT 来恢复裁剪模型的行为**，然后度量行为恢复了了多少。这是「以小验大」的训练侧一半：裁剪缩小模型，SFT 教它重新表现得像原模型。

## 核心思想

1. 部署**全模型**一次，对目标训练集生成思维链回答（教师轮；同一服务可顺带产出裁剪用的激活统计，一次扫描两用）。
2. 给教师输出打分，只保留干净收尾且长度预算内的样本。
3. 从存活样本构建小 SFT 集（几百条），多模态输入（图像）从源数据集回联。
4. 用该数据集 SFT **裁剪模型**——VL 模型的视觉塔冻结。
5. 在固定的、带种子的训练集子集（「吸收度」：小模型吸收了多少教师行为）与留出集（「泛化度」）上逐检查点评测。
6. 可选：从恢复后的检查点继续 RL（GRPO）——RL 一致性阶段。

## 已验证结果（试点）

试点模型：**Step-5-Preview**（352 专家多模态 MoE，92 文本层 + 47 视觉层），裁剪到 top-16 专家（1.21 TB → 75 GB，单机 8 NPU 可训练），200 样本 SFT 集，8×Ascend 950DT。

| 检查点 | base | 100 步 | 200 步 | 300 步 | 400 步 |
|---|---|---|---|---|---|
| 吸收度正确率（n=100） | 0% | 35% | 55% | 72% | 74% |
| 截断输出（每 100 题） | 100 | 7 | 1 | 0 | 0 |

读法：

- 裁剪模型在几百个 SFT 步内恢复教师行为的大部分并饱和（400 步 ≈ 74%）；证据是**学习曲线**，不是单个数字。
- base 裁剪模型 0% 不是能力下限，而是输出格式未驯化（每题跑到 token 上限）；截断随行为对齐 100 → 0。
- 从恢复后的检查点起跑的 RL（GRPO）在固定 100 题验证集上持续改进 reward / format-pass——小模型仍可训练，这正是以小验大的意义。

## 流程布局（规划）

```
sft_recovery/
├── README.md / README_zh.md     # 本文档（双语）
├── REQUIRED_VERL.txt            # verl 版本钉扎（RL 一致性阶段）
├── data_build/
│   ├── build_sft_dataset.py     # 教师池 -> 过滤/种子抽样/渲染验证 jsonl   [本骨架已含]
│   ├── collect_teacher_outputs.py  # 可断点续跑的教师生成 client           [TODO]
│   └── score_filter_eval_samples.py # 教师输出评分 + 拉黑                  [TODO]
├── eval/
│   ├── materialize_eval_split.py   # 同构 parquet + seed42 固定索引 + GT 回联 [TODO]
│   ├── eval_client.py              # 固定解码评测 client，原子写、可续跑       [TODO]
│   └── score_eval.py               # 数学等价评分（可插拔后端）               [TODO]
└── docs/
    └── absorption_curve.md         # 方法论：为什么学习曲线就是结果          [TODO]
```

## 数据构建细节（本骨架已含的部分）

`data_build/build_sft_dataset.py` 是方法的核心。给定逐样本教师输出（JSON：question / content / metrics / blacklisted）与源数据集（内嵌图像的 parquet），它：

1. 过滤：未拉黑、`finish_reason == stop`、教师 token 总量在预算内；
2. 用**固定种子**洗牌并抽 N；
3. 按样本 index 从源 parquet 回联图像；
4. （可选，强烈建议）用**训练器自己的 processor + chat template** 渲染每个候选，渲染长度超上限即拒——杜绝「jsonl 里好好的、训练时爆长」这类 bug；
5. 写 `sft<N>.jsonl`（messages 格式 + 供后续 GT 回联的 `source_id`）与 `selection_stats.json`（过滤链完整审计：池大小、逐原因跳过、验证期拒绝、最终长度分位）。

渲染验证可插拔（`--render-module`），工具本身不引任何训练器专有依赖；纯文本数据集（上游 gsm8k 场景）可免 parquet/pandas 直接使用。

## 路线图

- [ ] 补 `collect_teacher_outputs.py` + `score_filter_eval_samples.py`
- [ ] 补评测链（物化 + client + 评分）
- [ ] vision-freeze 零漂移审计工具（base 与各检查点的冻结模块权重比对）
- [ ] verl GRPO 上的 RL 一致性阶段：带预填 think 适配的 reward adapter、全模型 vs 裁剪模型同配置对比（进行中的 MiniMax-M3.1-Flash top16e 单机试点将为此阶段供数）
