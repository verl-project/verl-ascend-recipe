# MiniMax-M3 以小验大实践（Ascend NPU）

本目录是「以小验大」方法在 MiniMax-M3 上的一次完整落地：把 ~428B 的原生多模态 MoE 大模型裁剪成
单机可训的小模型，用「全尺寸评测产出 SFT 数据集 → 小模型 SFT → RL 验证」的闭环，替代直接在大模型上做
RL 调试与优化验证。与 [GLM-5.3-Flash 示例](../GLM5.3-Flash/README.md) 互为补充：那条路径走
MindSpeed-MM 训练栈 + F32→BF16 权重转换，本路径覆盖 **VeOmni 训练栈 + VeOmni checkpoint → HF
release 布局转换**。通用的专家裁剪方法论（vllm-ascend patch、专家激活统计 `stat_moe.py`、裁剪脚本
`expert_pruned.py`）见上级目录 [README_zh.md](../../README_zh.md)。

## MiniMax-M3 模型情况

MiniMax-M3 是 MiniMax 的原生多模态（图像/文本输入）MoE 大模型，主打代码、Agent 与长上下文推理
（来源：官方博客 [MiniMax M3](https://www.minimax.io/blog/minimax-m3)、
Hugging Face `MiniMaxAI/MiniMax-M3`）：

| 项目 | 数值 | 说明 |
| --- | --- | --- |
| 总参数量 | ~428B | MoE 架构 |
| 激活参数量 | ~23B（约 5.4%） | 稀疏激活，推理/训练成本远低于总参数量 |
| MoE 专家 | 每层 128 个路由专家 + 共享专家 | 路由专家占权重约 97%，正是"以小验大"的裁剪抓手 |
| 层数 | 60 层语言层（0-2 层为 dense，其余 57 层 MoE） | |
| 注意力 | MSA 稀疏注意力（`self_attn.index_*` indexer） | 1M 长上下文的关键 |
| 多模态 | 原生 VL：vision tower（32 层）+ `multi_modal_projector` + `patch_merge_mlp` | |
| 上下文 | 1M token | |
| 权重大小 | BF16 约 796 GiB | 推理需要 16 卡 × 64 GiB（TP=16，每卡 ~50 GiB） |

权重体量换算（`prune_moe_experts.py` 实测）：非专家权重固定 25.9 GiB，单个路由专家 113 MB，
总权重 ≈ 25.94 + 每层保留数 × 57 × 0.1054 GiB：

| 每层保留专家数 | 2 | 4 | 8（本示例） | 128（原始） |
| --- | --- | --- | --- | --- |
| 总权重 (GiB) | 37.9 | 49.9 | ~74.0 | 795.5 |

## 模型 RL 训练面临的挑战

- **模型大、训练慢**：BF16 权重约 796 GiB，连推理都要 TP=16 占满一台 16 卡机；RL 一个 step 包含完整的
  rollout 采样 + 参数更新，大模型单步耗时以小时计，一轮完整训练动辄数天到数周，调参反馈周期极长。
- **多机资源难申请**：训练需要按集群规模排队申请，实验排期不可控；一次失败的实验浪费的是整批机器。
- **问题定位成本高**：任何精度或性能问题（loss 异常、reward 不涨、精度偏差、算子性能）都需要在大规模
  多机任务上复现，每一次复现都是高成本操作。
- **性能优化验证慢**：性能优化（并行策略、通信、算子）改动后必须重新训练/推理验证，验证流程本身就把
  优化迭代拖得很慢。

结论：在大模型上直接做 RL 的精度验证、问题定位和性能优化，成本高到难以支撑快速迭代，因此需要
"以小验大"——保留架构和关键组件，裁剪出一个行为尽可能接近原模型、但单机可训的小模型。

## 以小验大的整体流程

```text
全尺寸 MiniMax-M3 (~428B, 每层 128 路由专家)
   │
   ├─ ① 全尺寸评测生成数据集：vllm_server.sh（TP=16）起服务 + run_aisbench_eval.sh 跑 geo3k 图文评测，
   │     保留"回答正确 + 带完整 think"的记录，gen_sft_dataset.py 组装成 VeOmni 可加载的 SFT 数据集
   │
   ├─ ② 随机减专家生成小模型：每层 128 → 8 个专家（moe8，约 74 GiB），并做基线评测
   │     确认仍具备基本语言能力
   │
   ├─ ③ 小模型 SFT 训练：VeOmni 用 ① 的数据集训练到过拟合（ckpt 为 DCP/hf_ckpt 布局），
   │     dcp_to_release_hf.py 转成 vLLM 可加载的 HF release 布局
   │
   ├─ ④ 合格性判定：gen_benchmark_dataset_from_sft.py 把 SFT 数据集还原成 ais_bench 评测集，
   │     起服务复评；训练集正确率落在 20%~40% 区间、loss 收敛到 1e-3 量级
   │
   └─ ⑤ RL 训练检验：gen_verl_dataset_from_sft.py 把 SFT 数据集转成 verl 格式做 RL；
         reward 上涨 + logdiff < 千分位 即通过；通过后小模型即可作为大模型精度/性能
         验证与优化的载体
```

### 1. 全尺寸模型评测生成数据集

用全尺寸 MiniMax-M3 对 benchmark（geometry3k，图文题）做推理评测，评测链路与 ais_bench +
mathruler 打分对齐，得到**带 think（思考链）**的完整回答。

筛选规则（`gen_sft_dataset.py` 自动执行）：

- **只保留带 think 的记录**：SFT 训练教的正是"怎么想"，没有思考链的样本无法用于 SFT；被截断
  （没有闭合 `</think>`）的记录自动丢弃；
- **只保留回答正确的记录**：全尺寸模型答对（mathruler 判分 + `\approx` 等变体兜底），才证明该能力
  存在于模型中，小模型 SFT/RL 才"有得学、有得涨"；
- **长度需符合要求**：过长的 think 会在训练和后续 RL rollout 中放大成本（`--max-tokens` 可按 token
  长度过滤）。

MiniMax-M3 的回答格式与 GLM 不同：思考内容以 `⏸`（思考起始标记）开头、`</think>` 结束，随后是
`\boxed{}` 结论——`vllm_server.sh` 的 `--reasoning-parser minimax_m3` 与之配套。SFT 样本的回答
组装为 `⏸\n{思考}\n</think>\n\n\boxed{X}`。

筛选后的记录组装成 VeOmni 可直接加载的 ShareGPT 数据集（`images/` + `geo3k_sft.json`，按
`--target-rows` 重复到目标条数）。

### 2. 随机减专家生成小模型

思路与 GLM-5.3-Flash 路径相同：把每层 128 个路由专家裁到少量几个，直接得到小尺寸模型。区别在
checkpoint 布局——GLM 是 `mlp.experts.<E>.{gate,up,down}_proj`，MiniMax 是
`block_sparse_moe.experts.<E>.w{1,2,3}`，所以本目录自带 `prune_moe_experts.py`（与 GLM 示例同名，
实现按 MiniMax 布局改写，参照上级目录的 `expert_pruned.py`）。两个注意点：

- **保留的每层专家数不能少于 `num_experts_per_tok`**，路由器才"真正有得选"（脚本会自动把 config 的
  `num_experts_per_tok` 钳到 ≤ 保留数，避免路由退化成 no-op）；
- **小模型必须先过一次基线评测**：裁剪后的模型要能正常推理、输出连贯的词句（`run_aisbench_eval.sh` 少量
  prompt 即可验证），连语言能力都丢掉的裁剪是无意义的。

目标是尽量减到**单机就能训练的最大专家数**——MiniMax-M3 这一次裁到了每层 8 个专家（即
`Minimax-M3-moe8`，约 74 GiB，8 卡可训）。选专家的方式有三种：默认按序号取前 N、
`--rank-by score_bias`（按路由偏置近似高频专家）、`--topk-json`（逐层指定，可与上级目录
`stat_moe.py` 的激活统计配合）。

### 3. 小模型 SFT 训练

裁剪后的小模型只有基本语言能力，答不出正确结果，需要先做 SFT：

- 数据集：第 1 步产出的带 think 数据集，让小模型把全尺寸模型"想过一遍"的推理过程学一遍；
- 训练由 **VeOmni**（`minimax_m3_vl`）完成（不在本目录，checkpoint 形如
  `CKPT/.../checkpoints/global_step_N`，DCP 布局或其 `hf_ckpt/` 导出）；
- **训练到过拟合再保存权重**：过拟合保证小模型对每条训练数据都完整学过，后续多次推理时有一定概率
  答对，RL 阶段才有"答对 → reward 上涨"的起点；
- VeOmni 导出的 checkpoint **不能直接给 vLLM 用**：VeOmni 自带的 `dcp_to_hf` 产物沿用训练侧命名
  （`model.multi_modal_projector.*`、`gate_up_proj` 融合等），vLLM 加载会直接报错。
  `dcp_to_release_hf.py` 把它逆映射成开源 release 布局（拆 `gate/up` 融合、专家按 id 展开、
  vision tower 补 `vision_model.encoder` 层级），并用 `--reference`（moe8 参考权重）校验输出
  名称集合与逐 tensor 形状，全部对上才落盘。

### 4. 什么样的小模型才是合格的小模型

SFT 之后、RL 之前，必须用**SFT 训练数据集本身**（也就是模型学过的那些题）对保存的 ckpt 做推理评测，
同时满足：

1. **训练 loss 收敛到 1e-3 量级**：说明已经出现一定程度的过拟合，训练数据被真正"记住"了；
2. **训练集正确率落在 20%~40%**：
   - 只有约 10%：学得不够，需要基于保存的权重**再训练一轮**；
   - 接近 70%：过拟合过头了。此时 RL 阶段 reward 很可能涨不动——模型已经"背下答案"，没有留给 RL
     通过自身努力上涨的空间，需要**回退到训练过程中的中间 ckpt**，或**从原始权重用更少的步数重新
     训练**，给 RL 预留提升空间。

复评数据集由 `gen_benchmark_dataset_from_sft.py` 生成：它把 SFT 数据集还原成 ais_bench 可用的
`problem/answer/images` parquet，其中原始 gold 答案通过**图片 md5 + 题面文本**从 rollout 记录中
匹配找回（SFT json 里只有模型的 `\boxed{}` 形式，不能当 gold 用）。

一句话：**评分低了就继续训，评分高了就回退 ckpt 或减步数重训**，把正确率调到 20%~40% 的区间再进入 RL。

### 5. RL 训练检验实际效果

这是最终的检验环节：

- **RL 数据集必须由 SFT 数据集转换成 verl 格式**（`gen_verl_dataset_from_sft.py` → verl
  `MultiTurnSFTDataset` 需要的 `messages` + `images` parquet），保证模型在同一个数据集上做强化
  学习，让小模型能"通过自身的努力"把 reward 涨上去。曾出现过 RL 数据集用错（与 SFT 不一致）导致
  reward 不涨的情况；
- **成功标准**：
  1. reward 持续上涨的曲线；
  2. logdiff 在千分位（1e-3）以下（训练与推理框架间的 logprob 一致性）。

达到标准即说明小模型的训练闭环是健康的，可以作为大模型训练的依据：大模型的精度验证、问题定位和性能
优化都可以先在小模型上进行，通过后再上大模型，成本极大降低。

## 脚本使用说明

### 目录内容

| 文件 | 作用 |
| --- | --- |
| `prune_moe_experts.py` | 裁剪路由专家：保留专家重编号、`gate`/`e_score_correction_bias` 切片、config 同步改写（`num_local_experts`、`num_experts_per_tok` 钳制，并修复公开 config 的 `num_hidden_layers` 12→60）；支持 `--rank-by score_bias` / `--topk-json` / `--stage-length`，`--dry-run` 先看计划与体积 |
| `vllm_server.sh` | 评测用 vLLM 服务：全尺寸（`TP_SIZE=16`，16 卡）或裁剪/SFT 产物（默认 TP=8）；`MODEL_PATH`/`MEDIA_PATH`/`TP_SIZE`/`ASCEND_RT_VISIBLE_DEVICES` 可覆盖 |
| `run_aisbench_eval.sh` | ais_bench 图文评测入口（与 GLM 示例同名）：通过 `--dataset-path` 指定评测集（GLM 示例写在数据集配置文件中）；`DATASET_PATH`/`NUM_PROMPTS` 等可用环境变量覆盖 |
| `gen_sft_dataset.py` | rollout JSONL → VeOmni 可加载的 ShareGPT SFT 数据集：mathruler 打分、保留正确 + 完整 think、图片本地化、按 `--target-rows` 重复 |
| `dcp_to_release_hf.py` | VeOmni ckpt（DCP step 目录或 `hf_ckpt`）→ vLLM 可加载的 HF release 布局：逆命名映射 + 拆 `gate/up` 融合 + 对参考权重校验名称/形状 |
| `gen_benchmark_dataset_from_sft.py` | SFT 数据集 → ais_bench 复评 parquet：原始 gold 经图片 md5 + 题面匹配从 rollout 找回 |
| `gen_verl_dataset_from_sft.py` | SFT 数据集 → verl `MultiTurnSFTDataset` parquet（`messages` + `images` 列，图片内嵌 bytes） |

> 环境依赖：vLLM + vllm-ascend、ais_bench（评测）、VeOmni（SFT 训练栈）、mathruler + pylatexenc
> （SFT 打分）；`dcp_to_release_hf.py` 的写入侧依赖 VeOmni，其余转换只需 CPU 上的 `torch` +
> `safetensors` / `pyarrow` + `datasets`。

### 目录约定

脚本不含任何用户/机器专属路径，统一用 `<...>` 占位符 + 环境变量覆盖：

```text
<model_dir>/                         # 原始权重根目录
├── MiniMax-M3/                      # 全尺寸权重（795.5 GiB）
├── Minimax-M3-moe8/                 # 裁剪产物（8 专家/层，~74 GiB；也是转换的参考权重）
└── MiniMax-M3-serve/                # 全尺寸 serve 目录（软链 + 修正 num_hidden_layers 的 config 副本）

<output_dir>/                        # 工作区根目录
├── outputs/default/geo3k_rollout/predictions/vllm-api-general-chat/geometry3k.jsonl
│                                    # 全尺寸 rollout（run_aisbench_eval.sh 的产物，SFT 数据来源）
├── dataset_vllm_rollout/
│   ├── geometry3k_vllm_rollout_sft.json     # gen_sft_dataset.py 产物（ShareGPT）
│   └── benchmark_test_after/geo3k/test.parquet  # gen_benchmark_dataset_from_sft.py 产物（复评集）
├── CKPT/hf_release/global_step_N/   # VeOmni ckpt → dcp_to_release_hf.py 产物（SFT 后的服务对象）
└── verl_start/geo3k-sft-dataset/train.parquet  # gen_verl_dataset_from_sft.py 产物（RL 用）
```

可覆盖的通用变量：`MODEL_PATH`/`TP_SIZE`/`ASCEND_RT_VISIBLE_DEVICES`/`MEDIA_PATH`（服务）、
`DATASET_PATH`/`NUM_PROMPTS`/`HOST_IP`/`HOST_PORT`（评测）；各转换脚本用 `--src`/`--out` 等
命令行参数指定输入输出。

### 典型操作顺序

```bash
# ① 全尺寸模型起服务（16 卡 TP=16；MODEL_PATH 指向 serve 目录）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 TP_SIZE=16 \
MODEL_PATH=<output_dir>/MiniMax-M3-serve ./vllm_server.sh

# ② 评测产出 rollout JSONL（--dataset-path 指向全量 geo3k 基准）
DATASET_PATH=<geo3k_dataset_dir>/geo3k_bench_519/geo3k/test.parquet ./run_aisbench_eval.sh

# ③ rollout → SFT 数据集（先 --dry-run 看筛选统计；--max-tokens 按长度过滤）
python gen_sft_dataset.py --dry-run
python gen_sft_dataset.py --max-tokens 4096

# ④ 裁剪出小模型：每层 128 → 8 个专家（先 --dry-run 看计划与体积）
python prune_moe_experts.py --dry-run --top-k 8
python prune_moe_experts.py --top-k 8 \
    --model-dir <model_dir>/MiniMax-M3 --output-dir <model_dir>/Minimax-M3-moe8

# ⑤ VeOmni SFT 训练（用 ③ 的数据集训练到过拟合，产出 global_step_N 的 DCP/hf_ckpt ckpt）

# ⑥ 训练 ckpt 转 HF release 布局（对参考权重校验名称与形状），再起服务复评
python dcp_to_release_hf.py \
    --ckpt <output_dir>/CKPT/Moe8-ckpt-.../checkpoints/global_step_500 \
    --out <output_dir>/CKPT/hf_release/global_step_500 \
    --reference <model_dir>/Minimax-M3-moe8
MODEL_PATH=<output_dir>/CKPT/hf_release/global_step_500 ./vllm_server.sh
./run_aisbench_eval.sh          # 默认 DATASET_PATH 即复评集 benchmark_test_after
# 按「合格小模型」标准判定（训练集正确率 20%~40%、loss 1e-3 量级）

# ⑦ SFT 数据集转 verl 格式，进入 RL 训练
python gen_verl_dataset_from_sft.py
```

> 本目录脚本与数据集无关：换数据集时只需替换 rollout 来源与 `DATASET_PATH`。geo3k（geometry3k）
> 是贯穿本示例的数据集（图文评测、SFT 数据来源）；裁剪与转换工具对任何 MiniMax-M3 架构的权重同样
> 适用，只要 `--reference` 指向同架构的 release 布局参考权重即可。
