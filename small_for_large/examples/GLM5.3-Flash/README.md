# GLM-5.3-Flash 以小验大实践（Ascend NPU）

本目录是「以小验大」方法在 GLM-5.3-Flash 上的一次完整落地：把 320B 的全尺寸模型裁剪成单机可训的
top16 小模型，用「全尺寸评测产出 SFT 数据集 → 小模型 SFT → RL 验证」的闭环，替代直接在大模型上做 RL
调试与优化验证。通用的专家裁剪方法论（vllm-ascend patch、专家激活统计）与上游裁剪脚本见上级目录
[README_zh.md](../../README_zh.md)，本文聚焦 GLM-5.3-Flash 这条路径本身；目录下各脚本的用法见文末
[脚本使用说明](#脚本使用说明)。

## GLM-5.3-Flash 模型情况

GLM-5.3-Flash 是 GLM-5 系列首个原生多模态模型（图像/视频/文本输入），主要参数如下（来源：智谱开放平台
[模型文档](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)、
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)）：

| 项目 | 数值 | 说明 |
| --- | --- | --- |
| 总参数量 | 320B | MoE 架构 |
| 激活参数量 | 18B（约 5.6%） | 稀疏激活，推理/训练成本远低于总参数量 |
| MoE 专家 | 每层 288 个路由专家 + 1 个共享专家，top-8 路由 | 288 个冗余专家正是"以小验大"的裁剪抓手 |
| 层数 | 45 层语言层（权重另含 MTP 多 token 预测模块） | |
| 注意力 | 线性注意力 + 稀疏注意力混合架构，mHC（流形约束超连接） | 相比 GLM-5.3 注意力计算量降 3.01 倍 |
| 上下文 | 1M token（最大输出 128K） | |
| 权重大小 | BF16 约 600 GiB（NVFP4 量化版约 181 GiB） | 开源权重需要先转成 BF16 才能用于训练/推理 |

训练规模（业界公开信息）：GLM-5.3-Flash 与 GLM-5.3（744B 总参 / 40B 激活）同属 GLM-5 系列，
GLM-5.3 的预训练在昇腾 + MindSpore 的大规模集群上完成（28.5T token，千卡量级）。这个规模的模型做 RL
后训练时，虽然激活参数不大，但训练态显存（权重 + 梯度 + 优化器状态，通常是推理权重的数倍）决定了它
依然只能在多机上训练，业界普遍按数十台多卡服务器的规模申请资源。

## 模型 RL 训练面临的挑战

- **模型大、训练慢**：RL 一个 step 包含完整的 rollout 采样 + 参数更新，大模型单步耗时以小时计，一轮
  完整训练动辄数天到数周，调参反馈周期极长。
- **多机资源难申请**：训练需要按集群规模排队申请，实验排期不可控；一次失败的实验浪费的是整批机器。
- **问题定位成本高**：任何精度或性能问题（loss 异常、reward 不涨、精度偏差、算子性能）都需要在大规模
  多机任务上复现，每一次复现都是高成本操作。
- **性能优化验证慢**：性能优化（并行策略、通信、算子）改动后必须重新训练/推理验证，验证流程本身就把
  优化迭代拖得很慢。

结论：在大模型上直接做 RL 的精度验证、问题定位和性能优化，成本高到难以支撑快速迭代，因此需要
"以小验大"——保留架构和关键组件，裁剪出一个行为尽可能接近原模型、但单机可训的小模型。

## 以小验大的整体流程

```text
全尺寸 GLM-5.3-Flash (320B, 每层 288 专家, top-8)
   │
   ├─ ① 全尺寸评测生成数据集：vLLM 起服务跑 geo3k 图文评测（对齐 verl RL 训练的 prompt 与 reward 逻辑），
   │     保留"回答正确 + 带完整 think + 长度合格"的记录，转成 MindSpeed-MM SFT 数据集
   │
   ├─ ② 随机减专家生成小模型：每层 288 → 16 个专家（top16，16 > top-k=8），并做冒烟/基线评测，
   │     确认仍具备基本语言能力
   │
   ├─ ③ 小模型 SFT 训练：用 ① 的数据集训练到过拟合，保存 ckpt（全 F32），再转成 BF16
   │
   ├─ ④ 合格性判定：SFT 训练集上推理正确率落在 20%~40% 区间、loss 收敛到 1e-3 量级
   │
   └─ ⑤ RL 训练检验：SFT 数据集转 verl 格式做 RL，reward 上涨 + logdiff < 千分位 即通过；
        通过后小模型即可作为大模型精度/性能验证与优化的载体
```

### 1. 全尺寸模型评测生成数据集

用全尺寸 GLM-5.3-Flash 对 benchmark（geometry3k，test split 共 601 题）做推理评测。评测链路必须与
verl RL 训练的 prompt 模板和 math reward 计算逻辑对齐，得到**带 think（思考链）**的完整回答。

筛选规则：

- **只保留带 think 的记录**：SFT 训练教的正是"怎么想"，没有思考链的样本无法用于 SFT；
- **只保留回答正确的记录**：全尺寸模型答对，才证明该能力存在于模型中，小模型 SFT/RL 才"有得学、有得涨"；
- **长度需符合要求**：过长的 think 会在训练和后续 RL rollout 中放大成本。

筛选后的记录组装成 MindSpeed-MM 可直接使用的 SFT 数据集（verl/RL 风格 parquet：
`images` / `prompt` / `reward_model.ground_truth` / `extra_info`）。

本步骤用到：`gen_benchmark_dataset_from_verl.py` 做数据格式转换 → `vllm_server.sh`（`MODE=stable|nightly`
切换镜像）起服务 → `run_aisbench_eval.sh` 评测。

### 2. 随机减专家生成小模型

思路很简单：把每层 288 个可选专家裁到 16 个，直接得到小尺寸模型。两个注意点：

- **保留的专家数必须大于 top-k**（GLM-5.3-Flash 的 top-k=8，所以保留 16 个），路由器才"真正有得选"，
  保留 MoE 的多选一结构，而不是退化为固定 FFN；
- **小模型必须先过一次评测**：裁剪后的模型要能正常推理、能输出连贯的词句，即使语无伦次，也必须保有
  基本语言能力。连语言能力都丢掉的裁剪是无意义的。

目标是尽量减到**单机就能训练的最大专家数**，才能真正起到降成本的作用——GLM-5.3-Flash 这一次裁到了
16 个专家（即 `GLM-5.3-Flash-BF16-top16`）。裁剪由本目录的 `prune_moe_experts.py` 完成：按
`--topk-json`（示例 `example_topk_glm53.json`）给出的逐层专家 id 保留前 `--top-k` 个，其它专家连同
路由权重一起丢弃；上游的裁剪脚本 `expert_pruned.py` 与激活统计 `stat_moe.py`（可按高频专家挑选）
见上级目录 [README_zh.md](../../README_zh.md)。减专家只解决"缩到单机可训"，能力要靠后续 SFT 过拟合
补回来。

### 3. 小模型 SFT 训练

裁剪后的小模型只有基本语言能力，答不出正确结果，需要先做 SFT：

- 数据集：第 1 步产出的带 think 数据集，让小模型把全尺寸模型"想过一遍"的推理过程学一遍，从而具备一定
  推理能力；
- 训练由 MindSpeed-MM 完成（不在本目录，checkpoint 形如 `CKPT/glm53_geo3k_sft/iter_0000400`）；
- **训练到过拟合再保存权重**：过拟合保证小模型对每条训练数据都完整学过，后续多次推理时有一定概率
  答对，RL 阶段才有"答对 → reward 上涨"的起点；
- MindSpeed-MM 导出的 ckpt 为全 F32，用 `convert_ckpt_bf16.py` 转成
  与基准权重一致的 BF16 布局后，才能拿去推理评测与后续训练。

### 4. 什么样的小模型才是合格的小模型

SFT 之后、RL 之前，必须用**SFT 训练数据集本身**（也就是模型学过的那些题）对保存的 ckpt 做推理评测，
同时满足：

1. **训练 loss 收敛到 1e-3 量级**：说明已经出现一定程度的过拟合，训练数据被真正"记住"了；
2. **训练集正确率落在 20%~40%**：
   - 只有约 10%：学得不够，需要基于保存的权重**再训练一轮**；
   - 接近 70%：过拟合过头了。此时 RL 阶段 reward 很可能涨不动——模型已经"背下答案"，没有留给 RL
     通过自身努力上涨的空间，需要**回退到训练过程中的中间 ckpt**，或**从原始权重用更少的步数重新训练**，
     给 RL 预留提升空间。

一句话：**评分低了就继续训，评分高了就回退 ckpt 或减步数重训**，把正确率调到 20%~40% 的区间再进入 RL。

### 5. RL 训练检验实际效果

这是最终的检验环节：

- **RL 数据集必须由 SFT 数据集转换成 verl 格式**，保证模型在同一个数据集上做强化学习，让小模型能
  "通过自身的努力"把 reward 涨上去。曾出现过 RL 数据集用错（与 SFT 不一致）导致 reward 不涨的情况；
- **成功标准**：
  1. reward 持续上涨的曲线；
  2. logdiff 在千分位（1e-3）以下（训练与推理框架间的 logprob 一致性）。

达到标准即说明小模型的训练闭环是健康的，可以作为大模型训练的依据：大模型的精度验证、问题定位和性能
优化都可以先在小模型上进行，通过后再上大模型，成本极大降低。

## 脚本使用说明

### 目录内容

| 文件 | 作用 |
| --- | --- |
| `gen_benchmark_dataset_from_verl.py` | 把 verl/RL 风格的数据集 parquet（`images`/`prompt`/`reward_model`/`extra_info`）转成 ais_bench `Geometry3KDataset` 需要的 HF 格式（`problem`/`answer`/`images`）；`--dataset` 选数据集，默认 geo3k |
| `glm53_train_prompt.jinja` | 与 MindSpeed-MM 训练侧渲染逐 token 一致的推理 chat template；vLLM 服务必须用它，评测产出的回答格式才能与训练对齐（注意：训练序列中**不预填** `<think>`，须由模型自己生成） |
| `vllm_server.sh` | 8 卡（TP=8）拉起 GLM-5.3-Flash-BF16 推理服务，供 ais_bench 评测；`MODE=stable`（默认：eager 模式、`--max-num-seqs 16`）/ `MODE=nightly`（nightly-main 镜像：图模式、`--max-num-seqs 32`），`BASE_DIR`/`MODEL`/`MEDIA_PATH` 可覆盖 |
| `run_aisbench_eval.sh` | ais_bench 图文评测入口（默认 `geometry3k_gen`，打分对齐 verl RL 训练的 math reward），`DATASETS`/`NUM_PROMPTS` 等可用环境变量覆盖 |
| `prune_moe_experts.py` | 按逐层 top-k 专家 id 裁剪 HF MoE 权重：保留专家重编号为 `0..N-1`、`mlp.gate`/`mlp.router`（含 `e_score_correction_bias`）同序切片、`config.json` 专家数同步改写；`shared_experts`/dense `gate_proj`/`model.visual` 原样拷贝，MTP 层复用最后一层 decoder 的 top-k；支持 `--dry-run`/`--top-k`/`--topk-json`/`--stage-length`/`--skip-layers`/`--drop-layers`/`--overwrite` |
| `example_topk_glm53.json` | GLM-5.3-Flash 逐层 top-k 专家 id 示例（`model.language_model.layers.3..44.mlp.experts`，每层 64 个，按层号滑动取窗），可直接作 `prune_moe_experts.py --topk-json` 的输入；配合 `--top-k 16` 即得到 top16 |
| `convert_ckpt_bf16.py` | F32→BF16 逐 tensor 转换引擎（`--src`/`--dst`/`--dtype-ref`）：按 `--dtype-ref`（top16 参考权重）决定哪些数值敏感 tensor 保持 F32（`hc_*_base/scale`、KDA `A_log`/`dt_bias`、MoE `e_score_correction_bias`），其余转 BF16；分片命名与源目录一致，非权重文件原样拷贝，重算 index `total_size`；支持 `--dry-run`/`--overwrite` |

> 环境依赖：vLLM + vllm-ascend（稳定/nightly 两套镜像）、ais_bench、MindSpeed-MM（SFT 训练侧）；
> 专家裁剪与 bf16 转换只需要 CPU 上的 `torch` + `safetensors`。

### 目录约定

脚本不含任何用户/机器专属路径，统一走环境变量覆盖：

```text
/home/weight/                          # 共享权重目录（MODEL 默认指向此处）
├── GLM-5.3-Flash-BF16/                # 全量 BF16 权重（vLLM 服务用）
└── GLM-5.3-Flash-BF16-top16/          # prune_moe_experts.py 裁剪产物（也是 --dtype-ref 参考权重）

$BASE_DIR/                             # 工作区根目录，默认 $HOME/glm5-3-s_b
├── start/                             # 本目录脚本放置处
│   ├── CKPT/glm53_geo3k_sft/          # MindSpeed-MM SFT 导出 ckpt（iter_XXXXXXX，全 F32）
│   └── logs/vllm/                     # vLLM 服务日志
└── dataset/benchmark/geo3k_bench/    # ais_bench 评测数据（gen_benchmark_dataset_from_verl.py 产物）
```

可覆盖的通用变量：`BASE_DIR`（工作区根目录）、`MODEL`（权重路径）、`MEDIA_PATH`（图文评测数据目录）、
`HOST_IP`/`HOST_PORT`（评测服务地址）；转换脚本 `convert_ckpt_bf16.py` 直接用命令行参数
（`--src <iter目录>` + `--dst <输出目录>` + `--dtype-ref <top16 参考权重>`）指定输入。

### 专家裁剪脚本（`prune_moe_experts.py`）

`prune_moe_experts.py` 负责第 2 步「减专家缩小模型」：输入一个 HF 分片 safetensors 权重目录和逐层
top-k 专家 id JSON，输出一个专家数变少的同构模型。它只改 MoE 专家与对应路由权重，**输入目录只读**；
产物只有基本语言能力，需要接着做第 3 步的 SFT 过拟合训练，才能得到可在第 5 步进入 RL 的小模型。

```bash
# 先 --dry-run 校验层覆盖/体积，再正式裁剪（输出目录已存在时加 --overwrite）
python prune_moe_experts.py --dry-run --model-dir /home/weight/GLM-5.3-Flash-BF16 \
    --topk-json example_topk_glm53.json --top-k 16 --output-dir /home/weight/GLM-5.3-Flash-BF16-top16
python prune_moe_experts.py --model-dir /home/weight/GLM-5.3-Flash-BF16 \
    --topk-json example_topk_glm53.json --top-k 16 --output-dir /home/weight/GLM-5.3-Flash-BF16-top16
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--model-dir` / `--output-dir` | 输入/输出 HF 权重目录；输出不能是输入目录或其子目录 |
| `--topk-json` | 逐层专家 id 列表，键支持 `model.language_model.layers.<L>.mlp.experts`、`layer_<L>` 等，值可为 id 数组或 `{"topk_ids": [...]}` |
| `--top-k` | 每层保留的专家数（默认 64），必须 ≥ config 的 `num_experts_per_tok`（GLM-5.3-Flash 为 8）；从 JSON 列表中取前 `--top-k` 个 |
| `--stage-length` | 截取后的保留列表按该长度确定性交织（默认 16），避免高频专家集中在靠前的 compact id；`0` 保持 JSON 原顺序 |
| `--skip-layers` / `--drop-layers` | 逗号分隔的层号：前者保持不裁剪，后者整层删除；注意 config 的专家数仍会全局改写 |
| `--original-num-experts` | 原模型每层专家数，默认从权重自动探测 |
| `--overwrite` | 覆盖已存在的输出目录 |
| `--dry-run` | 只报告计划（保留/丢弃 tensor 数、空分片、输出体积、磁盘余量），不落盘 |

产物：裁剪后的权重分片 + 与输入同名的 index + 改写后的 `config.json`（`n_routed_experts` 等专家数
字段统一改为 `--top-k`）+ `expert_prune_summary.json`（层覆盖、保留/丢弃 tensor 数、体积）；输入
目录若有 `sizes.json`/`local.sha256` 会一并重建。超过 `num_hidden_layers` 的 MTP 层自动复用最后
一层 decoder 的裁剪列表，保证整个 checkpoint 只有一个全局专家数。

### 典型操作顺序

```bash
# ① 数据准备：把 verl/RL 风格的数据集 parquet 转成 ais_bench 需要的格式
#    本示例用 geo3k（--dataset 默认值），产物 geo3k_bench/geo3k/test.parquet
python gen_benchmark_dataset_from_verl.py --dataset geo3k --splits test
# 并将 ais_bench 的 geometry3k_gen 数据集配置中 path 指向该 parquet：
#   benchmark/ais_bench/benchmark/configs/datasets/geometry3k/geometry3k_gen.py

# ② 拉起推理服务（BASE_DIR/MODEL/MEDIA_PATH 可用环境变量覆盖）
./vllm_server.sh                    # 稳定镜像（默认 MODE=stable: eager 模式）
MODE=nightly ./vllm_server.sh       # nightly-main 镜像（图模式, glm5next + KDA）

# ③ 评测：默认 geometry3k_gen、12 条；全量 601 题用 NUM_PROMPTS=601 覆盖
./run_aisbench_eval.sh
NUM_PROMPTS=601 HOST_IP=x.x.x.x ./run_aisbench_eval.sh

# ④ 随机减专家生成小模型：每层 288 → 16（先 --dry-run 校验计划与体积，再正式裁剪）
python prune_moe_experts.py --dry-run \
    --model-dir /home/weight/GLM-5.3-Flash-BF16 \
    --topk-json example_topk_glm53.json --top-k 16 \
    --output-dir /home/weight/GLM-5.3-Flash-BF16-top16
python prune_moe_experts.py \
    --model-dir /home/weight/GLM-5.3-Flash-BF16 \
    --topk-json example_topk_glm53.json --top-k 16 \
    --output-dir /home/weight/GLM-5.3-Flash-BF16-top16

# ⑤ 对裁剪产物做冒烟/基线评测（MODEL 指向 top16），确认仍有基本语言能力后再进入 SFT
MODEL=/home/weight/GLM-5.3-Flash-BF16-top16 ./vllm_server.sh
./run_aisbench_eval.sh

# ⑥ MindSpeed-MM SFT 训练（用 ① 的数据集训练到过拟合，产出 iter_0000400 等全 F32 ckpt）

# ⑦ 训练 ckpt 转 BF16：逐 tensor 按参考权重的 dtype 决定 F32/BF16
#    支持 --dry-run（只预览不写盘）；--dst 已存在时需 --overwrite
python convert_ckpt_bf16.py \
    --src $BASE_DIR/start/CKPT/glm53_geo3k_sft/iter_0000400 \
    --dst $BASE_DIR/start/CKPT/glm53_geo3k_sft/iter_0000400_bf16 \
    --dtype-ref /home/weight/GLM-5.3-Flash-BF16-top16
#    注意：MindSpeed-MM 导出若缺 generation_config.json / LICENSE 等 aux 文件，
#    引擎只原样拷贝源目录已有文件，需先从参考权重补齐

# ⑧ 用 SFT 数据集评测 ⑦ 的产物，按「合格小模型」标准判定后，转 verl 格式进入 RL 训练
```

> 本目录脚本全部与数据集无关：换数据集时只需 `--dataset` 指定、`MEDIA_PATH`/`DATASETS` 覆盖即可。
> geo3k 是贯穿本示例的数据集（601 题图文评测、SFT 数据来源）。`convert_ckpt_bf16.py` 对任何
> GLM-5.3-Flash 的全 F32 导出同样适用（geo3k SFT、dapo17k RL 等训练得到的 ckpt 一视同仁），
> 只要 `--dtype-ref` 指向同架构的 BF16 参考权重即可。
