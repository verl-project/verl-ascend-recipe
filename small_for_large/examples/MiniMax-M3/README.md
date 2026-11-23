# MiniMax-M3 以小验大实践（Ascend NPU）

本目录是「以小验大」方法在 MiniMax-M3 上的一次完整落地：把 ~428B 的原生多模态 MoE 大模型裁剪成
单机可训的小模型，用「全尺寸评测产出 SFT 数据集 → 小模型 SFT → RL 验证」的闭环，替代直接在大模型上做
RL 调试与优化验证。与 [GLM-5.3-Flash 示例](../GLM5.3-Flash/README.md) 互为补充：那条路径走
MindSpeed-MM 训练栈 + F32→BF16 权重转换，本路径覆盖 **VeOmni 训练栈 + VeOmni checkpoint → HF
release 布局转换**。通用的专家裁剪方法论（vllm-ascend patch、专家激活统计 `stat_moe.py`、裁剪脚本
`expert_pruned.py`）见上级目录 [README_zh.md](../../README_zh.md)。本文后文依次介绍方法流程、SFT 部署、
推理部署和脚本使用说明。

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
| 权重大小 | BF16 约 796 GiB | 单机推理需 16 卡 × 64 GiB（TP=16）；生产建议 2×A3 |

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

流程第 3 步的训练环境见「SFT 部署」，流程第 1、4 步的推理环境见「推理部署」，脚本用法见「脚本使用说明」。

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

筛选后的记录组装成 VeOmni 可加载的 ShareGPT 数据集（`images/` + `geo3k_sft.json`，按
`--target-rows` 重复到目标条数）。

### 2. 随机减专家生成小模型

思路与 GLM-5.3-Flash 路径相同：把每层 128 个路由专家裁到少量几个，直接得到小尺寸模型。区别在
checkpoint 布局——GLM 是 `mlp.experts.<E>.{gate,up,down}_proj`，MiniMax 是
`block_sparse_moe.experts.<E>.w{1,2,3}`，所以本目录自带 `prune_moe_experts.py`（与 GLM 示例同名，
实现按 MiniMax 布局改写，参照上级目录的 `expert_pruned.py`）。两个注意点：

- **建议保留的专家数不少于原始 `num_experts_per_tok`**，路由器才"真正有得选"；脚本会把输出 config 的
  `num_experts_per_tok` 钳到 ≤ 保留数，避免路由退化成 no-op；
- **小模型必须先过一次基线评测**：裁剪后的模型要能正常推理、输出连贯的词句（`run_aisbench_eval.sh` 少量
  prompt 即可验证），连语言能力都丢掉的裁剪是无意义的。

目标是尽量减到**单机就能训练的最大专家数**——MiniMax-M3 这一次裁到了每层 8 个专家（即
`Minimax-M3-moe8`，约 74 GiB，8 卡可训）。选专家的方式有三种：默认按序号取前 N、
`--rank-by score_bias`（按路由偏置近似高频专家）、`--topk-json`（逐层指定，可与上级目录
`stat_moe.py` 的激活统计配合）。

### 3. 小模型 SFT 训练

裁剪后的小模型只有基本语言能力，答不出正确结果，需要先做 SFT：

- 数据集：流程第 1 步产出的带 think 数据集，让小模型把全尺寸模型"想过一遍"的推理过程学一遍；
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

## SFT 部署（VeOmni，NPU）

本章对应流程第 3 步（小模型 SFT）。训练栈为 **VeOmni**，模型实现为 `minimax_m3_vl`；环境安装参考
[VeOmni Ascend ARM 安装](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/get_started/installation/install_ascend_arm.md)
或 [x86 安装](https://github.com/ByteDance-Seed/VeOmni/blob/main/docs/get_started/installation/install_ascend_x86.md)。

> 所用 VeOmni 需包含 `minimax_m3_vl`（`veomni/models/transformers/minimax_m3_vl/`，
> `dcp_to_release_hf.py` 依赖其中的 checkpoint converter）；若当前版本没有该目录，先切换到配套分支。

### 1. 版本与环境安装

| 组件 | main 分支推荐版本 |
| --- | --- |
| Python | 3.11 / 3.12 |
| torch | 2.10.0 |
| torch_npu | 2.10.0 |
| CANN | 9.0.0（CI）；仓库镜像也覆盖 8.3.RC2 |

安装前先加载 CANN 环境（`CANN_path` 默认 `/usr/local/Ascend`）：

```bash
export CANN_path=/usr/local/Ascend
source $CANN_path/ascend-toolkit/set_env.sh
```

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
```

方式一：uv（推荐；ARM / x86 二选一）

```bash
uv sync --frozen --extra npu_aarch64    # ARM
uv sync --locked --extra npu            # x86
source .venv/bin/activate
```

方式二：pip（ARM / x86 二选一）

```bash
pip install -e .[npu_aarch64]           # ARM
pip install -e .[npu]                   # x86
pip install transformers==5.16.1 datasets==2.21.0
```

图文 SFT 不需要视频解码；如后续训练视频/音频，再按 VeOmni 文档安装 torchcodec 与系统 ffmpeg。

### 2. 权重与数据

- **权重**：SFT 起点是裁剪后的 `Minimax-M3-moe8`，通过 `--model.model_path` 指定。
- **数据**：`gen_sft_dataset.py` 产出的 ShareGPT 数据集（`geo3k_sft.json` + `images/`），
  通过 `--data.train_path` 传给 VeOmni。脚本默认输出到 `dataset/geo3k_sft/geo3k_sft.json`；
  典型操作顺序用 `--out-dir/--out-name` 将其写到目录约定的
  `dataset_vllm_rollout/geometry3k_vllm_rollout_sft.json`。
- **训练配置**：使用 VeOmni 仓库中 `minimax_m3_vl` 的示例 YAML（路径以所用版本为准），
  本示例主要覆盖其中的数据与权重路径。

### 3. 关键配置

- **chunk loss**：在训练 YAML 中设
  `model.ops_implementation.cross_entropy_loss_implementation: chunk_loss`，降低长序列 loss 的显存尖刺。

### 4. 启动训练

VeOmni 用 `train.sh`（内部 `torchrun`）启动，按 `ASCEND_RT_VISIBLE_DEVICES` 自动设置每节点进程数
（训练入口与配置路径以所用 VeOmni 版本为准）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export CPU_AFFINITY_CONF=1

bash train.sh tasks/train_vlm.py <minimax_m3_vl_config.yaml> \
    --model.model_path <model_dir>/Minimax-M3-moe8 \
    --data.train_path <output_dir>/dataset_vllm_rollout/geometry3k_vllm_rollout_sft.json \
    --train.micro_batch_size 1
```

多机时设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`，每个节点执行同一命令。
训练到过拟合后保存的 ckpt 形如 `global_step_N`（DCP 或 `hf_ckpt/`），用 `dcp_to_release_hf.py`
转成 vLLM 可加载的 HF release 布局后再做流程第 4 步复评。

## 推理部署（vLLM-Ascend，NPU）

本章对应流程第 1、4 步的评测推理服务（流程第 5 步的 RL rollout 由 verl 内部调度 vLLM）。MiniMax-M3 在
vLLM-Ascend main 分支支持，官方细节见
[vLLM-Ascend MiniMax-M3 教程](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/MiniMax-M3.md)。

### 1. 权重与硬件

| 权重 | 推荐硬件 | 说明 |
| --- | --- | --- |
| `MiniMax-M3`（BF16，本示例） | 2×A3（64 GiB×16/节点）；单机 16 卡可起但不推荐 | 约 796 GiB；`vllm_server.sh` 全尺寸用 `TP_SIZE=16` |
| `MiniMax-M3-w8a8` | 1×A3（64 GiB×16）或 1×A2（64 GiB×8） | 需 `--quantization ascend` |
| `MiniMax-M3-MXFP8` | 950DT（96 GiB×8） | 需 `--quantization mxfp8`，可配 `--kv-cache-dtype fp8` |
| `MiniMax-M3-EAGLE3-GQA` | 配合上述权重 | 可选 EAGLE3 投机解码草稿模型 |

裁剪/SFT 产物（`Minimax-M3-moe8`、`hf_release/global_step_N`）约 74 GiB，8 卡 TP=8 即可。

### 2. 镜像与环境

用官方 vLLM-Ascend 镜像（版本以官方文档为准），或从源码安装 vLLM + vLLM-Ascend main 分支：

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm && pip install -v -e . && cd ..
git clone https://github.com/vllm-project/vllm-ascend.git && cd vllm-ascend
git checkout main && pip install -v -e . && cd ..
pip show vllm vllm-ascend
```

Docker 场景按 A2/A3/950DT 的官方 `docker run` 挂载驱动与 `davinci*` 设备（设备列表因硬件而异）。

### 3. 单机起服务

```bash
# 全尺寸 BF16：16 卡 TP=16
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 TP_SIZE=16 \
MODEL_PATH=<output_dir>/MiniMax-M3-serve ./vllm_server.sh
# 裁剪/SFT 产物：默认 8 卡 TP=8
MODEL_PATH=<output_dir>/CKPT/hf_release/global_step_1000 ./vllm_server.sh
```

全尺寸 serve 目录需用软链并修正 `config.json` 的 `num_hidden_layers`（12→60），见「目录约定」。

脚本核心命令（`MODEL_PATH`/`TP_SIZE`/`MEDIA_PATH` 可覆盖）：

```bash
vllm serve "$MODEL_PATH" --host 0.0.0.0 --port 12345 \
  --served-model-name auto --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1 --enable-expert-parallel \
  --max-model-len 8192 --max-num-seqs 16 \
  --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
  --reasoning-parser minimax_m3 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --allowed-local-media-path "$MEDIA_PATH" \
  --gpu-memory-utilization 0.90 --safetensors-load-strategy prefetch \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

NPU 环境变量：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True   # 脚本已设置
export HCCL_OP_EXPANSION_MODE=AIV                         # 官方示例建议
```

关键点：

- `--quantization`：只在量化权重时加——w8a8 用 `ascend`，MXFP8 用 `mxfp8`；BF16 权重**不要加**。
- `--reasoning-parser minimax_m3`：解析 `⏸ ... </think>` 思考段，评测/接口按需读取 `content`。
- `--enable-expert-parallel`：MoE 模型必须开启。
- `--limit-mm-per-prompt` / `--allowed-local-media-path`：图文评测必需，后者指向图片目录。
- `--max-model-len 8192`：本示例评测配置；长上下文/生产配置按显存与吞吐调整。
- 官方生产优化项（`--additional-config` 的 flashcomm1/multistream/weight_nz_mode、EAGLE3
  `--speculative-config`、DP>1）本示例默认未启用，按需参考官方教程。

### 4. 服务验证

```bash
curl -s http://127.0.0.1:12345/v1/models
curl -s http://127.0.0.1:12345/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":32}'
```

服务正常后即可运行流程第 1、4 步的 ais_bench 评测。多机、PD 分离、KV Cache Pool 等大规模部署
以官方教程为准。

## 脚本使用说明

> 「典型操作顺序」用 ①-⑦ 编号，与前文方法流程的第 1-5 步是两套编号。

### 目录内容

| 文件 | 作用 |
| --- | --- |
| `vllm_server.sh` | 评测用 vLLM 服务：全尺寸（`TP_SIZE=16`，16 卡）或裁剪/SFT 产物（默认 TP=8）；`MODEL_PATH`/`MEDIA_PATH`/`TP_SIZE`/`ASCEND_RT_VISIBLE_DEVICES` 可覆盖 |
| `run_aisbench_eval.sh` | ais_bench 图文评测入口（与 GLM 示例同名）：通过 `--dataset-path` 指定评测集（GLM 示例写在数据集配置文件中）；`DATASET_PATH`/`NUM_PROMPTS` 等可用环境变量覆盖 |
| `gen_sft_dataset.py` | rollout JSONL → VeOmni 可加载的 ShareGPT SFT 数据集：mathruler 打分、保留正确 + 完整 think、图片本地化、按 `--target-rows` 重复 |
| `prune_moe_experts.py` | 裁剪路由专家：保留专家重编号、`gate`/`e_score_correction_bias` 切片、config 同步改写（含 `num_hidden_layers` 12→60 修复）。参数见后文「专家裁剪脚本」小节 |
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

### 专家裁剪脚本（`prune_moe_experts.py`）

`prune_moe_experts.py` 负责流程第 2 步「减专家缩小模型」：把每层 128 个路由专家裁到 `--top-k` 个，
保留专家重编号为 `0..top-k-1`，`gate`/`e_score_correction_bias` 同序切片，并同步改写 `config.json`
（`num_local_experts`、`num_experts_per_tok` 钳制，修复公开 config 的 `num_hidden_layers` 12→60）。
完整命令见「典型操作顺序」④。

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--model-dir` / `--output-dir` | 输入/输出 HF 权重目录；`--output-dir` 在非 `--dry-run` 时必填 |
| `--top-k` | 每层保留的专家数（必填）；脚本会把输出 config 的 `num_experts_per_tok` 钳到 ≤ 该值 |
| `--topk-json` | 逐层指定专家 id（GLM 兼容格式），优先于 `--rank-by` |
| `--rank-by` | `index`（默认，取 0..N-1）/ `score_bias`（按路由偏置近似高频专家） |
| `--experts-per-tok` | 覆盖输出 config 的 `num_experts_per_tok`；默认 `min(原值, --top-k)` |
| `--stage-length` | 保留 id 按该长度交织（默认 0 关闭），避免高频专家集中在靠前 id |
| `--shard-size-gb` | 输出分片大小（默认 4.5） |
| `--dry-run` / `--overwrite` | 只报告计划与体积 / 覆盖已有输出目录 |

产物：保留专家重编号后的权重分片 + 改写后的 `config.json`。建议先用 `--dry-run` 确认每层专家数、
输出体积和磁盘余量。

### 典型操作顺序

以下命令在工作区根目录 `<output_dir>` 下执行。下面 ①-⑦ 是操作步骤，按执行先后排列。

```bash
# ① 全尺寸模型起服务（16 卡 TP=16；MODEL_PATH 指向 serve 目录）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 TP_SIZE=16 \
MODEL_PATH=<output_dir>/MiniMax-M3-serve ./vllm_server.sh

# ② 评测产出 rollout JSONL（--dataset-path 指向全量 geo3k 基准）
DATASET_PATH=<geo3k_dataset_dir>/geo3k_bench_519/geo3k/test.parquet ./run_aisbench_eval.sh

# ③ rollout → SFT 数据集（输出到目录约定的 dataset_vllm_rollout/；先 --dry-run 看筛选统计）
python gen_sft_dataset.py --out-dir dataset_vllm_rollout --out-name geometry3k_vllm_rollout_sft.json --dry-run
python gen_sft_dataset.py --out-dir dataset_vllm_rollout --out-name geometry3k_vllm_rollout_sft.json --max-tokens 4096

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
