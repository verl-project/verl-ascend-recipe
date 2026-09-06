# Megatron + vLLM Ascend On-Policy Distillation Recipes

This single practice document contains both the Chinese delivery guide and its
English counterpart. The recipes reproduce the two OPD chains assigned in
[community task #21](https://github.com/verl-project/verl-ascend-recipe/issues/21).

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the reviewed official release
branch and installation commands. The recipe uses `release/v0.8.0` plus the
bundled Ascend compatibility patch; it does not install from a personal fork.

---

## 中文实践文档

### 1. 范围与验收链路

本目录提供昇腾上的全参数 On-Policy Distillation（OPD）训练配方。student 使用
Megatron 后端训练，student rollout 与 teacher 使用 vLLM Ascend 推理。

正式交付包含两条四卡链路：

| 数据 | Student | Teacher | OPD 配置 | 四卡拓扑 |
| --- | --- | --- | --- | --- |
| GSM8K | Qwen2.5-0.5B | Qwen2.5-3B-Instruct | `forward_kl_topk`，top-k 64 | student TP2 + teacher TP2 |
| Geo3K | Qwen3-VL-2B-Instruct | Qwen3-VL-4B-Instruct | `k1`，top-k 64，policy-gradient weighting | student TP1/DP2 + teacher TP1/DP2 |

Qwen3-VL 的模型对和主要算法参数参考公开的
[`run_qwen3_vl_geo3k.sh`](https://pages.doit.wisc.edu/DMAQBOOL/verl/-/blob/main/examples/on_policy_distillation_trainer/run_qwen3_vl_geo3k.sh)
（2026-08-24 核对）。正式参数包括 global batch 128、prompt 1024、response 2048、
rollout 1、seed 42、无 task reward、训练前验证和每五步验证。硬件拓扑从 4+4 GPU
降级为四张 NPU，dynamic batch、micro batch、显存比例和 optimizer offload 等执行
参数采用实测的 NPU 配置；这些属于硬件适配，不宣称与公开 GPU 脚本逐项相同。

### 2. 文件

- `run_qwen2_5_0_5b_megatron_npu.sh`：Qwen2.5 / GSM8K 配方；
- `run_qwen3_vl_2b_megatron_npu.sh`：官方模型对齐的 Qwen3-VL / Geo3K 配方；
- [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt)：可复现 VeRL revision；
- `patch/verl_opd_ascend.patch`：VeRL 核心兼容改动的独立 patch。

核心改动已经提交到 [verl PR #7333](https://github.com/verl-project/verl/pull/7333)。
本配方以官方 `release/v0.8.0` 为基础，并在 editable install 前应用本目录携带的兼容
patch，因此不依赖个人 fork。相关修复进入正式 release 后即可删除 patch。

### 3. 环境

四卡验收使用 Atlas 800T A2 / Ascend 910B3 和以下固定软件栈：

| 组件 | 已验证版本或 revision |
| --- | --- |
| Python | 3.11.15 |
| PyTorch / torch-npu | 2.9.0 / 2.9.0 |
| vLLM | 0.18.0 (`bcf2be96120005e9aea171927f85055a6a5c0cf6`) |
| vLLM Ascend | 0.18.0 (`e18643f8a4d5bd9990727654318ad069ea0b56e2`) |
| triton-ascend | 3.2.1 |
| Ray | 2.56.1 |
| transformers | `cc7ab9be508ce6ed3637bba9e50367b29b742dc6` |
| Megatron Core | `core_r0.16.0` (`ddc0d6774783b032ddceacc5714e653651daecb9`) |
| MindSpeed | `core_r0.16.0` (`0bda3e134e1d8185b229d201b030757cdcb3ac36`) |
| mbridge | `a61943d7fcb34a190471cfeb0a0eb8bbda621ddf` |

启动 Ray 或 vLLM Ascend 前，同时加载 CANN 和 NNAL/ATB：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Qwen3-VL 需要包含 `qwen3_vl` bridge 的 mbridge revision：

```bash
pip install --no-deps \
  'git+https://github.com/ISEEKYAN/mbridge.git@a61943d7fcb34a190471cfeb0a0eb8bbda621ddf'

python - <<'PY'
from mbridge import AutoBridge

assert "qwen3_vl" in AutoBridge.list_supported_models()
PY
```

### 4. 安装固定 VeRL

从 `verl-ascend-recipe` 根目录运行：

```bash
./install_verl.sh --recipe opd --show
./install_verl.sh --recipe opd --method git --dest ./verl
cd verl
```

`--method git` 会检出官方 `release/v0.8.0`，应用兼容 patch，再执行 editable
install。也可在同一 release 源码树中独立审计或手动应用 patch：

```bash
git apply --check --unidiff-zero ../opd/patch/verl_opd_ascend.patch
git apply --unidiff-zero ../opd/patch/verl_opd_ascend.patch
```

### 5. 数据

在 VeRL 根目录准备 GSM8K 和 Geo3K：

```bash
python examples/data_preprocess/gsm8k.py \
  --local_save_dir "$HOME/data/gsm8k"

python examples/data_preprocess/geo3k.py \
  --local_save_dir "$HOME/data/geo3k"
```

Geo3K parquet 必须包含 `images` 列。若图片以路径而不是字节保存，所有 Ray worker
必须能访问相同路径。配方默认读取：

```text
$HOME/data/gsm8k/train.parquet
$HOME/data/gsm8k/test.parquet
$HOME/data/geo3k/train.parquet
$HOME/data/geo3k/test.parquet
```

### 6. 启动

两条配方默认各使用四张 NPU：两张 student/rollout 和两张 teacher。

```bash
# Qwen2.5-0.5B <- Qwen2.5-3B-Instruct
bash ../opd/run_qwen2_5_0_5b_megatron_npu.sh

# Qwen3-VL-2B-Instruct <- Qwen3-VL-4B-Instruct
bash ../opd/run_qwen3_vl_2b_megatron_npu.sh
```

模型、数据、batch、序列长度、并行度、显存比例、训练步数和保存频率均可通过同名
大写环境变量覆盖；额外 Hydra override 可以直接追加在命令末尾。例如：

```bash
STUDENT_MODEL=/models/Qwen3-VL-2B-Instruct \
TEACHER_MODEL=/models/Qwen3-VL-4B-Instruct \
TRAIN_FILE=/data/geo3k/train.parquet \
VAL_FILE=/data/geo3k/test.parquet \
TOTAL_TRAINING_STEPS=1 \
bash ../opd/run_qwen3_vl_2b_megatron_npu.sh trainer.save_freq=-1
```

### 7. 为什么需要兼容 patch

#### Qwen3-VL ragged MRoPE

Qwen3-VL 的 `position_ids` 为 `[batch, 4, sequence]`，可变长度维是最后的 sequence。
TensorDict consolidate/序列化后，NestedTensor 可能恢复为错误 ragged 维。只修改
`_ragged_idx` 不会同步内部 `values/offsets`，后续 unbind 或 mini-batch 切分会样本
错位或 indexing error。patch 从每个 `[4, seq_len]` 样本重建 NestedTensor，使
metadata 与底层存储一致；布局已正确时保持 no-op。

#### reverse-KL 有限值

K1 estimator 使用：

```text
k1 = student_log_prob - teacher_log_prob
```

后端在极低概率或精度边界返回 `-inf` 时，`-inf - (-inf)` 为 NaN，单侧 `-inf`
产生无穷值。patch 只在用户显式设置 `log_prob_min_clamp` 时，于 estimator 前对
student 和 teacher 应用相同下界。默认 `null` 的行为不变；本配方使用 `-10.0`。

### 8. 四卡100步结果

global TPS 统一按以下端到端口径统计：

```text
global TPS = perf/total_num_tokens / perf/time_per_step
```

| 链路 | 训练 | reward 前/后窗口 | loss 前/后窗口 | 稳态 global TPS | 下游 step0 / step100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5 / GSM8K | 100/100，exit 0 | 0.00833 / 0.16667 | 0.54703 / 0.27079 | 223.41 | 0.607% / 28.279% |
| Qwen3-VL / Geo3K | 100/100，exit 0 | 0.225000 / 0.248906 | 0.268061 / 0.207617 | 1368.02 | composite 0.251581 / 0.271048 |

两条链路 reward 均上升且 global TPS 均大于无匹配 GPU 标杆时约定的 100 token/s
门槛。Qwen3-VL 的下游值是 trainer-inline Geo3K composite，不是 raw accuracy；
step0 到 step100 提升 `+1.9468` 个百分点。

<div align="center">
  <img src="https://github.com/chengminhua/verl_data/blob/main/opd_megatron_vllm_ascend_reproduction.png" alt="Qwen2.5 and Qwen3-VL four-NPU OPD reproduction curves">
</div>

### 9. 完整训练日志

未截断原始日志发布在同一个
[GitHub Gist](https://gist.github.com/egangu/4469135b55c9d0a73a4ad6d076ff2b34)：

| 日志 | 内容 |
| --- | --- |
| `opd-qwen2.5-4npu-tp2-response1024-batch24-100step.log` | Qwen2.5 step 1--100 |
| `opd-qwen3-vl-official-aligned-2b4b-batch128-response2048-20step-v6.orchestrator.log` | Qwen3-VL step 1--20 |
| `opd-qwen3-vl-official-aligned-2b4b-batch128-response2048-100step-v7-resume20.orchestrator.log` | Qwen3-VL step 21--100 |

Qwen3-VL 是一次 checkpoint-resume 链路，因此分为 1--20 和 21--100 两个文件。
两条训练均为 launcher/trainer exit 0，核心指标有限，训练期无 OOM 或 worker restart，
最终 Megatron checkpoint 通过模型权重、optimizer 和 extra-state 校验。

### 10. 长跑前门禁与限制

长跑前至少执行：

- 一步 smoke，确认 loss、gradient 和 reward 有限；
- 确认 student 与 teacher tokenizer 对同一 token ID 使用相同词表映射；
- 检查 Qwen3-VL 样本已加载图片且 visual tokens 存在；
- 检查每个 optimizer step 只推进一次 global step；
- 验证 checkpoint 保存和恢复；
- 在固定短 batch 上验证任何性能修改不改变输出。

已知限制：

- Qwen3-VL mbridge 仍为 experimental，必须固定 revision；
- 当前正式结果覆盖 TP/DP，不覆盖 PP/EP；
- 四卡硬件降级与公开八卡配置的吞吐不能直接横向比较；
- vLLM Ascend、CANN 和 triton-ascend 必须采用兼容组合；
- release 分支尚未包含的两处兼容修复由本目录 patch 提供；正式 release 纳入修复后
  应删除该 patch。

---

## English Guide

### Scope

These recipes provide full-parameter OPD on four Ascend NPUs with Megatron
training and vLLM Ascend inference:

- Qwen2.5-0.5B student from Qwen2.5-3B-Instruct on GSM8K, using
  `forward_kl_topk` and top-k 64;
- Qwen3-VL-2B-Instruct student from Qwen3-VL-4B-Instruct on Geo3K, using
  K1, top-k 64, policy-gradient weighting, rollout one, and no task reward.

The Qwen3-VL model pair and main algorithm controls follow the public
[`run_qwen3_vl_geo3k.sh`](https://pages.doit.wisc.edu/DMAQBOOL/verl/-/blob/main/examples/on_policy_distillation_trainer/run_qwen3_vl_geo3k.sh),
checked on 2026-08-24. The 4+4 GPU allocation is reduced to two student and
two teacher NPUs. Dynamic batching, micro-batches, memory utilization, and
optimizer offload use the measured NPU settings and are not claimed to match
the public GPU script line by line.

### Install and run

Install the reviewed official VeRL release and bundled compatibility patch from
the recipe repository root. Use the git method; a wheel-only install cannot
apply the source patch:

```bash
./install_verl.sh --recipe opd --method git --dest ./verl
cd verl
git diff --check
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

bash ../opd/run_qwen2_5_0_5b_megatron_npu.sh
bash ../opd/run_qwen3_vl_2b_megatron_npu.sh
```

Model paths, data paths, topology, batch sizes, sequence limits, memory
utilization, step count, and checkpoint cadence can be overridden through
uppercase environment variables. Additional Hydra overrides can be appended
to either command.

### Compatibility changes

The compatibility patch rebuilds incorrectly deserialized Qwen3-VL ragged
MRoPE position IDs instead of changing private metadata alone. It also applies
the existing configurable log-probability floor symmetrically before the K1
estimator, preventing `-inf - (-inf)` from contaminating the loss and gradient.
Both changes are in [verl PR #7333](https://github.com/verl-project/verl/pull/7333).

### Results and evidence

Both four-NPU runs completed 100 optimizer steps with exit code zero, finite
core metrics, valid final checkpoints, increasing reward, and global throughput
above 100 token/s. Qwen2.5 achieved 223.41 global token/s and improved GSM8K
from 0.607% to 28.279%. Qwen3-VL achieved 1368.02 global token/s and improved
the trainer-inline Geo3K composite from 0.251581 to 0.271048 (+1.9468 pp).

The complete unabridged logs are listed in the Chinese section above and hosted
in the linked public Gist. Do not interpret the
trainer-inline Geo3K composite as raw accuracy, and do not extrapolate the
four-NPU result to a different model or topology.
