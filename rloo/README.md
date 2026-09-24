# Qwen3-8B RLOO FSDP2 + vLLM Ascend Recipe
---

### 1. 范围与验收链路

提供 Qwen3-8B 在昇腾 NPU 上的 RLOO训练配方：

- FSDP2 负责 Actor 和 Reference Model；
- vLLM Ascend 负责 rollout；
- GSM8K strict reward 负责结果评分；
- RLOO 使用同一 prompt 的多条 response 构造 leave-one-out baseline；
- KL penalty 直接计入 reward，不额外启用 Actor KL loss。

本次实测链路为单张 Ascend910_9382。
[`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md)。

### 2. 文件

- `run_qwen3_8b_rloo_fsdp_npu.sh`：RLOO + FSDP2 + vLLM Ascend 启动脚本；
- [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt)：已验证的官方 verl revision；
- `patch/verl_qwen3_8b_rloo_model.patch`：Qwen3-8B 模型接入配置 Patch；
- `patch/README.md`：Patch 的应用、验证和回滚方法；
- [`ACCEPTANCE_CRITERIA.md`](ACCEPTANCE_CRITERIA.md)：验收标准及当前实测结论。

### 3. 环境

当前结果使用以下软硬件栈：

| 组件 | 已验证版本或 revision |
| --- | --- |
| NPU | Ascend910_9382，64 GB HBM |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| vLLM | `0.23.0+empty` |
| vLLM Ascend | `0.23.0` |
| Transformers | `5.5.4` |
| 模型 | Qwen3-8B，BF16 |

启动训练前加载 CANN 环境；若环境同时安装 NNAL/ATB，也应加载对应环境脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
```

### 4. 安装固定 verl

从 `verl-ascend-recipe` 根目录检查安装计划：

```bash
./install_verl.sh --recipe rloo --show
./install_verl.sh --recipe rloo --method git --dest ./verl
```

也可在已检出 `REQUIRED_VERL.txt` 所列 commit 的 verl 仓库中手动应用 Patch：

```bash
git apply --check ../rloo/patch/verl_qwen3_8b_rloo_model.patch
git apply ../rloo/patch/verl_qwen3_8b_rloo_model.patch
git diff --check
```

Patch 已在上述 verl commit 上通过 `git apply --check`。它修改的是官方 RLOO 示例；
本目录的 NPU 脚本已直接包含相同配置，因此直接运行本目录脚本时不重复依赖 Patch。

### 5. 数据

使用 verl 的 GSM8K 预处理脚本生成数据：

```bash
python examples/data_preprocess/gsm8k.py \
  --local_save_dir "$HOME/data/gsm8k"
```

数据目录必须包含：

```text
$HOME/data/gsm8k/train.parquet
$HOME/data/gsm8k/test.parquet
```

### 6. 启动

在已安装 verl 的环境中执行：

```bash
bash /path/to/verl-ascend-recipe/rloo/run_qwen3_8b_rloo_fsdp_npu.sh \
  --data_path="$HOME/data/gsm8k" \
  --model_path="/models/Qwen3-8B"
```

脚本默认执行 100 步。可通过环境变量覆盖主要参数，并在末尾追加 Hydra override：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
total_training_steps=1 \
train_batch_size=32 \
rollout_n=2 \
bash rloo/run_qwen3_8b_rloo_fsdp_npu.sh \
  --data_path=/data/gsm8k \
  --model_path=/models/Qwen3-8B \
  trainer.default_local_dir=/checkpoints/rloo_smoke
```

正式训练前建议先运行一步 smoke，并确认 loss、gradient、reward 和 TPS 均为有限值。

### 7. 模型适配与算法配置

#### Qwen3 模型适配 Patch

Patch 向 verl 官方 Qwen3-8B RLOO 示例补充以下配置：

```text
data.trust_remote_code=True
+data.apply_chat_template_kwargs.enable_thinking=False
actor_rollout_ref.model.trust_remote_code=True
```

`trust_remote_code` 允许 tokenizer 和训练模型加载模型仓库提供的实现。关闭 thinking
使训练 rollout、GSM8K strict `####` 答案提取及下游评测使用相同模板口径。

#### RLOO 与 Ascend 配置

| 配置 | 实测值 | 目的 |
| --- | ---: | --- |
| `algorithm.adv_estimator` | `rloo` | 启用 leave-one-out advantage |
| `rollout_n` | 5 | 为每个 prompt 采样多条 response |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | 控制 reference KL reward penalty |
| `actor.strategy` / `ref.strategy` | `fsdp2` | 使用 FSDP2 训练和 reference 计算 |
| `model_dtype` | `bfloat16` | 匹配 Ascend BF16 路径 |
| `use_dynamic_bsz` | `True` | 按 token 数平衡动态 batch |
| `param_offload` / `optimizer_offload` | `True` | 满足单卡 8B 训练显存约束 |
| `rollout.name` | `vllm` | 使用 vLLM Ascend rollout |
| `free_cache_engine` | `True` | 训推切换时释放 KV cache |

当前交付中的“算法优化实现”是 RLOO 参数组合和 Ascend 执行配置，不包含对 verl RLOO
核心数学实现的改写。验收方若要求算法源码变更，应另行补充核心代码提交。

### 8. 实测结果

训练端 global TPS 沿用 verl 日志口径：

```text
global TPS = perf/total_num_tokens / perf/time_per_step
```

#### 20 步训练

| 指标 | 结果 |
| --- | ---: |
| 完成步数 | 20/20，checkpoint `global_step_20` |
| Reward 首步 / 末步 | 0.593750 / 0.656004 |
| Reward 前 5 步 / 后 5 步均值 | 0.571879 / 0.615449 |
| Reward 窗口提升 | +0.043570 |
| Reward 线性斜率 | +0.003911/步 |
| 平均 global TPS | 153.67 token/s |
| 平均 step time | 145.52 s |

Reward 呈上升趋势，训练 TPS 高于无匹配 A100 标杆时约定的 100 token/s 门槛。

#### GSM8K 下游评测

两组权重使用同一份 1,319 条 GSM8K test、greedy decoding、`max_tokens=512`、关闭
thinking，并采用 verl strict `####` 答案提取：

| 权重 | 正确数 | 准确率 | 输出 token | 输出 TPS |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-8B 基座 | 703/1319 | 53.30% | 343,572 | 389.35 |
| RLOO step 20 | 824/1319 | 62.47% | 341,096 | 381.31 |

RLOO 权重相对基座提升 121 道题，即 `+9.17` 个百分点。推理 TPS 与训练 global TPS
是不同口径，不能直接横向比较；前者仅用于验证训练权重的部署吞吐。

