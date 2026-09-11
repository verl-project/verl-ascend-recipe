# Qwen3-8B 的 FSDP2 LoRA merge 训练

本示例对应 [issue #78](https://github.com/verl-project/verl-ascend-recipe/issues/78)。
每次向推理模型同步权重前，先将 LoRA 权重合并到基座模型。

- 模型与算法：Qwen3-8B，GRPO + LoRA。
- 训练后端：FSDP2。
- 推理后端：vLLM-Ascend。

固定配置已在单台四张 Ascend 910B3 上完成连续100步训练。

| 指标 | 结果 |
| --- | --- |
| 首末十步平均奖励 | 0.39160 → 0.83555 |
| GSM8K 验证准确率 | 24.72% → 80.67% |
| 每卡训练吞吐量 | 655.51 tokens/s |

第80步验证准确率为89.08%，后期下降原因尚未确定。详见[适配与验证报告](ALGORITHM_TUNING_REPORT.md)。

## 离线复算

在仓库根目录执行以下命令；检查程序只依赖 Python 标准库，不需要 NPU。

```bash
python3 lora_rl_merge/tools/check_validation.py \
  lora_rl_merge/evidence/910b3-100step/metrics.log --devices 4
python3 -m unittest discover -s lora_rl_merge/tools -v
```

[证据目录](evidence/910b3-100step/README.md)保存指标日志、版本、文件哈希及检查点记录。
检查项目：

- 训练步连续，指标有限，梯度非零。
- 末十步平均奖励高于首十步，每卡吞吐量超过100 tokens/s。
- 初始和最终验证结果完整。

## 固定环境

| 组件 | 实测版本或范围 |
| --- | --- |
| 硬件 | 单台，4 × Ascend 910B3 |
| verl | `bc72e38edba78e778bfbd462638f9634b9140a76`，0.9.0.dev0 |
| vLLM | `bcf2be96120005e9aea171927f85055a6a5c0cf6`，0.18.0 |
| vLLM-Ascend | `a43c8cc8057f490ed1df2c6ed66253e2d7817da4`，0.18.1.dev44 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |
| CANN / 驱动 | 9.0.0 / 26.0.rc1 |
| transformers / peft / ray | 5.10.4 / 0.20.0 / 2.56.1 |

版本声明见 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。通用 `install_verl.sh` 只安装verl，
不会配置CANN、vLLM-Ascend或应用本示例的补丁，不能代替完整环境准备。

### 核对软件版本

复现要求是软件版本、源码提交和补丁一致，不要求镜像名称、image ID 或 RootFS 一致。
可使用已有容器或自行安装的环境。宿主机驱动须兼容 CANN；26.0.rc1 是本次实测版本。

在实际训练环境中核对以下内容：

- CANN：核对安装目录中的版本文件，要求9.0.0。
- Python 包：用下方命令查看版本，与上表比较。
- 源码：核对 verl、vLLM 和 vLLM-Ascend 的提交号。
- 补丁：应用本示例的两份补丁，或用反向检查确认已应用。

```bash
python3 -m pip show torch torch-npu verl vllm vllm-ascend transformers peft ray
npu-smi info
```

`npu-smi info` 用于核对驱动和设备，不代替 CANN 版本检查。
以下命令直接在训练环境执行。先设置实际目录；三个源码目录应对应当前 Python 环境安装的代码。

```bash
RECIPE_DIR=$(realpath lora_rl_merge)
WORK_DIR=/data/verl-ascend/issue78
VERL_DIR=/data/verl-ascend/src/verl
VLLM_DIR=/data/verl-ascend/src/vllm
VLLM_ASCEND_DIR=/data/verl-ascend/src/vllm-ascend
mkdir -p "$WORK_DIR"
```

### 模型与数据

将 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) 权重和分词器放到
`$WORK_DIR/models/Qwen3-8B/`，再用固定 verl 版本生成 GSM8K 数据：

```bash
python3 "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$WORK_DIR/data/gsm8k"
(cd "$WORK_DIR" && sha256sum -c "$RECIPE_DIR/evidence/910b3-100step/inputs.sha256")
```

校验覆盖模型权重、配置、分词器和两个 Parquet 数据文件。若哈希不同，需核对输入内容。
Parquet 文件也可能仅因序列化版本不同而产生字节差异。

### 应用两个依赖补丁

以下命令用于尚未打补丁的源码。若补丁已应用，改用 `git apply --reverse --check` 确认，不重复应用。

```bash
(
  set -e
  test "$(git -C "$VERL_DIR" rev-parse HEAD)" = bc72e38edba78e778bfbd462638f9634b9140a76
  test "$(git -C "$VLLM_DIR" rev-parse HEAD)" = bcf2be96120005e9aea171927f85055a6a5c0cf6
  test "$(git -C "$VLLM_ASCEND_DIR" rev-parse HEAD)" = a43c8cc8057f490ed1df2c6ed66253e2d7817da4
  git -C "$VERL_DIR" apply --check "$RECIPE_DIR/patches/0001-get_npu_versions-first-visible-npu-id.patch"
  git -C "$VERL_DIR" apply "$RECIPE_DIR/patches/0001-get_npu_versions-first-visible-npu-id.patch"
  git -C "$VLLM_ASCEND_DIR" apply --check "$RECIPE_DIR/patches/vllm-ascend/0001-record-sampler-stream.patch"
  git -C "$VLLM_ASCEND_DIR" apply "$RECIPE_DIR/patches/vllm-ascend/0001-record-sampler-stream.patch"
)
```

源码修改必须在当前 Python 环境中生效。可编辑安装会直接使用源码；若使用预构建安装包，需按该项目的安装流程重新构建或安装补丁版本。

两个补丁的作用：

- **设备探测**：查询首个可见物理卡，修复硬编码卡1导致的失败。原函数不读取 `ASCEND_VISIBLE_DEVICES`，设置该变量不能替代补丁。
- **采样器**：调用 `q.record_stream(...)`，回移上游 [PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394) 的张量生命周期修复。

## 运行完整100步

以下命令使用实测的训练脚本和超参数。示例假定物理卡4–7映射为当前环境的逻辑卡0–3。
`PHYS_CARDS` 记录物理编号，实际选择由 `ASCEND_RT_VISIBLE_DEVICES` 控制。
直接访问宿主机全部设备时，应按实际逻辑编号设置该变量，例如4,5,6,7。只使用获准的四张卡。

- 首次运行：使用新的检查点目录。
- 中断恢复：重跑同一命令，由 `RESUME_MODE=auto` 加载最近的检查点。

```bash
(
  set -e
  cd "$VERL_DIR"
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 PHYS_CARDS=4,5,6,7 TOTAL_TRAINING_STEPS=100 \
    MODEL_PATH="$WORK_DIR/models/Qwen3-8B" DATA_DIR="$WORK_DIR/data/gsm8k" \
    DEFAULT_LOCAL_DIR="$WORK_DIR/checkpoints/clean100" RAY_TEMP_DIR=/tmp/ray-l78 \
    SAVE_FREQ=10 TEST_FREQ=20 VAL_BEFORE_TRAIN=True RESUME_MODE=auto \
    bash "$RECIPE_DIR/run_qwen3_8b_lora_merge_fsdp2_npu.sh"
)
```

默认 `TOTAL_EPOCHS=TOTAL_TRAINING_STEPS` 确保数据遍历次数足够。训练仍在目标步数停止。
脚本保存 `training.*.log`、保留失败退出码，并检查训练步数。

### 日志与结束检查

| 情况 | 处理方法 |
| --- | --- |
| 训练未中断 | 用 `tools/check_validation.py` 检查完整日志 |
| 训练曾中断 | 按实际恢复的检查点整理第1–100步记录，排除重复计算的步骤 |
| Ray 转发日志缺行 | 保留 TaskRunner 原始日志 |

训练结束后，核对以下内容：

- 最终步数：`latest_checkpointed_iteration.txt`。
- 检查点：模型分片、优化器分片和额外状态。
- 执行结果：最终验证和进程退出码。

保存结果后释放本方占用的设备；使用容器时停止本方容器。本次只检查了第100步检查点文件，尚未测试从该检查点恢复训练。

吞吐量按 `sum(perf/total_num_tokens) / sum(perf/time_per_step) / 4` 计算。
它仅统计训练步，不代表全任务吞吐量或纯生成速度。本次未验证 GPU 对比、八卡或 A3 配置。
按照 issue 的要求，最终完成还需合入 PR 并向 issue 提交实践文档。
