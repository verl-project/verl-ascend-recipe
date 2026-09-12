# Qwen3-0.6B 的 FSDP 多教师在线蒸馏

本示例对应 [issue #73](https://github.com/verl-project/verl-ascend-recipe/issues/73)，
使用 Qwen3-0.6B 学生、Qwen3-4B 和 Qwen3-8B 教师，在 FSDP 与 vLLM-Ascend 上运行 MOPD。
每条样本按数据源选择一位教师，对学生生成的 token 序列计算 log-prob，并用于学生蒸馏损失。

指定配置已在单台四张 Ascend 910B3 上从初始权重连续完成 100 步训练。

| 指标 | 实测结果 |
| --- | --- |
| 首末十步平均 reward。 | 0.39961 → 0.47734。 |
| GSM8K 全量验证准确率。 | 37.83% → 49.43%。 |
| MATH 全量验证准确率。 | 50.12% → 51.24%。 |
| 每卡训练吞吐量。 | 153.49 tokens/s，包含学生和教师共四张卡。 |

脚本默认配置采用非思考模板、明确 GSM8K 最终答案格式和 batch256，与上述实测配置一致。
修改这些训练参数后不能继承本次结果。适配原理、完整曲线和结果边界见[实践文档](ALGORITHM_TUNING_REPORT.md)。

## 离线复算

从 recipe 仓库根目录执行，汇总工具只依赖 Python 标准库，不需要 NPU：

```bash
python3 mopd/tools/summarize_training.py \
  --segment mopd/results/2026-09-11-batch256/training_log.txt 1 100 \
  --expected-steps 100 --devices 4 --output /tmp/mopd-summary.json
python3 -m unittest discover -s mopd/tools -p 'test_*.py' -v
```

[证据目录](results/2026-09-11-batch256/README.md)包含完整日志、结果汇总、训练曲线、运行版本及检查点文件记录。
汇总结果分别检查连续步数、非有限指标、零梯度、首末十步 reward 和四卡吞吐量，不能只看程序退出码。
六次验证的得分均按原评分函数计算。

## 固定环境

版本声明见 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。

| 组件 | 实测版本 |
| --- | --- |
| 硬件。 | 单台，4 × Ascend 910B3，每卡 64 GB。 |
| verl。 | `bc72e38edba78e778bfbd462638f9634b9140a76`。 |
| vLLM。 | `bcf2be96120005e9aea171927f85055a6a5c0cf6`。 |
| vLLM-Ascend。 | `a43c8cc8057f490ed1df2c6ed66253e2d7817da4`。 |
| torch / torch_npu。 | 2.9.0 / 2.9.0.post2。 |
| CANN / 驱动。 | 9.0.0 / 26.0.rc1。 |
| transformers / peft / ray。 | 5.10.4 / 0.20.0 / 2.56.1。 |

复现按软件版本、源码提交和补丁核对。镜像 ID 在运行信息中保留，用于追溯本次实验，
不要求接收方使用相同镜像名称或 RootFS。其他环境仍须完成自己的配置和训练验证。
`install_verl.sh` 只安装 verl，不会配置 CANN、vLLM-Ascend 或应用这里的补丁。

```bash
python3 -m pip show torch torch-npu verl vllm vllm-ascend transformers peft ray
npu-smi info
```

CANN 版本应另行检查安装目录中的版本文件；`npu-smi` 主要用于核对设备和驱动。
在 recipe 仓库根目录设置下面的实际路径。三个源码目录应对应当前 Python 环境安装的代码：

```bash
RECIPE_DIR=$(realpath mopd)
VERL_DIR=/path/to/verl
VLLM_DIR=/path/to/vllm
VLLM_ASCEND_DIR=/path/to/vllm-ascend
```

### 应用依赖补丁

以下命令用于尚未打补丁的源码。补丁已存在时，用 `git apply --reverse --check` 确认，不重复应用。

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

设备探测补丁修复部分物理卡映射下的硬编码查询问题。采样器补丁回移上游
[PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394) 的跨 stream 张量生命周期修复。
修改应在当前 Python 环境中生效；使用预构建安装包时，需要重新构建或安装补丁版本。

### 模型、数据和资源

| 角色 | 模型 | 数据源 | 卡数 |
| --- | --- | --- | --- |
| 学生进行训练和在线生成。 | Qwen3-0.6B | 两套数据。 | 2 张。 |
| GSM8K 教师计算学生序列的 log-prob。 | Qwen3-4B | `openai/gsm8k` | 1 张，TP=1。 |
| MATH 教师计算学生序列的 log-prob。 | Qwen3-8B | `DigitalLearningGmbH/MATH-lighteval` | 1 张，TP=1。 |

将三个 Qwen 模型的权重、配置和 tokenizer 放到下面的路径。学生池的 `NGPUS_PER_NODE=2` 不包含教师卡；
运行环境需要能访问完整四张卡。共享设备应先确认无他人存活负载，并验证真实 open 和计算能力。
本次容器使用 bridge 网络和 private IPC，物理卡为 2、3、6、7。

```bash
export STUDENT_MODEL=/path/to/models/Qwen3-0.6B
export GSM8K_TEACHER_MODEL=/path/to/models/Qwen3-4B
export MATH_TEACHER_MODEL=/path/to/models/Qwen3-8B
export SOURCE_DATA_DIR=${SOURCE_DATA_DIR:-$HOME/data}
export DATA_DIR=${DATA_DIR:-$HOME/data_mopd_nonthinking}
export DEFAULT_LOCAL_DIR=/path/to/checkpoints/mopd_nonthinking_batch256_100

python3 "$VERL_DIR/examples/data_preprocess/gsm8k.py" --local_save_dir "$SOURCE_DATA_DIR/gsm8k"
python3 "$VERL_DIR/examples/data_preprocess/math_dataset.py" --local_save_dir "$SOURCE_DATA_DIR/math"
python3 "$RECIPE_DIR/tools/prepare_nonthinking_data.py" \
  --source "$SOURCE_DATA_DIR" --output "$DATA_DIR" --tokenizer "$STUDENT_MODEL"
```

数据准备要求输出目录不存在，只向 GSM8K prompt 追加相同的简短解答和最终单行 `#### <number>` 说明，
不读取答案；原问题、答案、行序、教师路由及其他字段不变，MATH 文件逐字节保留。
工具生成 `manifest.json`，记录输入输出哈希和提示长度。训练与验证都使用原生非思考模板及原评分函数。

## 运行完整 100 步

首次运行使用新的 checkpoint 目录，从初始学生权重开始。脚本中的默认训练参数对应已通过配置；
修改默认值属于新的实验，不能继承本次验证结论。

```bash
export RAY_TEMP_DIR=/tmp/ray-mopd

cd "$VERL_DIR"
bash "$RECIPE_DIR/run_qwen3_06b_mopd_fsdp_npu.sh" --cfg job --resolve > mopd_nonthinking.yaml
python3 "$RECIPE_DIR/tools/preflight.py" mopd_nonthinking.yaml
bash "$RECIPE_DIR/run_qwen3_06b_mopd_fsdp_npu.sh"
```

preflight 检查已验证的 batch256、非思考模板、k1 纯蒸馏配置，以及两条数据路由、三个 tokenizer 的
token ID 映射和长度过滤。修改这些配置时预检会明确失败，表示新配置需要独立验证。
本轮保留训练样本 14966 条，验证样本为 GSM8K 1319 条、MATH 4998 条；每 epoch 有 58 个完整 batch。
脚本默认 epoch 上限等于目标步数，实际在第 100 步停止，避免数据遍历次数不足导致提前结束。

Ray 临时路径宜短，本次环境的完整 Unix socket 路径上限为 107 字节。训练使用 BF16 参数及优化器状态；
小更新舍入的已知风险见[诊断记录](DIAGNOSTIC_EXPERIMENTS.md)，不能由 CPU 例子推断本次完整训练无效。

## 日志与结束检查

训练脚本保存 `training.*.log`、保留失败退出码，并检查实际训练更新是否连续达到目标步数。
只有验证输出的 step 不算训练更新；完成步数检查不能替代 reward、吞吐和数值检查。

每 10 步保存 checkpoint，`RESUME_MODE=auto` 用于中断后续训。本次连续完成，未执行实际恢复；
不能把文件存在称为恢复测试。保存完整日志、模型和优化器分片、数据加载器状态及终态后，释放本方占用的设备资源。

吞吐量按 `sum(perf/total_num_tokens) / sum(perf/time_per_step) / 4` 计算，计数包含 prompt 和 response。
它不等于纯生成速度，也不包含整个作业的全部启动和验证时间。
本次只有一条完整轨迹，MATH 最终分数低于第 80 步；没有 GPU 对照、八卡或其他模型组合的验证结论。
