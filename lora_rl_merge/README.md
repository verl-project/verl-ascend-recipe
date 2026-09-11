# Qwen3-8B 的 FSDP2 LoRA merge 训练

本 recipe 对应 [issue #78](https://github.com/verl-project/verl-ascend-recipe/issues/78)：
Qwen3-8B 使用 GRPO 和 LoRA 训练，每次 rollout 权重同步前将 adapter 合并进基座模型，
训练后端为 FSDP2，推理后端为 vLLM-Ascend。

2026-09-11，固定依赖及两个补丁的配置在单台 **4 × 910B3** 上从头完成连续100步。
首末十步 reward 均值由 **0.39160 升至 0.83555**，GSM8K 验证准确率由 **24.72% 升至 80.67%**，
按全部四卡计算的训练步吞吐为 **655.51 tokens/s/NPU**。所有已记录训练指标有限，梯度非零，
保存了 checkpoint100，进程退出码为0。结果满足 issue 在无GPU标杆时的训练长度、reward 和TPS条款。

第80步验证准确率为89.08%，第96–100步出现梯度范数增大与 reward 下降，原因尚未定位。
不把第100步称为最优checkpoint，也不保证每次重跑得到相同曲线。
完整方法、证据与限制见[算法适配调优文档](ALGORITHM_TUNING_REPORT.md)。

## 离线复算

从 recipe 仓库根目录执行，仅依赖Python标准库，不需要NPU：

```bash
python3 lora_rl_merge/tools/check_validation.py \
  lora_rl_merge/evidence/910b3-100step/metrics.log --devices 4
python3 -m unittest discover -s lora_rl_merge/tools -v
```

[证据目录](evidence/910b3-100step/README.md)保存从原始worker日志逐字抽取的全部训练和验证指标行、
哈希、摘要、版本来源、输入校验值、进程退出状态及checkpoint检查结果。
检查器拒绝缺步、重复步、验证行冒充训练、缺少rollout诊断、NaN/Inf、零梯度、错误卡数分母、
reward未上升、TPS不超过100，以及缺少初始或最终验证的结果。

## 固定环境

| 组件 | 实测版本或范围 |
| --- | --- |
| 硬件 | 主证据为单台4 × 910B3；另有910B1的历史冒烟及续训证据。8卡与A3未验证。 |
| verl | `bc72e38edba78e778bfbd462638f9634b9140a76`，0.9.0.dev0。 |
| vLLM | `bcf2be96120005e9aea171927f85055a6a5c0cf6`，0.18.0。 |
| vLLM-Ascend | `a43c8cc8057f490ed1df2c6ed66253e2d7817da4`，0.18.1.dev44。 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2。 |
| CANN / 驱动 | 9.0.0 / 26.0.rc1。 |
| transformers / peft / ray | 5.10.4 / 0.20.0 / 2.56.1。 |

版本声明见 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。通用 `install_verl.sh` 只安装verl，
不会配置CANN、vLLM-Ascend或应用本recipe的补丁，不能代替完整环境准备。

### 镜像取得与加载

原Quay标签 `latest-cann9.0.0-torch_npu2.9.0post2-910b-ubuntu22.04-py3.11-vllm` 在
2026-09-11返回 `TAG_EXPIRED`，原digest也返回404。**不要把拉取该标签作为可用的安装步骤。**
固定环境保存在离线Docker镜像 `verl-issue78-image.tar.zst` 中，大小为4,973,379,981字节，
SHA256为 `2e2f56293ed3b39f91a84dd0d28ffaff8d293dbaedb6d3ac88548e3eb2118685`。
接收方须先取得该归档或已经持有经校验确认的同一镜像。本Git仓库不含镜像二进制，
仅克隆仓库不能补齐此环境前置条件。

```bash
echo '2e2f56293ed3b39f91a84dd0d28ffaff8d293dbaedb6d3ac88548e3eb2118685  verl-issue78-image.tar.zst' | sha256sum -c -
set -o pipefail
zstd -dc verl-issue78-image.tar.zst | docker load | tee image-load.log
IMAGE_ID=$(sed -n 's/^Loaded image ID: //p' image-load.log | tail -n 1)
test -n "$IMAGE_ID"
python3 lora_rl_merge/tools/check_image.py "$IMAGE_ID"
```

归档在验证主机加载后的image ID为
`sha256:25cdc80fb496290267a07dd270023caf9a473c4167959b90b8ffe2bc3fbda27c`，
原主机为 `sha256:b25fae72319d0c16451fb27afd6391f97e8734452ce369901082dd5b8956ae91`。
导出/加载后的配置序列化存在差异，RootFS层和主要运行配置的比对见[镜像证据](evidence/910b3-100step/image.json)。
不同image ID不能单独证明环境一致或不一致。

### 容器、模型与数据

共享主机使用bridge网络和private IPC，只映射获准设备，不停止他人容器。
宿主驱动须满足该CANN环境要求。以下示例使用已分配的物理卡4–7：

```bash
WORK_DIR=/data/verl-ascend/issue78
mkdir -p "$WORK_DIR"
docker run -d --name verl-issue78 --network bridge --ipc private --shm-size 128g \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$WORK_DIR":/workspace/work -w /verl --entrypoint bash \
  "$IMAGE_ID" -lc 'sleep infinity'
```

将本仓库的 `lora_rl_merge/` 复制到 `$WORK_DIR/lora_rl_merge/`，将
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) 权重及tokenizer放到
`$WORK_DIR/models/Qwen3-8B/`。在容器内用固定verl预处理器生成GSM8K数据：

```bash
docker exec verl-issue78 bash -lc 'cd /verl && python3 examples/data_preprocess/gsm8k.py --local_save_dir /workspace/work/data/gsm8k'
docker exec verl-issue78 bash -lc 'cd /workspace/work && sha256sum -c lora_rl_merge/evidence/910b3-100step/inputs.sha256'
```

校验覆盖模型权重、配置、tokenizer和两个parquet文件。哈希不同意味着输入身份尚未核对，
不能直接继承本次实验结论；重新生成的parquet也可能因序列化版本不同而有不同字节内容。

### 应用两个依赖补丁

以下命令在新建容器内执行一次。已经应用补丁时，使用 `git apply --reverse --check`
确认补丁存在，不重复应用。

```bash
docker exec verl-issue78 bash -lc '
  set -e
  test "$(git -C /verl rev-parse HEAD)" = bc72e38edba78e778bfbd462638f9634b9140a76
  test "$(git -C /vllm rev-parse HEAD)" = bcf2be96120005e9aea171927f85055a6a5c0cf6
  test "$(git -C /vllm-ascend rev-parse HEAD)" = a43c8cc8057f490ed1df2c6ed66253e2d7817da4
  git -C /verl apply --check /workspace/work/lora_rl_merge/patches/0001-get_npu_versions-first-visible-npu-id.patch
  git -C /verl apply /workspace/work/lora_rl_merge/patches/0001-get_npu_versions-first-visible-npu-id.patch
  git -C /vllm-ascend apply --check /workspace/work/lora_rl_merge/patches/vllm-ascend/0001-record-sampler-stream.patch
  git -C /vllm-ascend apply /workspace/work/lora_rl_merge/patches/vllm-ascend/0001-record-sampler-stream.patch
'
```

设备探测补丁查询 `npu-smi info -m` 中首个可见物理卡，解决只挂部分卡时硬编码卡1的失败。
原函数忽略 `ASCEND_VISIBLE_DEVICES`，设置它不能替代补丁。
sampler补丁调用 `q.record_stream(...)`，回移上游
[PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394) 的张量生命周期修复。

## 运行完整100步

正式运行使用与单机实测相同的训练脚本及超参。首次使用全新checkpoint目录，
中断后重跑同一命令，由 `RESUME_MODE=auto` 恢复最近checkpoint。不要用已有实验目录冒充从头训练。

```bash
docker exec verl-issue78 bash -lc '
  set -e
  cd /verl
  NPROC_PER_NODE=4 PHYS_CARDS=4,5,6,7 TOTAL_TRAINING_STEPS=100 \
    MODEL_PATH=/workspace/work/models/Qwen3-8B DATA_DIR=/workspace/work/data/gsm8k \
    DEFAULT_LOCAL_DIR=/workspace/work/checkpoints/clean100 RAY_TEMP_DIR=/tmp/ray-l78 \
    SAVE_FREQ=10 TEST_FREQ=20 VAL_BEFORE_TRAIN=True RESUME_MODE=auto \
    bash /workspace/work/lora_rl_merge/run_qwen3_8b_lora_merge_fsdp2_npu.sh
'
```

默认 `TOTAL_EPOCHS=TOTAL_TRAINING_STEPS` 提供足够epoch容量，训练仍在目标步数停止。
脚本保存独立 `training.*.log`，保留训练失败退出码，并拒绝未到目标步的正常提前退出。
它只验证本次调用的连续更新和最终步号。无中断的完整日志可直接交给 `tools/check_validation.py`；
续训证据须先按实际checkpoint选择唯一的1–100步轨迹，剔除被重算的更新。
Ray转发日志缺行时，应保留对应TaskRunner的原始worker日志，不能用进度条补齐指标。

作业结束后核对 `latest_checkpointed_iteration.txt`、模型/优化器分片及额外状态、最终验证和退出码。
保存结果后停止本方容器释放设备。checkpoint100文件检查不等于实际加载恢复测试。

## 交付边界

吞吐定义为 `sum(perf/total_num_tokens) / sum(perf/time_per_step) / 4`，不含全部启动和验证开销，
不是纯生成速度。无GPU基线，不主张达到A100的某个比例。
历史FSDP2单测有6项NPU通过记录，不据此主张FSDP1、8卡、A3或当前上游CI已获验证。
按issue定义，最终完成还需要PR合入及实践文档提交到issue，本地结果不替代维护者接受。
