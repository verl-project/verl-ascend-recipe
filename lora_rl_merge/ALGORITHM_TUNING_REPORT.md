# Qwen3-8B LoRA merge 的昇腾适配与100步验证

固定配置在单台四张 Ascend 910B3 上完成连续100步训练，首末十步平均奖励从0.39160升至0.83555，
每卡训练吞吐量为655.51 tokens/s。这符合 [issue #78](https://github.com/verl-project/verl-ascend-recipe/issues/78)
在没有 GPU 基准结果时要求的训练长度、奖励上升和 TPS 大于100条件。
第80步后验证准确率下降，原因尚未确定；本次结果不能证明不同随机种子或其他配置都同样稳定。

## 配置与方法

本实验检验固定依赖和补丁能否支持 Qwen3-8B 的 GRPO + LoRA merge 训练。
训练使用 FSDP2，推理使用 vLLM-Ascend，并设置 `model.lora.merge=True`。

| 项目 | 配置 |
| --- | --- |
| 源码 | 实测提交为 `84142b1`，完整提交号和文件哈希见 [provenance.json](evidence/910b3-100step/provenance.json)。 |
| 依赖 | verl、vLLM、vLLM-Ascend 等版本固定在 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。 |
| 硬件与系统 | 使用四张 Ascend 910B3、CANN 9.0.0、驱动26.0.rc1、torch 2.9.0和 torch_npu 2.9.0.post2。 |
| 模型与数据 | 使用 Qwen3-8B，以及 GSM8K 的7473条训练样本和1319条测试样本；文件哈希见 [inputs.sha256](evidence/910b3-100step/inputs.sha256)。 |
| GRPO | 训练批量为128，优化小批量为64，每条提示采样8个响应，提示和响应长度上限均为1024，学习率为1e-5，KL 系数为0.001。 |
| LoRA | 秩为32，缩放参数 alpha 为64，目标模块为 `all-linear`，权重合并开启。 |
| 执行方式 | 使用 FSDP2 和 bf16，不卸载 actor，推理张量并行度为2，启用 eager 模式，`gpu_memory_utilization=0.6`。 |
| 训练与保存 | 从头训练100步，epoch 上限为100；每10步保存检查点，每20步验证，并在训练前验证。 |

算法和 LoRA 超参数沿用固定 verl 版本的
[`run_qwen3_8b_merge_fsdp.sh`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh)。
本示例调整了 NPU 设备、设备数量、eager 模式、分块熵计算、torch.compile、epoch 上限及保存和验证频率，
并增加了日志和训练步数检查。历史配置只遍历一次数据，导致 `7473 // 128 = 58` 步后提前结束；当前脚本已修正。

实验在新检查点目录中运行，没有中断或跨主机续训。完整准备和运行命令见 [README](README.md)。
训练后保存 TaskRunner 原始日志、容器退出状态和检查点文件记录，再抽取完整指标行进行复算。
后续整理仅修改训练脚本注释，两份依赖补丁保持不变。

## 适配实现

权重合并复用 verl 已有实现。
[`engine_workers.py`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/verl/workers/engine_workers.py)
读取合并配置；FSDP2 在
[`transformer_impl.py`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/verl/workers/engine/fsdp/transformer_impl.py)
的 `merged_lora_context` 中取得合并后的权重，退出上下文时恢复 LoRA 和基座模型状态。
[`vllm_async_server.py`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/verl/workers/rollout/vllm_rollout/vllm_async_server.py)
在此模式下设置 `lora_rank=0`，使推理端接收完整模型权重。参考策略的对数概率由禁用 LoRA 的 actor 计算。

设备探测补丁修复 `get_npu_versions()` 硬编码查询物理卡1的问题，改为从 `npu-smi info -m`
取得首个可见物理卡。原函数不读取 `ASCEND_VISIBLE_DEVICES`，因此设置该变量不能替代补丁。

采样器补丁回移上游 [PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394)
的张量生命周期修复，合入提交为 `fc0ce85b58f019e5a1988dbd3f39d014052fb44b`。
张量 `q` 在辅助执行流中创建、在当前执行流中使用。`wait_stream` 保证执行顺序，
`q.record_stream(torch.npu.current_stream())` 则防止分配器在使用完成前重用其内存。
历史同卡、张量并行度为1的对照中，原版生成224621个 token，出现4个越界 ID 及对应的 `-inf`；
补丁组生成225828个 token，两类异常均为0。该对照支持生命周期修复，但不能逐一解释原训练中的异常。
本次完整训练的张量并行度为2，没有开展只改变补丁的完整训练对照，因此不能将奖励或吞吐量变化归因于补丁。

## 实测结果

[metrics.log](evidence/910b3-100step/metrics.log)包含全部训练和验证指标行，
[summary.json](evidence/910b3-100step/summary.json)保存复算结果。

| 检查项 | 结果 |
| --- | --- |
| 训练步数 | 第1–100步连续，没有缺失或重复。 |
| 数值检查 | 已记录训练指标均为有限值，各步梯度范数均非零，原先异常的七项 `rollout_corr` 指标均存在且有限。 |
| 平均奖励 | 首十步为0.3916015625，末十步为0.835546875。 |
| 每卡吞吐量 | 按累计 token 数和训练步时间计算为655.5106 tokens/s，最低单步值为565.1711 tokens/s。 |
| GSM8K 准确率 | 初始为326/1319（24.72%），最终为1064/1319（80.67%）。 |
| 保存与退出 | 第100步的模型、优化器分片和额外状态存在；容器退出码为0，`OOMKilled=false`。 |

累计 token 数为71,621,758，训练步累计时间为27,315.25477820309秒。
每卡吞吐量按 `71621758 / 27315.25477820309 / 4` 计算，不是各步吞吐量的算术平均值，
也不包含全部启动、验证和保存时间。容器总运行时间约为8.04小时。

| 训练步 | 0 | 20 | 40 | 60 | 80 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K 准确率 | 24.7157% | 70.1289% | 80.2123% | 86.6566% | 89.0826% | 80.6672% |

![四张910B3上的奖励、验证准确率、吞吐量、梯度范数和训练步时间](evidence/910b3-100step/training-curves.png)

第96–100步的奖励依次为0.8174、0.8145、0.7881、0.6982、0.6113，梯度范数依次为0.6250、
2.1797、2.1289、2.0723、2.7148，响应长度同时增加。第100步验证准确率比第80步低8.42个百分点。
本次没有多随机种子实验，也没有学习率、KL 系数或梯度裁剪对照，尚不能解释后期下降的原因。

## 结果的适用范围

本次只验证了上述版本、四张910B3和100步配置，没有 GPU 基准、八卡或 A3 结果。
指标检查只覆盖日志中记录的数值，不能证明所有中间张量均正确。第100步检查点仅检查了文件，
尚未实际加载模型和优化器恢复训练；第80步检查点也未单独加载评测。

日志中的 NPU 内存指标按 `1024**3` 换算，实际单位为 GiB。CPU 内存指标来自系统统计，
不等于 actor 进程的常驻内存或锁页内存；本次未采集锁页内存峰值。

原 Quay 镜像标签已过期，复现需先取得 README 指定的离线镜像。归档已校验 SHA256，
加载后的 RootFS 层和主要运行配置核对结果见[镜像记录](evidence/910b3-100step/image.json)。
公共下载地址尚未提供，PR 合入和向 issue 提交实践文档也尚未完成。
