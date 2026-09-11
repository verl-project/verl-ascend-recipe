# Qwen3-8B LoRA merge 的昇腾适配与100步验证

## 1. 断言

在固定依赖上应用设备探测和sampler生命周期补丁后，Qwen3-8B的GRPO + LoRA merge配置
能够在单台4 × 910B3上完成连续100步，已记录指标全部有限、梯度非零、首末十步reward上升，
全部四卡分母的训练步吞吐超过100 tokens/s/NPU，并保存checkpoint100与最终验证。

本次单机实验支持该断言：reward为0.39160 → 0.83555，吞吐为655.51 tokens/s/NPU。
第96–100步出现训练不稳定，最终验证低于第80步，原因尚未定位。
这份报告证明一次固定配置的实测结果，不保证任意环境或随机种子的结果相同。

## 2. 背景与动机

[issue #78](https://github.com/verl-project/verl-ascend-recipe/issues/78) 要求在Atlas 800T A2/A3上支持
FSDP2 + vLLM-Ascend的Qwen3-8B GRPO + LoRA（`model.lora.merge=True`）。无GPU标杆时，
精度性能条件为训练100步或12小时、reward上升、TPS大于100，另外需要实践文档及PR合入。
本次没有GPU标杆，只采用上述绝对判据，不主张平均误差、下游误差或A100性能比例达标。

历史运行揭示两个缺口。第一，`total_training_steps=100` 配合一个epoch，实际只运行
`7473 // 128 = 58` 步后正常退出。第二，原版sampler的完整100步轨迹每步都有七项非有限
`rollout_corr`诊断。修复后曾完成跨910B1/910B3的续训轨迹，本报告以另一次从头运行的
单机轨迹为主要证据，避免依赖跨硬件拼接。

## 3. 实验设计与实现边界

### 3.1 固定输入

| 项目 | 本次取值 |
| --- | --- |
| 实测recipe | `84142b1`，完整提交和脚本/补丁哈希见 [provenance.json](evidence/910b3-100step/provenance.json)。 |
| verl | `bc72e38edba78e778bfbd462638f9634b9140a76`。 |
| vLLM / vLLM-Ascend | `bcf2be96120005e9aea171927f85055a6a5c0cf6` / `a43c8cc8057f490ed1df2c6ed66253e2d7817da4`。 |
| 平台 | 单台四张Ascend 910B3，CANN 9.0.0、驱动26.0.rc1、torch 2.9.0、torch_npu 2.9.0.post2。 |
| 模型与数据 | Qwen3-8B；GSM8K train 7473条、test 1319条。输入哈希见 [inputs.sha256](evidence/910b3-100step/inputs.sha256)。 |
| 算法 | GRPO、batch128、mini-batch64、rollout.n=8、prompt1024、response1024、lr=1e-5、KL系数0.001。 |
| LoRA | rank32、alpha64、默认all-linear、`model.lora.merge=True`。 |
| 并行与内存 | FSDP2、bf16、actor不offload、vLLM TP2、eager模式、`gpu_memory_utilization=0.6`。 |
| 长度与保存 | 100步、epoch上限100、save10、test20、初始验证开启、自动恢复开启；本次从第1步开始。 |

算法与LoRA超参沿用固定verl的
[`run_qwen3_8b_merge_fsdp.sh`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh)。
除NPU设备、四卡数、eager、分块熵计算和关闭torch.compile等启动项外，还调整了epoch容量、
保存/验证频率及日志完成检查。这些运行控制差异不能概括为“完全只改设备”。

### 3.2 合并权重路径

固定verl的 [`engine_workers.py`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/verl/workers/engine_workers.py)
读取 `model.lora.merge`。FSDP2在
[`transformer_impl.py`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/verl/workers/engine/fsdp/transformer_impl.py)
的 `merged_lora_context` 内取合并后的完整权重，并在退出上下文后恢复adapter和基座状态。
[`vllm_async_server.py`](https://github.com/verl-project/verl/blob/bc72e38edba78e778bfbd462638f9634b9140a76/verl/workers/rollout/vllm_rollout/vllm_async_server.py)
在merge模式下设置 `lora_rank=0`，推理端接收完整权重，不依赖推理侧adapter支持。
参考log-prob使用禁用adapter的actor，不另建参考模型。

本recipe复用上述已有算法实现，新增昇腾启动配置、依赖补丁及验证工具，不将上游已有能力描述为本次新实现。

### 3.3 两个补丁与控制边界

`get_npu_versions()` 原先硬编码查询物理卡1，部分设备映射中该卡不存在。补丁先解析
`npu-smi info -m` 的可见卡，再查询首个逻辑计算芯片对应的物理卡；它不负责设备隔离。
原函数不使用 `ASCEND_VISIBLE_DEVICES`，环境变量不能修复这一查询。

vLLM-Ascend sampler在辅助stream上生成张量 `q`、在当前stream上消费。
`wait_stream` 保证执行顺序，`q.record_stream(torch.npu.current_stream())` 则记录消费者stream，
避免分配器在消费完成前重用其内存。这一行回移自上游
[PR #13394](https://github.com/vllm-project/vllm-ascend/pull/13394)，合入提交为
`fc0ce85b58f019e5a1988dbd3f39d014052fb44b`。

历史同卡TP1对照中，原版224621个token有4个越界ID及对应 `-inf`，补丁组225828个token两类异常为0。
该小样本支持生命周期修复，但不能与原训练的异常token逐一对应。本次TP2完整训练检查补丁后的目标路径；
没有单变量训练对照，因此不把reward或吞吐变化归因于补丁，也不量化不同硬件的性能差异。

## 4. 执行步骤

1. 准备固定镜像、模型、GSM8K输入和recipe，逐文件核对SHA256，再核对依赖提交及两个补丁。
2. 在已分配的四张910B3上，以bridge网络和private IPC启动容器，使用全新的checkpoint目录。
3. 执行 [README的完整100步命令](README.md#运行完整100步)，保持上述训练配置。
4. 保存TaskRunner原始worker日志、进程退出状态及checkpoint文件检查，逐字抽取全部指标行，
   用 `tools/check_validation.py` 复算，保留原始日志和提取文件各自的哈希。

这不是新旧配方的性能对照，它回答固定补丁和配置能否满足本次训练判据。
后续交付整理保留训练脚本和两份补丁的内容，新增的离线工具不参与训练更新。

## 5. 观测结果

原始指标见 [metrics.log](evidence/910b3-100step/metrics.log)，复算结果见
[summary.json](evidence/910b3-100step/summary.json)。两者无需访问本方服务器即可检查。

| 判据 | 预期 | 实测 |
| --- | --- | --- |
| 训练长度 | 连续100步。 | 第1–100步无缺失、重复或替代更新。 |
| 数值 | 已记录指标有限，梯度非零。 | 全部训练指标有限，零梯度步为0；七项历史异常诊断逐步存在且有限。 |
| reward | 末十步均值高于首十步。 | 0.3916015625 → 0.835546875。 |
| 吞吐 | 四卡分母大于100。 | 加权655.5106165177748，最低逐步565.1711475341843 tokens/s/NPU。 |
| 最终验证 | 有初始和最终结果。 | GSM8K 326/1319 → 1064/1319，即24.72% → 80.67%。 |
| 保存与退出 | 有checkpoint100，进程成功退出。 | latest=100，模型/优化器分片和额外状态存在；ExitCode=0、OOMKilled=false。 |

吞吐原始分子为 **71,621,758 tokens**，训练步累计时间为 **27,315.25477820309秒**：
`71621758 / 27315.25477820309 / 4 = 655.5106165177748 tokens/s/NPU`。
它不是各步吞吐的算术平均值，也不是含全部启动、验证和保存开销的全作业吞吐或纯生成速度。
容器从 `2026-09-10T20:03:16.847377746Z` 至 `2026-09-11T04:05:31.321482907Z`，历时约8.04小时。

| step | 0 | 20 | 40 | 60 | 80 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K准确率 | 24.7157% | 70.1289% | 80.2123% | 86.6566% | 89.0826% | 80.6672% |

![单机四张910B3的训练reward、验证准确率、吞吐、梯度范数和步时](evidence/910b3-100step/training-curves.png)

## 6. 分析与异常

### 6.1 训练判据与后期下降

训练长度、首末十步reward和吞吐均达到预期，最终留出集准确率也高于初始值。
但第96–100步reward依次为0.8174、0.8145、0.7881、0.6982、0.6113，
梯度范数依次为0.6250、2.1797、2.1289、2.0723、2.7148，响应长度同时上升。
第100步验证比第80步低8.42个百分点。这是实际观测到的训练不稳定，不能由均值达标掩盖。

本次没有多随机种子重复，也没有学习率、KL系数或梯度裁剪对照，无法区分随机波动、优化不稳定、
恢复差异及硬件数值差异的作用。单机已测验证点中第80步最高；历史续训轨迹在恢复第90步时曾测得
更高准确率，因此不能泛称“所有轨迹峰值都是第80步”。checkpoint80也未做独立加载评测。

### 6.2 性能与资源

使用真实四卡分母；issue没有指定卡数或明确TPS是整机还是每卡，此处完整披露口径。
本次没有GPU或同负载跨硬件对照，不将不同阶段的TPS差异解释为硬件性能差异。
响应长度、每步token总量和训练阶段都会改变所报吞吐。

`actor/perf/max_memory_allocated_gb`、`max_memory_reserved_gb` 在固定源码中以 `1024**3` 换算，
单位实际为GiB；`cpu_memory_used_gb` 来自系统内存统计，不是actor进程RSS或主机锁页内存。
本实验未采集锁页分配器峰值，不能用系统内存统计补足。

### 6.3 复现环境的外部变化

交付整理时发现原Quay标签已过期，不能让接收方依赖旧 `latest` 拉取命令。
已保留的固定镜像归档经过哈希核对，加载后在两台910B3上曾用于实际训练。
加载后的image ID与原主机不同，RootFS层和主要运行配置一致，见[镜像记录](evidence/910b3-100step/image.json)。
接收方取得该归档是实机复现前置条件；没有公共下载地址时不能声称公共安装流程已经闭合。

## 7. 有边界的结论

单机实测支持第1节断言，覆盖固定版本、四卡、FSDP2、TP2、Qwen3-8B GRPO + LoRA merge。
原来的58步提前退出和补丁后完整训练证据缺失已得到补足。两份依赖补丁及目标训练结果可以一同评审。
本报告不对后期下降作未经验证的归因，不将本次结果称为所有配置上的稳定性保证。

## 8. 效度威胁与未覆盖项

- 只有一条单机完整轨迹，没有多随机种子统计或独立接受结论。
- checkpoint100检查了文件及额外状态，但未加载模型和优化器继续训练，不能称为该checkpoint的恢复测试。
- 没有GPU标杆、A3、八卡、FSDP1、ACL graph或更长响应配置的完整验证。
- 没有最佳超参或硬件等价性主张；更换依赖或调参后需重新验证。
- 指标检查器不能证明未记录张量全部正确，原始数值、版本和实际运行证据共同限定结论。

## 9. 复现与交付

完整准备、补丁和训练命令见 [README](README.md)。离线复算：

```bash
python3 lora_rl_merge/tools/check_validation.py lora_rl_merge/evidence/910b3-100step/metrics.log --devices 4
```

保留本次已验证超参作为复现基线。降低学习率、调整KL系数或加强梯度裁剪都是尚待实验的候选，
不直接写入已验证配方。若按验证点选模型，应明确选择规则并单独加载评测候选checkpoint。
按issue定义，PR合入与实践文档提交到issue仍属于外部完成步骤，本地整理不替代这些步骤。
