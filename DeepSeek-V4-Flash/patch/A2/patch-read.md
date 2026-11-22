# A2 patch 说明

## MindSpeed-LLM patch

### [01-rope-table-adaptation.patch](mindspeed-llm/01-rope-table-adaptation.patch)

集中处理 DeepSeek-V4 RoPE 表的数值、长度和缓存。

- 在 CPU 上使用 FP64 计算 RoPE，转换为 complex64 后移至 NPU，使计算方式与 vLLM-Ascend rollout 对齐。
- 压缩与非压缩 RoPE 表长度均使用 `max_position_embeddings`，覆盖扩展后的上下文。
- 按参数和 device 缓存单独的频率表，同时缓存模型中的堆叠表，避免每次 forward 重算或重建。

### [02-nonuniform-pipeline-layers.patch](mindspeed-llm/02-nonuniform-pipeline-layers.patch)

修复非均匀 PP 层数分配的配置解析与层偏移。

- 将字符串形式的 `num_layer_list` 解析为整数列表。
- 在配置缺失时构造 `layer_offset`，确保各 stage 的层编号正确。
- 注册 Megatron 实际使用的模块级 offset 函数。

### [03-variable-sequence-pipeline.patch](mindspeed-llm/03-variable-sequence-pipeline.patch)

注册 MindSpeed 的动态 P2P shape 通信实现，使流水线支持变长序列、普通 3D 激活以及 MHC 的 4D 激活。

### [04-hash-layer-validation.patch](mindspeed-llm/04-hash-layer-validation.patch)

将 hash 层数超过单个 PP stage 层数时的异常改为 warning，允许减层配置继续运行。

### [05-tnd-short-segment-padding.patch](mindspeed-llm/05-tnd-short-segment-padding.patch)

修复短 TND 片段的稀疏注意力 backward 问题。

- 补齐 query、原始 KV、压缩 KV，并生成有效的稀疏索引，规避 CANN 9.1 `SparseFlashMlaGrad` 的短片段越界。
- query 与原始 KV 增加相同长度，保持真实 token 的因果窗口，算子输出再裁回原长度。

### [06-tnd-cp-rope-offsets.patch](mindspeed-llm/06-tnd-cp-rope-offsets.patch)

修正 TP × CP 连续 TND 分片的 RoPE 位置。

- CP 偏移按完整的 TP-gathered block 长度计算，避免按单个 TP 分片长度推进位置。
- TP 偏移只在 sequence parallel 开启时使用；同时区分局部与全局的 RoPE 切片范围。

### [07-tnd-memory-optimization.patch](mindspeed-llm/07-tnd-memory-optimization.patch)

降低显存开销。

- indexer 和 compressor 共用一次可求导的 prefix hidden-state gather，indexer 使用 detached view，减少重复通信和大 tensor 分配。
- 前缀覆盖完整 gathered tensor 时使用 view，避免再次复制。
- 向 fused indexer 传递 `return_value`，在无需 score 时关闭其返回。
- 不需要 indexer loss 时关闭投影梯度，并释放不再使用的 score、query、key 和 weight tensor。

### [08-cp16-training-consistency.patch](mindspeed-llm/08-cp16-training-consistency.patch)

修复 CP16 训练一致性。

- 在训练/prefill 的非 overlap 压缩路径中，先将 KV、gate score 和 RoPE gather 成全局序列，再执行压缩。
- 记录 compressor 已 gather 的状态，避免 attention 重复 gather。
- 四条 fused 路径先计算全局 token 前缀，再除以压缩比例取整。例如 `2 * (704 // 128) = 10`，而 `(2 * 704) // 128 = 11`，修正后不会少算一个可见压缩块。

### [09-router-replay-r3.patch](mindspeed-llm/09-router-replay-r3.patch)

支持 RouterReplay 功能。

- 显式传递 `enable_routing_replay` 开关和 `router_replay` 对象。
- top-k helper 接入 RECORD、REPLAY_FORWARD、REPLAY_BACKWARD；具体的回放索引和 mask 由 verl RouterReplay 管理。
- hash 层保留 token ID 到 expert 的确定性查表并记录索引，当前实现关闭 hash 分支的 forward/backward replay 路径。

### [10-strict-checkpoint-resume.patch](mindspeed-llm/10-strict-checkpoint-resume.patch)

支持严格断点续训，将严格断点续训所需的 MindSpeed 实现注册到 Megatron 实际调用点。

- 注册 `FileSystemWriterAsync.preload_tensors` 的同步保存实现，保持 staticmethod 语义，避免额外绑定 `self`。
- 注册 distributed optimizer 的 `load_state_dict` 和 parameter-state load 实现。
- 同时覆盖 `DistributedOptimizer` 与 `ChainedOptimizer` 的参数状态恢复入口，确保 swap optimizer 使用正确的 CPU 权威状态。

## MindSpeed core patch

### [01-strict-checkpoint-resume.patch](mindspeed/01-strict-checkpoint-resume.patch)

支持严格断点续训，提供 swap optimizer CPU 权威状态映射与同步 DCP 保存的 preload 适配。

- 同步 DCP 保存的 preload 不再为整桶建立重复 CPU staging。
- 为 `SwapDistributedOptimizer` 重建模型参数到 optimizer param-group 的稳定索引映射。
- 保存和恢复时直接读写 optimizer 使用的 CPU main param 与 Adam 状态，避免只更新临时/设备侧副本。
- optimizer `load_state_dict` 后重新建立索引映射，保证严格续训后的参数与状态一一对应。

## mbridge patch

### [01-deepseek-v4-base.patch](mbridge/01-deepseek-v4-base.patch)

包含基础 Bridge 和 DeepSeek-V4 模型适配，修改 `core/bridge.py`、模型注册与 DeepSeek-V3 公共映射，并新增 V4 Bridge。

- 注册 `DeepseekV4Bridge`，增加 V4 配置和 HF/Megatron 参数映射，涵盖 attention sink、MHC、compressor、indexer、共享专家和 MTP。
- 支持普通 `local_experts` 的权重加载、TP/ETP 切分合并和 EP 全局专家编号。
- 兼容 `head.weight`、`embed.weight` 的加载逻辑，处理 attention sink 的 TP 分片和合并。
- 增加 `tid2eid` 映射并保留整数 dtype，将该路由表 buffer 纳入 V4 导出；保留 attention sink、APE 和 MHC 参数的原始 dtype。
- 将 dtype 判断、额外 state-dict 导出项及 TP 分片识别抽成可重写的基础接口。

### [02-expert-gemm-support.patch](mbridge/02-expert-gemm-support.patch)

支持 grouped GEMM。

- 将 `moe_grouped_gemm` 开启为 `True`，增加 `weight1`/`weight2` 的 GEMM 参数识别和 HF 专家名称映射。
- 实现 expert-major 布局下的 HF/Megatron 转换、gate/up 拼接，以及 GEMM 权重的 ETP 切片、合并和加载。
- 按专家块执行 EP gather 和流式导出，使各 rank 生成相同的全局专家权重名称流，减少导出时同时驻留的中间数据。
- 在 GEMM 导出路径中加入 CUDA/NPU 缓存清理和 NPU 同步。

### [03-fp8-current-device.patch](mbridge/03-fp8-current-device.patch)

修复 safetensors 读取的设备选择。

## vLLM patch

### [01-dspark-speculative-token-validation.patch](vllm/01-dspark-speculative-token-validation.patch)

调整 speculative config：DSpark 方法跳过 MTP 模块复用时的 token 数整除检查，允许 DSpark 使用该检查原先拒绝的 speculative token 数。其他方法继续使用原检查。

### [02-abort-spec-decode-metrics.patch](vllm/02-abort-spec-decode-metrics.patch)

保留 abort 请求的投机解码统计：按 client 分组时保存 Request 对象，在结束输出中附带累计的 `spec_decode_metrics`，同时兼容原来的 request ID 字符串调用方式，避免 abort 输出丢失已产生的统计。

## vLLM-Ascend patch

### [01-deepseek-v4-rope-alignment.patch](vllm-ascend/01-deepseek-v4-rope-alignment.patch)

对齐 DeepSeek-V4 rollout 与训练侧的 RoPE：

- 使用 `max_position_embeddings` 构造完整扩展上下文的表，原始上下文长度单独用于 YaRN 校正范围。
- 压缩层使用 YaRN 校正参数；非压缩层关闭 YaRN。
- 在 CPU 上以 FP64 计算频率和复数旋转值，之后保存为 FP32 cos/sin 表并移至目标设备。
- 表长已是扩展长度，生成位置序列时不再额外乘 scaling factor。

### [02-spec-decode-safe-prev-index.patch](vllm-ascend/02-spec-decode-safe-prev-index.patch)

修复 Triton speculative decode kernel 的前驱索引：加载前将 `offsets - 1` 截断到非负地址，再用 `tl.where` 将首项的前驱累计值设为 0，避免构造负的读取地址并保持原有累计值语义。

## Megatron-LM patch

### [01-router-replay-primitives.patch](megatron/01-router-replay-primitives.patch)

为 verl/MindSpeed-LLM 的 R3 router replay 提供 Megatron 侧基础接口。

- 增加 `apply_router_token_dropping()`，按 expert capacity、drop policy 和 padding 配置更新 routing probabilities 与 routing map。
- 增加 `compute_routing_scores_for_aux_loss()`，支持 `softmax`、`sigmoid` 和 DeepSeek-V4 使用的 `sqrtsoftplus`，并处理 padding mask。
- fused 路径单独探测 Transformer Engine 的 `fused_compute_score_for_moe_aux_loss`；依赖缺失时保持普通 PyTorch 路径可用，只有显式请求 fused 时才报错。
- 在 `MLATransformerConfig` 中声明 `enable_routing_replay`，使 verl 能显式开启 MoE 路由记录与回放。

该 patch 对应 verl 的 `08-router-replay.patch` 和 MindSpeed-LLM 的 `09-router-replay-r3.patch` 所依赖的 Megatron 接口；缺少它时 verl 会进入“不支持 NPU router replay”的降级路径。

### [02-preserve-fp32-expert-bias.patch](megatron/02-preserve-fp32-expert-bias.patch)

保护 DeepSeek-V4 的 expert bias 精度。

- `Float16Module` 包装模型前暂存所有 `expert_bias` buffer 及其 persistent 属性。
- 模型整体转换为 FP16/BF16 后，以原始数值重新注册 FP32 buffer。
- 避免 expert bias 先降为 BF16、再由 `_maintain_float32_expert_bias()` 转回 FP32 时只能恢复 dtype 而不能恢复已经丢失的尾数。

该 patch 对启用 `moe_router_enable_expert_bias` 的训练路径有直接影响，并保证 mbridge 写入的 HF `gate.bias` 落入真正的 FP32 buffer。

### [03-strict-checkpoint-resume.patch](megatron/03-strict-checkpoint-resume.patch)

提供严格断点续训所需的底层 DCP 分段读写、page-cache 回收、metadata/optimizer 布局支持及 GLU optimizer 内存优化。

- checkpoint 写盘按默认 512MiB 分段执行 `fdatasync` 和 `posix_fadvise(DONTNEED)`，限制共享文件系统 page cache 驻留。
- checkpoint 读取完成一个前缀后回收对应 clean page cache；可用 `MEGATRON_CKPT_READ_FADVISE_CHUNK_BYTES=0` 关闭。
- 同步保存允许 CPU view 直通，避免 plan 阶段整批 clone；保留诊断统计以定位大额 contiguous/clone 分配。
- 缺少 `mcore_data` 的 metadata 使用 live state-dict 布局回退，并兼容旧的一维 flattened tensor checkpoint。
- 修正 distributed optimizer 的 replica metadata 校验和 DeepSeek-V4 GLU optimizer 一维布局，恢复时原位写回状态。

## verl patch

### [01-mindspeed-model-config.patch](verl/01-mindspeed-model-config.patch)

接入 MindSpeed-LLM 的 DeepSeek-V4 模型入口，按 LOCAL_RANK 设置并检查 NPU 设备，统一模型配置和 num_layer_list 参数的传递。

### [02-thd-cp-input-layout.patch](verl/02-thd-cp-input-layout.patch)

对齐 THD/packed 输入与 DeepSeek-V4 的连续 CP 分片布局，使各 PP stage 的 hash router 获取正确的 token IDs，并兼容旧版 PackedSeqParams。

### [03-npu-bshd-sequence-alignment.patch](verl/03-npu-bshd-sequence-alignment.patch)

将 NPU BSHD 输入的序列长度向上对齐到 1024 的倍数，且至少为 2048，满足算子的长度要求。

### [04-npu-ipc-weight-transfer.patch](verl/04-npu-ipc-weight-transfer.patch)

放开 NPU IPC 版本检查，并对齐混合 dtype 权重在传输桶中的偏移，保证权重读写位置正确。

### [05-vllm-version-compatibility.patch](verl/05-vllm-version-compatibility.patch)

适配 vLLM 0.26/0.30 的 API、模型导入和指标格式，处理 Ascend MoE 权重重载、DP rank 配置及推理暂停与恢复。

### [06-ascend-rollout-sleep.patch](verl/06-ascend-rollout-sleep.patch)

管理 NPU rollout 的休眠与恢复：解除内存映射前等待 NPU 工作完成，恢复 KV cache 前同步各 rank，并为 MTP/LoRA 场景保留 sleep level 1。

### [07-rollout-task-queue.patch](verl/07-rollout-task-queue.patch)

读取 VLLM_ASCEND_TASK_QUEUE_ENABLE，并将其作为 TASK_QUEUE_ENABLE 传入 rollout server 的运行环境。

### [08-router-replay.patch](verl/08-router-replay.patch)

支持 R2/R3 路由回放、MindSpeed hash routing 和非均匀 PP，统一回放开关、CP 布局、层数偏移与路由 map 汇总，并检查 rollout 路由行数。

### [09-training-memory-optimization.patch](verl/09-training-memory-optimization.patch)

减少模型加载、训练输出和 checkpoint/RoPE 的显存驻留，按需保留 model output 和扩展 RoPE 表，并在 backward 结束后回收已完成的 MHC 输入存储。

### [10-async-server-port-reservation.patch](verl/10-async-server-port-reservation.patch)

按 IPv4/IPv6 地址族预留 DP server 所需端口，统一 master port 和端口列表，并在服务启动前或异常时释放预留 socket。

### [11-wandb-tracking.patch](verl/11-wandb-tracking.patch)

将 Hydra/OmegaConf 配置转换为 WandB 可序列化的格式，展平配置 key，并处理超长 key 和缩短后的名称冲突。

### [12-runtime-logging.patch](verl/12-runtime-logging.patch)

将 vLLM 日志等级调整为 INFO，提高内存日志的可见性，输出 allocated、reserved、device used 和 total 等内存信息。

### [13-train-validation-isolation.patch](verl/13-train-validation-isolation.patch)

在 colocate async 模式下隔离训练与验证请求，让验证独占推理阶段，并按入队屏障释放等待中的训练请求，避免 partial-rollout retry 干扰验证。

### [14-rollout-profiling-lifecycle.patch](verl/14-rollout-profiling-lifecycle.patch)

管理 rollout profiler 的启停，在 sleep 前结束采集，并在训练异常或重复清理时正确释放 profiler 状态。

### [15-speculative-metrics-aggregation.patch](verl/15-speculative-metrics-aggregation.patch)

累加异步分段生成中的 draft token、accepted token 和 verify step 计数，使 speculative decode 的完整统计进入训练指标。

### [16-strict-checkpoint-resume.patch](verl/16-strict-checkpoint-resume.patch)

支持严格断点续训，完整保存和恢复 model、optimizer、dataloader、extra 与 TransferQueue 状态。所有组成部分保存成功后才发布 ckpt_complete.json，分阶段汇总保存错误，并只恢复完整发布的 checkpoint；当前路径要求 actor_rollout_ref.actor.checkpoint.async_save=False。
