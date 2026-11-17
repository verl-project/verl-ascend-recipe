# A3 patch 说明

## A3 安装脚本

[install.sh](install.sh) 保留原脚本的 clone、cd、安装步骤，使用外部官方仓库的分支、tag 和 commit；第 7 步统一按各目录的文件编号顺序应用补丁。镜像源沿用原脚本的 Triton-Ascend 地址，原容器镜像说明保持不变。

| 仓库 | 官方分支 / tag | 安装时固定的官方 commit |
| --- | --- | --- |
| vLLM | `v0.30.0` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | `releases/v0.30.0` | `9ca3f1e433e13e1ab8285fd305ba61c3a40ea835` |
| mbridge | `v0.15.1` | `0cd4ae23f2425da77a80cb3f517828452fa8e984` |
| verl | `release/v0.9.0` | `91c297dc8d9991b79c90cad5b50683a5d6d8c02b` |
| MindSpeed | `26.2.0_core_r0.12.1` | `15023c2d0fed38d17f01b5fa7baf4ede63e0dd4e` |
| Megatron-LM | `core_v0.12.1` | `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17` |
| MindSpeed-LLM | `26.2.0` | `06dd5fa9c1f89d7c9728ab5733c39fcd027ed0d1` |

在希望保存这些源码仓库的目录中执行：

```bash
bash /path/to/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/install.sh
```

仓库 clone 到执行时的当前目录；请在这些源码仓库的父目录中运行。`PATCH_DIR` 从安装脚本所在位置定位 A3 补丁。

- vLLM 使用 `use_existing_torch.py --prefix` 和 `VLLM_TARGET_DEVICE=empty`，安装命令使用 `python -m pip`、`--user` 和 `--no-build-isolation`。
- vLLM-Ascend 保留 custom-kernel 编译设置，requirements 与 editable 安装沿用原 Triton 镜像源。Triton 版本由官方 requirements 管理，当前为 `3.6.0.post2`。
- vLLM/vLLM-Ascend 安装前按参考命令 unset `http_proxy`、`https_proxy`。
- 每个补丁先执行 `git apply --check`，通过后再应用；失败时安装脚本立即停止。
- Megatron-LM 使用 [megatron/](megatron/) 中按功能拆分的 A3 补丁，还原 `/home/tiger/Megatron-LM` 当前工作树的源码修改；不再依赖旧的 `../megatron.patch` 或 `patch/r3/megatron-lm.patch`。
- MindSpeed 使用 [mindspeed/](mindspeed/) 中的严格断点续训补丁，安装脚本在第 7 步与其他仓库一起应用。
- MindSpeed-LLM 包内 model provider 由根目录的 `pretrain_deepseek4.py` 直接复制。
- 不再单独安装 Triton-Ascend；Transformers 安装版本改为 `5.14.1`。
- verl、MindSpeed、MindSpeed-LLM 在进入仓库目录后，先 checkout 官方分支，再 checkout 上表官方 commit。

2026-11-17 已从官方仓库拉取上表源码并验证原有 33 个 A3 补丁。随后追加 4 个严格断点续训补丁，总数为 37。MindSpeed 的 `15023c2d` 已确认是官方分支中的提交；脚本未使用内部修复后的仓库 HEAD。脚本通过 `bash -n`、简单脚本的安装/checkouts/patch 顺序和失败中止检查，未执行实际 pip 安装。

## 严格断点续训补丁总览

`07_checkpoint_single_patch` 中的四份最终单补丁已经按 A3 目录编号重命名，并直接应用到当前 code 下对应仓库；四份补丁均无冲突，应用后通过 `git apply --reverse --check`。

| 仓库 | A3 补丁 | 主要职责 |
| --- | --- | --- |
| verl | [16-strict-checkpoint-resume.patch](verl/16-strict-checkpoint-resume.patch) | 完整发布标记、严格 resume、dataloader/TransferQueue 状态、分阶段保存错误汇总及进程内 DCP writer |
| MindSpeed | [01-strict-checkpoint-resume.patch](mindspeed/01-strict-checkpoint-resume.patch) | swap optimizer CPU 权威状态映射与同步保存 preload 适配 |
| MindSpeed-LLM | [10-strict-checkpoint-resume.patch](mindspeed-llm/10-strict-checkpoint-resume.patch) | 将 MindSpeed checkpoint/optimizer patch 注册到实际 Megatron 调用点 |
| Megatron-LM | [03-strict-checkpoint-resume.patch](megatron/03-strict-checkpoint-resume.patch) | DCP 分段同步写盘、page-cache 回收、metadata/optimizer 布局及 GLU optimizer 内存优化 |

严格续训目标是同代码版本、同并行布局下完整恢复 model、optimizer、dataloader、extra 和 TransferQueue 状态。当前路径要求 `actor_rollout_ref.actor.checkpoint.async_save=False`；同步保存默认使用 512MiB 分段 `fdatasync + posix_fadvise(DONTNEED)`，并只在全部组成部分成功后发布 `ckpt_complete.json`。

## MindSpeed-LLM 补丁

补丁位于 [mindspeed-llm/](mindspeed-llm/) 目录，按 **01 → 10** 的编号顺序应用，每个 patch 都基于前一个应用后的代码。

原来的 `01-deepseekv4-adaptation.patch` 按功能拆为 01–04；原来的 `02-tnd-dynamic-batch-cp.patch` 拆为两个修复补丁 05、06 和一个显存优化补丁 07。原来的 CP16 修复和 R3 路由回放补丁内容保持不变，编号顺延为 08、09；严格断点续训注册补丁追加为 10。

本系列仅包含内部提交的源码修改，忽略已有文件的权限变更；相互抵消的 RoPE 长度限制修改与撤销已省略。包内 model provider 通过下文的 `cp` 命令生成。

## [01-rope-table-adaptation.patch](mindspeed-llm/01-rope-table-adaptation.patch)

来源：`ff802915`。集中处理 DeepSeek-V4 RoPE 表的数值、长度和缓存。

- 在 CPU 上使用 FP64 计算 RoPE，转换为 complex64 后移至 NPU，使计算方式与 vLLM-Ascend rollout 对齐。
- 压缩与非压缩 RoPE 表长度均使用 `max_position_embeddings`，覆盖扩展后的上下文。
- 按参数和 device 缓存单独的频率表，同时缓存模型中的堆叠表，避免每次 forward 重算或重建。

## [02-nonuniform-pipeline-layers.patch](mindspeed-llm/02-nonuniform-pipeline-layers.patch)

来源：`ff802915`。修复非均匀 PP 层数分配的配置解析与层偏移。

- 将字符串形式的 `num_layer_list` 解析为整数列表。
- 在配置缺失时构造 `layer_offset`，确保各 stage 的层编号正确。
- 注册 Megatron 实际使用的模块级 offset 函数。

## [03-variable-sequence-pipeline.patch](mindspeed-llm/03-variable-sequence-pipeline.patch)

来源：`ff802915`。注册 MindSpeed 的动态 P2P shape 通信实现，使流水线支持变长序列、普通 3D 激活以及 MHC 的 4D 激活。

## [04-hash-layer-validation.patch](mindspeed-llm/04-hash-layer-validation.patch)

来源：`ff802915`。将 hash 层数超过单个 PP stage 层数时的异常改为 warning，允许对应的小模型配置继续运行。

## [05-tnd-short-segment-padding.patch](mindspeed-llm/05-tnd-short-segment-padding.patch)

来源：`bb785ab6`。修复短 TND 片段的稀疏注意力 backward 问题。

- 补齐 query、原始 KV、压缩 KV，并生成有效的稀疏索引，规避 CANN 9.1 `SparseFlashMlaGrad` 的短片段越界。
- query 与原始 KV 增加相同长度，保持真实 token 的因果窗口，算子输出再裁回原长度。

## [06-tnd-cp-rope-offsets.patch](mindspeed-llm/06-tnd-cp-rope-offsets.patch)

来源：`bb785ab6`。修正 TP × CP 连续 TND 分片的 RoPE 位置。

- CP 偏移按完整的 TP-gathered block 长度计算，避免按单个 TP 分片长度推进位置。
- TP 偏移只在 sequence parallel 开启时使用；同时区分局部与全局的 RoPE 切片范围。

## [07-tnd-memory-optimization.patch](mindspeed-llm/07-tnd-memory-optimization.patch)

来源：`bb785ab6`。将原 02 中降低显存开销的修改集中在一个 patch 中。

- indexer 和 compressor 共用一次可求导的 prefix hidden-state gather，indexer 使用 detached view，减少重复通信和大 tensor 分配。
- 前缀覆盖完整 gathered tensor 时使用 view，避免再次复制。
- 向 fused indexer 传递 `return_value`，在无需 score 时关闭其返回。
- 不需要 indexer loss 时关闭投影梯度，并释放不再使用的 score、query、key 和 weight tensor。

## [08-cp16-training-consistency.patch](mindspeed-llm/08-cp16-training-consistency.patch)

来源：`747d6585`。原 CP16 补丁保持完整内容，修复 CP/SP 分片长度不能整除压缩比例时的全局对齐和因果前缀。

- 在训练/prefill 的非 overlap 压缩路径中，先将 KV、gate score 和 RoPE gather 成全局序列，再执行压缩。
- 记录 compressor 已 gather 的状态，避免 attention 重复 gather。
- 四条 fused 路径先计算全局 token 前缀，再除以压缩比例取整。例如 `2 * (704 // 128) = 10`，而 `(2 * 704) // 128 = 11`，修正后不会少算一个可见压缩块。

同样适用于其他触发上述分片不对齐条件的并行配置。

## [09-router-replay-r3.patch](mindspeed-llm/09-router-replay-r3.patch)

来源：`816f19b7`。原 R3 补丁保持完整内容，将 MoE 专家选择接入 verl 的 RouterReplay。

- 显式传递 `enable_routing_replay` 开关和 `router_replay` 对象。
- top-k helper 接入 RECORD、REPLAY_FORWARD、REPLAY_BACKWARD；具体的回放索引和 mask 由 verl RouterReplay 管理。
- hash 层保留 token ID 到 expert 的确定性查表并记录索引，当前实现关闭 hash 分支的 forward/backward replay 路径。

## [10-strict-checkpoint-resume.patch](mindspeed-llm/10-strict-checkpoint-resume.patch)

将严格断点续训所需的 MindSpeed 实现注册到 Megatron 实际调用点。

- 注册 `FileSystemWriterAsync.preload_tensors` 的同步保存实现，保持 staticmethod 语义，避免额外绑定 `self`。
- 注册 distributed optimizer 的 `load_state_dict` 和 parameter-state load 实现。
- 同时覆盖 `DistributedOptimizer` 与 `ChainedOptimizer` 的参数状态恢复入口，确保 swap optimizer 使用正确的 CPU 权威状态。

## 应用示例

在 MindSpeed-LLM 仓库根目录中执行：

```bash
PATCH_DIR=/mnt/bn/lrm-hl/zhibei.ye/code/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/mindspeed-llm
for patch in "$PATCH_DIR"/*.patch; do
    git apply --check "$patch" || break
    git apply --whitespace=nowarn "$patch" || break
done
```

包内 model provider 直接从当前仓库根目录复制：

```bash
cp pretrain_deepseek4.py mindspeed_llm/pretrain_deepseek4.py
```

以 `e4a75c1` 为验证基线，依次应用九个 patch 并执行上述复制后，全部主仓文件内容与本地 `816f19b7` 一致；已有文件权限保留基线状态。

## 公开 26.2.0 分支验证

2026-11-16 已从 `https://gitcode.com/Ascend/MindSpeed-LLM.git` 克隆到 `/opt/tiger/code/MindSpeed-LLM`，执行 `git checkout 26.2.0`；验证时分支 HEAD 为 `06dd5fa9c1f89d7c9728ab5733c39fcd027ed0d1`。

01–09 的 `git apply --check` 和实际应用全部通过，没有补丁冲突。执行 model provider 复制后，17 个涉及修改或复制的 Python 文件均通过语法解析，已有文件权限保持不变。

该验证覆盖公开分支上的补丁应用和 Python 语法，未执行训练。公开分支的上游修改保留，其中 `transformer_block.py` 的 checkpoint FP8 context 实现与内部历史基线不同。

逐个补丁的校验和、返回码及检查结果记录在 `/opt/tiger/code/mindspeed-llm-patch-validation.json`。

## MindSpeed core 补丁

补丁位于 [mindspeed/](mindspeed/) 目录，基线为 `15023c2d0fed38d17f01b5fa7baf4ede63e0dd4e`。

### [01-strict-checkpoint-resume.patch](mindspeed/01-strict-checkpoint-resume.patch)

- 同步 DCP 保存的 preload 不再为整桶建立重复 CPU staging。
- 为 `SwapDistributedOptimizer` 重建模型参数到 optimizer param-group 的稳定索引映射。
- 保存和恢复时直接读写 optimizer 使用的 CPU main param 与 Adam 状态，避免只更新临时/设备侧副本。
- optimizer `load_state_dict` 后重新建立索引映射，保证严格续训后的参数与状态一一对应。


## mbridge 补丁

补丁位于 [mbridge/](mbridge/) 目录，来自 `/home/tiger/mbridge` 的未提交修改，包括 4 个已修改文件和新增的 `mbridge/models/deepseek_v4.py`。按 **01 → 02 → 03** 顺序应用。

应用基线为 `0cd4ae23f2425da77a80cb3f517828452fa8e984`（`release 0.15.1`），仓库地址为 `https://github.com/ISEEKYAN/mbridge.git`。

### [01-deepseek-v4-base.patch](mbridge/01-deepseek-v4-base.patch)

包含基础 Bridge 和 DeepSeek-V4 模型适配，修改 `core/bridge.py`、模型注册与 DeepSeek-V3 公共映射，并新增 V4 Bridge。

- 注册 `DeepseekV4Bridge`，增加 V4 配置和 HF/Megatron 参数映射，涵盖 attention sink、MHC、compressor、indexer、共享专家和 MTP。
- 支持普通 `local_experts` 的权重加载、TP/ETP 切分合并和 EP 全局专家编号。
- 兼容 `head.weight`、`embed.weight` 的加载逻辑，处理 attention sink 的 TP 分片和合并。
- 增加 `tid2eid` 映射并保留整数 dtype，将该路由表 buffer 纳入 V4 导出；保留 attention sink、APE 和 MHC 参数的原始 dtype。
- 将 dtype 判断、额外 state-dict 导出项及 TP 分片识别抽成可重写的基础接口。

本补丁中 `moe_grouped_gemm=False`，基础实现不引用 GEMM 专属接口。

### [02-expert-gemm-support.patch](mbridge/02-expert-gemm-support.patch)

仅包含 grouped GEMM 相关支持，基于 01 修改 `core/bridge.py` 和 `models/deepseek_v4.py`。

- 将 `moe_grouped_gemm` 开启为 `True`，增加 `weight1`/`weight2` 的 GEMM 参数识别和 HF 专家名称映射。
- 实现 expert-major 布局下的 HF/Megatron 转换、gate/up 拼接，以及 GEMM 权重的 ETP 切片、合并和加载。
- 按专家块执行 EP gather 和流式导出，使各 rank 生成相同的全局专家权重名称流，减少导出时同时驻留的中间数据。
- 在 GEMM 导出路径中加入 CUDA/NPU 缓存清理和 NPU 同步。

### [03-fp8-current-device.patch](mbridge/03-fp8-current-device.patch)

保持原补丁内容不变，修复 safetensors 读取的设备选择。

读取文件时显式使用 `cuda:<torch.cuda.current_device()>`，原始权重文件与跨文件存储的 scale tensor 使用同一当前设备。

### mbridge 应用示例

在干净的 mbridge checkout 中执行：

```bash
git checkout 0cd4ae23f2425da77a80cb3f517828452fa8e984
PATCH_DIR=/mnt/bn/lrm-hl/zhibei.ye/code/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/mbridge
for patch in "$PATCH_DIR"/*.patch; do
    git apply --check "$patch" || break
    git apply --whitespace=nowarn "$patch" || break
done
```

也可按同样顺序使用 `git am "$PATCH_DIR"/*.patch`，每个 patch 生成一个提交。

已在该基线的干净临时 checkout 中验证 `git apply` 和 `git am`，每一步均成功；只应用 01 时普通专家与 hash 路由的基础映射检查通过。应用全部补丁后完整还原当前未提交修改，5 个文件的内容和权限一致，并通过 Python 语法解析。原 `/home/tiger/mbridge` 的 HEAD、index 和未提交状态保持不变。本次验证覆盖补丁重建、基础映射与语法，未执行训练。



## vLLM 补丁

补丁位于 [vllm/](vllm/) 目录，来自 `/home/tiger/vllm` 的 2 个源码文件修改，按 01 → 02 顺序应用；依赖文件修改不纳入补丁。基线为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，仓库为 `https://github.com/vllm-project/vllm.git`。

### [01-dspark-speculative-token-validation.patch](vllm/01-dspark-speculative-token-validation.patch)

调整 speculative config：DSpark 方法跳过 MTP 模块复用时的 token 数整除检查，允许 DSpark 使用该检查原先拒绝的 speculative token 数。其他方法继续使用原检查。

### [02-abort-spec-decode-metrics.patch](vllm/02-abort-spec-decode-metrics.patch)

保留 abort 请求的投机解码统计：按 client 分组时保存 Request 对象，在结束输出中附带累计的 `spec_decode_metrics`，同时兼容原来的 request ID 字符串调用方式，避免 abort 输出丢失已产生的统计。

## vLLM-Ascend 补丁

补丁位于 [vllm-ascend/](vllm-ascend/) 目录，来自 `/home/tiger/vllm-ascend` 的 3 个源码文件修改，按 01 → 02 顺序应用；依赖文件修改不纳入补丁。基线为 `2df8721f4ed5d581136d7a5e3dd2487ef5036899`（`releases/v0.30.0`），仓库为 `https://github.com/vllm-project/vllm-ascend.git`。

### [01-deepseek-v4-rope-alignment.patch](vllm-ascend/01-deepseek-v4-rope-alignment.patch)

对齐 DeepSeek-V4 rollout 与训练侧的 RoPE：

- 使用 `max_position_embeddings` 构造完整扩展上下文的表，原始上下文长度单独用于 YaRN 校正范围。
- 压缩层使用 YaRN 校正参数；非压缩层关闭 YaRN。
- 在 CPU 上以 FP64 计算频率和复数旋转值，之后保存为 FP32 cos/sin 表并移至目标设备。
- 表长已是扩展长度，生成位置序列时不再额外乘 scaling factor。

### [02-spec-decode-safe-prev-index.patch](vllm-ascend/02-spec-decode-safe-prev-index.patch)

修复 Triton speculative decode kernel 的前驱索引：加载前将 `offsets - 1` 截断到非负地址，再用 `tl.where` 将首项的前驱累计值设为 0，避免构造负的读取地址并保持原有累计值语义。

## Megatron-LM 补丁

补丁位于 [megatron/](megatron/) 目录，按 **01 → 03** 顺序应用。基线为 `a845aa7e12b3a117e24c2352b9e3e60bad2e3a17`（`core_v0.12.1`），仓库为 `https://github.com/NVIDIA/Megatron-LM.git`。01–02 来自当前工作树的 DeepSeek-V4 修改，03 追加严格断点续训能力。

这些补丁不包含可重新生成的 `helpers_cpp` 扩展、`egg-info` 或 Python 缓存，也不叠加旧的 `../megatron.patch`。

### [01-router-replay-primitives.patch](megatron/01-router-replay-primitives.patch)

为 verl/MindSpeed-LLM 的 R3 router replay 提供 Megatron 侧基础接口。

- 增加 `apply_router_token_dropping()`，按 expert capacity、drop policy 和 padding 配置更新 routing probabilities 与 routing map。
- 增加 `compute_routing_scores_for_aux_loss()`，支持 `softmax`、`sigmoid` 和 DeepSeek-V4 使用的 `sqrtsoftplus`，并处理 padding mask。
- fused 路径单独探测 Transformer Engine 的 `fused_compute_score_for_moe_aux_loss`；依赖缺失时保持普通 PyTorch 路径可用，只有显式请求 fused 时才报错。
- 在 `MLATransformerConfig` 中声明 `enable_routing_replay`，使 verl 能显式开启 MoE 路由记录与回放。

该补丁对应 verl 的 `08-router-replay.patch` 和 MindSpeed-LLM 的 `09-router-replay-r3.patch` 所依赖的 Megatron 接口；缺少它时 verl 会进入“不支持 NPU router replay”的降级路径。

### [02-preserve-fp32-expert-bias.patch](megatron/02-preserve-fp32-expert-bias.patch)

保护 DeepSeek-V4 的 expert bias 精度。

- `Float16Module` 包装模型前暂存所有 `expert_bias` buffer 及其 persistent 属性。
- 模型整体转换为 FP16/BF16 后，以原始数值重新注册 FP32 buffer。
- 避免 expert bias 先降为 BF16、再由 `_maintain_float32_expert_bias()` 转回 FP32 时只能恢复 dtype 而不能恢复已经丢失的尾数。

该补丁对启用 `moe_router_enable_expert_bias` 的训练路径有直接影响，并保证 mbridge 写入的 HF `gate.bias` 落入真正的 FP32 buffer。

### [03-strict-checkpoint-resume.patch](megatron/03-strict-checkpoint-resume.patch)

提供严格断点续训所需的底层 DCP 读写与 optimizer 布局支持。

- checkpoint 写盘按默认 512MiB 分段执行 `fdatasync` 和 `posix_fadvise(DONTNEED)`，限制共享文件系统 page cache 驻留。
- checkpoint 读取完成一个前缀后回收对应 clean page cache；可用 `MEGATRON_CKPT_READ_FADVISE_CHUNK_BYTES=0` 关闭。
- 同步保存允许 CPU view 直通，避免 plan 阶段整批 clone；保留诊断统计以定位大额 contiguous/clone 分配。
- 缺少 `mcore_data` 的 metadata 使用 live state-dict 布局回退，并兼容旧的一维 flattened tensor checkpoint。
- 修正 distributed optimizer 的 replica metadata 校验和 DeepSeek-V4 GLU optimizer 一维布局，恢复时原位写回状态。

### Megatron-LM 应用示例

在干净的 Megatron-LM checkout 中执行：

```bash
git checkout a845aa7e12b3a117e24c2352b9e3e60bad2e3a17
PATCH_DIR=/path/to/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/megatron
for patch in "$PATCH_DIR"/*.patch; do
    git apply --check "$patch" || break
    git apply --whitespace=nowarn "$patch" || break
done
```

安装脚本在 clone 并 checkout `core_v0.12.1` 后，于第 7 步按文件名顺序执行上述三份补丁。

## verl 补丁

补丁位于 [verl/](verl/) 目录，按 **01 → 16** 顺序应用。每个补丁基于前面补丁应用后的代码，以下说明对应当前最终内容。

应用基线：`91c297dc8d9991b79c90cad5b50683a5d6d8c02b`。源仓库根目录是 `/opt/tiger/verl_npu`，其内部 HEAD 为 `08a8864c5888d140c9df9d0fa0f1ba01e3ed2d89`。补丁在目标仓库根目录应用。

范围仍排除 `runtime_env.yaml`、`.sh`、UT/test、FSDPTurbo 集成、`.gitignore` 和本地新增文件。独立 profiler 收集脚本和相互抵消的修改已剔除。`[VALWIN]` 探针函数与调用已删除，原有 step 指标保留。

### 功能清单

| Patch | 功能 |
| --- | --- |
| [01-mindspeed-model-config.patch](verl/01-mindspeed-model-config.patch) | MindSpeed-LLM 模型入口与配置 |
| [02-thd-cp-input-layout.patch](verl/02-thd-cp-input-layout.patch) | THD 输入与连续 CP 布局 |
| [03-npu-bshd-sequence-alignment.patch](verl/03-npu-bshd-sequence-alignment.patch) | NPU BSHD 序列长度对齐 |
| [04-npu-ipc-weight-transfer.patch](verl/04-npu-ipc-weight-transfer.patch) | NPU IPC 判定与权重桶偏移对齐 |
| [05-vllm-version-compatibility.patch](verl/05-vllm-version-compatibility.patch) | 推理版本、MoE 重载和后端指标格式兼容 |
| [06-ascend-rollout-sleep.patch](verl/06-ascend-rollout-sleep.patch) | NPU 休眠、内存解除映射与恢复同步 |
| [07-rollout-task-queue.patch](verl/07-rollout-task-queue.patch) | rollout task queue 开关传递 |
| [08-router-replay.patch](verl/08-router-replay.patch) | R2/R3 回放、hash routing 与非均匀 PP |
| [09-training-memory-optimization.patch](verl/09-training-memory-optimization.patch) | 模型加载、训练输出及 checkpoint/RoPE 显存 |
| [10-async-server-port-reservation.patch](verl/10-async-server-port-reservation.patch) | DP server IPv4/IPv6 端口预留 |
| [11-wandb-tracking.patch](verl/11-wandb-tracking.patch) | WandB 配置序列化和 key 处理 |
| [12-runtime-logging.patch](verl/12-runtime-logging.patch) | vLLM INFO 日志与内存日志 |
| [13-train-validation-isolation.patch](verl/13-train-validation-isolation.patch) | 训练/验证请求准入与释放屏障 |
| [14-rollout-profiling-lifecycle.patch](verl/14-rollout-profiling-lifecycle.patch) | rollout profiler 启停与异常清理 |
| [15-speculative-metrics-aggregation.patch](verl/15-speculative-metrics-aggregation.patch) | speculative decode 分段计数聚合 |
| [16-strict-checkpoint-resume.patch](verl/16-strict-checkpoint-resume.patch) | checkpoint 原子发布、严格恢复、TransferQueue 状态和进程内 DCP writer |

### [01-mindspeed-model-config.patch](verl/01-mindspeed-model-config.patch)

**目的**：让 DeepSeek-V4 通过 MindSpeed-LLM 的模型入口构建，保证模型参数落在当前 worker 的 NPU 上，并统一配置传递方式。

**涉及文件**：`verl/trainer/config/engine/mindspeed.yaml`、`verl/workers/engine/mindspeed/transformer_impl.py`、`verl/workers/engine/mindspeed/utils.py`。

- 将模型构建入口切换到 `mindspeed_llm.pretrain_deepseek4.model_provider`。
- 构建前按 `LOCAL_RANK` 设置设备，构建后检查模型参数的 device index；不匹配时立即报错，帮助定位 Ray 可见设备映射问题。
- 清空默认 `mcore_kwargs`，由外部配置提供 MindSpeed-LLM 参数，避免默认模型配置覆盖 V4 的配置。
- `apply_patch()` 使用 engine 配置组装 mcore 参数，不再混入原来的 HF model/optimizer 配置转换结果。
- 将列表形式的 `num_layer_list` 转为逗号分隔字符串，以适配 MindSpeed 的 argparse 接口。

**适用与依赖**：MindSpeed-LLM 的 DeepSeek-V4 模型路径。需要按 MindSpeed-LLM 说明准备包内 `pretrain_deepseek4.py`；该复制操作不包含在 verl 补丁中。加载显存优化与 router replay 分别在 09 和 08 中。

### [02-thd-cp-input-layout.patch](verl/02-thd-cp-input-layout.patch)

**目的**：保持 THD/packed 输入与模型的 CP 分片布局一致，使所有 PP stage 的 hash router 都能取得对应 token IDs。

**涉及文件**：`verl/models/mcore/model_forward.py`、`model_forward_fused.py`、`util.py`，以及 `verl/workers/engine/megatron/transformer_impl.py`。

- 普通与 fused forward 都使用 `pre_process=True` 生成 packed token IDs，去掉早期先注释、再恢复该参数的中间状态。
- DeepSeek4 的 `position_embedding_type` 或 `dsv4_hybrid` attention variant 使用连续 CP 布局；其他模型保留 zigzag 布局。
- 在旧版 `PackedSeqParams` 构造函数不提供 `cp_partition_mode` 时，构造后补充该属性，避免连续分片配置被旧接口拒绝。
- 对 NPU/MindSpeed 路径，不注入原生 MCore 的 DSV4 attention variant，让 MindSpeed 自己的 attention 实现管理该配置。

**职责边界**：这份补丁负责 token/CP 布局；PP 的每 stage 层数、offset 和 replay map gather 由 08 处理。两者解决不同问题。

### [03-npu-bshd-sequence-alignment.patch](verl/03-npu-bshd-sequence-alignment.patch)

**目的**：为 NPU 的 BSHD kernel 提供满足长度对齐和最小长度要求的 padded 输入。

**涉及文件**：`verl/models/mcore/util.py` 中 `preprocess_bshd_engine()`。

- 在 NPU 路径中，将 `max_seqlen` 向上对齐到 1024 的倍数。
- 对齐结果至少为 2048，例如 1500 → 2048、2500 → 3072。
- 后续 TP/CP 分片的整除约束仍按原逻辑处理。

**适用与影响**：只在 NPU BSHD 预处理路径生效；会增加 padding 行数以满足算子要求，THD 布局修复属于 02。

### [04-npu-ipc-weight-transfer.patch](verl/04-npu-ipc-weight-transfer.patch)

**目的**：保留当前环境的 IPC 放行逻辑，并修正混合 dtype 权重写入 IPC bucket 的偏移对齐。

**涉及文件**：`verl/utils/device.py`、`verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`。

- `check_ipc_version_support()` 直接返回 `True`，绕过原软件/CANN 版本检查。这是当前源修改的行为，并不是新增的硬件能力探测。
- 每个权重写入 bucket 前，将偏移对齐到 `max(weight.element_size(), 4)`。
- bucket metadata 记录对齐后的真实偏移，避免 NPU IPC 消费方从不满足对齐要求的位置读取权重。

**职责边界**：负责 IPC 能力判定和数据布局；多节点 RPC 端口由 10 处理，推理版本 API 由 05 处理。

### [05-vllm-version-compatibility.patch](verl/05-vllm-version-compatibility.patch)

**目的**：集中处理 vLLM 旧版到 0.26/0.30 的推理接口变化，避免相同文件在多个版本补丁里反复修改。

**涉及文件**：`verl/utils/vllm/ascend_moe.py`、`npu_vllm_patch.py`、`patch.py`，`verl/workers/engine_workers.py`，以及 rollout 下的 `async_llm.py`、`utils.py`、`vllm_async_server.py`。

- 仅对 vLLM < 0.15 使用旧 FusedMoE 类的 weight-loader 包装，避免新版工厂/布局被旧逻辑改写。
- 对已初始化的 Ascend BF16 MoE 专家，重载时转置输入权重并保留推理存储布局；只跳过专家重复 post-process，其他层仍刷新派生权重。
- 兼容 `UnquantizedFusedMoEMethod` 的模块拆分和 DeepSeek-V4 模型类的导入位置变化。新模块内部缺依赖时保留错误，不把它当作旧版本回退条件。
- GDN 可选依赖缺失时，临时跳过无关的 Qwen3.5/GDN 注册，再恢复模块和 registry 状态，让其他模型的 Ascend patch 能初始化。
- 从 MindSpeed 训练进程的环境路径中移除 vLLM-Ascend inference-only 算子 vendor 条目，保留训练侧 native CANN 路径。
- vLLM 0.30 的 API head 节点不设置显式 `data_parallel_start_rank`，headless 节点保留 offset，避免被识别为 hybrid DP 拓扑。
- 使用 `PauseAwareAsyncLLM`：请求提交与 pause 串行化，resume 后重新开放准入，复用已完成的 pause/drain 屏障，避免等待过期的 DP running 状态。
- 兼容不可解析的自定义版本字符串；现有代码回退到 0.23.0 的版本判断。
- 读取 0.30 completion 的 speculative histogram：draft 数直接读取，accepted 数为 `sum(i * count)`，verify 次数为 `sum(histogram)`；旧后端回退到 request 统计。缺少完整计数时跳过并 warning。

**职责边界**：这里只转换后端 API/格式；跨 resume 累计和 trainer 的统计开关在 15 中。原 FSDPTurbo 提交的权重归一化及专属 loader helper 没有带回。

### [06-ascend-rollout-sleep.patch](verl/06-ascend-rollout-sleep.patch)

**目的**：避免 NPU kernel 尚在使用内存映射时解除映射，并保证各 rank 在 KV cache 恢复阶段同步。

**涉及文件**：`verl/third_party/vllm/__init__.py`、`verl/utils/vllm/ascend_sleep.py`、`verl/workers/engine_workers.py`，以及 rollout 的 `utils.py`、`vllm_async_server.py`。

- NPU sleep level 通过 `VLLM_SLEEP_LEVEL` 配置，默认仍为 1；不是自动开启 level 2。
- MTP rollout 或 LoRA adapter 场景强制使用 level 1，保护未必通过 actor 权重同步恢复的 drafter/adapter 状态。
- 包装 `CaMemAllocator.sleep()`，在首次 unmap 前调用 `torch.npu.synchronize()`，等待所有 stream 的已提交工作。
- KV cache resume 前执行分布式 barrier，并用 `asyncio.to_thread()` 等待，避免阻塞当前 async event loop。

**依赖与边界**：承接 05 的推理运行路径；验证/训练请求的完整 admission 协议在 13 中。

### [07-rollout-task-queue.patch](verl/07-rollout-task-queue.patch)

**目的**：将外部配置的 Ascend task queue 开关传给 rollout server actor。

**涉及文件**：`verl/workers/rollout/vllm_rollout/vllm_async_server.py`。

- 从 `VLLM_ASCEND_TASK_QUEUE_ENABLE` 读取值，默认 `"0"`。
- 创建 server actor 时写入 runtime environment 的 `TASK_QUEUE_ENABLE`。
- 作用于 rollout server 的环境；不修改已排除的 `runtime_env.yaml`。

### [08-router-replay.patch](verl/08-router-replay.patch)

**目的**：将 R2/R3 的回放状态、路由行对齐、MindSpeed hash routing 和非均匀 PP 支持集中在一份补丁中。

**涉及文件**：agent loop 的 `agent_loop.py`，`verl/utils/megatron/router_replay_patch.py`、`router_replay_utils.py`，Megatron/MindSpeed engine 的 `transformer_impl.py`、MindSpeed `utils.py` 和 `engine_workers.py`。

- 六处 action 检查先判断 `enable_routing_replay`，关闭回放时不调用依赖回放配置的 helper。覆盖 R2 record、forward/backward 切换和输出收集等位置。
- 支持 sqrtsoftplus 路由分数。MindSpeed routing 含 `input_ids` 时只安装 replay 状态，不覆盖原生 hash routing；原生 MCore 路径仍安装对应 routing wrapper。
- 将 MindSpeed 的 replay 开关传到全局 Megatron args 和 config；worker 从相应 engine 配置读取 replay mode。
- 检查 rollout 路由行数，只接受总 token 数或少一个最终采样 token 的行数；默认对更短数据报错，避免错位回放。调试开关可放宽该检查。
- 回放采用 engine 实际传入的 CP layout 和最小局部行数，与模型 THD packing 保持一致。
- `num_layer_list` 统一从配置或全局 args 解析，在均分断言前处理；offset 优先用已有 layer_offset，否则累计前面 stage 的层数，并兼容不同 MCore offset 接口签名。
- 非均匀 PP 的路由 map 使用 object gather，再按 PP rank 顺序合并，支持不同 stage 的 layer 维长度。
- 去掉重复 CP layout helper，统一 DeepSeek4 的连续布局判断。

**为什么保留开关保护**：它负责“关闭回放时不执行 helper”；非均匀 PP 修复负责“启用回放时正确计算层数/offset”，两者不同。43 层分 4 个 stage 的 `[10,11,11,11]` 和 offset `[0,10,21,32]` 已验证，包括从全局 args 读取列表的情况。

### [09-training-memory-optimization.patch](verl/09-training-memory-optimization.patch)

**目的**：减少模型加载、中间训练输出和 DeepSeek4 checkpoint/RoPE 的显存驻留。

**涉及文件**：Megatron engine `transformer_impl.py`，MindSpeed engine `deepseek4_memory.py`、`transformer_impl.py`，以及 `engine_workers.py`。

- vanilla bridge 模型加载使用 `memory_efficient=True`。
- NPU training worker 在首次模型分配前开启 expandable segments；vLLM server 保留自己的 allocator 环境。
- 普通训练仅在调用方明确要求 `return_model_output` 时保留 model output；forward-only 仍保留。避免带计算图的逐 token 输出留在 forward_data_store 中直到 optimizer step。
- RoPE 表长度随当前输入需求增长；需要增长时清模型 cache，CPU offload 时清模型属性和全局 RoPE cache。
- 以 storage 弱引用跟踪 BF16、4D、需要梯度的设备端 MHC checkpoint 输入及输入梯度，不额外持有 tensor。
- 完整 backward schedule 结束后再释放已结束的存储，排除参数和参数梯度存储并去重；异常路径清理 hook/上下文，不提前释放未结束的输入。
- `forward_only` 或 MTP 启用时跳过该输入存储回收 wrapper，继续走原路径。

**依赖与边界**：基于 01 的模型构建和 02 的输入路径；回放修改只在 08。本补丁保留按需输出和 MTP 分支，不能把所有 model output 一概删除。

### [10-async-server-port-reservation.patch](verl/10-async-server-port-reservation.patch)

**目的**：解决多节点 DP server 的 IPv4/IPv6 端口探测与实际绑定不一致问题。

**涉及文件**：`verl/utils/net_utils.py`、`verl/workers/rollout/vllm_rollout/vllm_async_server.py`。

- 按服务地址族预留一组本地端口；IPv6 socket 同时覆盖 IPv4，避免另一地址族的探测再次选中该端口。
- 提供 vLLM 使用的五端口池，并设置 master port 和其余 port list。
- 异常时关闭已预留 socket；正常启动实际服务前也关闭预留 socket，避免预留动作阻塞 TCPStore/RPC 的正式 bind。

**依赖**：与 05 的 DP rank/API 兼容配合。它处理地址和端口生命周期，不处理模型权重或 router replay。

### [11-wandb-tracking.patch](verl/11-wandb-tracking.patch)

**目的**：让复杂 Hydra/OmegaConf 配置能可靠地作为实验配置写入 WandB，处理过长或重名的 key。

**涉及文件**：`verl/utils/tracking.py`。

- 将 DictConfig/ListConfig 转为普通容器，并解析 interpolation，再转换成 JSON 可序列化形式。
- 展平嵌套配置，使用点分隔 key；按既定缩写缩短长组件。
- key 上限为 64；仍过长时截断并附加原始 key 的 hash 后缀，缩短后冲突时也使用 hash 区分。
- `wandb.init()` 使用处理后的配置，保留 project、name、entity、settings 等参数。

**职责边界**：修改实验配置上传，不改每个训练 step 的 loss/reward 数值，也不生成 VALWIN 指标。

### [12-runtime-logging.patch](verl/12-runtime-logging.patch)

**目的**：提高现有运行日志的可见性。当前只修改两个文件，已经没有 `trainer_base.py` 的探针修改。

**涉及文件**：`verl/trainer/constants_ppo.py`、`verl/utils/profiler/performance.py`。

- PPO Ray runtime environment 的 `VLLM_LOGGING_LEVEL` 从 WARN 改为 INFO。
- `log_gpu_memory_usage()` 默认等级从 DEBUG 改为 WARN，并增加直接 stdout 输出。
- 内存日志包含 allocated、reserved、device used 和 total；按原有 rank 条件输出，默认关注 rank 0。
- 这些是文本日志。`[VALWIN]` 函数和调用已删除，也不再为它额外计算 response score 的 sum。

**step 指标**：原有 `trajectory_spans`、`trajectory_staleness`、`trajectory_staleness_worst` 的 mean/min/max 仍通过 `metrics.update()` 写入 step 指标，删除探针不影响这些已有指标。12 本身不增加新的标量指标。

### [13-train-validation-isolation.patch](verl/13-train-validation-isolation.patch)

**目的**：在 colocate async 模式下让验证独占推理阶段，防止训练 partial-rollout retry 在验证期间重新进入暂停的引擎，并按屏障释放训练请求。

**涉及文件**：checkpoint engine `base.py`；agent loop 的 `agent_loop.py`、`single_turn_agent_loop.py`、`tool_agent_loop.py`；Trainer `trainer_base.py`、`trainer_colocate_async.py`；rollout `llm_server.py`、`replica.py` 和 `vllm_async_server.py`。

- Trainer 预判本 step 是否验证；验证前关闭 train admission，最后通过 finally 释放。`VERL_R3_ISOLATE_VAL_TRAIN=0` 可关闭该 hold 行为。
- 共享 load balancer、client 和 server 都区分训练请求与 `validate=True` 请求；被 hold 的训练尝试在引擎外等待，不把空尝试记作有效权重版本。
- agent loop 向 server 转发 validate 标记，并在后处理中去掉重复关键字，避免 multiple-values 参数错误。
- 释放时先暂停调度但允许重入队，安装 add_request acknowledgement，等待 held requests 真正进入 engine queue 后再恢复正常准入，避免下一批新请求抢先进入。
- replica/checkpoint manager 增加对应的 hold、prepare、complete 和 release 调用，并为长时间等待提供心跳/状态日志。

**依赖与边界**：基于 05 的 PauseAwareAsyncLLM、06 的生命周期同步和 10 的 server 端口处理。它是调度/准入协议，独立于 08 的 MoE 路由回放算法；启用 profiling 时配合 14 的清理逻辑。

### [14-rollout-profiling-lifecycle.patch](verl/14-rollout-profiling-lifecycle.patch)

**目的**：修复 Trainer 实际调用的 profiler 启停，使 trace 能在 sleep 前结束，异常退出时也清理采集状态。

**涉及文件**：`verl/trainer/ppo/v1/trainer_base.py`。

- 用 `_rollout_profiler_active` 跟踪是否已启动 rollout profiler。
- 在 fit 提交请求前，根据当前 profiling step 启动 server profiler。
- step 的采样区间通过 finally 停止 rollout profiler，再执行 on_sample_end 和后续 sleep/release 流程。
- fit 的 step 执行通过 finally 调用 `_stop_profiling()`，覆盖采样前失败、训练异常等路径。
- 重复清理不会重复发送 server stop；未启用 profiling 时不发送相关 profiler RPC。

**保留原因**：上述方法由训练循环直接调用，不是独立脚本。独立的 `scripts/collect_npu_profile.py` 已排除。此补丁承接 13 的 fit 重排，使用全套顺序应用；已验证启用、关闭、重复清理以及两条异常 finally 路径。

### [15-speculative-metrics-aggregation.patch](verl/15-speculative-metrics-aggregation.patch)

**目的**：让 MTP/speculative decode 的计数在异步分段生成后完整进入训练统计。

**涉及文件**：`verl/trainer/ppo/v1/trainer_base.py`、`verl/workers/rollout/llm_server.py`。

- Trainer 从 `actor_rollout_ref.rollout.mtp` 读取开关，替代原来的 model 配置路径；只有 rollout 启用相应统计时才取 extra_fields。
- 每个 resume 都是新的后端请求，将分段的 `spec_num_draft_tokens`、`spec_num_accepted_tokens`、`spec_num_verify_steps` 累加到最终输出，避免只保留最后一段。
- 后端格式转换在 05 中；这里使用转换后的统一字段，不再重复处理版本差异。
- 原有 `compute_spec_decode_metrics()` 消费这些计数并写入 step 指标。它与已删除的 VALWIN 文本探针不同，属于有效指标链路。

**依赖**：需要 05 产出后端统计格式，并承接 13 的异步 client；也可与 vLLM 仓库的 abort-spec-decode-metrics 补丁配合，保留被中止后端请求已经产生的计数。

### [16-strict-checkpoint-resume.patch](verl/16-strict-checkpoint-resume.patch)

**目的**：只恢复完整发布的 checkpoint，并把训练、dataloader、TransferQueue 与 Megatron DCP 状态作为同一个原子发布单元。

**涉及文件**：v1 三种 Trainer、Megatron checkpoint manager、`verl/utils/megatron/dist_checkpointing.py`、Megatron engine 和 rollout replica。

- 保存前使旧的完成标记失效；DCP、`data.pt` 和 TransferQueue 全部成功后原子写入 `ckpt_complete.json` 与 latest step。
- 自动恢复遇到 partial checkpoint 时回退前一个完整 step；严格模式缺少 dataloader 或 TransferQueue 状态直接失败。
- 禁止没有 CPU snapshot 保障的 `actor.checkpoint.async_save=True`，同步路径采用进程内逐 tensor writer。
- plan、write、finalize 分阶段汇总所有 rank 的错误；任一阶段失败都不发布半成品。
- 修复 DeepSeek-V4 异构 PP layer key、replica metadata 和 optimizer sharded state；保存/加载 checkpoint 时不搬运无关梯度。
- async trainer 恢复在途 prompt 后跳过重复 warmup，并为 rollout 控制 RPC 保留额外 actor 并发余量。
- checkpoint retention 只清理已经发布且不受当前恢复源保护的旧 step。


## A3 补丁应用与验证

各目录编号独立，请分别在对应仓库的干净 checkout 中应用。先 checkout 该仓库上述基线 SHA，再将 `PATCH_DIR` 指向对应目录，例如在 verl 仓库根目录执行：

```bash
git checkout 91c297dc8d9991b79c90cad5b50683a5d6d8c02b
PATCH_DIR=/mnt/bn/lrm-hl/zhibei.ye/code/verl-ascend-recipe/DeepSeek-V4-Flash/patch/A3/verl
for patch in "$PATCH_DIR"/*.patch; do
    git apply --check "$patch" || break
    git apply --whitespace=nowarn "$patch" || break
done
```

verl 已按最终功能重新分组，功能补丁之间共享源码上下文；在上述基线上按 01–16 顺序应用整套。

原有 vLLM、vLLM-Ascend 和 verl 01–15 补丁均已在干净临时 checkout 中逐个验证 `git apply` 和 `git am`。新增的四份严格断点续训补丁在当前 A3 前序代码树上通过 `git apply --check`、实际应用、`git apply --reverse --check` 和 Python 语法检查；重命名后的 A3 文件与 `07_checkpoint_single_patch` 原文件逐字一致。

另外执行了隔离的必要行为检查：非均匀 PP 层数/offset（含全局参数 fallback）、关闭回放的六个短路保护、profiler 启停与异常清理，以及 vLLM completion histogram/旧请求统计格式。未执行 NPU 推理或训练。

原仓库的 HEAD、index 和未提交修改均保持不变。验证覆盖补丁应用、内容重建和语法，未执行 NPU 推理或训练。详细逐步 tree、校验和与结果见 [验证记录](/opt/tiger/code/a3-vllm-ascend-verl-patch-validation.json)。

2026-11-17 已重新验证移除 VALWIN 后的整套 15 个 verl 补丁，并确认原有 step 指标 key 保持不变；12 及后续 Trainer 补丁上下文已经重新生成。
