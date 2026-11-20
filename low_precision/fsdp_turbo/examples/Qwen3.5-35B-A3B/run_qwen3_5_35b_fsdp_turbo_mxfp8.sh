#!/usr/bin/env bash
# dependency: GPU vllm==0.30.0, transformers==5.22.2
# dependency: NPU vllm==0.30.0, vllm-ascend v0.30.0 官方基点 @59ebc4384＋recipe patch 层, transformers==5.22.2
# Qwen3.5-35B-A3B GRPO 低精 launcher：FSDPTurbo 训练侧动态 MXFP8 ＋ vLLM-Ascend 推理侧 MXFP8 权重加载。
# 与高精件 run_qwen3_5_35b_fsdp_turbo_a5.sh 成对，差异只在量化面：
#   训练侧＝FSDPTurbo 动态 MXFP8 开关（本文件 gate 组，默认开）；
#   推理侧＝rollout.quantization 行＋MXFP8 权重目录（用 tool/smart_create_quant_weight_dir.py 生成，
#   排除 linear_attn，原因见 readme.md）。
# E 六参（MODEL_NAME＋use_fused_causal_conv1d/use_fused_gdn × actor/ref）已烙进本件（与高精件同款同位，
# 属两精度共用建模层底座）；外部点火层勿再追加——撞本件 + 键会在 hydra config append 处 fail-fast。
# 机器适配：weight_nz_mode=0 已烙进 npu 块（rollout 追加段），无需点火口令再传。
# bf16 复现臂（run67 型）：QUANT_MODE 显式置空（空串走条件门省略 quantization 行）＋
#   MXFP8=0 ＋ MODEL_PATH 指裸 bf16 目录，见 readme「bf16 双侧高精对照臂」。
# mm-键：npu 块 mm_device_do_normalize=false 为训推一致性修复键，高低精 launcher 均保留。
set -xeuo pipefail

# 本地路径配置层（用户令 11-19）：launcher 自身保持「环境变量读取＋通用默认」、零真实内部路径；
# 本机真实值可放同目录 env.local.sh（仓库不提供该文件，由使用者按需自建）。
# 存在才 source，且置于 user-adjustable 段之前：env.local.sh 已设的项优先生效，未设的回落通用默认。
LOCAL_ENV_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.local.sh"
if [ -f "${LOCAL_ENV_SH}" ]; then
    set -a
    # shellcheck disable=SC1090
    . "${LOCAL_ENV_SH}"
    set +a
fi

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3_5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Qwen3_5-35B}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-4}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-8}
FSDP_SIZE=${FSDP_SIZE:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-35B-A3B-MXFP8"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# ModelSlim 原生字符串路由保持（PR #8511 原样；模板修复在导出器侧，不动路由）。
# 注意用 "-" 不用 ":-"：仅 unset 才回落 ascend，显式置空（QUANT_MODE=）被保留，
# 供条件门省略 quantization 行＝bf16 复现臂（"："型会把空串换成 ascend，置空失效）。
QUANT_MODE=${QUANT_MODE-ascend}
# 训练侧 FSDPTurbo 动态 MXFP8 开关：1=开（actor+ref），0=纯 bf16 对照
MXFP8=${MXFP8:-1}
# 推理侧在线 MXFP8 重载门（vllm-ascend-scalefix.patch 的 env 开关，默认关）：
# bf16 盘上权重在引擎加载时在线量化，RL 每步 refit 复用同链。不设＝scalefix 装了但门关着
# （静默跑 bf16），因此这里默认 1 并在 npu 块显式 export。
VLLM_ASCEND_ONLINE_WEIGHT_QUANT=${VLLM_ASCEND_ONLINE_WEIGHT_QUANT:-1}
########################### MXFP8 quantization plan (训练侧) ###########################
QUANT_APPLY_MODULES="['model.language_model.layers.{*}']"
QUANT_IGNORED_MODULES="['model.language_model.layers.{*}.linear_attn.*']"
QUANT_CONVERTERS="['quantize.linear.mx','quantize.moe.mx']"
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}
fsdp_size=${FSDP_SIZE:-8}
ep_dispatcher=all2all
ep_use_fused_ops=False

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        # 内存/加载稳定性 env（env.local.sh 或外部已设则优先生效）：
        # PYTORCH_NPU_ALLOC_CONF 防宿主内存压力下 DataLoader worker 被 SIGKILL；
        # HF_DEACTIVATE_ASYNC_LOAD 关闭异步加载竞态
        export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:1024}
        export HF_DEACTIVATE_ASYNC_LOAD=${HF_DEACTIVATE_ASYNC_LOAD:-1}
        # 量化三件之一：scalefix env 门显式 export（ray 子进程继承链）
        export VLLM_ASCEND_ONLINE_WEIGHT_QUANT
        n_devices_per_node=8
        fsdp_size=8
        EP_SIZE=8
        ep_use_fused_ops=True
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

########################### shared turbo config values ###########################
ACTOR_TURBO="actor_rollout_ref.actor.fsdp_config.turbo_config"
REF_TURBO="actor_rollout_ref.ref.fsdp_config.turbo_config"

MODEL_NAME="qwen3_5_moe"

FSDP_APPLY_MODULES='{model.visual.blocks.\{*\}:{},'\
'model.visual.patch_embed:{},'\
'model.visual.pos_embed:{},'\
'model.language_model.embed_tokens:{},'\
'model.language_model.layers.\{*\}:{},'\
'lm_head:{}}'

HOOK_MODULES="['model.language_model.layers.{*}']"
EXPERT_APPLY_MODULES="['model.language_model.layers.{*}.mlp.experts']"
RECOMPUTE_PLAN="['model.language_model.layers.{*}','model.visual.blocks.{*}']"

ULYSSES_FUNCTION_PATCHES="[{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.eager_attention_forward'],"\
"type:full_attention},"\
"{target_functions:['transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet.forward'],"\
"type:gated_delta_net}]"

LOSS_FUNCTION_PATCHES="[{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],"\
"type:causal_lm_loss}]"

MODULE_PATCHES="[{target:transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward,"\
"replacement:fsdp_turbo.models.qwen.qwen3_5_moe.qwen3_5_moe_model_forward}]"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=16
    data.max_prompt_length=2048
    data.max_response_length=6144
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.strategy=fsdp_turbo
    "${ACTOR_TURBO}.distributed.fully_shard_parallel_size=${fsdp_size}"
    "${ACTOR_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${ACTOR_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${ACTOR_TURBO}.distributed.ep_plan.apply_modules=${EXPERT_APPLY_MODULES}"
    "+${ACTOR_TURBO}.distributed.ep_plan.dispatcher=${ep_dispatcher}"
    "++${ACTOR_TURBO}.distributed.ep_plan.use_fused_ops=${ep_use_fused_ops}"
    "+${ACTOR_TURBO}.memory.recompute=True"
    "+${ACTOR_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
    "+${ACTOR_TURBO}.model.use_fused_causal_conv1d=True"
    "+${ACTOR_TURBO}.model.use_fused_gdn=True"
    "+${ACTOR_TURBO}.model.model_name_or_path=${MODEL_NAME}"
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.strategy=fsdp_turbo
    "${REF_TURBO}.distributed.fully_shard_parallel_size=${fsdp_size}"
    "${REF_TURBO}.distributed.expert_parallel_size=${EP_SIZE}"
    "${REF_TURBO}.distributed.ulysses_parallel_size=${SP_SIZE}"
    "+${REF_TURBO}.distributed.fsdp_plan.apply_modules=${FSDP_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.fsdp_plan.hook_modules=${HOOK_MODULES}"
    "+${REF_TURBO}.distributed.ep_plan.apply_modules=${EXPERT_APPLY_MODULES}"
    "+${REF_TURBO}.distributed.ep_plan.dispatcher=${ep_dispatcher}"
    "++${REF_TURBO}.distributed.ep_plan.use_fused_ops=${ep_use_fused_ops}"
    "+${REF_TURBO}.memory.recompute=True"
    "+${REF_TURBO}.memory.recompute_plan=${RECOMPUTE_PLAN}"
    "+${REF_TURBO}.model.use_fused_causal_conv1d=True"
    "+${REF_TURBO}.model.use_fused_gdn=True"
    "+${REF_TURBO}.model.model_name_or_path=${MODEL_NAME}"
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

########################### conditional cp_plan & module_patches ###########################
if [ "${SP_SIZE}" -gt 1 ]; then
    ACTOR+=(
        "+${ACTOR_TURBO}.distributed.cp_plan.ulysses_function_patches=${ULYSSES_FUNCTION_PATCHES}"
        "+${ACTOR_TURBO}.distributed.cp_plan.loss_function_patches=${LOSS_FUNCTION_PATCHES}"
        "+${ACTOR_TURBO}.module_patches=${MODULE_PATCHES}"
    )
    REF+=(
        "+${REF_TURBO}.distributed.cp_plan.ulysses_function_patches=${ULYSSES_FUNCTION_PATCHES}"
        "+${REF_TURBO}.distributed.cp_plan.loss_function_patches=${LOSS_FUNCTION_PATCHES}"
        "+${REF_TURBO}.module_patches=${MODULE_PATCHES}"
    )
fi

########################### training-side FSDPTurbo dynamic MXFP8 ###########################
if [ "${MXFP8}" = "1" ]; then
    ACTOR+=(
        "+${ACTOR_TURBO}.quantization.quantization_plan.quant_recipe=mxfp8"
        "+${ACTOR_TURBO}.quantization.quantization_plan.quant_format=E4M3"
        "+${ACTOR_TURBO}.quantization.quantization_plan.converters=${QUANT_CONVERTERS}"
        "+${ACTOR_TURBO}.quantization.quantization_plan.quant_apply_modules=${QUANT_APPLY_MODULES}"
        "+${ACTOR_TURBO}.quantization.quantization_plan.quant_ignored_modules=${QUANT_IGNORED_MODULES}"
        "+${ACTOR_TURBO}.quantization.quantization_plan.enable_fsdp_low_precision_all_gather=False"
        "+${ACTOR_TURBO}.quantization.quantization_plan.fsdp_low_precision_all_gather_mode=on-demand"
    )
    REF+=(
        "+${REF_TURBO}.quantization.quantization_plan.quant_recipe=mxfp8"
        "+${REF_TURBO}.quantization.quantization_plan.quant_format=E4M3"
        "+${REF_TURBO}.quantization.quantization_plan.converters=${QUANT_CONVERTERS}"
        "+${REF_TURBO}.quantization.quantization_plan.quant_apply_modules=${QUANT_APPLY_MODULES}"
        "+${REF_TURBO}.quantization.quantization_plan.quant_ignored_modules=${QUANT_IGNORED_MODULES}"
        "+${REF_TURBO}.quantization.quantization_plan.enable_fsdp_low_precision_all_gather=False"
        "+${REF_TURBO}.quantization.quantization_plan.fsdp_low_precision_all_gather_mode=on-demand"
    )
    echo "[MXFP8] 训练侧 (actor/ref) FSDPTurbo 动态 MXFP8 已启用；rollout 引擎走 vllm mxfp8（quantization=${QUANT_MODE}）"
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,12,16,24,32,48,56,64]"
)

# ===== ModelSlim 导出器模板铺满（路由 ascend 不变，PR #8511 原样）=====
# 条件门：QUANT_MODE 非空才追加 quantization 行，置空＝整行省略（引擎不挂
# W8A8_MXFP8，裸 bf16 权重目录无需 quant_model_description.json）＝bf16 复现臂。
if [ -n "${QUANT_MODE}" ]; then
    ROLLOUT+=(++actor_rollout_ref.rollout.engine_kwargs.vllm.quantization=${QUANT_MODE})
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.resume_from_path=checkpoints/
    trainer.val_before_train=True
    trainer.save_freq=-1
    trainer.test_freq=10
    trainer.total_epochs=100
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_device_do_normalize=false
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048
            # 机器适配：weight_nz_mode=0（原点火口令项收编进 npu 块，含高精件）
            +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee logs/qwen3_5-35b-exp08-pr8511-${start_time}.log
