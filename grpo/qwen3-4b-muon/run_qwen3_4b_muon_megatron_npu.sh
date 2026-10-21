#!/usr/bin/env bash
# GRPO | Qwen3-4B | Muon | Megatron actor | vLLM-Ascend rollout | Ascend NPU

set -xeuo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
DEVICE=${DEVICE:-npu}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-4B checkout}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to a data directory}
VERL_ROOT=${VERL_ROOT:?Set VERL_ROOT to the current upstream verl checkout}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ACTOR_TP=${ACTOR_TP:-2}
ACTOR_PP=${ACTOR_PP:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-48}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-20000000000}

GSM8K_DIR=${GSM8K_DIR:-${DATA_ROOT}/gsm8k}
MATH_DIR=${MATH_DIR:-${DATA_ROOT}/math}
TRAIN_FILE_GSM8K=${TRAIN_FILE_GSM8K:-${GSM8K_DIR}/train.parquet}
VAL_FILE_GSM8K=${VAL_FILE_GSM8K:-${GSM8K_DIR}/test.parquet}
TRAIN_FILE_MATH=${TRAIN_FILE_MATH:-${MATH_DIR}/train.parquet}
VAL_FILE_MATH=${VAL_FILE_MATH:-${MATH_DIR}/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-${DATA_ROOT}/checkpoints/qwen3-4b-muon}
LOG_FILE=${LOG_FILE:-${DATA_ROOT}/logs/qwen3-4b-muon/train.log}
RAY_TMPDIR=${RAY_TMPDIR:-${DATA_ROOT}/ray-tmp}
RUNTIME_COMPAT_DIR=${RUNTIME_COMPAT_DIR:-${DATA_ROOT}/runtime/qwen3-4b-muon}
RUNTIME_PYTHON=${RUNTIME_PYTHON:-${RUNTIME_COMPAT_DIR}/python}

if [[ ! -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]]; then
    echo "VERL_ROOT does not contain the upstream verl trainer: ${VERL_ROOT}" >&2
    exit 1
fi
if [[ "${DEVICE}" != npu ]]; then
    echo "This recipe is intended for DEVICE=npu" >&2
    exit 1
fi

export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RAY_DATA_HOME=${DATA_ROOT}
export TMPDIR=${TMPDIR:-${DATA_ROOT}/tmp}
mkdir -p "${TMPDIR}"
export RAY_TMPDIR
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
unset RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "${GSM8K_DIR}" "${MATH_DIR}" "${CKPTS_DIR}" "$(dirname "${LOG_FILE}")" "${RAY_TMPDIR}" "${RUNTIME_COMPAT_DIR}"

# The official MindSpeed wheel in the NPU environment predates the current
# Megatron-Bridge provider fields.  Keep the compatibility shim in the recipe
# runtime; installed packages and the upstream checkout remain untouched.
cat > "${RUNTIME_COMPAT_DIR}/sitecustomize.py" <<'PY'
"""Lazy compatibility hook for the installed MindSpeed/Megatron bridge."""
import importlib.abc
import importlib.machinery
import sys


_TARGET = "megatron.bridge.models.gpt_provider"
_OPT_TARGET = "megatron.core.optimizer"
_ENGINE_TARGET = "verl.workers.engine.megatron.transformer_impl"
_DEFAULTS = {
    "moe_zero_memory_num_layers": None,
    "moe_zero_memory": "disable",
    "moe_alltoall_overlap_comm": False,
    "moe_fb_overlap": False,
    "optimizer_selection": None,
    "ema_decay": 0.9999,
    "noop_layers": None,
    "use_ascend_coc": False,
    "use_ascend_mc2": False,
    "use_gmm_fp8": True,
    "fp8_reuse_quantized_weight": False,
    "context_parallel_kv_cache_policy": None,
    "context_parallel_cache_interval": 0,
    "use_ulysses_allgather_kv": False,
    "use_fused_rotary_pos_emb": False,
    "op_cal_tflops": False,
    "recompute_activation_function_num_layers": None,
    "recompute_norm_num_layers": None,
    "use_fused_moe_token_permute_and_unpermute": False,
    "unaligned_linear": False,
    "multi_head_latent_attention": False,
    "multi_latent_attention": False,
    "group_query_attention": False,
    "adaptive_recompute_device_swap": False,
    "enable_fine_grained_expert_placement": False,
    "fix_router": False,
    "gemm_gradient_accumulation_fusion": False,
    "moe_allgather_overlap_comm": False,
    "moe_alltoall_mc2": False,
    "moe_permutation_async_comm": False,
    "moe_tp_extend_ep": False,
    "moe_unperm2_mem_optim_swap": False,
    "quant_states_enabled": False,
    "recompute_in_advance": False,
    "recompute_in_bubble": False,
    "reduce_recompute_for_last_chunk": False,
    "reset_attention_mask": False,
    "reuse_fp32_param": False,
    "smart_swap": False,
    "sparse_mode": False,
    "swap_attention": False,
    "ttp_enabled": False,
    "pipeline_num_transformer_layers": None,
    "num_layers_per_virtual_pipeline_stage": None,
    "schedules_method": None,
    "n_shared_experts": None,
    "moe_shared_expert_intermediate_size": None,
    "virtual_optimizer": None,
    "dist_train": False,
    "tokenizer_name_or_path": None,
    "tokenizer_type": None,
    "param_and_grad_buffer_pad": None,
    "compress_dense": "disable",
    "compress_activation": "",
    "aiqos": False,
    "aiqos_mode": None,
    "aiqos_schedule": None,
    "attention_mask_type": "causal",
    "alibi_fusion_attn_type": None,
    "auto_settings": False,
    "context_parallel_algo": None,
    "cp_window_size": None,
    "data_parallel_size": None,
    "expert_placement_freq": None,
    "fine_grained_expert_placement_thre": None,
    "global_batch_size": None,
    "micro_batch_size": None,
    "moe_expert_capacity_factor": None,
    "moe_per_layer_logging": False,
    "npu_datadump": False,
    "npu_deterministic": False,
    "num_virtual_stages_per_pipeline_rank": None,
    "quant_grads": False,
    "quant_grads_dtype": None,
    "ring_degree": None,
    "tp_2d": False,
    "tp_x": None,
    "tp_y": None,
    "untie_embeddings_and_output_weights": False,
    "use_custom_fsdp": False,
    "use_dsa_absorb": False,
    "use_fused_lightning_indexer": False,
    "use_fused_lightning_indexer_kl_loss": False,
    "use_fused_sparse_flash_attention": False,
    "use_fusion_attn_v2": False,
    "use_g2_attention": False,
    "use_quant_optimizer": False,
    "use_legacy_models": False,
    "ulysses_degree_in_cp": None,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, fullname, loader):
        self._fullname = fullname
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        provider = getattr(module, "GPTModelProvider", None)
        if provider is not None:
            for name, value in _DEFAULTS.items():
                if not hasattr(provider, name):
                    setattr(provider, name, value)
        if self._fullname == _OPT_TARGET and hasattr(module, "get_megatron_optimizer"):
            native = module.get_megatron_optimizer
            from megatron.core.optimizer.muon import get_megatron_muon_optimizer

            def _dispatch(config, model_chunks, config_overrides=None,
                          use_gloo_process_groups=True, pg_collection=None,
                          dump_param_to_param_group_map=None):
                if str(getattr(config, "optimizer", "")).lower() == "muon":
                    return get_megatron_muon_optimizer(
                        config=config,
                        model_chunks=model_chunks,
                        config_overrides=config_overrides,
                        use_gloo_process_groups=use_gloo_process_groups,
                        layer_wise_distributed_optimizer=False,
                        pg_collection=pg_collection,
                    )
                return native(
                    config=config,
                    model_chunks=model_chunks,
                    config_overrides=config_overrides,
                    use_gloo_process_groups=use_gloo_process_groups,
                    pg_collection=pg_collection,
                    dump_param_to_param_group_map=dump_param_to_param_group_map,
                )

            module.get_megatron_optimizer = _dispatch
        if self._fullname == _ENGINE_TARGET and hasattr(module, "load_megatron_model_to_gpu"):
            native_load = module.load_megatron_model_to_gpu

            def _safe_load(models, load_grad=True, load_frozen_params=True):
                try:
                    return native_load(models, load_grad=load_grad, load_frozen_params=load_frozen_params)
                except AttributeError as exc:
                    if "storage" not in str(exc) or "NoneType" not in str(exc):
                        raise
                    # The non-layerwise DDP path on older Megatron-Core leaves
                    # param_data unset; weights are already resident on NPU.
                    for model_chunk in models:
                        for param in model_chunk.parameters():
                            if param.device.type != "npu":
                                param.data = param.data.to("npu")

            module.load_megatron_model_to_gpu = _safe_load


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in {_TARGET, _OPT_TARGET, _ENGINE_TARGET}:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(fullname, spec.loader)
        return spec


sys.meta_path.insert(0, _Finder())
PY
cat > "${RUNTIME_PYTHON}" <<EOF
#!/usr/bin/env bash
export PYTHONPATH="${RUNTIME_COMPAT_DIR}:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" "\$@"
EOF
chmod +x "${RUNTIME_PYTHON}"
export PYTHONPATH="${RUNTIME_COMPAT_DIR}:${PYTHONPATH}"
if [[ ! -f "${TRAIN_FILE_GSM8K}" || ! -f "${VAL_FILE_GSM8K}" ]]; then
    "${PYTHON_BIN}" "${VERL_ROOT}/examples/data_preprocess/gsm8k.py" \
        --local_save_dir "${GSM8K_DIR}"
fi
if [[ ! -f "${TRAIN_FILE_MATH}" || ! -f "${VAL_FILE_MATH}" ]]; then
    "${PYTHON_BIN}" "${VERL_ROOT}/examples/data_preprocess/math_dataset.py" \
        --local_save_dir "${MATH_DIR}"
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE_GSM8K}','${TRAIN_FILE_MATH}']"
    data.val_files="['${VAL_FILE_GSM8K}','${VAL_FILE_MATH}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.optimizer=muon
    actor_rollout_ref.actor.optim.use_layer_wise_distributed_optimizer=False
    actor_rollout_ref.actor.optim.muon_momentum=0.95
    actor_rollout_ref.actor.optim.muon_nesterov=False
    actor_rollout_ref.actor.optim.muon_split_qkv=True
    actor_rollout_ref.actor.optim.muon_scale_mode=spectral
    actor_rollout_ref.actor.optim.muon_coefficient_type=quintic
    actor_rollout_ref.actor.optim.muon_num_ns_steps=5
    actor_rollout_ref.actor.optim.muon_tp_mode=blockwise
    actor_rollout_ref.actor.optim.muon_match_adamw_update_rms=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=False
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,16,32,64,128,256]"
)

TRAINER=(
    trainer.logger=['console']
    trainer.project_name=qwen3-4b-muon
    trainer.experiment_name=qwen3-4b-muon-megatron-npu
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=disable
    trainer.resume_from_path=null
    trainer.val_before_train=False
    trainer.max_actor_ckpt_to_keep=1
    trainer.save_freq=${TOTAL_TRAINING_STEPS}
    trainer.test_freq=10
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

cd "${VERL_ROOT}"
set +e
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" \
    "${ROLLOUT[@]}" "${TRAINER[@]}" model_engine=megatron \
    ray_kwargs.ray_init.runtime_env.py_executable="${RUNTIME_PYTHON}" \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
    +ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY} \
    "$@" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
