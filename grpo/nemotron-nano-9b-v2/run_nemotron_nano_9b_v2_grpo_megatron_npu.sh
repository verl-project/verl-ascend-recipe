#!/usr/bin/env bash
set -xeo pipefail

################################################### environment ###################################################
# Adapted from verl's canonical Nemotron Nano v3 Megatron recipe:
# examples/grpo_trainer/run_nemotron_nano_v3_30b_a3b_megatron.sh
# Run in an official verl NPU environment with CANN, torch-npu,
# vLLM-Ascend, Megatron-LM, MindSpeed, and Megatron-Bridge installed.

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
set -u

export DEVICE=npu
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export TRAINING_BACKEND=verl

# vLLM 0.23.0's Mamba prefill Triton kernel is not supported by the official
# Triton-Ascend 3.2.2 wheel. MindSpeed-LLM provides the official pure PyTorch
# NPU SSD implementation. This temporary module bridges the prefill API and
# provides a graph-safe NPU tensor implementation for one-token Mamba decode;
# it does not modify an installed package.
# Install MindSpeed-LLM from https://github.com/Ascend/MindSpeed-LLM first.
mamba_npu_compat_dir=$(mktemp -d "${TMPDIR:-/tmp}/verl-mamba-npu.XXXXXX")
cat > "${mamba_npu_compat_dir}/verl_mamba_npu_compat.py" <<'PY'
import torch
import torch_npu
import triton

if not hasattr(triton, "set_allocator"):
    triton.set_allocator = lambda allocator: None
torch.cuda.device = torch.npu.device
torch.cuda.get_device_properties = torch.npu.get_device_properties

import mindspeed.megatron_adaptor
import mindspeed_llm.tasks.models.ssm.state_space_duality as ssd


class _SingleContextParallel:
    get_context_parallel_world_size = staticmethod(lambda: 1)
    get_context_parallel_rank = staticmethod(lambda: 0)
    get_context_parallel_group = staticmethod(lambda: None)


# The recipe fixes Megatron context parallelism at one. Rollout workers do not
# initialize Megatron parallel state, so isolate that fact to this SSD module.
ssd.mpu = _SingleContextParallel()


def _run_ssd(x, dt, A, B, C, chunk_size, D, dt_bias, initial_states,
             dt_softplus, dt_limit, return_state):
    if not dt_softplus or D is None or D.ndim != 1 or dt_bias is None:
        raise NotImplementedError("This NPU adapter supports the Nemotron Mamba2 layout only")
    options = ssd.StateOptions(
        initial_states=initial_states, return_final_state=return_state
    )
    processor = ssd.StateSpaceProcessor(
        {
            "nheads_local": x.shape[2],
            "ngroups_local": B.shape[2],
            "dt_min": dt_limit[0],
            "dt_max": dt_limit[1],
            "dt_bias": dt_bias,
            "headdim": x.shape[3],
            "d_state": B.shape[3],
            "chunk_size": chunk_size,
            "D_has_hdim": False,
        }
    )
    result = processor.process(
        ssd.ProcessInputs(
            x=x.flatten(2),
            dt=dt,
            A=A,
            B=B.flatten(2),
            C=C.flatten(2),
            D=D,
        ),
        options,
    )
    if return_state:
        y, state = result
        return y.to(x.dtype), state
    return result.to(x.dtype), None


def _mcore_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
                initial_states=None, seq_idx=None, cu_seqlens=None,
                dt_softplus=False, dt_limit=(0.0, float("inf")),
                return_final_states=False, return_varlen_states=False,
                state_dtype=None):
    if z is not None or seq_idx is not None or cu_seqlens is not None:
        raise NotImplementedError("Packed Megatron Mamba training is not enabled")
    if return_varlen_states:
        raise NotImplementedError("Variable-length Megatron state return is not enabled")
    y, state = _run_ssd(
        x, dt, A, B, C, chunk_size, D, dt_bias, initial_states,
        dt_softplus, dt_limit, return_final_states,
    )
    return (y, state) if return_final_states else y


_plan_input = None
_plan = None


def _varlen_plan(cu_seqlens):
    global _plan_input, _plan
    if cu_seqlens is _plan_input:
        return _plan
    boundaries = cu_seqlens.detach().cpu().tolist()
    length_values = [
        end - start for start, end in zip(boundaries, boundaries[1:])
    ]
    if not length_values or min(length_values) <= 0:
        raise ValueError("Mamba prefill requires non-empty sequences")
    starts = torch.tensor(boundaries[:-1], device=cu_seqlens.device)
    lengths = torch.tensor(length_values, device=cu_seqlens.device)
    positions = torch.arange(max(length_values), device=cu_seqlens.device)
    gather = starts[:, None] + positions[None, :]
    valid = positions[None, :] < lengths[:, None]
    _plan_input, _plan = cu_seqlens, (gather, valid)
    return _plan


def _vllm_varlen(x, dt, A, B, C, chunk_size, cu_seqlens,
                 cu_chunk_seqlens, last_chunk_indices, seq_idx, out, D=None,
                 z=None, dt_bias=None, initial_states=None, dt_softplus=False,
                 dt_limit=(0.0, float("inf")), return_intermediate_states=False,
                 state_dtype=None):
    if z is not None or return_intermediate_states:
        raise NotImplementedError("Mamba prefix caching is disabled by this recipe")
    gather, valid = _varlen_plan(cu_seqlens)
    safe = gather.clamp(max=x.shape[0] - 1)
    y, states = _run_ssd(
        torch.where(valid[..., None, None], x[safe], 0),
        torch.where(valid[..., None], dt[safe], -100.0),
        A,
        torch.where(valid[..., None, None], B[safe], 0),
        torch.where(valid[..., None, None], C[safe], 0),
        chunk_size, D, dt_bias, initial_states, dt_softplus, dt_limit, True,
    )
    out.index_copy_(0, gather[valid], y[valid].to(out.dtype))
    return states.to(state_dtype or C.dtype)


def _npu_selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    z=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    null_block_id=-1,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    is_blackwell=False,
):
    """NPU implementation of vLLM's non-speculative Mamba2 decode step."""
    import torch.nn.functional as F

    del is_blackwell
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is None:
        raise ValueError("vLLM Mamba2 decode requires a preallocated output")
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)
    if dst_state_batch_indices is not None and dst_state_batch_indices.dim() == 1:
        dst_state_batch_indices = dst_state_batch_indices.unsqueeze(1)

    cache_batch, nheads, headdim, dstate = state.shape
    token_count = x.shape[0]
    sequence_count = len(cu_seqlens) - 1 if cu_seqlens is not None else token_count
    if token_count != sequence_count:
        raise NotImplementedError(
            "The issue-70 recipe supports vLLM's non-speculative one-token decode"
        )
    if x.shape != (token_count, nheads, headdim) or dt.shape != x.shape:
        raise ValueError("Unexpected Mamba2 decode x/dt shapes")
    if A.shape != (nheads, headdim, dstate):
        raise ValueError("Unexpected Mamba2 decode A shape")
    ngroups = B.shape[1]
    if B.shape != (token_count, ngroups, dstate) or C.shape != B.shape:
        raise ValueError("Unexpected Mamba2 decode B/C shapes")
    if nheads % ngroups or D.shape != (nheads, headdim):
        raise ValueError("Unexpected Mamba2 decode head/group shapes")
    if dt_bias.shape != (nheads, headdim) or out.shape != x.shape:
        raise ValueError("Unexpected Mamba2 decode bias/output shapes")
    if z is not None and z.shape != x.shape:
        raise ValueError("Unexpected Mamba2 decode gate shape")
    if num_accepted_tokens is not None and num_accepted_tokens.shape != (sequence_count,):
        raise ValueError("Unexpected Mamba2 accepted-token metadata shape")

    if state_batch_indices is None:
        source_indices = torch.arange(sequence_count, device=x.device)
    else:
        source_indices = state_batch_indices[:, 0].to(device=x.device, dtype=torch.long)
    if dst_state_batch_indices is None:
        destination_indices = source_indices
    else:
        destination_indices = dst_state_batch_indices[:, 0].to(
            device=x.device, dtype=torch.long
        )
    if source_indices.numel() != sequence_count or destination_indices.numel() != sequence_count:
        raise ValueError("Unexpected Mamba2 state-index metadata shape")

    valid = (source_indices != null_block_id) & (destination_indices != null_block_id)
    safe_sources = source_indices.clamp(min=0, max=cache_batch - 1)
    current_state = state.index_select(0, safe_sources).float()
    current_state = current_state * valid[:, None, None, None]

    heads_per_group = nheads // ngroups
    b_heads = B.repeat_interleave(heads_per_group, dim=1).float()
    c_heads = C.repeat_interleave(heads_per_group, dim=1).float()

    # Nemotron expands these tensors over the head dimension. Match vLLM's
    # tied-head fast path without materializing the repeated values.
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )
    if tie_hdim:
        dt_scalar = dt[..., 0].float() + dt_bias[..., 0].float().unsqueeze(0)
        if dt_softplus:
            dt_scalar = F.softplus(dt_scalar)
        decay = torch.exp(A[:, 0, 0].float().unsqueeze(0) * dt_scalar)
        x_discrete = x.float() * dt_scalar.unsqueeze(-1)
        next_state = current_state * decay[:, :, None, None]
        next_state = next_state + x_discrete.unsqueeze(-1) * b_heads.unsqueeze(2)
    else:
        dt_fp32 = dt.float() + dt_bias.float().unsqueeze(0)
        if dt_softplus:
            dt_fp32 = F.softplus(dt_fp32)
        decay = torch.exp(A.float().unsqueeze(0) * dt_fp32.unsqueeze(-1))
        next_state = current_state * decay
        next_state = next_state + (
            x.float().unsqueeze(-1) * dt_fp32.unsqueeze(-1) * b_heads.unsqueeze(2)
        )

    output = (next_state * c_heads.unsqueeze(2)).sum(dim=-1)
    output = output + x.float() * D.float().unsqueeze(0)
    if z is not None:
        output = output * F.silu(z.float())
    output = output * valid[:, None, None]
    out.copy_(output.to(out.dtype))

    # Avoid aclnnNonzero, which synchronizes the stream and cannot be captured
    # in an Ascend graph. Padded rows use cache row zero, restored after write.
    safe_destinations = destination_indices.clamp(min=0, max=cache_batch - 1)
    row_zero_before = state[0].clone()
    row_zero_mask = valid & (destination_indices == 0)
    row_zero_candidate = (
        next_state * row_zero_mask[:, None, None, None]
    ).sum(dim=0)
    row_zero_after = torch.where(
        row_zero_mask.any(), row_zero_candidate, row_zero_before
    )
    state.index_copy_(0, safe_destinations, next_state.to(state.dtype))
    state[0].copy_(row_zero_after.to(state.dtype))


from megatron.core.ssm import mamba_mixer as mcore_mamba
from vllm.model_executor.layers.mamba import mamba_mixer2 as vllm_mamba
from vllm.model_executor.layers.mamba.ops import ssd_combined as vllm_ssd
from vllm.model_executor.layers.mamba.ops import ssu_dispatch as vllm_ssu

mcore_mamba.mamba_chunk_scan_combined = _mcore_scan
vllm_ssd.mamba_chunk_scan_combined_varlen = _vllm_varlen
vllm_mamba.mamba_chunk_scan_combined_varlen = _vllm_varlen
vllm_ssu.selective_state_update = _npu_selective_state_update
vllm_mamba.selective_state_update = _npu_selective_state_update

print("[issue-70] enabled official MindSpeed SSD and NPU tensor SSU on NPU")
PY
export PYTHONPATH="${mamba_npu_compat_dir}:${PYTHONPATH:-}"
if [[ -n "${VERL_USE_EXTERNAL_MODULES:-}" ]]; then
    export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES},verl_mamba_npu_compat"
else
    export VERL_USE_EXTERNAL_MODULES=verl_mamba_npu_compat
fi

IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [[ ${#visible_devices[@]} -ne 8 ]]; then
    echo "This recipe requires exactly 8 visible NPUs; got ${ASCEND_RT_VISIBLE_DEVICES}" >&2
    exit 2
fi

################################################### quick config ###################################################

# ---- user-adjustable ----
rollout_mode=${ROLLOUT_MODE:-"async"}
return_raw_chat="False"
rollout_name="vllm"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
dtype="bfloat16"

project_name='DAPO'
exp_name=${EXP_NAME:-'nemotron_nano_9b_v2_grpo_npu'}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
enable_overlong_buffer=True
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-512}
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-32}

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:?Set RAY_DATA_HOME to the model, data, and checkpoint storage root}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/NVIDIA-Nemotron-Nano-9B-v2"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:--1}
RESUME_MODE=${RESUME_MODE:-disable}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 10 / 10))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
param_offload=${PARAM_OFFLOAD:-False}
optimizer_offload=${OPTIMIZER_OFFLOAD:-True}
actor_ppo_micro_bsz=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-512}
cudagraph_capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-"[1,2,4,8,16,32,64,128,256,512]"}
gen_tp=${ROLLOUT_TP:-8}
train_tp=${TRAIN_TP:-8}
train_pp=1
EP=1
ETP=1
if (( 8 % gen_tp != 0 || 8 % train_tp != 0 )); then
    echo "ROLLOUT_TP and TRAIN_TP must divide 8; got ${gen_tp} and ${train_tp}" >&2
    exit 2
fi
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
################################################### start of config ###################################################

FP8=(
    # train
    # +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="e4m3" # e4m3 or hybrid
    # +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    # +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"
    # rollout
    actor_rollout_ref.actor.megatron.dtype=${dtype}
    actor_rollout_ref.rollout.dtype=${dtype}
    # +actor_rollout_ref.rollout.quantization="fp8"
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=$return_raw_chat
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
)

REWARD_MODEL=(
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
    reward_model.reward_manager=dapo
)

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    # The fused Mamba training path requires the CUDA-only causal-conv1d extension.
    # Use the official unfused Mamba path on NPU.
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_mamba_mem_eff_path=False
    actor_rollout_ref.model.use_fused_kernels=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_ppo_micro_bsz}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${param_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${optimizer_offload}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.cudagraph_capture_sizes="${cudagraph_capture_sizes}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
)

TRAINER=(
    trainer.logger=['console']
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.test_freq=${TEST_FREQ}
    trainer.save_freq=${SAVE_FREQ}
    trainer.total_epochs=10
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=${RESUME_MODE}
    trainer.log_val_generations=10
    trainer.device=npu
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.use_torch_compile=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)
EXTRA=(
    model_engine=megatron
)

################################################### start script ###################################################

# uv (set VERL_USE_UV=0 for system python): GPU vllm/sglang × megatron run the driver and every Ray worker
# (runtime_env.py_executable) through `uv run` on the matching extras of the committed uv.lock;
# other backends / NPU fall back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ] && { [ "${rollout_name}" = vllm ] || [ "${rollout_name}" = sglang ]; }; then
    LAUNCH=(uv run --frozen --all-packages --extra "${rollout_name}" --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${rollout_name} --extra megatron")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REWARD_MODEL[@]}" \
    "${FP8[@]}" \
    "${PERF_OPT[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
