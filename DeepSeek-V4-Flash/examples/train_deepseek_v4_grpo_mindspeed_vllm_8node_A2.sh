#!/bin/bash
# DeepSeek-V4-Flash GRPO — A2(910B) 8机 128 NPU 全层 43 层（关闭 mcore ckpt，从 HF 加载）
#
# 配置要点：
#   - actor/ref: use_dist_checkpointing=False，经 mbridge 从 HF 加载
#   - 训练并行：tp2 x pp4 x cp4 x ep32（128 卡）；DeepSeek4 开 CP 用 kvallgather_cp_algo
#   - 优化器：仅 swap_optimizer，勿开 optimizer_cpu_offload
#     （param_offload=True 时 HybridDeviceOptimizer 会对 non-leaf / 未分配参数报错）
#   - Rollout：gen_tp8 x gen_dp8 x gen_ep64 + FULL_DECODE_ONLY 开图，共 2 replica
#
# 启动（Ray 集群已就绪 — 仅在 head 节点提交一次）：
#   bash train_deepseek_v4_grpo_mindspeed_vllm_8node_A2.sh
#
# runtime env 说明：
#   本脚本通过 ray job submit 提交，worker 不继承本 shell 环境，环境变量来自
#   --runtime-env 指定的 yaml。默认使用同目录 runtime_env_A2.yaml（CANN/HCCL 等，
#   路径按 /workspace-verl 安装目录编写）；如需调整，通过 RUNTIME_ENV 覆盖：
#   RUNTIME_ENV=/path/to/runtime_env.yaml bash train_deepseek_v4_grpo_mindspeed_vllm_8node_A2.sh
#
#set -xeuo pipefail

VERL_PKG_ROOT=$(python3 -c "import os, verl; print(os.path.dirname(verl.__file__))")
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
WORKING_DIR=${WORKING_DIR:-"${VERL_PKG_ROOT}"}
# 默认使用 A2 专用 runtime env（包含 CANN/HCCL 等必要环境变量，路径按 /workspace-verl 安装目录编写）；
# 可通过 RUNTIME_ENV 环境变量覆盖为自定义 yaml
RUNTIME_ENV_TEMPLATE=${RUNTIME_ENV:-"${SCRIPT_DIR}/runtime_env_A2.yaml"}
echo "WORKING_DIR=${WORKING_DIR}"
echo "RUNTIME_ENV_TEMPLATE=${RUNTIME_ENV_TEMPLATE}"

NNODES=${NNODES:-8}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
RUNTIME_ENV="${RUNTIME_ENV_TEMPLATE}"

# ---------- 训练配置（全层 43）----------
project_name='DeepSeek-V4-Flash'

NUM_LAYERS=${NUM_LAYERS:-43}
exp_name="DeepSeek-V4-Flash-8node-A2-L${NUM_LAYERS}-hf"

TOTAL_NPUS=$((NNODES * NPUS_PER_NODE))

# 模型/数据/ckpt 路径，请按实际环境修改
HF_MODEL_PATH=${HF_MODEL_PATH:-/model/DeepSeek-V4-Flash-Base-BF16}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-/ckpt}

TRAIN_FILE=${TRAIN_FILE:-/data/math-17k/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/data/math-17k/aime-2024.parquet}

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))

# real_train_batch_size = train_prompt_bsz x n_resp_per_prompt
n_resp_per_prompt=8
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-128}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-128}
echo "batch: train_prompt_bsz=${train_prompt_bsz} n_resp_per_prompt=${n_resp_per_prompt} real_train_batch_size=$((train_prompt_bsz * n_resp_per_prompt)) (n_gpus=${TOTAL_NPUS})"

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

all_offload=${ALL_OFFLOAD:-True}
# 仅 swap_optimizer 扛 Adam 状态，默认开启
swap_optimizer=${SWAP_OPTIMIZER:-True}
train_tp=${TRAIN_TP:-2}
train_pp=${TRAIN_PP:-4}
train_ep=${TRAIN_EP:-32}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.7}

use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

train_etp=1
# DeepSeek4 开 CP 时必须用 kvallgather_cp_algo（ulysses 会被 MindSpeed 拒绝）
train_cp=${TRAIN_CP:-4}

echo "NUM_LAYERS=${NUM_LAYERS} TOTAL_NPUS=${TOTAL_NPUS} train_tp=${train_tp} train_pp=${train_pp} train_ep=${train_ep} train_cp=${train_cp} all_offload=${all_offload} swap_optimizer=${swap_optimizer} gpu_memory_utilization=${gpu_memory_utilization}"

# num_layer_list 支持非整除切分（43 层 x pp4 -> 10,10,10,13）
if (( train_pp > 1 )); then
    NUM_LAYER_LIST=$(python3 - <<PY
n, pp = ${NUM_LAYERS}, ${train_pp}
base = n // pp
layers = [base] * (pp - 1)
layers.append(n - sum(layers))
print(",".join(map(str, layers)))
PY
)
    echo "num_layer_list=${NUM_LAYER_LIST}"
else
    NUM_LAYER_LIST=""
fi

FULL_COMPRESS_RATIOS='[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4]'
COMPRESS_RATIOS=$(python3 - <<PY
import ast
ratios = ast.literal_eval("${FULL_COMPRESS_RATIOS}")
print(str(ratios[:${NUM_LAYERS}]).replace(" ", ""))
PY
)

# 128 卡：gen_tp8 x gen_dp8 = 64 卡/replica，gen_ep=64，共 2 replica
# verl RolloutConfig 硬性要求：gen_ep == gen_tp * gen_dp
gen_tp=${GEN_TP:-8}
gen_dp=${GEN_DP:-8}
gen_ep=${GEN_EP:-$((gen_tp * gen_dp))}
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

rollout_world_size=$((gen_tp * gen_dp))
num_rollout_replicas=$((TOTAL_NPUS / rollout_world_size))
if (( rollout_world_size * num_rollout_replicas != TOTAL_NPUS )); then
    echo "ERROR: gen_tp x gen_dp=${gen_tp} x ${gen_dp} must divide TOTAL_NPUS=${TOTAL_NPUS}" >&2
    exit 1
fi
if (( gen_ep != gen_tp * gen_dp )); then
    echo "ERROR: gen_ep=${gen_ep} must equal gen_tp x gen_dp=${gen_tp} x ${gen_dp}" >&2
    exit 1
fi
echo "rollout: gen_tp=${gen_tp} gen_dp=${gen_dp} gen_ep=${gen_ep} replicas=${num_rollout_replicas}"

global_batch_size=$((train_prompt_bsz * n_resp_per_prompt))

DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.tokenizer_path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.hf_config_path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.actor.mindspeed.use_remove_padding=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.rollout_correction.rollout_is="token"
    algorithm.rollout_correction.rollout_is_threshold=2.0
    algorithm.rollout_correction.rollout_is_batch_normalize=false
    algorithm.rollout_correction.rollout_rs="token_k1"
    algorithm.rollout_correction.rollout_rs_threshold="0.5_2.0"
    algorithm.rollout_correction.bypass_mode=false
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.mindspeed.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    actor_rollout_ref.actor.mindspeed.grad_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=False
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True

    # 仅 swap_optimizer，勿开 optimizer_cpu_offload。
    # param_offload=True 时 HybridDeviceOptimizer 会报
    # "can't optimize a non-leaf Tensor" / "data is not allocated yet"。
    # +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    # +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    # +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer=${swap_optimizer}
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.enable_dsa_indexer=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_lightning_indexer=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_lightning_indexer_loss=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_sparse_flash_attn=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.index_n_heads=64
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.index_head_dim=128
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.index_topk=512
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.hc_mult=4
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.enable_mhc=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_mhc=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.kv_compress=True

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.multi_latent_attention=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_pos_emb_head_dim=64
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_head_dim=512
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.q_lora_rank=1024
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.o_lora_rank=1024
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.kv_lora_rank=512
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.v_head_dim=128
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_layernorm=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.mla_fa_without_pad=True

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.o_groups=8
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.sliding_window_size=128
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.original_seq_len=65536
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_factor=16
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_rope_theta=160000.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.max_batch_size=4
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_ratios="${COMPRESS_RATIOS}"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_permutation_async_comm=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_token_dispatcher_type=alltoall
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_layer_freq=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.first_k_dense_replace=-1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.num_experts=256
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk=6
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_ffn_hidden_size=2048
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_load_balancing_type=none
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_group_topk=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_num_groups=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk_scaling_factor=1.5
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_aux=False
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_aux_loss_coeff=0.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_score_function=sqrtsoftplus
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_enable_expert_bias=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_intermediate_size=2048
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.fix_router=False
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_dtype=fp32
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.n_hash_layers=3

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_num_layers=0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_loss_scaling_factor=0.3
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_mem_efficient_logits=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.beta_fast=32
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.beta_slow=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_factor=16
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_mscale=1.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_mscale_all_dim=1.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_original_max_position_embeddings=65536
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_theta=10000.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_type=yarn

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.transformer_impl=local
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.spec="['mindspeed_llm.tasks.models.spec.deepseek4_spec', 'layer_spec']"
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc_interval=50
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_shared_storage=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_distributed_optimizer=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_flash_attn=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_mcore_models=True

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layers=${NUM_LAYERS}
    "++actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layer_list=\"${NUM_LAYER_LIST}\""
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_size=4096
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.ffn_hidden_size=4096
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.num_attention_heads=64
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_type=PretrainedFromHF
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_name_or_path=${HF_MODEL_PATH}
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length=$actor_ppo_max_token_len
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.max_position_embeddings=1048576
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.global_batch_size=${global_batch_size}
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.make_vocab_size_divisible_by=1
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.lr=1e-6
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.train_iters=2000
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.lr_decay_style=constant
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.untie_embeddings_and_output_weights=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.disable_bias_linear=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.add_bias_linear=False
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.attention_dropout=0.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.init_method_std=0.02
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_dropout=0.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.position_embedding_type=deepseek4
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.normalization=RMSNorm
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_swiglu=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_rmsnorm=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swiglu=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.swiglu_limit=10.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_masked_softmax_fusion=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.attention_softmax_in_fp32=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.min_lr=1e-6
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.weight_decay=1e-2
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.clip_grad=1.0
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.adam_beta1=0.9
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.adam_beta2=0.999
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.initial_loss_scale=65536
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.vocab_size=129280
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.padded_vocab_size=129280
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.rotary_base=10000
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.norm_epsilon=1e-6
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_load_optim=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_load_rng=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.bf16=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.distributed_timeout_minutes=120
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_gradient_accumulation_fusion=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.gradient_accumulation_fusion=False
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_save_optim=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_save_rng=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.context_parallel_algo=kvallgather_cp_algo
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.masked_softmax_fusion=False
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_overlap=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.indexer_loss_coeff=0.0

    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_mhc=True
    ++actor_rollout_ref.actor.mindspeed.llm_kwargs.no_pad_to_seq_lengths=True
)

REF_CONFIG=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.mindspeed.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.mindspeed.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.mindspeed.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.ref.mindspeed.use_dist_checkpointing=False
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

# ref 复用 actor 的 llm_kwargs（除 swap_optimizer）
REF_LLM_KWARGS=()
for arg in "${ACTOR_CONFIG[@]}"; do
    if [[ "$arg" == *".actor.mindspeed.llm_kwargs."* ]]; then
        if [[ "$arg" == *".llm_kwargs.swap_optimizer="* ]]; then
            continue
        fi
        REF_LLM_KWARGS+=("${arg/actor_rollout_ref.actor.mindspeed.llm_kwargs/actor_rollout_ref.ref.mindspeed.llm_kwargs}")
    fi
done
REF_CONFIG=( "${REF_CONFIG[@]}" "${REF_LLM_KWARGS[@]}" )

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.max_num_seqs=32
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY'
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1, 2, 4, 8, 16, 32]"
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    # DSV4+Ascend：默认 yaml 为 True，必须显式关掉，否则易在 rollout/sleep 卡死 (507011)
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
)

TRAINER_CONFIG=(
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    trainer.total_epochs=300
    trainer.val_before_train=False
    trainer.test_freq=-5
    trainer.save_freq=-1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    actor_rollout_ref.actor.checkpoint.save_contents="['model']"
)

submit_training_job() {
    mkdir -p logs
    LOG_FILE=logs/run_deepseek_v4_8node_A2_L${NUM_LAYERS}_tp${train_tp}pp${train_pp}ep${train_ep}_$(date +%Y%m%d_%H%M%S).log

    ray job submit --runtime-env="${RUNTIME_ENV}" \
        --working-dir "${WORKING_DIR}" \
        -- python3 -m verl.trainer.main_ppo \
        --config-path=config \
        --config-name='ppo_trainer.yaml' \
        model_engine=mindspeed \
        "${DATA_CONFIG[@]}" \
        "${MODEL_CONFIG[@]}" \
        "${ACTOR_CONFIG[@]}" \
        "${REF_CONFIG[@]}" \
        "${ROLLOUT_CONFIG[@]}" \
        "${ALGORITHM_CONFIG[@]}" \
        "${TRAINER_CONFIG[@]}" \
        "$@" | tee "${LOG_FILE}"
}

# ---------- 入口 ----------
submit_training_job "$@"
