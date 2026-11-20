#!/bin/bash
set -euo pipefail
# DeepSeek-V4-Flash DSpark GRPO：4 节点 / 64 NPU，完整 43 层。
# 配置来源：dspark-20k.sh。
# prompt 2k + response 20k = 22k（22528 tokens）；actor TP2/PP4/CP4/EP16。
# rollout TP4/DP8/EP32，两份 replica；DSpark K5；关闭 reference/KL。
# 按 train_deepseek_v4_grpo_mindspeed_vllm.sh 的格式直接启动 Python。
# 在已配置好环境、能导入 torch_npu 的 Ray 主节点运行：
#   bash examples/train_deepseek_v4_grpo_mindspeed_vllm_A2.sh
# Hydra 覆盖参数可以追加在命令末尾，优先级最高。

# Project Configuration
project_name='DeepSeekV4'
trainer_logger='["console","wandb"]'

# Node Info
NNODES=4
NPUS_PER_NODE=16
TOTAL_NPUS=$((NNODES * NPUS_PER_NODE))

# Model Weights Paths
NUM_LAYERS=${NUM_LAYERS:-43}
HF_MODEL_PATH=${HF_MODEL_PATH:-/model/DeepSeek-V4-Flash-DSpark-bf16}
DSPARK_DRAFT_MODEL_PATH=${DSPARK_DRAFT_MODEL_PATH:-${DRAFT_MODEL_PATH:-${HF_MODEL_PATH}}}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/model/DeepSeek-V4-Flash-Base-BF16/chat_template.jinja}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=/ckpt

# File System Paths
TRAIN_FILE=/data/math-17k/dapo-math-17k.parquet
TEST_FILE=/data/math-17k/aime-2024.parquet

# Data Length Configuration
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 20))}
full_sequence_length=$((max_prompt_length + max_response_length))
max_model_len=${full_sequence_length}
# RoPE 必须覆盖完整序列；动态分批预算不能替代位置表长度。
rope_table_len=${full_sequence_length}

# Training Batch Configuration
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-128}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-128}
n_resp_per_prompt=8
global_batch_size=$((train_prompt_bsz * n_resp_per_prompt))

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=0.0

# Performance and Memory Management Configuration
all_offload=${ALL_OFFLOAD:-True}
swap_optimizer=${SWAP_OPTIMIZER:-True}
use_dynamic_bsz=True
use_remove_padding=True
# CP4 下动态分批预算为 10240×4；保留源脚本的 10k 预算。
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-10240}
infer_ppo_max_token_len=10240

# Megatron Parallelism Configuration
train_tp=${TRAIN_TP:-2}
train_pp=${TRAIN_PP:-4}
train_cp=${TRAIN_CP:-4}
train_ep=${TRAIN_EP:-16}
train_etp=1
train_model_parallel_size=$((train_tp * train_pp * train_cp))
train_dp=$((TOTAL_NPUS / train_model_parallel_size))
# PP4 的非均匀切分保留为 10/11/11/11，降低 PP0 的 activation 峰值。
if (( train_pp > 1 )); then
    NUM_LAYER_LIST="10,11,11,11"
else
    NUM_LAYER_LIST=""
fi

# DeepSeek-V4 Compression Configuration
FULL_COMPRESS_RATIOS='[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4]'
COMPRESS_RATIOS=$(python3 - <<PY_RATIOS
import ast
ratios = ast.literal_eval("${FULL_COMPRESS_RATIOS}")
print(str(ratios[:${NUM_LAYERS}]).replace(" ", ""))
PY_RATIOS
)

# Generation Configuration
DSPARK_ENABLE=${DSPARK_ENABLE:-1}
case "${DSPARK_ENABLE}" in
    0|1) ;;
    *) echo "ERROR: DSPARK_ENABLE must be 0 or 1" >&2; exit 2 ;;
esac
gen_tp=${GEN_TP:-4}
gen_dp=${GEN_DP:-8}
gen_ep=${GEN_EP:-$((gen_tp * gen_dp))}
# 保留训练后残留设备内存和 MoE workspace 的余量。
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.7}
# chunked prefill 单次 token 预算，不限制最大 response/context。
max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-64}
rollout_world_size=$((gen_tp * gen_dp))
num_rollout_replicas=$((TOTAL_NPUS / rollout_world_size))
if (( rollout_world_size * num_rollout_replicas != TOTAL_NPUS )); then
    echo "ERROR: gen_tp×gen_dp=${gen_tp}×${gen_dp} must divide TOTAL_NPUS=${TOTAL_NPUS}" >&2
    exit 1
fi
if (( gen_ep != gen_tp * gen_dp )); then
    echo "ERROR: gen_ep=${gen_ep} must equal gen_tp×gen_dp=${gen_tp}×${gen_dp}" >&2
    exit 1
fi

# Run Naming
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
experiment_tag="$(date +%Y%m%d_%H%M%S)_$$"
exp_name="colocate_async_tnd_dspark${DSPARK_ENABLE}_resp${max_response_length}_ctx${max_model_len}_seqs${rollout_max_num_seqs}_gentp${gen_tp}ep${gen_ep}_traintp${train_tp}pp${train_pp}cp${train_cp}ep${train_ep}_ppo${actor_ppo_max_token_len}_030_R3_${experiment_tag}"

# Data Configuration
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

# Model Configuration
# 完整 DSpark checkpoint 未附带 chat template，复用原模型的提示词格式。
# Hydra 参数传模板正文，保留其中的原始换行和反斜杠。
CHAT_TEMPLATE_LITERAL=$(python3 - "${CHAT_TEMPLATE_PATH}" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1]).read_text()
if not template.strip():
    raise SystemExit('chat template must not be empty')
quote = chr(34)
backslash = chr(92)
escaped = template.replace(backslash, backslash * 2).replace(quote, backslash + quote)
print(quote + escaped + quote, end='')
PY
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.tokenizer_path="${HF_MODEL_PATH}"
    actor_rollout_ref.model.hf_config_path="${HF_MODEL_PATH}"
    "actor_rollout_ref.model.custom_chat_template=${CHAT_TEMPLATE_LITERAL}"
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding}
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.actor.mindspeed.use_remove_padding=${use_remove_padding}
    # 每个 micro-batch 按自己的序列长度处理，避免按整批最长样本 padding。
    actor_rollout_ref.actor.mindspeed.pad_bshd_to_minibatch_max=False
)

# Reinforcement Learning Algorithm Configuration
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

# Actor Configuration
# 当前环境使用 mcore_kwargs；保留源配置的命名空间。
ACTOR_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    # Loss Function Configuration
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.lr=1e-6
    # Megatron Parallelism Strategy
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.mindspeed.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.mindspeed.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.mindspeed.expert_tensor_parallel_size=${train_etp}
    # Memory Optimization
    actor_rollout_ref.actor.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    actor_rollout_ref.actor.mindspeed.grad_offload=${all_offload}
    # Model Weights Management
    actor_rollout_ref.actor.mindspeed.use_dist_checkpointing=False
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    actor_rollout_ref.actor.mindspeed.router_replay.mode=R3

    # CPU optimizer 状态由 swap_optimizer 管理；optimizer_cpu_offload 保持关闭。

    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.swap_optimizer=${swap_optimizer}
    # DSA_ARGS
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.enable_dsa_indexer=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_fused_lightning_indexer=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_fused_lightning_indexer_loss=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_sparse_flash_attn=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.index_n_heads=64
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.index_head_dim=128
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.index_topk=512
    # MHC_ARGS
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.hc_mult=4
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.enable_mhc=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_fused_mhc=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.kv_compress=True

    # ATTN_ARGS
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.multi_latent_attention=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.qk_pos_emb_head_dim=64
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.qk_head_dim=512
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.q_lora_rank=1024
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.o_lora_rank=1024
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.kv_lora_rank=512
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.v_head_dim=128
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.qk_layernorm=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.mla_fa_without_pad=True

    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.o_groups=8
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.sliding_window_size=128
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.original_seq_len=65536
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_factor=16
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.compress_rope_theta=160000.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.max_batch_size=4
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.compress_ratios="${COMPRESS_RATIOS}"
    # MOE_ARGS
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_grouped_gemm=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_permutation_async_comm=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_token_dispatcher_type=alltoall
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_layer_freq=1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.first_k_dense_replace=-1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.num_experts=256
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_topk=6
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_ffn_hidden_size=2048
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_load_balancing_type=none
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_group_topk=1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_num_groups=1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_topk_scaling_factor=1.5
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.seq_aux=False
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_aux_loss_coeff=0.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_score_function=sqrtsoftplus
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_enable_expert_bias=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_shared_expert_intermediate_size=2048
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.fix_router=False
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_router_dtype=fp32
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.n_hash_layers=3

    # MTP and Recompute Configuration
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.mtp_num_layers=0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.mtp_loss_scaling_factor=0.3
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.mtp_mem_efficient_logits=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.recompute_granularity=full
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.recompute_method=uniform
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.recompute_num_layers=1
    # RoPE Configuration
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.beta_fast=32
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.beta_slow=1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_scaling_factor=16
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_scaling_mscale=1.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_scaling_mscale_all_dim=1.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_scaling_original_max_position_embeddings=${rope_table_len}
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_theta=10000.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rope_scaling_type=yarn

    # Runtime and Model Specification
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.transformer_impl=local
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.spec="['mindspeed_llm.tasks.models.spec.deepseek4_spec', 'layer_spec']"
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.manual_gc=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.manual_gc_interval=50
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_shared_storage=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_distributed_optimizer=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_flash_attn=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_mcore_models=True

    # Model Dimensions and Training Defaults
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.num_layers=${NUM_LAYERS}
    "++actor_rollout_ref.actor.mindspeed.mcore_kwargs.num_layer_list=\"${NUM_LAYER_LIST}\""
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.hidden_size=4096
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.ffn_hidden_size=4096
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.num_attention_heads=64
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.tokenizer_type=PretrainedFromHF
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.tokenizer_name_or_path=${HF_MODEL_PATH}
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.seq_length=$rope_table_len
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.max_position_embeddings=1048576
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.micro_batch_size=1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.global_batch_size=${global_batch_size}
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.make_vocab_size_divisible_by=1
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.lr=1e-6
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.train_iters=2000
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.lr_decay_style=constant
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.untie_embeddings_and_output_weights=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.disable_bias_linear=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.add_bias_linear=False
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.attention_dropout=0.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.init_method_std=0.02
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.hidden_dropout=0.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.position_embedding_type=deepseek4
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.normalization=RMSNorm
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_fused_swiglu=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_fused_rmsnorm=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.swiglu=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.swiglu_limit=10.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_masked_softmax_fusion=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.attention_softmax_in_fp32=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.min_lr=1e-6
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.weight_decay=1e-2
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.clip_grad=1.0
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.adam_beta1=0.9
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.adam_beta2=0.999
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.initial_loss_scale=65536
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.vocab_size=129280
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.padded_vocab_size=129280
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.rotary_base=10000
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.norm_epsilon=1e-6
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_load_optim=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_load_rng=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.bf16=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.distributed_timeout_minutes=120
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_gradient_accumulation_fusion=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.gradient_accumulation_fusion=False
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_save_optim=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_save_rng=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.context_parallel_algo=kvallgather_cp_algo
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.masked_softmax_fusion=False
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.moe_shared_expert_overlap=True
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.indexer_loss_coeff=0.0
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0 
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.use_triton_mhc=True
    # SwiGLU-limit backward 保留当前环境的原生实现。
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.no_pad_to_seq_lengths=True
    # reset_attention_mask 与 packed TND/remove-padding 开关保持一致。
    ++actor_rollout_ref.actor.mindspeed.mcore_kwargs.reset_attention_mask=${use_remove_padding}
)

# Reference Configuration
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
    actor_rollout_ref.ref.mindspeed.pad_bshd_to_minibatch_max=False
)

# Reference 复用 actor 的模型参数，但不启用训练用 swap_optimizer。
REF_mcore_kwargs=()
for arg in "${ACTOR_CONFIG[@]}"; do
    if [[ "$arg" == *".actor.mindspeed.mcore_kwargs."* ]]; then
        if [[ "$arg" == *".mcore_kwargs.swap_optimizer="* ]]; then
            continue
        fi
        REF_mcore_kwargs+=("${arg/actor_rollout_ref.actor.mindspeed.mcore_kwargs/actor_rollout_ref.ref.mindspeed.mcore_kwargs}")
    fi
done
REF_CONFIG=( "${REF_CONFIG[@]}" "${REF_mcore_kwargs[@]}" )

# Rollout Configuration
ROLLOUT_CONFIG=(
    # K5 capture sizes, max_num_seqs and Ascend graph settings come from
    # DSPARK_TP4_GRAPH_ARGS, appended after this array at launch.
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_dsa_cp=False
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_flashcomm1=False
    # ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.multistream_overlap_shard_expert=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_fused_mc2=1
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
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
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    # prefix：yaml 默认 True，必须关。每步 weight sync + free_cache_engine 会清掉；DSV4 MLA KV 前缀缓存未验证。
    actor_rollout_ref.rollout.enable_prefix_caching=False
    # chunked prefill 按上述单次 token 预算处理完整 22k 上下文。
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    # Disable rollout replay dumps for long runs.
    skip.rollout_tq.enable=${ROLLOUT_CACHE_ENABLE:-False}
    skip.rollout_tq.dump_dir=${ROLLOUT_CACHE_DIR:-/data/rollout_dump/deepseek}
    skip.rollout_tq.steps='[1,2,3,4]'
    skip.rollout_tq.action=repeat
)

# Trainer Configuration
TRAINER_CONFIG=(
    trainer.logger="${trainer_logger}"
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    trainer.total_epochs=300
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.default_local_dir="${CKPTS_DIR}"
    #trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    actor_rollout_ref.actor.checkpoint.save_contents="['model']"
    trainer.v1.trainer_mode=colocate_async
    # 当前 head 不提供 CPU，SimpleStorage 的 1 CPU actor 必须调度到 worker。
    ++transfer_queue.backend.SimpleStorage.required_node_resource=worker
)

# DSpark Speculative Decoding and Graph Configuration
NUM_SPECULATIVE_TOKENS=5
DSPARK_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    ++actor_rollout_ref.rollout.mtp._target_=verl.workers.config.MtpConfig
    ++actor_rollout_ref.rollout.mtp.enable_train=False
)
if [[ "${DSPARK_ENABLE}" == 1 ]]; then
    DRAFT_MODEL_LITERAL=$(python3 - "${DSPARK_DRAFT_MODEL_PATH}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
    )
    DSPARK_CONFIG+=(
        ++actor_rollout_ref.rollout.mtp.enable=True
        ++actor_rollout_ref.rollout.mtp.enable_rollout=True
        ++actor_rollout_ref.rollout.mtp.method=dspark
        ++actor_rollout_ref.rollout.mtp.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
        actor_rollout_ref.rollout.disable_log_stats=False
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.per_request_spec_decode_metrics=summary
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.method=dspark
        "++actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.model=${DRAFT_MODEL_LITERAL}"
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.draft_sample_method=greedy
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config.enforce_eager=True
    )
    # K5 每个请求最多 6 个 decode tokens；64 seqs 对应最大 graph 384。
    GRAPH_CAPTURE_SIZES='[12,24,48,96,192,384]'
    MAX_GRAPH_CAPTURE_SIZE=384
else
    DSPARK_CONFIG+=(
        ++actor_rollout_ref.rollout.mtp.enable=False
        ++actor_rollout_ref.rollout.mtp.enable_rollout=False
        ++actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config=null
    )
    GRAPH_CAPTURE_SIZES='[1,2,4,8,16,32,64]'
    MAX_GRAPH_CAPTURE_SIZE=64
fi

# TP4，目标模型 decode graph；草稿模型保持 eager。
DSPARK_TP4_GRAPH_ARGS=(
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.cudagraph_capture_sizes=null
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=${GRAPH_CAPTURE_SIZES}"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=${MAX_GRAPH_CAPTURE_SIZE}
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.ascend_compilation_config.enable_npugraph_ex=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.ascend_compilation_config.enable_static_kernel=False
)

# Training Entrypoint
# 保留源脚本的参数顺序：DSpark/graph 在通用 rollout 后覆盖，用户参数最后覆盖。
PPO_ARGS=(
    model_engine=mindspeed
    "${DATA_CONFIG[@]}"
    "${MODEL_CONFIG[@]}"
    "${ACTOR_CONFIG[@]}"
    "${REF_CONFIG[@]}"
    "${ROLLOUT_CONFIG[@]}"
    "${ALGORITHM_CONFIG[@]}"
    "${TRAINER_CONFIG[@]}"
    "${DSPARK_CONFIG[@]}"
    "${DSPARK_TP4_GRAPH_ARGS[@]}"
    "$@"
)

mkdir -p logs
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    "${PPO_ARGS[@]}" 2>&1 | tee "logs/run_deepseek_v4_npu_A2_${RUN_TIMESTAMP}_$$.log"
