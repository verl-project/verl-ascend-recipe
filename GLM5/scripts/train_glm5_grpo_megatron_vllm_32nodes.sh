set -x

# Project Configuration
project_name='GLM5'
exp_name='GLM5-32-nodes'

# Node Info
NNODES=${NNODES:-32}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# Model Weights Paths
MODEL_PATH=/model/GLM5
CKPTS_DIR=/ckpt

# File System Paths
TRAIN_FILE=/data/dapo-math-17k.parquet
TEST_FILE=/data/dapo-math-17k.parquet

# Data Length Configuration
max_prompt_length=$((1024 * 24))
max_response_length=$((1024 * 8))

# Training Batch Configuration
train_prompt_bsz=64
train_prompt_mini_bsz=16
n_resp_per_prompt=8

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
clip_ratio_low=0.2
clip_ratio_high=0.28
balance_batch=False

# Performance and Memory Management Configuration
all_offload=True
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

# Megatron Parallelism Configuration
first_layer=9
last_layer=9
train_pp=8
train_tp=2
train_ep=64
train_etp=1
train_cp=2

# Generation Configuration
gen_tp=8
gen_dp=8
gen_ep=64
gpu_memory_utilization=0.7
max_num_seqs=128
max_num_batched_tokens=4096
max_model_len=$((max_prompt_length + max_response_length))

# Important Sample Configuration
ROLLOUT_IS=${ROLLOUT_IS:-sequence}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-true}
ROLLOUT_RS=${ROLLOUT_RS:-token_k1}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-0.6_1.6}

DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
    +data.apply_chat_template_kwargs.enable_thinking=False
    data.shuffle=True
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE}
    algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS}
    algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD}
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='fused'
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$train_pp
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$train_tp
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$train_ep
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$train_etp
    actor_rollout_ref.actor.megatron.context_parallel_size=$train_cp
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=$first_layer
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=$last_layer
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.experimental_attention_variant="dsa"
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_dsa_absorb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_lightning_indexer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_sparse_flash_attention=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_lightning_indexer_kl_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${train_cp}
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=False
)

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$train_pp
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$train_tp
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$train_ep
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$train_etp
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=$gen_tp
    actor_rollout_ref.rollout.data_parallel_size=$gen_dp
    actor_rollout_ref.rollout.expert_parallel_size=$gen_ep
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.load_format='dummy'
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32]"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
)

TRAINER_CONFIG=(
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.rollout_data_dir="logs/rollout_data_dir/$(date +%Y%m%d_%H%M%S)"
    trainer.resume_mode="auto"
    trainer.balance_batch=${balance_batch}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.total_epochs=100
)

EXTRA=(
    model_engine=megatron
    actor_rollout_ref.nccl_timeout=7200
)


python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer' \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "${EXTRA[@]}" \
    2>&1 | tee logs/run_glm5_$(date -d '+8 hours' +%Y%m%d_%H%M%S).log.log \
    "$@" 
