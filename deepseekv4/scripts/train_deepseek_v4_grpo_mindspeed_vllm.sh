#!/bin/bash
#set -xeuo pipefail
# Project Configuration
project_name='DeepSeekV4'
exp_name='DeepSeekV4-8-node'

# Node Info
NNODES=${NNODES:-8}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# Model Weights Paths
MODEL_PATH=/model/DeepSeek-V4-Flash-Base-BF16
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=/ckpt

# File System Paths
TRAIN_FILE=/data/gsm8k/train.parquet
TEST_FILE=/data/gsm8k/test.parquet
# Data Length Configuration
max_prompt_length=$((1024*2))
max_response_length=$((1024*2))

# Training Batch Configuration
train_prompt_bsz=32
train_prompt_mini_bsz=32
n_resp_per_prompt=8

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# Performance and Memory Management Configuration
all_offload=True
use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

# Megatron Parallelism Configuration
train_tp=2
train_ep=64
train_etp=1
train_pp=2
train_cp=1

# Generation Configuration
gen_tp=8
gen_dp=8
gen_ep=64
gpu_memory_utilization=0.5
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
    # data.shuffle=False
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.actor.mindspeed.use_remove_padding=False
)

ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
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

    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
	+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swap_optimizer=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.enable_dsa_indexer=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.index_n_heads=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.index_head_dim=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.index_topk=512
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hc_mult=4
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.enable_mhc=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.kv_compress=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.norm_eps=1e-6

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.multi_latent_attention=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_pos_emb_head_dim=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_head_dim=512
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.q_lora_rank=1024
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.kv_lora_rank=512
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.v_head_dim=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.qk_layernorm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mla_fa_without_pad=True

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_g2_attention=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.o_groups=8
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.g2_window_size=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_head_dim=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.original_seq_len=65536
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_factor=16
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_rope_theta=160000.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.max_batch_size=4
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.compress_ratios="[0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4]"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_grouped_gemm=False  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_permutation_async_comm=True 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_token_dispatcher_type=alltoall 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_layer_freq=1 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.first_k_dense_replace=-1 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_experts=256 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk=6 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_ffn_hidden_size=2048 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_load_balancing_type=none 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_group_topk=1 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_num_groups=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_topk_scaling_factor=1.5
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_aux=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_aux_loss_coeff=0.001
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_score_function=sqrtsoftplus
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_enable_expert_bias=True 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_intermediate_size=2048
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.fix_router=False  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_router_dtype=fp32
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.n_hash_layers=3

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_num_layers=0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_loss_scaling_factor=0.3 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.mtp_mem_efficient_logits=True  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.beta_fast=32  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.beta_slow=1  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_factor=16  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_mscale=1.0  
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_mscale_all_dim=1.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_original_max_position_embeddings=65536
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rope_scaling_type=yarn

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.transformer_impl=local
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.spec="['mindspeed_llm.tasks.models.spec.deepseek4_spec', 'layer_spec']"
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc=True 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.manual_gc_interval=50 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_shared_storage=True 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_distributed_optimizer=True 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_flash_attn=True 
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_mcore_models=True   

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layers=43
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_layer_list=\'21,22\'
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_size=4096
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.ffn_hidden_size=4096
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_attention_heads=64
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_type=PretrainedFromHF
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.tokenizer_name_or_path=$MODEL_PATH
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length=$actor_ppo_max_token_len
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.max_position_embeddings=163840
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.global_batch_size=128
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.make_vocab_size_divisible_by=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.lr=1e-6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.train_iters=2000
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.lr_decay_style=constant
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.untie_embeddings_and_output_weights=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.disable_bias_linear=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.add_bias_linear=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.attention_dropout=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.init_method_std=0.02
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.hidden_dropout=0.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.position_embedding_type=g2
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.normalization=RMSNorm
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_rotary_position_embeddings=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_swiglu=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swiglu=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.swiglu_limit=10.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_masked_softmax_fusion=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.attention_softmax_in_fp32=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.min_lr=1e-6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.weight_decay=1e-2
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.clip_grad=1.0
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.adam_beta1=0.9
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.adam_beta2=0.999
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.initial_loss_scale=65536
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.vocab_size=129280
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.padded_vocab_size=129280
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.rotary_base=10000
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.norm_epsilon=1e-6
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_load_optim=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_load_rng=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.bf16=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.distributed_timeout_minutes=120
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_save_optim=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_save_rng=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.context_parallel_algo=ulysses_cp_algo
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.masked_softmax_fusion=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.moe_shared_expert_overlap=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.indexer_loss_coeff=0.0

    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_sfa=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_sinkhorn=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_triton_rmsnorm_without_weight=False
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.no_pad_to_seq_lengths=True

    # +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_lightning_indexer="${ascend_c_key}"
    # +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_fused_lightning_indexer_loss="${ascend_c_key}"
    # +actor_rollout_ref.actor.mindspeed.llm_kwargs.use_sparse_flash_attn="${ascend_c_key}"
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
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.max_num_seqs=32
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode='FULL_DECODE_ONLY'
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[1,2,4,8,16,32]"
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.load_format="safetensors"
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
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=20
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.use_legacy_worker_impl=disable
    trainer.resume_mode=disable
    actor_rollout_ref.actor.checkpoint.save_contents="['model']" 
)


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@" | tee logs/run_deepseek_v4_npu_$(date +%Y%m%d_%H%M%S).log
