#!/usr/bin/env bash
set -xeuo pipefail

# Qwen3-8B RLOO (REINFORCE Leave-One-Out) with FSDP2 training and
# vLLM-Ascend rollout on Ascend NPU.
#
# Usage:
#   bash run_qwen3_8b_rloo_fsdp_npu.sh \
#     --data_path=/path/to/data \
#     --model_path=/path/to/Qwen3-8B
#
# The data directory must contain train.parquet and test.parquet.
data_path=""
model_path=""
extra_args=()

for para in "$@"; do
    if [[ $para == --data_path=* ]]; then
        data_path="${para#*=}"
    elif [[ $para == --model_path=* ]]; then
        model_path="${para#*=}"
    else
        extra_args+=("$para")
    fi
done

if [[ -z "$data_path" || -z "$model_path" ]]; then
    echo "Error: Both --data_path and --model_path must be provided."
    exit 1
fi

ENGINE=vllm
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400
export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-0}

# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0

project_name=${project_name:-rloo_qwen3_8b_ascend}
experiment_name=${experiment_name:-qwen3_8b_rloo_fsdp_1npu}

# prompt&response length
max_prompt_length=${max_prompt_length:=1024}
max_response_length=${max_response_length:=4096}
max_model_len=$((max_prompt_length + max_response_length))

# vLLM-Ascend parameters
free_cache_engine=True
enable_chunked_prefill=True
gpu_memory_utilization=${gpu_memory_utilization:=0.82}
tensor_model_parallel_size=${tensor_model_parallel_size:=1}
rollout_n=${rollout_n:=5}
rollout_agent_num_workers=${rollout_agent_num_workers:=24}
max_num_batched_tokens=${max_num_batched_tokens:=16384}
max_num_seqs=${max_num_seqs:=32}

# batch size
train_batch_size=${train_batch_size:=256}
ppo_mini_batch_size=${ppo_mini_batch_size:=16}

# training params
nnodes=1
n_gpus_per_node=${n_gpus_per_node:-1}
total_training_steps=${total_training_steps:=100}
save_freq=${save_freq:=10}
test_freq=${test_freq:=10}
val_before_train=${val_before_train:=True}
val_max_samples=${val_max_samples:=16}
val_n=${val_n:=16}

# RLOO parameters
adv_estimator=rloo
use_kl_in_reward=True
kl_penalty=kl
kl_coef=0.001
use_kl_loss=False
entropy_coeff=0

# Performance parameters. Offload is required for the verified 1-NPU setup.
use_dynamic_bsz=True
offload=True
fsdp_size=-1
loss_agg_mode="token-mean"
actor_ppo_max_token_len=8192
log_prob_max_token_len=8192

# reward
overlong_buffer_enable=False
overlong_buffer_len=4096
overlong_buffer_penalty_factor=0.0

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_penalty=${kl_penalty} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="$data_path/train.parquet" \
    data.val_files="$data_path/test.parquet" \
    data.prompt_key=prompt \
    data.train_batch_size=${train_batch_size} \
    data.train_max_samples=-1 \
    data.val_max_samples=${val_max_samples} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.dataloader_num_workers=8 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=8 \
    data.truncation=left \
    data.shuffle=True \
    data.return_raw_chat=True \
    data.return_full_prompt=False \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.offload_policy=${offload} \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.agent.num_workers=${rollout_agent_num_workers} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=${free_cache_engine} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes='[1,2,4,8,16]' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=1 \
    reward.reward_manager.name=dapo \
    reward.num_workers=8 \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${overlong_buffer_enable} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_buffer_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=True \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    ray_kwargs.ray_init.num_cpus=32 \
    trainer.device=npu \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.log_val_generations=16 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.resume_mode=auto \
    critic.enable=False \
    "${extra_args[@]}"
