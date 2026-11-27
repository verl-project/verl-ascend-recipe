#!/bin/bash
set -eo pipefail
source env/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export n_gpus_per_node=8 train_batch_size=32 ppo_mini_batch_size=8 rollout_n=4
export max_prompt_length=512 max_response_length=512 max_num_batched_tokens=2048 max_num_seqs=32 gpu_memory_utilization=0.5
export val_max_samples=128 val_n=1 save_freq=100 test_freq=20 total_training_steps=100
bash /rloo/run_qwen3_8b_rloo_fsdp_npu.sh --model_path=/workspace/Qwen3_8b --data_path=$PWD/data trainer.default_local_dir=$PWD/checkpoints trainer.resume_mode=disable trainer.total_epochs=2 actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.actor.fsdp_config.offload_policy=False actor_rollout_ref.ref.fsdp_config.param_offload=False actor_rollout_ref.ref.fsdp_config.offload_policy=False "$@"
