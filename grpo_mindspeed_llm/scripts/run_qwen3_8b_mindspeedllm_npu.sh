#!/bin/bash
set -xeuo pipefail
# Project Configuration
project_name='GRPO-Qwen3-8B-BASE-MATH'
exp_name='GRPO-Qwen3-8B-BASE-MindSpeedLLM-SGLang'

# Necessary env
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DISABLE_L2_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export TORCH_COMPILE_DISABLE=True

# 让 verl 自动发现并注册 mindspeed_llm 引擎 + 链接 hydra yaml 配置
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH="${SCRIPT_DIR}/..:$PYTHONPATH"
export VERL_USE_EXTERNAL_MODULES=backend

# Node Info
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# Model Weights Paths
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=$RAY_DATA_HOME/gsm8k/train.parquet
TEST_FILE=$RAY_DATA_HOME/gsm8k/test.parquet
# Data Length Configuration
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))

# Training Batch Configuration
train_prompt_bsz=16
train_prompt_mini_bsz=16
n_resp_per_prompt=8
micro_batch_size=1

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
train_tp=4
train_pp=2

# SGLang Generation Configuration
gen_tp=4
gen_dp=1
gpu_memory_utilization=0.5
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

# Data Configuration
DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='left'
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.grad_offload=${all_offload}
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    actor_rollout_ref.actor.mindspeed.llm_kwargs.spec='[mindspeed_llm.tasks.models.spec.qwen3_spec, layer_spec]'
    actor_rollout_ref.actor.mindspeed.llm_kwargs.seq_length=${max_model_len}
    actor_rollout_ref.actor.mindspeed.llm_kwargs.micro_batch_size=${micro_batch_size}
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.num_query_groups=8
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_method=uniform
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_granularity=full
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.recompute_num_layers=1
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.overlap_grad_reduce=True
    +actor_rollout_ref.actor.mindspeed.llm_kwargs.overlap_param_gather=True
)

REF_CONFIG=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.mindspeed.param_offload=${all_offload}
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=sglang
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
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
    trainer.save_freq=-1
    trainer.default_local_dir="${CKPTS_DIR}"
)

# profiling configuration
PROF_CONFIG=(
    global_profiler.tool=npu
    global_profiler.steps=null
    global_profiler.save_path=/profpath
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks="[0]"
    actor_rollout_ref.actor.profiler.all_ranks=False
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=True
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=['npu','cpu']
    actor_rollout_ref.actor.profiler.tool_config.npu.level=level0
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=True
    actor_rollout_ref.rollout.profiler.enable=True
    actor_rollout_ref.rollout.profiler.ranks="[0]"
    actor_rollout_ref.rollout.profiler.all_ranks=False
)

python3 -m verl.trainer.main_ppo \
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
    "${PROF_CONFIG[@]}" \
    "$@"
