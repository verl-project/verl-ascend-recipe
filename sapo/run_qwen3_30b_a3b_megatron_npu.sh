#!/usr/bin/env bash
# SAPO | Qwen3-30B-A3B-Base (MoE, 128 experts) | Megatron | vLLM rollout | Ascend NPU
#
# SAPO replaces ratio clipping with a smooth tau-parameterized surrogate
# (arXiv:2511.20347). This script carries the exact configuration that passed a
# 100-step acceptance run on 16 x Ascend 910B3 (2 nodes x 8). See README.md for
# the measured results and the probe series behind each non-obvious default.
#
#   NNODES=2 NDEVICES_PER_NODE=8 \
#   MODEL_PATH=/path/to/Qwen3-30B-A3B-Base \
#   MCORE_MODEL_PATH=/path/to/Qwen3-30B-A3B-Base-mcore \
#   TRAIN_FILE=/path/to/dapo-math-17k/train.parquet \
#   VAL_FILE=/path/to/aime-2024/test.parquet \
#   bash sapo/run_qwen3_30b_a3b_megatron_npu.sh

set -xeuo pipefail

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
# Pre-converted Megatron dist checkpoint. The accepted run used one; produce it
# with scripts/converter_hf_to_mcore.py --use_cpu_initialization from the verl
# tree. Leave empty to let mbridge load the HF weights directly.
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}

NNODES=${NNODES:-2}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

# SAPO smoothing temperatures (paper defaults).
# tau_pos/tau_neg are ActorConfig fields, NOT policy_loss fields:
# compute_policy_loss_sapo reads config.tau_pos off ActorConfig, so overriding
# them under actor.policy_loss.* silently has no effect and leaves the defaults
# in place with no error and no warning.
TAU_POS=${TAU_POS:-1.0}
TAU_NEG=${TAU_NEG:-1.05}

# Megatron parallelism. world must be divisible by EP*ETP*PP; the data-parallel
# and expert-data-parallel sizes are derived by megatron's mpu from world_size
# (dp = world/(TP*PP*CP), edp = world/(EP*ETP*PP)) and are never set here.
#
# EP=8/ETP=1 over EP=4/ETP=4 is the communication lever on this model: measured
# at the operator level, AlltoAllV absolute duration dropped 70% (16.1s -> 4.9s)
# and total device time 66% (42.1s -> 14.4s). EP compresses the all-to-all
# volume itself, not merely its share of the step.
TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-4}

# "full" recomputes every layer, "selective" only the cheap ops, "none"
# disables it. selective was measured at -10.4% step time for +7.6 GB device
# memory on this model, which does not pay for itself once the colocated
# rollout needs the headroom. Megatron rejects recompute_method and
# recompute_num_layers when granularity is selective, so each mode emits its
# own flag set rather than sharing one.
RECOMPUTE=${RECOMPUTE:-full}

# Fraction of optimizer state held on the host by the HybridDeviceOptimizer.
# This is load-bearing and must stay at 1 on this configuration: with all three
# offloads disabled the run OOMs in step 1. By contrast verl's own
# megatron.optimizer_offload is redundant here -- toggling it moved update_actor
# by 1.5%, which is noise -- so it is left False below.
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
# Packing ceiling rather than a fixed split: dynamic_bsz below repacks by token
# budget. micro2 -> micro4 bought -28.7% on update_actor at zero memory cost;
# micro8 bought a further -26.1% but still could not reach the throughput
# target on its own.
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}

# The decisive throughput lever on this recipe: token-budget packing via
# rearrange_micro_batches instead of a fixed sequence-count split. Throughput
# went 80.9 -> 113.2 (+40%) when this was enabled on top of the otherwise
# best static configuration, with the bulk of the gain in old_log_prob
# (173.9s -> 30.5s, -82.5%) because packing equalises the token count each rank
# actually computes.
#
# All three flags must agree. verl asserts actor against rollout and refuses to
# start when they disagree; ref is not validated but silently runs a different
# batching scheme, so it is set here too.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_N=${ROLLOUT_N:-8}
# Rollout and training share device memory; leave headroom for the offload
# traffic Megatron generates on Ascend.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}

PROJECT_NAME=${PROJECT_NAME:-verl_sapo_qwen3_moe}
# resume_mode defaults to auto, which picks up a checkpoint left by any earlier
# run sharing this name. Change it for every new run.
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_megatron_16npu}

# Checkpointing is disabled by default because saving is not yet reliable on
# this configuration -- see README.md "Known limitation". Set a positive value
# only if you have validated a save end to end on your cluster.
SAVE_FREQ=${SAVE_FREQ:--1}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
# Excluding 'optimizer' keeps a checkpoint at weight size (~57 GB) instead of
# ~374 GB on 16 ranks; the cost is that a resumed run restarts Adam from
# scratch.
SAVE_CONTENTS=${SAVE_CONTENTS:-'["model","extra"]'}

# Profiling is opt-in and costs nothing when off. Leave it off for any run whose
# timings you intend to quote: discrete per-role tracing inflated step time by
# ~3.3x in our measurements.
PROFILE=${PROFILE:-0}
PROFILE_STEPS=${PROFILE_STEPS:-"[2,3]"}
PROFILE_RANKS=${PROFILE_RANKS:-"[0]"}
PROFILE_ALL_RANKS=${PROFILE_ALL_RANKS:-False}
PROFILE_DISCRETE=${PROFILE_DISCRETE:-True}
PROFILE_LEVEL=${PROFILE_LEVEL:-level1}
PROFILE_ANALYSIS=${PROFILE_ANALYSIS:-True}
PROFILE_SAVE_PATH=${PROFILE_SAVE_PATH:-./profile_data}

# Prompt filtering runs to completion before any device work starts, and the
# shipped data config pins it to a single process (see
# trainer/config/data/legacy_data.yaml), so filtering the full dataset costs
# ~28 minutes with every accelerator idle. Restore the code default so the wait
# scales with the machine.
FILTER_WORKERS=${FILTER_WORKERS:-$(python3 -c 'import os; print(max(1, os.cpu_count() // 4))' 2>/dev/null || echo 8)}
TEST_FREQ=${TEST_FREQ:--1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
########################### end user-adjustable ###########################

########################### Ascend runtime ###########################
export CUDA_DEVICE_MAX_CONNECTIONS=1  # for megatron comm/compute overlap
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}  # more streams than FFTS+
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${FILTER_WORKERS}
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo
    actor_rollout_ref.actor.tau_pos=${TAU_POS}
    actor_rollout_ref.actor.tau_neg=${TAU_NEG}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    # SAPO drops ratio clipping, and the paper trains without a KL penalty.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    # Redundant on this configuration (measured at noise level) while the
    # HybridDeviceOptimizer below is doing the actual work.
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    # megatron.optimizer_offload alone does not move the distributed optimizer
    # state off-device: without these, Adam lazily allocates exp_avg/exp_avg_sq
    # on the accelerator during the first step() and a 30B MoE runs out of
    # memory there.
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    # 128 experts without fp32 routing is numerically fragile (Megatron warns).
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# Trailing extras array; stays non-empty-safe under `set -u`.
EXTRA=(
    model_engine=megatron
)

# Activation recomputation. Megatron validates these against each other:
# selective granularity requires recompute_num_layers/method to be unset, so
# the modes cannot share one flag set.
case "${RECOMPUTE}" in
    full)
        EXTRA+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        )
        ;;
    selective)
        EXTRA+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective
        )
        ;;
    none)
        ;;
    *)
        echo "Unsupported RECOMPUTE=${RECOMPUTE}. Expected 'full', 'selective' or 'none'." >&2
        exit 1
        ;;
esac

# All three roles are traced so the per-stage split is complete; tracing only
# the actor tells you update_actor is slow but not what it is competing with.
if [ "${PROFILE}" != 0 ]; then
    EXTRA+=(
        global_profiler.tool=npu
        global_profiler.steps=${PROFILE_STEPS}
        global_profiler.save_path="${PROFILE_SAVE_PATH}"
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.ranks=${PROFILE_RANKS}
        actor_rollout_ref.actor.profiler.all_ranks=${PROFILE_ALL_RANKS}
        actor_rollout_ref.actor.profiler.tool_config.npu.discrete=${PROFILE_DISCRETE}
        actor_rollout_ref.actor.profiler.tool_config.npu.contents="['npu','cpu']"
        actor_rollout_ref.actor.profiler.tool_config.npu.level=${PROFILE_LEVEL}
        actor_rollout_ref.actor.profiler.tool_config.npu.analysis=${PROFILE_ANALYSIS}
        actor_rollout_ref.rollout.profiler.enable=True
        actor_rollout_ref.rollout.profiler.ranks=${PROFILE_RANKS}
        actor_rollout_ref.rollout.profiler.all_ranks=${PROFILE_ALL_RANKS}
        actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=${PROFILE_DISCRETE}
        actor_rollout_ref.rollout.profiler.tool_config.npu.contents="['npu','cpu']"
        actor_rollout_ref.rollout.profiler.tool_config.npu.level=${PROFILE_LEVEL}
        actor_rollout_ref.rollout.profiler.tool_config.npu.analysis=${PROFILE_ANALYSIS}
        actor_rollout_ref.ref.profiler.enable=True
        actor_rollout_ref.ref.profiler.ranks=${PROFILE_RANKS}
        actor_rollout_ref.ref.profiler.all_ranks=${PROFILE_ALL_RANKS}
        actor_rollout_ref.ref.profiler.tool_config.npu.discrete=${PROFILE_DISCRETE}
        actor_rollout_ref.ref.profiler.tool_config.npu.contents="['npu','cpu']"
        actor_rollout_ref.ref.profiler.tool_config.npu.level=${PROFILE_LEVEL}
        actor_rollout_ref.ref.profiler.tool_config.npu.analysis=${PROFILE_ANALYSIS}
    )
fi

# Load from a pre-converted Megatron dist checkpoint when one is supplied.
if [ -n "${MCORE_MODEL_PATH}" ]; then
    EXTRA+=(
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}"
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}"
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
