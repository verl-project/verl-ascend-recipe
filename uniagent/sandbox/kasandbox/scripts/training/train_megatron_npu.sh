#!/usr/bin/env bash
# Validate and submit the colleague recipe through our KASandbox deployment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
set -a
source "$ROOT/examples/e2b_swe_grpo/zmj.sh"
set +a
: "${RUNTIME_ENV:?Generate a private Ray runtime JSON before submission}"

if (( TOTAL_NPUS != PHYSICAL_NPUS )); then
    echo "Colocated trainer and rollout request ${TOTAL_NPUS} NPUs, but PHYSICAL_NPUS=${PHYSICAL_NPUS}" >&2
    exit 1
fi
if (( TOTAL_NPUS % TRAIN_MODEL_PARALLEL_SIZE != 0 )); then
    echo "Total NPUs (${TOTAL_NPUS}) must be divisible by TP*PP*CP (${TRAIN_MODEL_PARALLEL_SIZE})" >&2
    exit 1
fi
if (( ROLLOUT_NPUS % ROLLOUT_TP != 0 )); then
    echo "Rollout NPUs (${ROLLOUT_NPUS}) must be divisible by ROLLOUT_TP (${ROLLOUT_TP})" >&2
    exit 1
fi
if (( TRAIN_PROMPT_BSZ % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "TRAIN_PROMPT_BSZ (${TRAIN_PROMPT_BSZ}) must be divisible by PPO_MINI_BATCH_SIZE (${PPO_MINI_BATCH_SIZE})" >&2
    exit 1
fi
if [[ ! -r "${TASK_CONFIG}" ]]; then
    echo "Task config does not exist or is not readable: ${TASK_CONFIG}" >&2
    exit 1
fi
if [[ "${TRAINING_DRY_RUN:-false}" != true && ! -r "${RUNTIME_ENV}" ]]; then
    echo "Ray runtime-env does not exist or is not readable: ${RUNTIME_ENV}" >&2
    exit 1
fi
if [[ "${TRAINING_DRY_RUN:-false}" == true ]]; then
    TRAIN_DATA_ROWS=not-checked
else
if [[ ! -r "${TRAIN_FILE}" ]]; then
    echo "Training data does not exist or is not readable: ${TRAIN_FILE}" >&2
    exit 1
fi
if ! TRAIN_DATA_ROWS="$(python3 - "${TRAIN_FILE}" <<'PY'
import sys

import pyarrow.parquet as pq

print(pq.read_metadata(sys.argv[1]).num_rows)
PY
)"; then
    echo "Failed to read the training-data row count: ${TRAIN_FILE}" >&2
    exit 1
fi
if [[ ! "${TRAIN_DATA_ROWS}" =~ ^[0-9]+$ ]]; then
    echo "Invalid training-data row count: ${TRAIN_DATA_ROWS}" >&2
    exit 1
fi
if (( TRAIN_DATA_ROWS < TRAIN_PROMPT_BSZ )); then
    echo "Training data has ${TRAIN_DATA_ROWS} rows, fewer than TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}." >&2
    echo "Lower TRAIN_PROMPT_BSZ or provide a larger dataset; otherwise trainer v1 gets steps_per_epoch=0." >&2
    exit 1
fi

fi

RAY_VERSION_URL="${RAY_JOB_ADDRESS%/}/api/version"
if [[ "${TRAINING_DRY_RUN:-false}" != true ]] && ! curl -fsS --connect-timeout 3 --max-time 10 "${RAY_VERSION_URL}" >/dev/null; then
    echo "Ray Jobs API is unavailable: ${RAY_VERSION_URL}" >&2
    echo "Run this script on the Ray head node or set RAY_JOB_ADDRESS to its dashboard URL." >&2
    exit 1
fi

echo "===== Two-node 32-card Qwen3.5-9B Dense training topology ====="
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}, rows=${TRAIN_DATA_ROWS}"
echo "Resources: trainer=${NNODES}x${NGPUS_PER_NODE}, rollout=${ROLLOUT_NNODES}x${ROLLOUT_NGPUS_PER_NODE}, total_npu=${PHYSICAL_NPUS}"
echo "TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}, N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}, trajectories_per_step=$((TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT))"
echo "CONCURRENCY=${CONCURRENCY}, NUM_AGENT_WORKERS=${NUM_AGENT_WORKERS}, GATEWAY_COUNT=${GATEWAY_COUNT}"
echo "TRAIN TP/PP/CP=${TRAIN_TP}/${TRAIN_PP}/${TRAIN_CP}, DP=$((TOTAL_NPUS / TRAIN_MODEL_PARALLEL_SIZE)), ROLLOUT_TP=${ROLLOUT_TP}, rollout_replicas=${ROLLOUT_REPLICAS}"
echo "Algorithm: adv_estimator=${ADV_ESTIMATOR}, loss_mode=${LOSS_MODE}, clip=${CLIP_RATIO_LOW}/${CLIP_RATIO_HIGH}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "Ray job address=${RAY_JOB_ADDRESS}"
echo "GDN backend: AscendC (use_ascend_gdn=True); all-0/all-1 group filtering disabled"

ray job submit --address="${RAY_JOB_ADDRESS}" --runtime-env "${RUNTIME_ENV}" \
    -- env PYTHONPATH="${PYTHONPATH}" PYTHONUNBUFFERED=1 RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
    VLLM_LOGGING_LEVEL=INFO \
    VLLM_LOG_STATS_INTERVAL=10 \
    python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    transfer_queue.enable=True \
    transfer_queue.metrics.enabled=True \
    trainer.device=npu \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings="${MAX_MODEL_LEN}" \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${TEST_FILE}']" \
    data.prompt_key=prompt \
    data.truncation=error \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    data.gen_batch_size="${TRAIN_PROMPT_BSZ}" \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection="${TRAJECTORY_SELECTION}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="${MASK_UNFINISHED_EPISODE}" \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=false \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}" \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    algorithm.rollout_correction.bypass_mode="${BYPASS_MODE}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    algorithm.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE}" \
    algorithm.rollout_correction.rollout_rs="${ROLLOUT_RS}" \
    algorithm.rollout_correction.rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD}" \
    algorithm.rollout_correction.loss_type="${BYPASS_LOSS_TYPE}" \
    actor_rollout_ref.actor.checkpoint.strict=False \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode="${BYPASS_MODE}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs="${ROLLOUT_RS}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type="${BYPASS_LOSS_TYPE}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16 \
    '+actor_rollout_ref.actor.megatron.override_ddp_config={grad_reduce_in_fp32: false, overlap_grad_reduce: true, bucket_size: 50000000}' \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.ref.megatron.use_remove_padding=True \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    'trainer.logger=["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
