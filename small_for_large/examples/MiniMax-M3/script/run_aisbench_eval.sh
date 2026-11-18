#!/bin/bash
# MiniMax-M3 ais_bench 评测入口（打分规则与 gen_sft_dataset.py 的 mathruler 判分对齐）。
# 示例数据集为 geo3k（见 README）。
#
# 可用环境变量覆盖参数，例如:
#   NUM_PROMPTS=100 ./run_aisbench_eval.sh
#   DATASET_PATH=<geo3k_dataset_dir>/geo3k_bench_519/geo3k/test.parquet ./run_aisbench_eval.sh

HOST_IP=${HOST_IP:-127.0.0.1}
HOST_PORT=${HOST_PORT:-12345}
DATASETS=${DATASETS:-geometry3k_gen}
# 评测集二选一: 全量 geo3k_bench（阶段① 起服务时）或
# gen_benchmark_dataset_from_sft.py 的产物（阶段④ 用 SFT 集复评时）
DATASET_PATH=${DATASET_PATH:-<output_dir>/dataset_vllm_rollout/benchmark_test_after/geo3k/test.parquet}
NUM_PROMPTS=${NUM_PROMPTS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_OUT_LEN=${MAX_OUT_LEN:-4096}

GEO3K_GRADER=relaxed ais_bench --models vllm_api_general_chat \
  --datasets "$DATASETS" \
  --dataset-path "$DATASET_PATH" \
  --num-warmups 0 \
  --num-prompts "$NUM_PROMPTS" \
  --max-out-len "$MAX_OUT_LEN" \
  --generation-kwargs '{"temperature": 0}' \
  --host-ip "$HOST_IP" \
  --host-port "$HOST_PORT" \
  --batch-size "$BATCH_SIZE"
