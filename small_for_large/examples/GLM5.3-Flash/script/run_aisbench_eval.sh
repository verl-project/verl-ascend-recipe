#!/bin/bash
# GLM-5.3-Flash ais_bench 评测入口（打分规则已对齐 verl RL 训练的 math reward）
# 示例数据集为 geo3k（见 README）。数据集路径已在 benchmark 配置中指定，
# 无法通过命令行覆盖：
#   benchmark/ais_bench/benchmark/configs/datasets/geometry3k/geometry3k_gen.py
#   -> path="<BASE_DIR>/dataset/benchmark/geo3k_bench/geo3k/test.parquet"
#
# 可用环境变量覆盖参数，例如全量 601 题：
#   NUM_PROMPTS=601 ./run_aisbench_eval.sh

HOST_IP=${HOST_IP:-127.0.0.1}
HOST_PORT=${HOST_PORT:-12345}
DATASETS=${DATASETS:-geometry3k_gen}
NUM_PROMPTS=${NUM_PROMPTS:-12}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_OUT_LEN=${MAX_OUT_LEN:-12288}

ais_bench --models vllm_api_general_chat \
  --datasets "$DATASETS" \
  --num-prompts "$NUM_PROMPTS" \
  --num-warmups 0 \
  --max-out-len "$MAX_OUT_LEN" \
  --generation-kwargs '{"temperature": 0}' \
  --host-ip "$HOST_IP" \
  --host-port "$HOST_PORT" \
  --batch-size "$BATCH_SIZE"
