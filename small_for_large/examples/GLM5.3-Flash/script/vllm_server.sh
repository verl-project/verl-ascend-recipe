#!/bin/bash
# GLM-5.3-Flash 评测用 vLLM 服务（8 卡 TP=8, 供 ais_bench 评测）
#
# 两种镜像模式, 用 MODE 环境变量切换:
#   MODE=stable  (默认) 稳定镜像: eager 模式（--enforce-eager）, --max-num-seqs 16
#   MODE=nightly       nightly-main 镜像（重构版 glm5next + KDA）: 图模式, --max-num-seqs 32
#
# 示例数据集为 geo3k（见 README）；评测配套脚本: run_aisbench_eval.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=400

MODE=${MODE:-stable}
# workspace root: holds the model workspace (start/, dataset/, logs/ ...); override as needed
BASE_DIR=${BASE_DIR:-$HOME/glm5-3-s_b}
MODEL=${MODEL:-/home/weight/GLM-5.3-Flash-BF16}
# ais_bench 通过 file:// 引用图片, 需指向 gen_benchmark_dataset_from_verl.py 的输出目录
MEDIA_PATH=${MEDIA_PATH:-$BASE_DIR/dataset/benchmark/geo3k_bench}
LOG_DIR="$BASE_DIR/start/logs/vllm"
mkdir -p "$LOG_DIR"

case "$MODE" in
  stable)
    MAX_NUM_SEQS=16
    extra_args=(--enforce-eager)   # 稳定镜像上图模式不可用, 强制 eager
    LOG_TAG="vllm"
    ;;
  nightly)
    MAX_NUM_SEQS=32               # nightly 图模式可用, 并发也更高
    extra_args=()
    LOG_TAG="vllm_nightly"
    ;;
  *)
    echo "ERROR: MODE 必须是 stable 或 nightly, 当前: $MODE" >&2
    exit 1
    ;;
esac

LOG_FILE="$LOG_DIR/${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log"
echo "MODE=$MODE, vllm log -> $LOG_FILE"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 12345 \
  --data-parallel-size 1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --served-model-name auto \
  --safetensors-load-strategy prefetch \
  --max-num-seqs "$MAX_NUM_SEQS" \
  "${extra_args[@]}" \
  --max-model-len 16384 \
  --chat-template "$BASE_DIR/start/glm53_train_prompt.jinja" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --allowed-local-media-path "$MEDIA_PATH" \
  --gpu-memory-utilization 0.9 \
  --api-server-count 1 2>&1 | tee "$LOG_FILE"

echo "vllm exited, log saved to $LOG_FILE"

# 图文评测 (geo3k test split, 601 题): 直接运行 ./run_aisbench_eval.sh
# 注意: ais_bench 无 --dataset-path 参数, 数据集路径写在
# benchmark/ais_bench/benchmark/configs/datasets/<dataset>/ 的配置 path 中
