#!/bin/bash
# MiniMax-M3 评测用 vLLM 服务，供 ais_bench 图文评测（示例数据集为 geo3k，见 README）。
#
# 两种服务对象（按阶段二选一）:
#   - 全尺寸 MiniMax-M3（~428B/23B 激活，BF16 约 796 GiB）:
#       ASCEND_RT_VISIBLE_DEVICES=0,...,15 TP_SIZE=16 MODEL_PATH=<output_dir>/MiniMax-M3-serve
#       （8 卡时每卡 ~100 GiB > 64 GiB，放不下；全尺寸目录为软链 + 修正过 config.json
#        （num_hidden_layers 12 -> 60）的副本）
#   - 裁剪/SFT 产物（如 Minimax-M3-moe8，或 dcp_to_release_hf.py 的输出）:
#       默认 8 卡 TP=8 即可
#
# 评测配套脚本: run_aisbench_eval.sh
set -e
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 全尺寸服务时改为 0,1,...,15（并把 TP_SIZE 提到 16）
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
TP_SIZE=${TP_SIZE:-8}

# 服务对象: SFT 训练后的 hf_release 产物（默认）或全尺寸 serve 目录
MODEL_PATH=${MODEL_PATH:-<output_dir>/CKPT/hf_release/global_step_1000}
# ais_bench 通过 file:// 引用图片, 需指向评测数据目录（gen_benchmark_dataset_from_sft.py 的输出）
MEDIA_PATH=${MEDIA_PATH:-<output_dir>/dataset_vllm_rollout/benchmark_test_after}
echo "MODEL_PATH : ${MODEL_PATH}"

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 12345 \
  --served-model-name auto \
  --trust-remote-code \
  --tensor-parallel-size "$TP_SIZE" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --reasoning-parser minimax_m3 \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --allowed-local-media-path "$MEDIA_PATH" \
  --gpu-memory-utilization 0.90 \
  --safetensors-load-strategy prefetch \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
