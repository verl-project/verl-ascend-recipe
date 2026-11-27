#!/usr/bin/env bash
# Qwen3.5-9B Dense, matching the colleague two-node topology.
export MODEL_PATH="${MODEL_PATH:-${HOME}/models/Qwen3.5-9B}"
export TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
export NNODES="${NNODES:-2}" NGPUS_PER_NODE="${NGPUS_PER_NODE:-16}"
MODEL_ARGS=()
