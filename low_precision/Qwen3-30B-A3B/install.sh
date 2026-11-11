#!/bin/bash
set -ex

cd /vllm-ascend
git checkout 59ebc4384b340065127e9b1e1bb73a257c51895e
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/Qwen3-30B-A3B/patch/vllm-ascend.patch

cd /verl
git checkout 07ba1a854e57356ccf1c9d604898607a88e3da3e
git apply --whitespace=nowarn ../verl-ascend-recipe/low_precision/Qwen3-30B-A3B/patch/verl.patch
