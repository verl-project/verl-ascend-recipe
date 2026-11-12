# DeepSeek-V4-Flash Small-for-Large Validation Recipe (Ascend NPU)

## Background

Industry-wide, RL single-step training for large models like DeepSeekV4 is extremely time-consuming and requires massive compute resources.

Therefore, this recipe attempts to prune a smaller model while maintaining the base model architecture, for rapid RL training accuracy verification, issue localization, and custom feature debugging.

This recipe is a post-training example for DeepSeek-V4-Flash on NPU, based on GRPO using the gsm8k dataset.

## 1. Preparation

### Environment Setup

#### Inference Scenario

**Option 1: Use Pre-built Image**

Refer to the official vllm-ascend [DeepSeek-V4](https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html) deployment documentation and use an image that supports DeepSeek-V4.

**Option 2: Build from Source**

```bash
# vLLM (v0.13.0)
git clone --branch v0.13.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e . -i https://repo.huaweicloud.com/repository/pypi/simple

# vLLM-Ascend (tag: v0.13.0rc3)
git clone --branch v0.13.0rc3 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e . -i https://repo.huaweicloud.com/repository/pypi/simple
```

#### Training-Inference Scenario (TODO)


### Prepare Model Weights

[DeepSeek-V4-Flash Open Weights](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)

- Weights need to be converted to BF16. This step requires approximately **692 GB** of disk space (149 GB source + 543 GB output).
- Under the `--output_hf_path` directory, you need to:
  - Edit `config.json` and remove the `quantization_config` dictionary.
  - Create `chat_template.jinja` and copy the content from: https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp/file/view/master/chat_template.jinja?status=0

```bash
# Model weight conversion script
python convert_model_flash.py --input_fp8_hf_path=/path/DeepSeek-V4-Flash --output_hf_path=/path/DeepSeek-V4-Flash-bf16 --quant_type=bfloat16
```

### Prepare Model Evaluation Tools

Follow the [vllm-ascend official documentation](https://docs.vllm.ai/projects/ascend/en/v0.13.0/developer_guide/evaluation/index.html) to select and use the evaluation tool.


## 2. Prune a Small Model

### Core Idea

DeepSeek-V4-Flash has 43 MoE layers with 256 experts each. The first 3 layers use hash routing to select 6 active experts, while subsequent layers compute 6 active experts based on input tokens and a scoring function.

With 256 experts, the model achieves high scores across diverse datasets. For a specific dataset, redundant experts can be pruned by retaining only high-frequency experts, maintaining reasonable performance.

- Add high-frequency expert saving code to the vllm-ascend repository.
- Launch inference service with the original model, run inference on the target dataset, and collect token-level active expert data for each MoE layer.
- Convert token-level active expert data into dataset-level high-frequency experts per MoE layer.
- Prune redundant experts from the latter 40 layers based on high-frequency experts.
- The first 3 layers cannot be pruned because they use hash routing to select active experts.
- For the latter 40 layers, prune to retain only the top-64 high-frequency experts.

### Steps

**1. Apply the small-for-large patch**

```bash
# Check vllm-ascend installation path
pip show vllm-ascend

# Copy vllm-ascend.patch to the vllm-ascend directory and apply it
cd /path/to/vllm-ascend
git apply vllm-ascend.patch
```

**2. Launch inference service on single node**

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1

export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Note: Set this environment variable to save token-level active experts for each MoE layer.
# Comment it out when not needed.
export TOPK_SAVE_PATH="/path/to/topk_ids_by_layer"

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/DeepSeek-V4-Flash-bf16 \
  --host 0.0.0.0 \
  --max_model_len 22528 \
  --max-num-batched-tokens 22528 \
  --served-model-name auto \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --max-num-seqs 48 \
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template /path/to/DeepSeek-V4-Flash-bf16/chat_template.jinja \
  --port 8006 \
  --enforce-eager
```

**3. Clear activation expert cache from dummy run**

After the inference service is ready, delete all `.txt` files in the `TOPK_SAVE_PATH` folder.

```bash
rm -rf topk_ids_by_layer/*
```

**4. Collect activation expert cache for the target dataset**

Take `ais_bench` as an example for dataset evaluation.

Create `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_V4.py` with the following content:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/path/to/DeepSeek-V4-Flash-bf16",
        model="auto",   # Must match the served-model-name used when launching the inference service
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="localhost",
        host_port=8006,
        max_out_len=1024,
        batch_size=48,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

```bash
# Verify configuration exists
ais_bench --models vllm_api_general_V4 --dataset gsm8k_gen --search
# Evaluate 500 prompts; adjust as needed for full dataset evaluation
ais_bench --models vllm_api_general_V4 --dataset gsm8k_gen --num-prompt 500 --num-warmups=0
```

Original model scores:

| dataset | #Experts | #Active | num-prompt | vllm-api-general-chat |
| ------- | -------- | ------- | ---------- | --------------------- |
| gsm8k   | 256      | 6       | 500        | 92.60                 |

After evaluation, `TOPK_SAVE_PATH` will contain the activated experts for all MoE layers across the `gsm8k-500` dataset.

**5. Identify high-frequency experts**

```bash
# --log-dir should point to the path previously specified in TOPK_SAVE_PATH.
# This generates the top64_experts.json file.
python stat_moe.py --top-k=64 --log-dir=""
```

**6. Prune model by high-frequency experts**

```bash
# --json-path points to the top64_experts.json generated in the previous step.
python expert_pruned.py --top-k=64 --json-path=top64_experts.json --model-dir=/path/to/DeepSeek-V4-Flash-bf16
```

**7. Validate the pruned model**

Launch inference service with the pruned model:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export USE_MULTI_BLOCK_POOL=1

export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Comment out this environment variable
# export TOPK_SAVE_PATH="/path/to/topk_ids_by_layer"

# Use the pruned model path
vllm serve /path/to/DeepSeek-V4-Flash-bf16-top64-pruned \
  --host 0.0.0.0 \
  --max_model_len 22528 \
  --max-num-batched-tokens 22528 \
  --served-model-name auto \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --max-num-seqs 48 \
  --async-scheduling \
  --enable-expert-parallel \
  --chat-template /path/to/DeepSeek-V4-Flash-bf16-top64-pruned/chat_template.jinja \
  --port 8006 \
  --enforce-eager
```

Evaluate the pruned model with `ais_bench`:

Create `benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_V4_pruned.py` with the following content:

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="/path/to/DeepSeek-V4-Flash-bf16-top64-pruned",
        model="auto",   # Must match the served-model-name used when launching the inference service
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="localhost",
        host_port=8006,
        max_out_len=1024,
        batch_size=48,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

Run evaluation:

```bash
# Verify configuration exists
ais_bench --models vllm_api_general_V4_pruned --dataset gsm8k_gen --search
# Evaluate 500 prompts; adjust as needed for full dataset evaluation
ais_bench --models vllm_api_general_V4_pruned --dataset gsm8k_gen --num-prompt 500 --num-warmups=0
```

Pruned model scores:

| dataset | #Experts | #Active | num-prompt | vllm-api-general-chat |
| ------- | -------- | ------- | ---------- | --------------------- |
| gsm8k   | 64       | 6       | 500        | 39.58                 |


**8. Verify the pruned checkpoint (recommended)**

`expert_pruned.py` rewrites shards, renumbers experts and rewrites `config.json` in one pass. We recommend verifying the pruned output before serving it, with the companion gate `verify_pruned.py`:

```bash
# 8.1 BEFORE pruning: snapshot the source directory fingerprint
python verify_pruned.py --source /path/DeepSeek-V4-Flash-bf16 --snapshot source_fp.json --fingerprint-only

# 8.2 prune as usual (step 6), then verify:
python verify_pruned.py --source /path/DeepSeek-V4-Flash-bf16 --pruned /path/DeepSeek-V4-Flash-bf16-top64-pruned --json-path top64_experts.json --top-k 64 --snapshot source_fp.json
```

The gate verifies, without loading the full model into memory:

| Check | What it proves |
|---|---|
| source integrity | the source directory is unchanged since the pre-prune snapshot (the prune provably never touched the original weights) |
| kept-expert fidelity | every kept expert tensor in the pruned model is byte-equal to the mapped slice of the source, using the same remapping `expert_pruned.py` applies |
| router fidelity | `gate.weight` / `gate.bias` rows of kept experts are byte-equal to the source rows |
| coverage | every prunable layer is present in the topk JSON — a missing layer would otherwise silently keep experts `0..N-1` |
| config diff | only `n_routed_experts` changed |
| stale expert dims | passthrough tensors in pruned layers that still carry a dimension of the old expert count are reported for review |

Results are written to `PROVENANCE.json` next to the pruned model (source fingerprint, keep-list hash, per-check stats, verdict). The tool exits non-zero on any failure, so it can be used as a hard gate in automated pipelines.

## 3. Training-Side Patch Adaptation (TODO)