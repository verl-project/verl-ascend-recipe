# Qwen3-8B ReMax with FSDP and vLLM-Ascend

This recipe runs Qwen3-8B ReMax training on an Atlas 800T A2/A3 node with FSDP for training and vLLM-Ascend for rollout. It is the Ascend counterpart of `examples/remax_trainer/run_qwen3_8b_fsdp.sh` in the core [verl](https://github.com/verl-project/verl) repository.

- Tracking issue: [verl-ascend-recipe #23](https://github.com/verl-project/verl-ascend-recipe/issues/23)
- Script: [`run_qwen3_8b_remax_fsdp_npu.sh`](run_qwen3_8b_remax_fsdp_npu.sh)
- Required core version: [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt)

## Validated environment

| Component | Version or configuration |
| --- | --- |
| Hardware | Atlas 800T A2, 8 x Ascend 910B3 64 GB |
| CANN | 25.5.1 |
| torch_npu | 2.9.0.post2 |
| vLLM | 0.18 |
| vLLM-Ascend | 0.18.1.dev41 |
| transformers | 5.3.0.dev0 |
| Model | Qwen3-8B |
| Training backend | FSDP |
| Rollout backend | vLLM-Ascend |
| Dataset | GSM8K and MATH |

## Adaptation and tuning

ReMax generates sampled responses together with one greedy baseline per prompt. The existing verl agent-loop path already combines these into one backend-independent rollout request, so no core trainer or rollout changes are needed for vLLM-Ascend.

The Ascend recipe applies the following platform-specific settings:

- sets `trainer.device=npu` and uses the vLLM backend, which resolves to vLLM-Ascend in the NPU environment;
- disables torch compilation for the actor and reference model;
- enables actor parameter and optimizer offload and reference parameter offload to leave HBM for the colocated rollout engine;
- uses a conservative rollout HBM utilization of `0.6` because ReMax produces sampled responses plus a greedy baseline;
- enables full-decode-only ACL graph capture and exports HCCL timeout, task queue, and CPU-affinity settings.

## Data preparation

Run these commands from the core verl checkout:

```bash
python3 examples/data_preprocess/gsm8k.py --local_save_dir "$HOME/data/gsm8k"
python3 examples/data_preprocess/math_dataset.py --local_save_dir "$HOME/data/math"
```

Use `GSM8K_DATA_DIR` and `MATH_DATA_DIR` when the Parquet files are stored elsewhere. Each directory must contain `train.parquet` and `test.parquet`.

## Run

From the core verl checkout:

```bash
MODEL_PATH=/data/models/Qwen3-8B \
bash /path/to/verl-ascend-recipe/remax/run_qwen3_8b_remax_fsdp_npu.sh
```

The script accepts Hydra overrides after the environment-variable configuration. For example:

```bash
TRAIN_BATCH_SIZE=128 \
ROLLOUT_TP=2 \
ROLLOUT_GPU_MEM_UTIL=0.6 \
MODEL_PATH=/data/models/Qwen3-8B \
bash /path/to/verl-ascend-recipe/remax/run_qwen3_8b_remax_fsdp_npu.sh \
    'trainer.logger=["console"]'
```

## Validation results

The following results were collected on the validated eight-NPU A2 environment:

| Test | Result |
| --- | --- |
| Smoke test | Qwen3-0.6B completed 2 steps; reward KL, ReMax advantages, and temperature-zero greedy baseline requests were observed |
| Qwen3-8B stability | Batch size 128 and response length 1024 completed 5 consecutive steps without OOM or HCCL timeout |
| Qwen3-8B training | A 60-step GSM8K run completed in approximately 2 hours 25 minutes |
| Reward trend | `critic/score/mean` increased from approximately 0.26 at step 1 to the 0.60-0.76 range, peaking at 0.764 |
| Throughput | End-to-end `perf/throughput` averaged approximately 381 tokens/s over 60 steps, with a 337-425 tokens/s range |
| Stress test | Response length 2048 completed 3 steps without OOM at 343-411 tokens/s |

The measured throughput exceeds the issue's absolute target of 100 tokens/s, and the reward trend is positive. The recorded 60-step run does not by itself meet the separate 100-step or 12-hour duration threshold in the issue; a longer validation log should be attached to the issue when available.

## Known limitations

- Parameter and optimizer offload provide safe HBM headroom on 64 GB devices at the cost of CPU transfer overhead.
- Larger-HBM devices can retune offload and `ROLLOUT_GPU_MEM_UTIL` for higher throughput.
- The reported long-run metrics use GSM8K rather than the combined default GSM8K and MATH training input.
- This contribution is configuration-level enablement and does not add a core algorithm or backend patch.
