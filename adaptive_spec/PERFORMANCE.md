# Performance

These measurements compare fixed K and AdaptiveSpec with the same model and deployment. Throughput is generated tokens divided by generation time. RL rollout paths are stochastic, so individual runs are also reported.

| Model / task | Fixed-K throughput (tokens/s) | AdaptiveSpec throughput (tokens/s) | Improvement |
|---|---:|---:|---:|
| Qwen3.5-9B / DAPO-Math | 1676.802 (K12) | 1896.725 | **13.12%** |
| Qwen3.5-9B / APPS | 1107.476 (K7) | 1229.569 | **11.02%** |

Both use the original BF16 precision, 4 TP2 Engines, up to 16 sequences per Engine, eager execution, a 65536-token response limit, a model length of 69632, 8192 batched tokens, and memory utilization of 0.5. Each run contains 32 prompts with 8 responses each, for 256 responses. Both APPS arms use the same Elastic scheduling.

## DAPO-Math

The input is length-stratified (H32), with thinking enabled. The summary reports the median throughput of three runs. K12 has the best median among the fixed K values measured.

| Run | Fixed K12 | AdaptiveSpec |
|---|---:|---:|
| 1 | 1676.802 | 2050.003 |
| 2 | 1308.530 | 1563.704 |
| 3 | 1751.769 | 1896.725 |

Units: tokens/s. The autoregressive median is 750.449 tokens/s; AdaptiveSpec achieves 2.53 times that throughput.

## APPS

The workload consists of 32 randomly sampled, frozen training prompts, with thinking disabled. A separate, non-overlapping set of 8 prompts selects fixed K7 through independent calibration before the runs below. Each seed forms a paired comparison.

| Generation seed | Fixed K7: seconds / tokens / tokens/s | AdaptiveSpec: seconds / tokens / tokens/s |
|---|---|---|
| 2026110201 | 930.761 / 1154205 / 1240.066 | 904.968 / 1062969 / 1174.593 |
| 2026110202 | 1086.026 / 1079339 / 993.842 | 765.724 / 991262 / 1294.543 |

Pooling total tokens over total time gives an 11.02% throughput improvement. Total time decreases by 17.16% and total output by 8.03%. The individual throughput changes are -5.28% and +30.26%. One response reaches the output limit in the first AdaptiveSpec run and one in the second fixed-K run.

AdaptiveSpec is suited to finite rollouts with mixed response lengths, multiple responses per prompt, and a gradually decreasing active batch. Evidence can be reused across sibling responses, and the drain phase provides room to adjust K.

See [VALIDATION.md](VALIDATION.md) for the test environment and validation results.
