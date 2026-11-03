# Rollout performance

## Workload and metric

The selected evaluation uses Qwen3.5-9B-Base with DAPO-Math, thinking enabled, temperature 1, top-p 1 and top-k -1. The response cap is 65,536 tokens. A frozen trace contains 16 prompts, 8 responses per prompt and 2 rollout steps: 256 requests and 7,291,056 generated tokens per cell.

Each policy replays the same requested completion lengths. The metric is generated tokens divided by rollout generation wall time; it excludes actor updates, predictor fitting, model loading and the initial prompt-feature pass. A cell must finish with all 256 requests matching the trace.

The measurement represents an activated predictor after warm-up, not ten RL epochs executed as part of each timed cell. Each host row is one point estimate; the second host repeats the same trace and is not an independent prompt population.

## Results

| Host | FCFS | D1 + EWS | Change vs FCFS | D2 + EWS | Change vs FCFS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 232 | 487.54 tokens/s | 514.65 tokens/s | +5.56% | 547.07 tokens/s | +12.21% |
| 222 | 480.61 tokens/s | 525.11 tokens/s | +9.26% | 547.68 tokens/s | +13.96% |

Improvement is `candidate_throughput / baseline_throughput - 1`. The hosts use Ascend 910 hardware and the vLLM 0.19.1 / vLLM-Ascend 0.19.1rc1 rollout stack.

## Measured dispatch configuration

The frozen measurement uses four TP=2 engines, request-level least-load placement and K=1 forced anchors. The public recipe uses no forced anchors and leaves server placement to verl's global load balancer. It also adds D2 admission and monitoring. The table measures the frozen dispatch configuration, not the full-training throughput of those public integration settings.

D1 is the community Linear/ListMLE predictor with token-scale calibration. D2 is the independently fitted multi-tap LogNormal predictor. In a separate unseen-prompt group, D1 + EWS improved by 3.59% while D2 + EWS changed by -3.20%. The stronger in-distribution D2 result therefore does not remove the need to validate ranking and tail calibration on the deployment workload.

EWS is most useful for finite batches with mixed response lengths, remaining waiting work and enough concurrency to overlap long and short requests. Extra predictor forwards and per-request dispatch add overhead. Measure generation time and full training-step time separately when choosing capacity and warm-up.

## Reproducing a deployment comparison

1. Install the revision and runtime in [README.md](README.md).
2. Freeze dataset/model revisions, prompt identities, sampling seeds, response cap, server topology and capacity.
3. Compare the standard-DAPO path (`PREDICTOR_ENABLE=false`) with D1 + EWS using the same launch arguments. Use separate output directories.
4. Let D1 fit on earlier completed outputs before measuring activated batches. For D2, supply a held-out candidate report as described in [D2_LIFECYCLE.md](D2_LIFECYCLE.md).
5. Record actual generated tokens, cap hits, generation wall time, predictor-forward/update time, full step wall time and exit status. Repeat both orders to assess drift.
6. Keep naturally generated outputs distinct from exact-length replay. Do not combine partial runs or select favorable seeds into one wall-time measurement.

The current verification stages and device-run status are in [VALIDATION.md](VALIDATION.md).
