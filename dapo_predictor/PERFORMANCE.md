# Rollout performance

## Online D2 + EWS during RL

The public launcher was evaluated with Qwen3-1.7B, 32 fixed DAPO-Math prompts, 8 responses per prompt, batches of 16 and 20 RL epochs: 40 actor updates and 5120 responses per arm. Both arms used the same initial model, prompt IDs, data seed 1, engine base seed 0, temperature 1, top-p 1, top-k -1, a 1024-token prompt cap and an 8192-token response cap. Four TP=2 engines ran on eight Ascend logical devices, with eight sequences per engine. EWS used a global window of 32, without forced anchors or live progress polling.

Online D2 fitted asynchronously from preceding outputs. Epochs 1–10 used FCFS admission; the first predictive batch was step 21, in epoch 11, after 2560 observed responses. The baseline disabled the predictor and used native DAPO. Each arm completed once, without resume or seed selection.

| Measurement interval | Native FCFS | D2 + EWS | Throughput change |
| --- | ---: | ---: | ---: |
| Epochs 11–20: response tokens / generation time | 730.51 tokens/s | 743.76 tokens/s | +1.81% |
| Epochs 11–20: response tokens / full training-step time | 705.61 tokens/s | 717.33 tokens/s | +1.66% |
| All 20 epochs: response tokens / generation time | 730.48 tokens/s | 747.47 tokens/s | +2.33% |
| All 20 epochs: response tokens / full training-step time | 705.72 tokens/s | 720.89 tokens/s | +2.15% |

For epochs 11–20, FCFS generated 17,801,876 tokens in 24,368.94 seconds of generation and 25,228.96 seconds of full training steps. D2 generated 17,886,844 tokens in 24,049.09 and 24,935.22 seconds respectively. Aggregates use total tokens divided by summed times, not the mean of per-step rates. Full-step timing includes actor updates and synchronous predictor work; installation and model startup are excluded.

Mean training reward in this interval was -0.49609 for FCFS and -0.49531 for D2; response cap-hit rates were 51.76% and 53.28%. These are natural-generation RL trajectories, not identical completion traces or independent held-out quality evaluations. There was one run per arm on a shared host; another workload briefly used unallocated boards during the baseline. Timing also differed during warm-up, before predictive admission. The full-run difference therefore cannot be attributed entirely to scheduling.

The pre-response D2 mean absolute error was 824.15 tokens on 37 fully uncensored prompt observations out of 320 in the activated interval. This subset excludes groups with capped responses. The p90 coverage bounds were 46.25%–98.05%; censoring prevents a narrow overall calibration estimate.

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
4. Let the selected predictor fit on earlier completed outputs before measuring activated batches. Default D2 fits asynchronously during RL; an external frozen candidate is an explicit compatibility option described in [D2_LIFECYCLE.md](D2_LIFECYCLE.md).
5. Record actual generated tokens, cap hits, generation wall time, predictor-forward/update time, full step wall time and exit status. Repeat both orders to assess drift.
6. Keep naturally generated outputs distinct from exact-length replay. Do not combine partial runs or select favorable seeds into one wall-time measurement.

The current verification stages and device-run status are in [VALIDATION.md](VALIDATION.md).
