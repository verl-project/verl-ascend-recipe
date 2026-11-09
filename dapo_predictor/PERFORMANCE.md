# Rollout performance and length prediction

This page records historical length-prediction and rollout scheduling experiments. Functional validation of online D2 + EWS is documented in [VALIDATION.md](VALIDATION.md).

## 1. Workload and measurement

The evaluation uses Qwen3.5-9B-Base with DAPO-Math, thinking enabled, temperature 1, top-p 1 and top-k -1. The response cap is 65,536 tokens. A frozen trace contains 16 prompts, 8 responses per prompt and 2 rollout steps: **256 requests and 7,291,056 generated tokens** per cell.

Each policy replays the same requested completion lengths. Throughput is total generated tokens divided by rollout generation wall time. It excludes actor updates, predictor fitting, model loading and the initial prompt-feature pass. All 256 requests must match the trace.

The measurement simulates predictive scheduling after a ten-epoch FCFS warm-up; the timed cells do not execute those RL epochs. Each host row is one completed measurement. The second host repeats the same trace rather than using a new prompt population.

Both hosts use Ascend 910 hardware, vLLM 0.19.1 and vLLM-Ascend 0.19.1rc1. The measured dispatch configuration uses four TP=2 engines, request-level least-load placement and K=1 forced anchors. The public recipe has no forced anchors and delegates server placement to verl. These results measure the historical rollout configuration, not end-to-end throughput of the default online D2 training path.

## 2. Rollout throughput

| Host | FCFS, tokens/s | D1 + EWS, tokens/s | D2 + EWS, tokens/s |
| --- | ---: | ---: | ---: |
| 232 | 487.54 | 514.65 (+5.56%) | 547.07 (+12.21%) |
| 222 | 480.61 | 525.11 (+9.26%) | 547.68 (+13.96%) |

Changes are relative to FCFS: `candidate_throughput / baseline_throughput - 1`.

Long First starts predicted long requests earlier, while completion-driven refill lets shorter waiting requests use freed capacity. This reduces waiting at the end of a finite rollout batch.

## 3. Length-prediction quality

D1 is the Linear/ListMLE predictor with PAVA calibration. D2 is the independently fitted multi-tap censored LogNormal predictor.

Spearman correlation compares predictions with actual mean response length per prompt:

| Evaluation group | D1 ranking score | D2 expected length |
| --- | ---: | ---: |
| Main trace | 0.3229 | 0.7595 |
| Unseen prompts, Group A | 0.4545 | 0.7401 |

On the main trace, D1's calibrated token estimates have a correlation of 0.3583.

D2 improves length-ordering correlation on both groups. Spearman measures ordering, not a percentage of correctly predicted lengths or model answer correctness. These are evaluations of the historical fitted predictors, not accuracy measurements of the online training run.

## 4. Workload variation

Better overall ranking does not guarantee the same throughput improvement on every workload.

On the unseen Group A prompts, D1 + EWS improved throughput by 3.59% over FCFS, while D2 + EWS changed by -3.20%. D2 underestimated several extreme-length responses, delaying their admission and increasing the batch's final waiting time.

Alongside overall ranking, identifying and calibrating the longest responses matters for Long First scheduling.
