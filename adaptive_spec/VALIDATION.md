# Validation

## Code and CPU

The recipe passes 116 CPU tests and Ruff. Tests cover policy decisions, TP messages, calibration, APPS preparation/scoring, GDN state indexing and Qwen method binding, and bounded worker cleanup.

On Linux with PyTorch 2.10, a two-process Gloo check produces matching results for 284 decisions, covering K switches, message-buffer reuse, and error propagation.

## NPU and verl integration

Environment: Qwen3.5-9B, original BF16 precision, TP2, capacity 16 per Engine, eager execution, and asynchronous scheduling with serial MTP; Python 3.11.10, torch 2.10.0, and torch_npu 2.10.0.post2. Backend versions are listed in [BACKEND.md](BACKEND.md).

- Dynamic K: startup capacity K12, with a constructed cost table covering K5 -> K12 -> K5. Both TP ranks agree on all 293 decisions. Saturation/drain phases, physical draft width, two confirmations, 64-round dwell, and packed messages pass their checks.
- verl: commit `d040717b21af2e23e8e789a3e354cff2394ae2de`, Ray 2.48.0, through the manager/worker, YAML Elastic router, client, and vLLM server. Two steps generate 16 responses each. Step metadata, TP agreement, returned tensors, logprobs, and router accounting pass their checks.

## Paired answer checks

Both arms use the same inputs, sampling configuration, and BF16 precision.

| Check | Fixed-K correct answers | AdaptiveSpec correct answers | Exact output-token agreement |
|---|---:|---:|---:|
| Native generation: 16 distinct arithmetic questions, startup capacity K12 in both arms | 16/16 | 16/16 | 16/16 |
| verl + Elastic: 4 questions with 4 responses each, two steps, physical K5 in both arms | 32/32 | 31/32 | 31/32 |

The verl check uses startup capacity K5 for fixed K and K12 for AdaptiveSpec. One answer differs. The APPS comparisons below use startup capacity K12 in both arms and check normal worker shutdown.

## Qwen3.5-9B/APPS regression

A fresh CANN 9.0.1 A3 environment uses Python 3.12, torch 2.10.0, torch_npu 2.10.0.post2, vLLM 0.23.0, the Ascend commit in [BACKEND.md](BACKEND.md), and Ray 2.59.0. The setup uses original BF16, four TP2 Engines, maximum 16 requests per Engine, eager execution, and Elastic routing.

The paired input contains 32 APPS test prompts (selection seed 42), eight responses per prompt, generation seed 20261103, and a 65,536-token response limit. Fixed K12 passes 137/256 response tests. AdaptiveSpec with maximum-K aggregation passes 150/256; all 256 responses are scorable. Eight workers execute the GDN state adapter. Native NPU state checks cover ten variable-width and state-transition cases.

The formal plugin run produces 986,765 response tokens in 1,411.12 rollout seconds (699.28 tokens/s). All four Engines complete worker cleanup normally; the driver exits with code 0. This regression covers 32 prompts.

An independently installed environment follows the README with Ray 2.60.0, NumPy 2.2.6 and SciPy 1.13.1. It generates 64 training responses for calibration, collects the K5/7/9/12 cost table, and runs the same 32-test-prompt comparison. Fixed K12 passes 149/256 response tests; AdaptiveSpec passes 144/256. Both return all 256 responses, with matching input identities and per-response seeds. Both complete worker cleanup normally and exit with code 0. Throughput is 847.83 versus 1001.10 tokens/s (+18.08%).

The cost collector also passes a targeted K5, B1, 512-token-context NPU exit check: both workers finish cleanup within the 30-second grace period, without worker termination signals, and the process exits with code 0.

See [PERFORMANCE.md](PERFORMANCE.md) for performance results.
