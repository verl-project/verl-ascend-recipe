# Validation

## Code and CPU

425 tests pass on Python 3.12.14: 331 upstream tests and 94 recipe tests. Ruff, Python 3.10 syntax checks, wheel/sdist builds, and isolated installation pass. After simplifying documentation and comments, the 94 recipe tests, Ruff, and wheel build pass again, with executable logic unchanged.

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

The verl check uses startup capacity K5 for fixed K and K12 for AdaptiveSpec. One answer differs. This difference and normal test-harness shutdown will be checked separately.

See [PERFORMANCE.md](PERFORMANCE.md) for performance results.
