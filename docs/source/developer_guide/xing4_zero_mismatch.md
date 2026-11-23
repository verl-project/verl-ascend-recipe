# Xing4 numerical recipe adapters

`vllm_ascend.patch.worker.xing4.install()` explicitly installs the inference
adapters used by the companion verl Xing4 recipe. The recipe provides the vLLM
general-plugin entry point and the native training bridge. Importing this package
does not activate the adapters by itself.

The implementation preserves BF16 arithmetic boundaries for expanded MLA attention,
RoPE, dense/shared row reductions, SwiGLU and expert weighting; router logits use
FP32 persistent GEMM and native sigmoid/top-k normalization. mHC uses the accepted
batched mixing kernel and the ordered, non-fused FP32 Sinkhorn reduction sequence.
Keep zero initialization in the mixing output: removing it was not validated.

The accepted topology is eager BF16 Xing4.0-29B-A4B, inference TP8/EP8/DP1, training
TP2/PP2/EP8/ETP1/CP1, block size 128, rotary dimension 64, no prefix caching or MTP.
Unsupported arithmetic/topology combinations fail with errors. Other model types
delegate to their original implementations. The recipe adds no environment
variable to vllm-ascend and makes no vLLM core modification.

Run the new NPU arithmetic regression test alongside the existing training-math
tests, then use the companion recipe to verify raw log probabilities and real IPC
weight updates. Historical acceptance does not replace validation after relocation.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_xing4_training_math.py
```

AI assistance was used to organize this integration. Architectural review should
cover the explicitly installed method wrappers and their model/config guards.
