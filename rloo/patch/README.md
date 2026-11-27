# Qwen3 RLOO patches
Status: this documentation-only PR update does not publish the new patch files. The following commands apply to the prepared files on the experiment server; code review and a separate code submission are still pending.

Base: verl 1a8a0f5ffd9d3f169ae4432b68526233ba028102.

From the verl checkout:
```bash
git apply --check /workspace/rloo/rloo/patch/verl_qwen3_8b_rloo_model.patch
git apply /workspace/rloo/rloo/patch/verl_qwen3_8b_rloo_model.patch
git apply --check /workspace/rloo/rloo/patch/verl_rloo_vectorized.patch
git apply /workspace/rloo/rloo/patch/verl_rloo_vectorized.patch
```
The model patch aligns the Qwen3 chat template with strict GSM8K scoring by disabling thinking. The NPU recipe already contains these model settings.

The algorithm patch computes group sums and counts with scatter_add instead of launching per-sample tensor arithmetic. It preserves arbitrary hashable prompt IDs, unequal group sizes, singleton zero baselines and response masks. It changes reduction order; comparisons use floating-point tolerances. No end-to-end speedup is claimed without paired benchmarks.

CPU and Ascend NPU equation/regression tests passed. Evidence: `/workspace/rloo/validation/issue79/test_rloo.log`. Test source: `/workspace/rloo/validation/issue79/test_rloo.py`.

Rollback (only these changes): `git apply -R <patch>` in reverse application order. Do not apply twice.
