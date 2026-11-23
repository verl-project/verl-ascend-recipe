# Ascend zero-mismatch recipes

The `rl-zero-mismatch-verl` branch contains the native Megatron GEMM and vocabulary
reduction controls, FP32 normalization/export handling, and resident vLLM IPC weight
synchronization support. They are opt-in through the existing engine configuration.

Use the matching `rl_zero_mismatch_vllm_ascend` branch of this repository as the
vllm-ascend source tree. That branch already includes the Qwen3.5 training arithmetic
implementation and its eleven NPU regression tests.

The [Xing4 recipe](xing4/README.md) adds a training bridge and a portable continuous
acceptance runner. Inference arithmetic lives in vllm-ascend, and a small vLLM
general-plugin entry point activates it without editing the upstream model plugin.

For the pinned Qwen3.5-35B-A3B BF16 training setup, enable these engine controls:

```yaml
override_mcore_model_config:
  preserve_gdn_a_log_fp32: true
  preserve_gdn_norm_export_fp32: true
  npu_batch_invariant_matmul: true
  npu_batch_invariant_reduction: true
override_transformer_config:
  use_triton_gdn: false
  use_ascend_gdn: true
```

The matching eager vLLM configuration uses `rl_config.enabled: true`,
`training_tensor_parallel_size: 2`, and all four `training_router_math`,
`training_expert_math`, `training_attention_math`, and `training_logprob_math`
switches enabled. The historical complete runner is preserved in the source
repository at `wucong25/verl@6b4e86ccc4d30996620bd694daa7c2cbbd7135bb`
under `.zero_diff/full_online_002.py`. Intermediate trace/probe scripts and model
weights are not runtime dependencies of this integration.

These changes were organized with AI assistance. Historical acceptance evidence
applies to the pinned numerical workload described in the recipe; it does not
establish general equivalence for other kernels, model versions, or topologies.
