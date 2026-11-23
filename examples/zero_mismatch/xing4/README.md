# Xing4.0 native training / rollout agreement

This recipe uses all 40 layers of Xing4.0-29B-A4B with native Megatron
TP2 / PP2 / EP8 / ETP1 / CP1 on 16 NPU ranks, and eager vLLM TP8 / EP8 on
8 resident inference ranks. The devices are reused between training and inference.
It compares raw finite FP32 response log-probability bits for decode vs prefill,
mixed vs single batching, native recomputation, and gradient-enabled forward.
Each optimizer step changes real parameters and synchronizes through device IPC.

## Code ownership

| Location | Responsibility |
| --- | --- |
| verl native engine and NPU helpers | GEMM/reduction selection, FP32 export, IPC synchronization |
| `xing_bridge.py` | Recipe-local Megatron model provider, hyper-connections and weight mapping |
| vllm-ascend `patch/worker/xing4/` | mHC, Sinkhorn, attention, RoPE, activation, row reduction, router and expert arithmetic |
| `plugin/` | Opt-in general-plugin registration; upstream model plugin remains unmodified |
| `acceptance.py`, `online_worker.py`, audit scripts | Continuous update and independent raw-tensor verification |

## Pinned dependencies

Use Python 3.12 with torch 2.10.0 and torch-npu 2.10.0.post7 in an Ascend environment.
The historical verified image was
`sha256:bdc20cc234a54139fe8b66c871e7b7c625bf0288b783c77a01ec14b00a269c6c`.

| Repository | Historical tested revision | Additional source changes required |
| --- | --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | None |
| XingChen-AGI/xing4-vllm-plugin | `b087d14fb517a1000cd9458e7784fb6b910cedda` | None with this recipe entry point |
| MindSpeed | `d0d5955ba445e8dc3ed8b6aa4a76c053364fba7a` | None |
| MindSpeed-Bridge | `eaea54b2faf3223aa2cb9495c5506b438e22dd03` | None; supplies GLM hyper-connection classes used by the local bridge |
| Megatron-LM | `571370c829ca768fe37244f4e2e7f28d8accc4ab` | None |
| Megatron-Bridge | `fcbb6031103d0ca845c1a54d4fee55ecfcca17b6` | See environment note below |

The inspected Megatron-Bridge checkout has a local edit to
`src/megatron/bridge/models/qwen3_asr/hf_qwen3_asr/__init__.py`. This is an
environment compatibility change, outside Xing model arithmetic; inspect the
`compat/megatron-bridge-qwen3-asr-registration.patch` before reproducing on a
different dependency image. The patch checks `CONFIG_MAPPING` before registration;
apply it only when the pinned dependency otherwise fails on duplicate `qwen3_asr`
registration. It is supplied separately and is not automatically applied.

Install both recipe branches into the same environment, then install the original
Xing model plugin and this opt-in entry point. Use `uv` for environment management:

```bash
uv venv --python 3.12 --system-site-packages .venv-xing
source .venv-xing/bin/activate
uv pip install --no-deps -e /path/to/verl-tree -e /path/to/vllm-ascend-tree
uv pip install --no-deps 'git+https://github.com/XingChen-AGI/xing4-vllm-plugin.git@b087d14fb517a1000cd9458e7784fb6b910cedda'
uv pip install --no-deps -e examples/zero_mismatch/xing4/plugin
```

Source the CANN and ATB environment scripts appropriate for the machine before
running. If `VLLM_PLUGINS` is configured, include `xing4_zero_mismatch` in its
allowlist. The upstream plugin's `xing4_0` entry point may remain enabled.

```bash
export XING_MODEL_PATH=/path/to/Xing4.0-29B-A4B
export XING_WORK_DIR=/path/to/new-run-directory
bash examples/zero_mismatch/xing4/run_acceptance.sh acceptance-001 30
```

The launcher refuses an existing run name. It saves source hashes, resolved
configuration, parameter fingerprints, raw tensors, synchronization receipts and
audits under `$XING_WORK_DIR/runs/acceptance-001`. Set `XING_WORK_DIR` to a short
absolute path to accommodate Ray/IPC socket path limits. Model weights and runtime
credentials are not included in the recipe. The prompt fixture contains the four
historically tested token sequences of lengths 65 / 129 / 257 / 513.

## Evidence and limits

The historical `full-opt20-30steps-001` implementation passed 30 true optimizer
updates / 31 weight versions, with 155 raw comparison groups, 15,872 FP32 elements,
zero bit mismatches, and finite outputs throughout. This publication relocates
that accepted implementation and removes machine-specific paths and tracing.
It requires its own smoke validation; the historical run is not a fresh run of
the reorganized source.

Later `pre_adaptive` experiments reduced 16/64-token median inference times to
6.605 / 23.896 seconds, but did not complete the same 30-step acceptance and still
regressed against the ordinary baseline. They are excluded from the default
integration, along with failed and slower experimental kernels. No claim of
zero performance overhead is made.
