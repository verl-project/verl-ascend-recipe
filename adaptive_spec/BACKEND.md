# Pinned Ascend backend

Use these public source revisions for this adapter:

| Component | Source |
| --- | --- |
| vLLM | [`v0.23.0`](https://github.com/vllm-project/vllm/tree/v0.23.0) |
| vLLM-Ascend | [`5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb`](https://github.com/vllm-project/vllm-ascend/tree/5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb) |

The [installation guide at that Ascend revision](https://github.com/vllm-project/vllm-ascend/blob/5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb/docs/source/installation.md)
specifies Linux, Python 3.10–3.12, PyTorch 2.10.0, torch-npu 2.10.0.post2,
and CANN/NNAL 9.0.1 with compatible host drivers. Prepare that environment and its
build dependencies separately; this recipe does not provision or upgrade them.
Do not substitute the current `main` branches or a similarly named vendor wheel.

## Source installation

In a dedicated environment with those prerequisites, use the upstream source-build
procedure with explicit revisions:

```bash
git clone --branch v0.23.0 --depth 1 https://github.com/vllm-project/vllm.git
VLLM_TARGET_DEVICE=empty python -m pip install -e ./vllm

git clone https://github.com/vllm-project/vllm-ascend.git
git -C vllm-ascend checkout --detach 5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb
git -C vllm-ascend submodule update --init --recursive
python -m pip install -e ./vllm-ascend
```

Install the recipe on the driver and workers, then check each environment:

```bash
python -m pip install -e ./adaptive_spec
adaptive-spec-check-backend
# Also check the scheduler when opting into targeted KV preemption:
adaptive-spec-check-backend --targeted-kv
```

The command reads distribution versions and source files without importing torch,
vLLM or loading a model. Exit status 0 means the checked source/version contract
matches; it does not validate compiled operators, drivers, model support or generation.
Verify ordinary model generation before collecting a deployment-specific phase map.

## Audited fingerprints

SHA-256 values are checked against the entire installed files, not only their signatures:

| File | SHA-256 |
| --- | --- |
| Ascend `worker/model_runner_v1.py` | `e4ad64b16d3707189ee54bb73be06b87a04423c1d7189e2dd8122530c605bb50` |
| vLLM `v1/worker/gpu_model_runner.py` | `6250ca0670c4593aa2f7c4567c9c71d6cb3d8182624a985d4d4e33d5bf054666` |
| vLLM `v1/core/sched/scheduler.py` (optional KV adapter) | `9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9` |

The allowlist also retains one previously tested Ascend runner fingerprint,
`cd0abab15168fa7cb6b094b8f1861975e7c9ab562d29fc0fde9906435f1e96cf`.
It adds telemetry to the public runner; inference statements and patched method
contracts are unchanged. That telemetry is unnecessary for this recipe and is not
distributed here. The public revision above is sufficient.

Unknown source remains an error. Audit a different backend before adding support;
do not disable the guards to make a preflight pass. See [VALIDATION.md](VALIDATION.md)
for the tested Python/runtime configurations and remaining integration coverage.
