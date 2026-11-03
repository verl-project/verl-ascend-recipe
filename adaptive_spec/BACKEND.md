# Inference environment

| Component | Version / source |
|---|---|
| vLLM | `v0.23.0` |
| vLLM-Ascend | `5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb` |
| PyTorch / torch_npu | `2.10.0` / `2.10.0.post2` |
| Python | 3.10-3.12 supported; NPU checks use 3.11.10 |
| verl | Fixed commit in [REQUIRED_VERL.txt](REQUIRED_VERL.txt) |

Prepare the backend, CANN/NNAL, and drivers using the [Ascend installation guide for this revision](https://github.com/vllm-project/vllm-ascend/blob/5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb/docs/source/installation.md). Install these source revisions rather than main:

```bash
git clone --branch v0.23.0 --depth 1 https://github.com/vllm-project/vllm.git
VLLM_TARGET_DEVICE=empty python -m pip install -e ./vllm
git clone https://github.com/vllm-project/vllm-ascend.git
git -C vllm-ascend checkout --detach 5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb
git -C vllm-ascend submodule update --init --recursive
python -m pip install -e ./vllm-ascend
python -m pip install -e ./adaptive_spec
adaptive-spec-check-backend
```

The check verifies versions and fingerprints of the source files containing adapted methods. Then verify ordinary model generation before collecting the cost table.

Supported execution is eager serial MTP with PP1/CP1. K must fit the startup capacity; K11 is excluded. Optional `VERL_MTP_KV_PREEMPT_MODE=targeted` handles KV pressure on the checked scheduler. When enabling it, also run `adaptive-spec-check-backend --targeted-kv` and use the same setting in performance comparison arms.

See [UPGRADING.md](UPGRADING.md) for backend upgrades and [VALIDATION.md](VALIDATION.md) for the tested environment.
