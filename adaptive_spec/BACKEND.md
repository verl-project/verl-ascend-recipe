# Inference environment

| Component | Version / source |
|---|---|
| vLLM | `v0.23.0` |
| vLLM-Ascend | `5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb` |
| PyTorch / torch_npu | `2.10.0` / `2.10.0.post2` |
| CANN / NNAL | `9.0.1` / `9.0.1` |
| Python | 3.10-3.12 supported; fresh-environment acceptance uses 3.12 |
| verl | Fixed commit in [REQUIRED_VERL.txt](REQUIRED_VERL.txt) |

Start on a host with the Ascend driver and Docker installed. For arm64 Atlas A3, the following CANN image supplies Python 3.12 and CANN/NNAL 9.0.1. Use the `a3` image for Ascend910_93-series devices and the `910b` image for 910B devices; their operator kernels differ. The digest below pins `9.0.1-a3-ubuntu22.04-py3.12`:

```bash
export AS_IMAGE=quay.io/ascend/cann@sha256:004a33d8634c191399418ea81e0c6a367b7459a5713c206c4f9088294337f149
docker pull "$AS_IMAGE"
```

For a server without registry access, save the image on a connected machine with `docker save "$AS_IMAGE" -o cann901.tar`, transfer the archive, and run `docker load -i cann901.tar` on the target. The arm64 image ID is `sha256:1572a3447f8eb74778d9956cb79cb38c39734f81b20ed58f5d9f0ba04c78f807`; set `AS_IMAGE` to that ID after loading.

From the recipe repository root, start a container with the selected logical devices. On the A3 setup, logical devices 0–7 correspond to physical cards 0–3:

```bash
device_args=()
for device in {0..7}; do device_args+=(--device "/dev/davinci${device}"); done
docker run --name adaptive-spec --network host --shm-size 32g -it \
  "${device_args[@]}" \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "$PWD":/workspace -w /workspace "$AS_IMAGE" bash
```

Run the remaining installation commands inside this container, from `/workspace`:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" "cmake>=3.26.1" ninja packaging jinja2
python -m pip install pyyaml numpy psutil attrs decorator scipy
python -m pip install torch==2.10.0 torch-npu==2.10.0.post2
export SOC_VERSION=ascend910_9391
```

Other Ascend platforms use the corresponding driver and CANN setup in the [upstream installation guide](https://github.com/vllm-project/vllm-ascend/blob/5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb/docs/source/installation.md). Install the fixed backend sources:

```bash
git clone --branch v0.23.0 --depth 1 https://github.com/vllm-project/vllm.git
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e ./vllm
git clone https://github.com/vllm-project/vllm-ascend.git
git -C vllm-ascend checkout --detach 5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb
git -C vllm-ascend submodule update --init --recursive
python -m pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
python -m pip install nanobind grpcio-tools
export CMAKE_PREFIX_PATH=/workspace/venv/lib/python3.12/site-packages/torch/share/cmake:${CMAKE_PREFIX_PATH:-}
python -m pip install --no-build-isolation -e ./vllm-ascend
python -m pip install "prometheus-fastapi-instrumentator>=7,<8"
bash ./install_verl.sh --recipe adaptive_spec --yes
python -m pip install -e ./adaptive_spec
python -m pip install "numpy==2.2.6" "scipy==1.13.1"
adaptive-spec-check-backend
```

verl requires NumPy 2.x; the tested combination uses NumPy 2.2.6 and SciPy 1.13.1. Triton-Ascend 3.2.1 declares a NumPy 1.26.4 requirement, so `pip check` reports this metadata conflict. The paired NPU rollout uses the versions pinned above.

The check verifies versions and fingerprints of the source files containing adapted methods. Then verify ordinary model generation before collecting the cost table.

Continue with the [Qwen3.5-9B/APPS example](README.md#qwen35-9bapps-example) for model download, calibration and rollout. Proxy and CA settings, when needed, belong to this container's environment. Remove download proxies before starting local Ray services; the example does this in its rollout process.

Supported execution is eager serial MTP with PP1/CP1. K must fit the startup capacity; K11 is excluded. Optional `VERL_MTP_KV_PREEMPT_MODE=targeted` handles KV pressure on the checked scheduler. When enabling it, also run `adaptive-spec-check-backend --targeted-kv` and use the same setting in performance comparison arms.

See [UPGRADING.md](UPGRADING.md) for backend upgrades and [VALIDATION.md](VALIDATION.md) for the tested environment.
