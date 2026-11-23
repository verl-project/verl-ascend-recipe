# FSDPTurbo MXFP8 环境安装指南

本配方在 Atlas A5（Ascend950DT）上运行 Qwen3.5-35B-A3B GRPO。安装流程参考 [verl 官方 A5 Dockerfile](https://github.com/verl-project/verl/blob/main/docker/ascend/Dockerfile.ascend_9.2.1_a5)，配方版本及补丁清单见 [readme.md](readme.md)。

## 前置条件

- Python 3.12，以及 Python 的 `venv` 模块。
- 已安装 CANN 9.2.0，以及 `git`、`gcc`、`g++` 和系统开发工具（包括 NUMA 开发库）。脚本会在虚拟环境安装 CMake、Ninja 和 Python 构建依赖。
- 工作目录至少预留 30 GB，需要能访问源代码仓和 Python 包索引。

## 安装

先检出包含本配方及其补丁的仓库版本，然后执行：

```bash
set -o pipefail
recipe=/path/to/verl-ascend-recipe
export WORK_DIR=/path/to/fsdp-turbo-workspace
export PYTHON_BIN=/path/to/python3.12
export CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$WORK_DIR"
bash "$recipe/low_precision/fsdp_turbo/install.sh" 2>&1 | tee "$WORK_DIR/install.log"
```

脚本按顺序准备构建环境、安装 vLLM、编译 vLLM-Ascend、安装训练依赖、恢复配方版本并校验导入。源码仓与虚拟环境默认放在 `WORK_DIR` 下。默认虚拟环境是 `$WORK_DIR/.venv`，可用 `VENV_DIR` 指定绝对路径。若已激活虚拟环境，脚本会使用它，显式设置的 `VENV_DIR` 优先。

- CANN 布局不同的机器用 `CANN_ENV` 指定实际脚本。若 ATB 安装在其他位置，可用 `ATB_ENV` 指定。
- vLLM 的隔离构建会请求更新的 PyTorch，因此本配方先安装 PyTorch 2.10，再使用 `--no-build-isolation` 构建 vLLM。训练依赖装完后，脚本重新校验并恢复运行时版本。vLLM 与 vLLM-Ascend 的 FastAPI/Starlette 声明存在冲突，最终按配方版本用 `--no-deps` 恢复。

## 安装验证

脚本结束时应输出 `FSDPTurbo recipe imports OK`。它会检查两项配方必需的实现、关键包版本以及 vLLM、verl、ACL 和训练扩展的导入。

卡空闲时，再检查实际设备可用性：

```bash
source "$CANN_ENV"
source "${VENV_DIR:-$WORK_DIR/.venv}/bin/activate"
export PYTHONPATH="$WORK_DIR/FSDPTurbo${PYTHONPATH:+:$PYTHONPATH}"
cd "$VIRTUAL_ENV"
python -c "import torch, torch_npu; assert torch.npu.is_available(), 'NPU unavailable'"
```

训练启动方式见 [readme.md](readme.md)。启动前将 FSDPTurbo 路径追加到 CANN 设置后的 `PYTHONPATH`。
