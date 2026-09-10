# MindSpeed-LLM Backend Recipe for Ascend NPU

基于 verl 框架和 MindSpeed-LLM 后端在 Ascend NPU 上进行 RLHF 训练的样例。

## 目录结构

```
grpo_mindspeed_llm/
├── backend/                    # MindSpeed-LLM 后端引擎（自包含）
│   ├── __init__.py             # 引擎注册 + hydra yaml 自动链接
│   ├── engine.py               # MindSpeedLLMEngineWithLMHead 引擎类
│   ├── config.py               # MindSpeedEngineConfig/ActorConfig/CriticConfig dataclass
│   └── utils.py                # apply_patch / gpt_model_provider 等 helper
├── configs/                    # hydra yaml 配置（_target_ 指向 backend.config.*）
│   ├── model_engine/mindspeed.yaml
│   ├── engine/mindspeed.yaml
│   ├── actor/mindspeed_actor.yaml
│   ├── critic/mindspeed_critic.yaml
│   └── ref/mindspeed_ref.yaml
├── scripts/                    # 训练脚本
│   └── run_qwen3_8b_mindspeedllm_npu.sh
└── readme.md
```

## 架构原理

本 recipe 自包含 MindSpeed-LLM 后端引擎，不依赖 verl-hardware-plugin。核心机制如下：

### 1. 引擎自动注册

verl 的 `verl/__init__.py` 支持 `VERL_USE_EXTERNAL_MODULES` 环境变量，启动时自动 `importlib.import_module("backend")`。

训练脚本中设置：
```bash
export PYTHONPATH="${SCRIPT_DIR}/..:$PYTHONPATH"
export VERL_USE_EXTERNAL_MODULES=backend
```

`backend/__init__.py` 被 import 时触发：
```python
from backend.engine import MindSpeedLLMEngineWithLMHead  # @EngineRegistry.register 生效
```

引擎注册到 verl 的 `EngineRegistry`，key 为 `(backend="mindspeed_llm", device="npu")`。

### 2. Hydra 配置自动链接

`backend/__init__.py` import 时自动将 `configs/` 下的 5 个 yaml 软链到 verl 的 `verl/trainer/config/` 目录，使 hydra 能发现 `model_engine=mindspeed` 及其派生的 actor/critic/ref 配置。

这些 yaml 的 `_target_` 指向 `backend.config.*`（recipe 中的 dataclass），`strategy` 为 `mindspeed_llm`，与 verl 核心自带的 `mindspeed_megatron`（用 `mcore_kwargs`）区分开。

### 3. 训练入口

直接用 verl 标准入口，无需自定义 main：
```bash
python3 -m verl.trainer.main_ppo \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    ...
```

## 环境准备

### 基础镜像

```bash
docker pull quay.io/ascend/verl:verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest

docker run -dit --ipc=host --network host --name 'rl_test' --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /home/:/home/ -v /data/:/data/ \
    quay.io/ascend/verl:verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest /bin/bash

docker exec -it rl_test bash
```

### 依赖版本

| 依赖组件       | 版本            |
| :------------- | :------------- |
| Docker 镜像    | quay.io/ascend/verl:verl-sglang-8.3.rc1-910b-ubuntu22.04-py3.11-latest |
| MindSpeed      | master         |
| MindSpeed-LLM  | master         |
| Megatron       | core_v0.12.1   |

### 安装 MindSpeed 依赖

镜像中已包含 verl、vllm、CANN 等基础环境，只需额外安装 MindSpeed 三件套：

```bash
cd /workspace

# Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git checkout core_v0.12.1 && cd ..

# MindSpeed
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed && pip3 install -r requirements.txt && cd ..

# MindSpeed-LLM
git clone https://gitcode.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM && pip3 install -r requirements.txt && cd ..
```

### 设置 PYTHONPATH

MindSpeed 三件套需要通过 PYTHONPATH 让 verl 能 import 到（镜像中默认未配置）：

```bash
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/workspace/MindSpeed
export PYTHONPATH=$PYTHONPATH:/workspace/MindSpeed-LLM
```

建议写入 `~/.bashrc` 永久生效。

### 准备 recipe

```bash
cd /workspace
git clone https://github.com/verl-project/verl-ascend-recipe.git
git clone https://github.com/verl-project/verl.git

# 将 recipe 复制到 verl/recipe 下
cp -r verl-ascend-recipe/grpo_mindspeed_llm verl/recipe/

# 安装TransferQueue
pip install "TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@main" --no-deps
```

## 启动训练

### Qwen3-8B GRPO（8 NPU）

```bash
cd /workspace/verl

# 准备数据（gsm8k parquet）
mkdir -p $HOME/verl/gsm8k
# 将 train.parquet / test.parquet 放到 $HOME/verl/gsm8k/

# 准备权重
export MODEL_PATH=/path/to/Qwen3-8B

# 启动训练
bash recipe/grpo_mindspeed_llm/scripts/run_qwen3_8b_mindspeedllm_npu.sh
```

脚本会自动完成：
1. 设置 `PYTHONPATH` 指向 `recipe/grpo_mindspeed_llm/`（包含 `backend/` 包）
2. 设置 `VERL_USE_EXTERNAL_MODULES=backend` 让 verl 自动注册引擎
3. 设置 NPU 相关环境变量（HCCL、CANN 等）
4. 调用 `python3 -m verl.trainer.main_ppo` 启动训练

## 调试验证

### 验证引擎注册

```bash
export PYTHONPATH="/workspace/verl/recipe/grpo_mindspeed_llm:$PYTHONPATH"
export VERL_USE_EXTERNAL_MODULES=backend

python3 -c "
import verl
from verl.workers.engine.base import EngineRegistry
engines = EngineRegistry._engines
print('Registered engines:', engines)
assert 'mindspeed_llm' in engines.get('language_model', {}), 'engine not registered!'
print('OK: MindSpeedLLMEngineWithLMHead registered')
"
```

### 验证 yaml 配置链接

```bash
python3 -c "
import verl
import verl.trainer.config as cfg
from pathlib import Path
cfg_dir = Path(list(cfg.__path__)[0])
for rel in ['model_engine/mindspeed.yaml', 'engine/mindspeed.yaml',
            'actor/mindspeed_actor.yaml', 'critic/mindspeed_critic.yaml',
            'ref/mindspeed_ref.yaml']:
    p = cfg_dir / rel
    assert p.exists(), f'Config not linked: {rel}'
    print(f'  OK: {rel}')
print('All configs linked')
"
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'backend'` | PYTHONPATH 未包含 recipe 目录 | 确认脚本中 `SCRIPT_DIR/..` 指向 `grpo_mindspeed_llm/` |
| `ModuleNotFoundError: No module named 'mindspeed_llm'` | MindSpeed-LLM 未安装或 PYTHONPATH 未设置 | `export PYTHONPATH=$PYTHONPATH:/workspace/MindSpeed-LLM` |
| `ModuleNotFoundError: No module named 'megatron'` | Megatron-LM 未安装或 PYTHONPATH 未设置 | `export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM` |
| EngineRegistry 找不到 mindspeed_llm | `VERL_USE_EXTERNAL_MODULES` 未设置 | 确认 `export VERL_USE_EXTERNAL_MODULES=backend` |
| hydra 找不到 `model_engine=mindspeed` | yaml 配置未链接 | 检查 `verl/trainer/config/model_engine/mindspeed.yaml` 是否存在 |
