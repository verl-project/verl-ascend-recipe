# 环境安装指南 ｜ low_precision/fsdp_turbo（Qwen3.5-35B-A3B MXFP8 GRPO · FSDPTurbo 链路）

从零搭建本用例运行环境：**FSDPTurbo（训练侧动态 MXFP8）＋ vLLM-Ascend（推理侧 MXFP8 权重加载）**。用例定位、版本矩阵与 patch 清单见 `readme.md`。

## 1. 安装总览

```bash
# 1) 在准备好的工作目录中获取本仓库（目录名须保持 verl-ascend-recipe）
git clone <本仓库地址> verl-ascend-recipe
# 2) 在仓库的父目录执行安装脚本（目录布局要求见 §3.1）
cd <工作目录>
source /usr/local/Ascend/ascend-toolkit/set_env.sh      # CANN 环境（编译步骤依赖，见 §3.2）
python3 -m venv <虚拟环境目录> && source <虚拟环境目录>/bin/activate
# 3) 一键安装
bash verl-ascend-recipe/low_precision/fsdp_turbo/install.sh
# 4) 按 §4 完成装后验证
# 5) 训练启动前设置 PYTHONPATH（必须追加式——覆盖式会丢 CANN 注入路径，vLLM 子进程报 No module named 'acl'）
export PYTHONPATH=$PWD/FSDPTurbo:$PYTHONPATH
```

安装完成的判据：`install.sh` 末尾自检输出 `自检通过：translate_weight_names_for_vllm / quant_ignored_modules 均在位` 且 vllm import 冒烟 OK，且 §4.2 验证清单全部通过。

## 2. 环境要求

| 项 | 要求 |
|---|---|
| NPU | 1 × Atlas A5（8 × Ascend950DT 96G） |
| CANN | 9.2.0（`ascend-toolkit/set_env.sh` 可正常 source） |
| Python | 3.12 |
| 磁盘 | 预留 ≥ 30 GB（依赖仓克隆＋虚拟环境） |

全部依赖的版本矩阵见 `readme.md`「环境版本」；`install.sh` 内已按该矩阵固定版本，无需手动指定。

## 3. 安装步骤

### 3.1 工作目录布局（重要）

`install.sh` 中的补丁与依赖清单引用均使用 `../verl-ascend-recipe/low_precision/fsdp_turbo/...` 相对路径，依赖仓会克隆到执行脚本时的当前目录。因此脚本必须在**本仓库的父目录**下执行，且仓库目录名保持克隆默认名 `verl-ascend-recipe`：

```
<工作目录>/                      ← 在此执行 install.sh
├── verl-ascend-recipe/         ← 本仓库
└── vllm-ascend/  vllm/  FSDPTurbo/  verl/  torchao-ascend/  MindSpeed-Ops/   ← 安装过程自动生成
```

在其他目录执行会导致补丁与清单的相对路径解析失败。

### 3.2 编译前置：CANN 环境

`vllm-ascend` 与 `MindSpeed-Ops` 的安装包含本地编译（CMake / AscendC），编译期需要 `torch_npu` 的 CANN 运行环境。执行 `install.sh` 前先 source：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

未 source 时编译步骤会因找不到 CANN 头文件/库而失败。

### 3.3 执行安装

```bash
cd <工作目录>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m venv <虚拟环境目录> && source <虚拟环境目录>/bin/activate
bash verl-ascend-recipe/low_precision/fsdp_turbo/install.sh
```

安装脚本共九步：

| 步 | 内容 |
|---|---|
| 1 | vllm-ascend requirements（torch 前置） |
| 2 | vllm editable 安装（运行时依赖由 pip 按 metadata 自动解析）＋冲突件强制钉死（fastapi／starlette／prometheus-fastapi-instrumentator／numpy）＋numpy 断言 |
| 3 | vllm-ascend 补丁应用＋editable 编译 |
| 4 | FSDPTurbo 官方 main commit 检出＋editable 安装 |
| 5 | verl 官方 commit 检出＋editable 安装（`--no-deps`） |
| 6 | verl 侧补丁两步应用（公共层 verl.patch＋fsdp 专属件） |
| 7 | transformers / triton-ascend 版本钉 |
| 8 | NPU kernel 层五件（fla / fla_npu / torchao / torchao_npu / mindspeed_ops） |
| 9 | 关键特性自检（硬性失败）＋ vllm import 冒烟 |

硬提示（不照做会装不上或静默出错）：

- 步骤 2 的运行时依赖由 pip 按 vLLM metadata 自动解析，随后**强制钉死已知冲突件**：vLLM 依赖面要求 fastapi `>=0.133`（连带 starlette `>=1.0`），vLLM-Ascend requirements 要求 fastapi `<0.124`，两层区间互斥，联合解析会产出毒解（fastapi 0.1.17、numpy 2.3.5）——分开安装＋事后回钉＝「分层安装」语义，冲突面仅 pip 警告、不阻断。钉死行带 `--no-deps` 为语义必需：钉死值之间的声明约束互斥（instrumentator 8.1.0 声明 starlette `>=1.0`），带依赖解析会直接 ResolutionImpossible。以后遇到新的冲突件（pip 警告或装后验证暴露）就往钉死行追加；不在清单里的交给 pip。numpy 断言失败＝某包传递依赖抬高了版本，定位该包改 `--no-deps` 重装。
- 步骤 3/5/8 各 editable 安装的 `--no-deps` / `--no-build-isolation` 均为语义必需：`--no-deps` 阻止依赖解析顶替已钉版本（torchao 声明 `torch>=2.11.0`、verl 的 `setup.py` 对 transformers 设上界）；`--no-build-isolation` 复用已就位的构建依赖（setuptools / setuptools-rust / cmake / pybind11 / torch / torch_npu / triton-ascend），沿用本脚本时须确保这些构建依赖在位。
- FSDPTurbo 官方基线把 `fla-core` / `flash-linear-attention` 列入主依赖且未钉版本（步骤 4 装入最新版），步骤 8 再钉回 `0.5.2`——两步顺序不可颠倒。

## 4. 装后验证

### 4.1 脚本内置自检

`install.sh` 末尾对两个关键特性做硬性检查并执行 vllm import 冒烟（冒烟自 `mktemp -d` 新建的空目录运行，规避克隆目录 `vllm/` 的 namespace 包影子），任一失败即中止安装：

- verl 侧 `translate_weight_names_for_vllm`（vLLM 量化权重名映射）；
- FSDPTurbo 侧 `quant_ignored_modules`（训练侧排除 `linear_attn` 依赖此项）。

这两项缺失时链路**不会报错，而是静默产出错误结果**（权重名映射失败会导致 rollout 输出全乱码），因此自检失败时按 `readme.md`「环境版本」矩阵核对版本后重装，**不要跳过**。

### 4.2 验证清单（安装完成后逐条执行）

在虚拟环境激活、CANN 已 source、`PYTHONPATH=$PWD/FSDPTurbo:$PYTHONPATH` 已设置的状态下：

```bash
python -c "import fla; print('fla OK')"
python -c "import fla_npu; print('fla_npu OK')"
python -c "import mindspeed_ops; print('mindspeed_ops OK')"
python -c "import torchao, torchao_npu; print('torchao OK')"
python -c "from fsdp_turbo.models.utils import get_model_by_name; print('get_model_by_name OK')"
python -c "import verl, vllm; print(vllm.__version__)"        # 预期 0.30.0 系
python -c "import torch; print(torch.npu.is_available())"     # 预期 True
python -c "import transformers; print(transformers.__version__)"   # 预期 5.22.2
python -c "import numpy; assert numpy.__version__=='1.26.4', numpy.__version__; print('numpy OK')"
python -c "import fastapi, starlette, prometheus_client, cachetools; print('fastapi face OK')"
```

全部通过即环境就绪。注意 `import triton_ascend` 与 `import fla_core` 并非有效校验方式：这两个包不提供同名顶层模块，`ModuleNotFoundError` 属正常现象；triton 侧可用性以 `import triton; triton.language.extra.cann` 为准，fla 侧以 `import fla` 为准。

## 5. 相关文档

- `readme.md`：用例定位、版本矩阵、patch 清单；
- `install.sh`：安装脚本本体；
- `examples/`：训练启动脚本（高低精成对：`run_qwen3_5_35b_fsdp_turbo_a5.sh`（bf16）／`run_qwen3_5_35b_fsdp_turbo_mxfp8.sh`（MXFP8））。
