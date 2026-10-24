# GLM5 on Ascend NPU（vLLM 0.23.0）
本recipe是基于GLM5模型在NPU上进行RLHF后训练的样例，基于GRPO与规则奖励，使用dapo-math17k数据集进行训练，使用数据集mmlu、gpqa、aime2024、aime2025、BFCL进行评测，同样适用于GLM5.1模型

本目录提供 GLM5/GLM5.1 在 Ascend NPU 上进行 GRPO 后训练的 vLLM 0.23.0 配套配置。


## 环境配套

详细的版本配套可以参考该指导文件 [verl github](https://github.com/verl-project/verl)，对于vllm-ascend需要加相关的patch文件，其安装方式可以参考下节 [安装 vLLM Ascend 0.23.0](#安装依赖)

## 安装 vLLM Ascend 0.23.0

```bash

#安装之前要先source cann环境： source /usr/local/Ascend/cann/set_env.sh
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout releases/v0.23.0

PATCH=../verl-ascend-recipe/GLM5/vllm_0230/patch/vllm-ascend.patch
git apply --check "$PATCH"
git apply "$PATCH"

pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..
```

### patch 作用

`patch/vllm-ascend.patch` 修改 `vllm_ascend/attention/sfa_v1.py`，保留：

```text
kv_b_proj
fused_qkv_a_proj
q_proj
```

vLLM Ascend 默认在 `process_weights_after_loading()` 后通过 `dispose_layer()` 将这些层的 Tensor 置为空 Tensor，以节省静态推理显存。VERL 每个训练 step 后还需要再次写入新权重；如果参数已经释放，下一轮同步会发生源参数与 `(0,)` 目标参数的 shape mismatch。

该 patch 允许 VERL 重复写入参数，并在所有权重 bucket 完成后重新执行 `process_weights_after_loading()`，刷新 `W_UV`、`W_UK_T` 等 SFA 推理权重。代价是增加一部分 NPU 显存占用。

确认 patch 已应用：

```bash
git apply --reverse --check   ../verl-ascend-recipe/GLM5/vllm_0230/patch/vllm-ascend.patch
```

命令成功表示已应用，不要重复应用。


## jemalloc安装

为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。

### Ubuntu 操作系统

通过操作系统源安装jemalloc（注意：要求ubuntu版本>=20.04）：

```shell
sudo apt install libjemalloc2
```

在启动任务前执行如下命令通过环境变量导入jemalloc，需先通过 **find /usr -name libjemalloc.so.2** 确认文件是否存在 ：

```shell
# arm64架构
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
# x86_64架构
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
```

### OpenEuler 操作系统

执行如下命令通过操作系统源安装jemalloc

```shell
yum install jemalloc
```

如果上述方法无法正常安装，可以通过源码编译安装 前往jemalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/

```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```

在启动任务前执行如下命令通过环境变量导入jemalloc：

```shell
#根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2,可通过以下命令来设置环境变量(可通过 find /usr -name libjemalloc.so.2 确认文件是否存在)
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
```

# 训练启动

```bash
cd verl
# 修改ray_start.sh中对应的网卡、主节点IP、权重、数据集地址
bash ../verl-ascend-recipe/GLM5/vllm_0230/scripts/ray_start.sh
```
