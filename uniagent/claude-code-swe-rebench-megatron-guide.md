# NPU 上使用 Megatron 后端运行 Claude Code SWE-Rebench 训练

主要介绍如何在 Ascend NPU 环境中，使用 Uni-Agent 调度 Claude Code 解决 SWE-Rebench 任务，接入已有的任务沙箱，并使用 **VERL Megatron 后端训练 actor**、vLLM 执行 rollout，完成 Qwen3.5-9B GRPO 训练。操作以**单节点 16 张 NPU**的脚本为基准；其他模型或卡数需要重新核对并行度、显存和吞吐参数。

按顺序完成环境检查、沙箱适配、Ray/NPU 检查和两步训练烟测。**首次 actor 更新完成且训练步数继续推进**是确认 Megatron 训练链路跑通的关键标准。文中的路径、镜像和网络参数须替换为实际环境中的值。

## 1. 环境要求与所需文件

这个参考以下面为基准：

**参考镜像（A3）**：docker pull swr.cn-north-4.myhuaweicloud.com/mindspeed/uniagent:v0.1.0-verlv0.9.1-a3-vllm

| 类别 | 要求 |
| --- | --- |
| 计算资源 | 单节点 16 张可用 Ascend NPU；驱动、CANN、PyTorch/torch-npu 与训练镜像匹配。 |
| 训练组件 | VERL、Megatron-LM、Megatron-Bridge、vLLM 和 vLLM-Ascend 的**同一套兼容版本**。建议使用已验证的训练镜像和版本清单。 |
| Uni-Agent | `0.1.0` 包及对应源码；本文提供完整训练脚本，另需与 Megatron 版本匹配的补丁。 |
| 任务依赖 | `swebench==4.1.0`；AscendC GDN 需要可用的 `fla_npu`；若沙箱由专用 SDK 管理，还需在执行任务的 Ray Worker 环境安装相应 SDK。 |
| 模型与数据 | 可读取的 Qwen3.5-9B 模型目录；训练、验证 Parquet 文件；训练数据包含 `prompt`、`data_source`、`extra_info` 列。 |
| 沙箱 | 已有沙箱平台与 Uni-Agent provider 适配器；能按 SWE-Rebench 数据中的镜像创建沙箱，提供 `/testbed` 仓库、`bash`、`git`、`claude` 命令和任务评测所需依赖。 |
| 网络与存储 | Ray 节点之间互通；训练节点可访问沙箱控制面及镜像仓库；模型、数据、日志与 checkpoint 目录有足够空间。 |

本指南在第 3 节给出自有沙箱任务 YAML，在第 6 节给出基于仓库官方 Megatron 示例修改的完整训练脚本。Megatron 兼容补丁需放在 Uni-Agent 仓库根目录，文件名为 `nvrx-version-0.6.0-compat.patch`；应先确认补丁与 Megatron-LM 提交匹配。

训练脚本要求显式设置 `MODEL_PATH`、`TRAIN_FILE`、`TEST_FILE`、`RUNTIME_ENV`、`TASK_CONFIG`、`RUNTIME_DIR`、`VERL_ROOT`、`MBRIDGE_ROOT` 和 `RAY_JOB_ADDRESS`。沙箱类型由任务配置的 `sandbox.provider` 决定，脚本不绑定某一种沙箱服务。

源码仓库可能包含其他 provider 的实现与示例 YAML。本流程只读取 `TASK_CONFIG` 指向的文件；其他示例文件不会自动参与任务。还需检查 Parquet 样本的沙箱字段，因为样本配置会覆盖 YAML 中同名字段。

## 2. 准备目录和版本清单

以下命令在训练容器内执行。把尖括号内容替换为实际值；代码块中的路径仅展示目录结构。

```bash
export UNI_AGENT_ROOT=/workspace/uni-agent
export VERL_ROOT=/workspace/verl
export MEGATRON_ROOT=/workspace/Megatron-LM
export MBRIDGE_ROOT=/workspace/Megatron-Bridge/src
export RUNTIME_DIR=/data/uni-agent
export MODEL_PATH=/models/Qwen3.5-9B
export TRAIN_FILE=/data/datasets/swe_rebench_train.parquet
export TEST_FILE=/data/datasets/swe_bench_verified.parquet
export RAY_JOB_ADDRESS=http://127.0.0.1:8265

test -d "$VERL_ROOT" && test -d "$MEGATRON_ROOT" && test -d "$MBRIDGE_ROOT"
test -f "$MODEL_PATH/config.json" && test -f "$TRAIN_FILE" && test -f "$TEST_FILE"
mkdir -p "$RUNTIME_DIR/logs" "$RUNTIME_DIR/ckpts" "$RUNTIME_DIR/config"
```

***使用镜像的话，下面不涉及，仅供参考**

如果尚未取得 Uni-Agent 源码，可克隆到一个**不存在的目标目录**，然后切换到与训练镜像匹配的固定提交或标签；已有源码时，从 `git checkout` 开始执行：

```bash
git clone https://github.com/verl-project/uni-agent.git "$UNI_AGENT_ROOT"
git -C "$UNI_AGENT_ROOT" checkout <已验证的提交号或标签>
git -C "$UNI_AGENT_ROOT" submodule update --init --recursive
python -m pip install --no-deps -e "$UNI_AGENT_ROOT"
python -c 'from importlib.metadata import version; assert version("uni-agent") == "0.1.0"'
test -f "$UNI_AGENT_ROOT/nvrx-version-0.6.0-compat.patch"
```

VERL、vLLM、vLLM-Ascend 和 Megatron 应按已验证的镜像版本清单安装。重新构建训练镜像时，先锁定各组件源码提交、CANN/PyTorch 版本和安装顺序，再在干净镜像中构建；Ascend 适配的 vLLM 应使用与 vLLM-Ascend 匹配的构建方式。

安装 SWE 任务依赖，并核对训练组件：

```bash
python -m pip install 'swebench==4.1.0'
python - <<'PY'
from importlib.metadata import version
for package in (
    "uni-agent", "verl", "vllm", "vllm-ascend", "swebench",
    "nvidia-resiliency-ext",
):
    print(f"{package}: {version(package)}")
PY
```

记录上述输出与 `git -C "$UNI_AGENT_ROOT" rev-parse HEAD`。使用专用沙箱 SDK 时，还应在 Ray Worker 使用的同一 Python 环境中验证该 SDK 可导入。

### Megatron 的 nvidia-resiliency-ext 0.6.0 兼容补丁

镜像若已包含补丁，只需验证；源码未打补丁且 `nvidia-resiliency-ext` 为 0.6.0 时，在**与补丁匹配的 Megatron-LM 提交**上执行：

```bash
export NVRX_PATCH="$UNI_AGENT_ROOT/nvrx-version-0.6.0-compat.patch"
if git -C "$MEGATRON_ROOT" apply --reverse --check "$NVRX_PATCH"; then
    echo 'Megatron 补丁已应用'
else
    git -C "$MEGATRON_ROOT" apply --check "$NVRX_PATCH"
    git -C "$MEGATRON_ROOT" apply "$NVRX_PATCH"
    echo 'Megatron 补丁已应用'
fi
python -c 'import megatron.core.dist_checkpointing.strategies.torch; print("Megatron 导入成功")'
```

该补丁让版本判断读取安装包元数据，并移除异步能力检测时的强制版本断言，用于解决 `nvidia_resiliency_ext` 未提供 `__version__` 导致的导入失败。`git apply --check` 不通过时，应核对 Megatron-LM 提交是否与补丁匹配，再使用对应补丁；不要跳过检查修改其他位置。

## 3. 接入已有沙箱

Uni-Agent 从每条数据的 `tools_kwargs.task` 与任务 YAML 合成运行配置。`sandbox.provider` 决定使用哪个实现，`sandbox.image` 指定任务镜像；`image_map` 可把数据集中的镜像名映射到实际镜像仓库。SWE-Rebench 运行时会在沙箱的 `/testbed` 执行 Git 清理、启动 CC，再执行评测。因此，沙箱必须支持：

训练数据的 `extra_info.tools_kwargs.task` 至少应给出任务名、任务镜像和该样本的元数据；示意结构如下，其余评测字段按数据集格式保留：

```json
{
  "name": "swe_rebench",
  "sandbox": {"image": "<样本镜像>"},
  "metadata": {"instance_id": "<实例 ID>", "base_commit": "<Git 提交>"}
}
```

若数据行另有 `sandbox.provider` 或旧的 `sandbox_kwargs`，它们可能覆盖任务 YAML。启动前应核对样本与 YAML 合成后的 provider、镜像和参数；不符合预期时，应调整数据准备流程并重新生成 Parquet。

1. 按任务镜像创建和释放实例，且并发配额能覆盖计划的 rollout 数量。
2. 在 `/testbed` 提供与样本 `base_commit` 对应的 Git 仓库，以及 `bash`、`git`、评测依赖。
3. 在沙箱的 `PATH` 中提供可运行的 `claude` 命令。
4. 从沙箱内连接本次 rollout 的模型网关地址；地址由 `run_task` 通过 `session.base_url` 写入 agent 模型配置。
5. 支持带工作目录和环境变量的命令执行、文件读写，以及停止后的资源释放。

### 3.1 配置自有沙箱

先将 `SANDBOX_PROVIDER` 设置为当前平台已注册的 provider 名称。以下任务配置只填写 Uni-Agent 与 CC SWE 共通的字段；根据适配器要求，在两个任务的 `sandbox_kwargs` 中加入资源、镜像和网络参数。训练数据使用 `swe_rebench`，验证数据使用 `swe_bench`。

```bash
: "${SANDBOX_PROVIDER:?请先设置自有沙箱的 provider 名称}"
export TASK_CONFIG="$RUNTIME_DIR/config/task_config_claude_code.yaml"
cat > "$TASK_CONFIG" <<YAML
- name: swe_bench
  sandbox:
    provider: ${SANDBOX_PROVIDER}
    runtime_timeout: 7200
    sandbox_kwargs: {}
  agent:
    name: claude_code
    max_turns: 200
    run_timeout: 4800
    model:
      temperature: 1.0
      top_p: 1.0
      max_total_tokens: 131072
  prompt_template: &swe_claude_prompt
    - role: user
      content: |-
        Read the following task description and resolve the issue in the current directory.

        Task description:
        {problem_statement}

        Inspect the relevant code, make the minimal correct changes, and verify the result with appropriate tests or checks. Do not modify tests or commit changes. When finished, briefly summarize what changed and how it was verified.

- name: swe_rebench
  sandbox:
    provider: ${SANDBOX_PROVIDER}
    runtime_timeout: 7200
    sandbox_kwargs: {}
  agent:
    name: claude_code
    max_turns: 200
    run_timeout: 4800
    model:
      temperature: 1.0
      top_p: 1.0
      max_total_tokens: 131072
  prompt_template: *swe_claude_prompt
YAML
```

如果数据集中的镜像名与平台镜像仓库不同，可在相应任务的 `sandbox` 下增加 `image_map`。映射后的镜像仍需包含对应样本的 `/testbed`、评测依赖与 CC 命令；请根据平台的镜像命名规则填写映射，不要直接套用其他平台的路径。

### 3.2 适配 provider 与模型网关

已有平台可实现 `Sandbox` 适配器并注册 provider 名称；适配器至少实现 `start`、`stop`、`_exec`，返回 `ExecResult(exit_code, stdout, stderr)`，并将 `sandbox_kwargs` 映射到平台 SDK 参数。基础类提供 `exec_shell`、文件读写和目录传输的默认实现；如平台提供原生文件通道，可覆盖相应方法。把 provider 模块注册到 `uni_agent.sandbox.registry.SANDBOX_MODULES`，确保模块与平台 SDK 在提交端及执行任务的 Ray Worker 上均可导入。

| 平台能力 | Uni-Agent 接口及检查点 |
| --- | --- |
| 创建/销毁实例 | `start()` / `stop()`；失败时也要释放已申请资源。 |
| 执行命令 | `_exec(argv, timeout, workdir, env)`；返回退出码、标准输出和标准错误。 |
| 文件读写 | 基础类默认实现 `read_file` / `write_file`；平台可提供原生实现。 |
| 实例健康与超时 | 建议实现 `is_alive()`，并确认创建超时与任务运行超时能被正确处理。 |
| 模型网关连接 | 沙箱内的 CC 必须能访问任务运行时传入的 `session.base_url`。 |

认证信息应通过受控的 Worker 环境或密钥管理服务提供，不要写入任务 YAML。任务镜像内应能在 `/testbed` 执行 Git 与评测命令，并通过 `command -v claude` 找到 CC。沙箱须能直连 `session.base_url`；若网络隔离要求隧道，需要在适配器与任务执行器中实现并验证，单独添加 `proxy_port` 不会给自有 provider 自动建立隧道。

## 4. 配置 Ray runtime 环境

**Ray Worker 不会自动继承提交终端的全部环境变量**。下面的 runtime 文件把 Megatron-Bridge、VERL 和 Uni-Agent 源码目录加入 Worker 的 `PYTHONPATH`；这些路径必须在实际 Worker 环境中存在。

```bash
export RUNTIME_ENV="$RUNTIME_DIR/config/runtime_env_swe_megatron.yaml"
cat > "$RUNTIME_ENV" <<YAML
env_vars:
  PYTHONPATH: "${MBRIDGE_ROOT}:${VERL_ROOT}:${UNI_AGENT_ROOT}"
  PYTHONUNBUFFERED: "1"
  RAY_OVERRIDE_JOB_RUNTIME_ENV: "1"
  VLLM_USE_V1: "1"
YAML
chmod 600 "$RUNTIME_ENV"
```

专用沙箱 SDK 的依赖和非敏感配置也必须进入实际 Ray Worker 环境。密钥应由运行平台注入；限制 Ray dashboard、Job 元数据和日志的读取权限，排障时不要输出完整环境变量。

## 5. 检查 Ray 与 NPU

```bash
npu-smi info
ray status
curl -fsS --connect-timeout 3 --max-time 10 "$RAY_JOB_ADDRESS/api/version" >/dev/null
echo 'Ray Jobs API 可访问'
```

确认 Ray 集群的可调度 NPU 资源数至少为 16，且没有与本实验冲突的作业。Ray head 若由平台管理，请使用平台分配的 Jobs API 地址；自行启动时，按集群部署手册启动后再执行上述检查。`8265` 的连接失败或 504 属于 Ray dashboard/Jobs API 问题，应先恢复该服务。

仅在**该容器尚无 Ray head、且确认不会影响现有作业**时，可用以下单机命令建立示例集群；实际资源名称和端口须与训练镜像、Ray 配置一致：

```bash
ray start --head --resources='{"NPU":16}' \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 \
  --disable-usage-stats
ray status
curl -fsS "$RAY_JOB_ADDRESS/api/version" >/dev/null
```

## 6. 生成训练脚本

以下脚本以官方 `examples/quickstart/training/train_qwen3p5_dense.sh` 为基线，保留其 Megatron、CC SWE rollout 与 reward 链路，仅调整单机 16 卡 Ascend/Qwen3.5-9B 参数、自有沙箱配置入口、AscendC GDN 和两步烟测设置。

```bash
mkdir -p "$UNI_AGENT_ROOT/examples/quickstart/training"
cat > "$UNI_AGENT_ROOT/examples/quickstart/training/train_npu_qwen3p5_9b_claude_code_swe_megatron.sh" <<'BASH'
#!/usr/bin/env bash
# Adapted from examples/quickstart/training/train_qwen3p5_dense.sh.
set -euo pipefail

project_name=${PROJECT_NAME:-"CC-SWE-Qwen3.5-9B-Megatron-Ascend"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE:?Set TRAIN_FILE}
TEST_FILE=${TEST_FILE:?Set TEST_FILE}

RUNTIME_DIR=${RUNTIME_DIR:?Set RUNTIME_DIR}
RUNTIME_ENV=${RUNTIME_ENV:?Set RUNTIME_ENV}
RAY_JOB_ADDRESS=${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS}
UNI_AGENT_ROOT=${UNI_AGENT_ROOT:?Set UNI_AGENT_ROOT}
cd "${UNI_AGENT_ROOT}"
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}
# Submit from the Uni-Agent root; VERL and Megatron-Bridge are supplied by the Ray runtime environment.
# --- Agent-framework rollout (replaces the swe_agent agent-loop) --------------
# Run-wide task base (agent + sandbox + sampling), loaded from this YAML by
# uni_agent.framework.task_runner.run_task and deep-merged onto each row's task.
# Same file-path idea as the old agent_loop_config_path; new (task-config) schema.
TASK_CONFIG=${TASK_CONFIG:?Set TASK_CONFIG}
TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}    # gateway tool-call parser; MUST match the model chat template
GATEWAY_COUNT=${GATEWAY_COUNT:-8}            # gateway actors fronting the engine
CONCURRENCY=${CONCURRENCY:-8}              # max in-flight rollout sessions (runner cap)
NUM_AGENT_WORKERS=${NUM_AGENT_WORKERS:-8}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-False}  # opt-in: zero the loss mask for unfinished episodes

rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"} # sglang or vllm

# Algorithm parameters
adv_estimator=${ADV_ESTIMATOR:-grpo}

use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}

# Response length parameters
max_prompt_length=${MAX_PROMPT_LENGTH:-8000}
max_response_length=${MAX_RESPONSE_LENGTH:-128000}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}  # unused
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
loss_mode=${LOSS_MODE:-vanilla}

# Algorithm
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# Performance Related Parameter
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False}
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-2}
train_tp=${TP:-2}
train_pp=${PP:-1}
train_cp=${CP:-4}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.0}

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

# V1 colocate_async topology. colocate_async colocates actor + rollout on the same
# GPUs (rollout replicas sleep during the train step), so NNODES is the TOTAL node
# count (replaces the old fully-async NNODES_ROLLOUT + NNODES_TRAIN split).
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

# parameter_sync_step defaults to 1 for colocate_async, so train_batch_size
# (prompts/step) only needs to be > 0 (the old async mode used train_batch_size=0).
# num_warmup_batches pre-fills the rollout pipeline before the first train step.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-8}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
lr_decay_steps=${LR_DECAY_STEPS:-2000}
test_freq=${TEST_FREQ:--1}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-34000}
total_training_steps=${TOTAL_TRAINING_STEPS:-200}

# ============================================================================
# Rollout correction is disabled by default for the standard GRPO + PPO
# baseline. Override these variables to enable behavior-anchor or decoupled
# rollout-correction experiments.
# ============================================================================
bypass_mode=${BYPASS_MODE:-False}                                # True => old_log_prob = rollout_log_prob
bypass_loss_type=${BYPASS_LOSS_TYPE:-ppo_clip}                   # ppo_clip | reinforce
rollout_is=${ROLLOUT_IS:-null}                                   # PPO clip already applies the IS ratio
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}                # single float => TIS upper clamp; "lo_hi" string => IcePop
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}  # normalize IS weights to mean=1.0 within a batch
rollout_rs=${ROLLOUT_RS:-null}                                   # no rejection sampling
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

# A Python import check does not prove kernel execution; verify the first actor update.
python3 - <<'PY_FLA'
import fla_npu
from fla_npu.ops import ascendc
print('fla_npu AscendC API import: OK')
PY_FLA

ray job submit --address="${RAY_JOB_ADDRESS}" --no-wait --runtime-env "${RUNTIME_ENV}" --working-dir "${UNI_AGENT_ROOT}" \
    -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
    python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.device=npu \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches} \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.filter_groups.enable=False \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_triton_gdn=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    algorithm.rollout_correction.loss_type=${bypass_loss_type} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${bypass_mode} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_rs} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${bypass_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS} \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.megatron.pad_bshd_to_minibatch_max=True \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    'trainer.logger=["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}"
BASH
chmod +x "$UNI_AGENT_ROOT/examples/quickstart/training/train_npu_qwen3p5_9b_claude_code_swe_megatron.sh"
bash -n "$UNI_AGENT_ROOT/examples/quickstart/training/train_npu_qwen3p5_9b_claude_code_swe_megatron.sh"
```

## 7. 两步训练烟测

确认沙箱资源、任务镜像及模型网关网络已就绪后，在 Ray head 所在环境执行。每个实验使用唯一的 `EXP_NAME`，尤其在多个节点共享日志和 checkpoint 目录时。训练脚本默认每个 prompt 产生 4 个响应，关闭全 0/全 1 组过滤，并对 actor 启用 AscendC GDN：

```bash
cd "$UNI_AGENT_ROOT"
export PROJECT_NAME=cc-megatron-qwen35-9b
export EXP_NAME="cc-megatron-smoke-$(date +%Y%m%d-%H%M%S)"
export NNODES=1 NGPUS_PER_NODE=16
export TRAIN_PROMPT_BSZ=8 N_RESP_PER_PROMPT=4
export TOTAL_TRAINING_STEPS=2
export CONCURRENCY=4 NUM_AGENT_WORKERS=4 GATEWAY_COUNT=1

bash examples/quickstart/training/train_npu_qwen3p5_9b_claude_code_swe_megatron.sh 2>&1 | tee "$RUNTIME_DIR/logs/$EXP_NAME.submit.log"
```

脚本沿用官方示例的 `ray job submit --no-wait`：提交日志只用于记录 Ray Job ID，训练日志由 Ray Jobs API 查询。`fla_npu` 导入成功只确认 Python 接口存在；需在第 8 节核对首次 actor 更新，才能确认 AscendC GDN 在当前 NPU 环境可用。

脚本的 `trainer.resume_mode=auto` 会参考同名实验目录；烟测和正式训练应使用不同的 `EXP_NAME`。

## 8. 监控与验收

在另一个终端设置烟测的实际实验名，从提交日志取得 Job ID，再读取 Ray Job 日志、轨迹和训练步数：

```bash
export EXP_NAME='<第 7 节使用的实验名>'
export PROJECT_NAME=cc-megatron-qwen35-9b
export SUBMIT_LOG="$RUNTIME_DIR/logs/$EXP_NAME.submit.log"
export LOG="$RUNTIME_DIR/logs/$EXP_NAME.ray.log"
export AGENT_DIR="$RUNTIME_DIR/logs/$PROJECT_NAME/$EXP_NAME"
test -f "$SUBMIT_LOG"
JOB_ID=$(grep -oE 'raysubmit_[A-Za-z0-9]+' "$SUBMIT_LOG" | tail -n 1)
test -n "$JOB_ID" || { echo '未找到 Ray Job ID，请检查提交日志'; exit 1; }
printf 'Ray Job ID: %s\n' "$JOB_ID"
ray job status --address "$RAY_JOB_ADDRESS" "$JOB_ID"
ray job logs --address "$RAY_JOB_ADDRESS" "$JOB_ID" > "$LOG"
find "$AGENT_DIR" -type f -name trajectory.json 2>/dev/null | wc -l
grep -E 'global_steps|update_actor|actor/|cu_seqlens|sandbox|connection|Traceback|ERROR' "$LOG" | tail -n 60
```

按以下顺序验收：

1. Ray Job 能启动；实际样本使用预期的沙箱 provider 和镜像，CC 能从沙箱访问模型网关，并产生轨迹文件。
2. SWE-Rebench 的 Git 清理、CC 执行和 reward/scoring 均完成，没有 `swebench` 导入失败或打分异常。
3. 首次 actor 更新完成，出现相应的 actor 指标；训练步数从 1 推进到 2，没有 `cu_seqlens is not supported for torch_chunk_gated_delta_rule`。
4. 两步烟测按预期结束，日志中无未处理的致命异常；按需要核对 checkpoint 和资源释放情况。

只有轨迹文件或 CC 进程运行，不能证明训练已通过 actor 更新。若首次 actor 更新尚未发生，继续观察并保留日志，不能将其记为完成。

## 9. 正式训练

两步烟测通过，且沙箱资源池能够承载计划并发后，设置新的实验名并逐步提高并发。下面的并发数 8 是示例值，应根据资源配额、单沙箱请求和观察到的创建成功率调整：

```bash
unset TOTAL_TRAINING_STEPS
export EXP_NAME="cc-megatron-full-$(date +%Y%m%d-%H%M%S)"
export CONCURRENCY=8 NUM_AGENT_WORKERS=8 GATEWAY_COUNT=2
bash examples/quickstart/training/train_npu_qwen3p5_9b_claude_code_swe_megatron.sh 2>&1 | tee "$RUNTIME_DIR/logs/$EXP_NAME.submit.log"
```

保持与烟测相同的模型、数据、任务配置、SDK、runtime 和训练组件版本。保存实验名、Job ID、配置文件校验值、源码提交号、Ray 日志路径和 checkpoint 路径，以便复现。多节点部署需使用对应的多节点脚本，单独验证共享存储、Ray 资源、网卡选择和节点间通信；本文的单节点命令不直接推广到多节点。

## 10. 常见问题

| 现象 | 定位和处理 |
| --- | --- |
| `Unknown sandbox provider` 或 SDK 导入失败 | 检查 `SANDBOX_MODULES` 注册、provider 模块及 SDK 是否在实际 Ray Worker 环境可导入。 |
| 镜像拉取或沙箱创建失败 | 检查 `image_map` 解析后的镜像名、镜像仓库权限及平台资源配额。 |
| `/testbed` 不存在或 `base_commit` 无法校验 | 检查样本镜像内容、数据行中的 `base_commit` 及任务镜像映射。 |
| `claude` 在沙箱中不存在 | 检查任务镜像是否包含 CC，或适配器是否正确挂载工具文件并设置沙箱内 `PATH`。 |
| CC 无法访问模型网关 | 检查沙箱到 `session.base_url` 的路由、DNS、防火墙和端口；自定义隧道需要在适配器与任务执行器中实现。 |
| Ray Jobs API 连接失败或 504 | 检查 head、dashboard、端口和代理配置；先恢复 Jobs API。 |
| `nvidia_resiliency_ext` 没有 `__version__` | 核对已安装版本、Megatron 源码提交与本指南第 2 节的兼容补丁。 |
| Hydra 不识别配置键 | 核对实际安装的 VERL、Megatron 版本与示例脚本；以该版本的配置结构修正。 |
| `No module named swebench` | 在实际运行的 Ray Worker 环境安装并验证 `swebench==4.1.0`；继续检查 reward/scoring。 |
| `steps_per_epoch=0` | 检查训练数据行数是否小于 `TRAIN_PROMPT_BSZ`，以及数据是否被预处理过滤。 |
| `cu_seqlens ... torch_chunk_gated_delta_rule` | 核对 actor 的 GDN 后端选择是否进入实际 Hydra/Megatron 配置；确认 AscendC 算子支持当前序列布局，再验证首次 actor 更新。 |
| 轨迹生成但训练停在第一步 | 查首个 reward/scoring 或 actor 更新异常；记录 Job ID 与首次错误位置。 |

## 参考文档

- [Uni-Agent Sandbox 配置与自定义 provider](https://uni-agent.readthedocs.io/en/latest/concepts/sandbox.html)
- [Ray runtime 环境与 Jobs API](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html)
