# Ascend Uni-Agent sandbox training recipes

Two portable launch bundles for **Qwen3.5-9B, two nodes × 16 Ascend NPUs,
Megatron actor training, vLLM rollout and Claude Code SWE tasks**:

| Recipe | Sandbox control/data plane | Model callback path |
| --- | --- | --- |
| [AgentENV](agentenv/README.md) | E2B-compatible AgentENV API, through independent SSH lanes | Sandbox → AgentENV host → PC → relay → actual Gateway listener |
| [KASandbox](kasandbox/README.md) | Separate E2B API and client-proxy, SSH or direct | Sandbox → PC → relay → actual Gateway listener |

These belong beside the existing [Ascend Megatron guide](../claude-code-swe-rebench-megatron-guide.md):
they add concrete sandbox integration and runnable launch scripts to that training workflow.
Each directory is independently usable and carries its own dependency patch and setup script.

## Training prerequisites

- Use the compatible Ascend training stack described in the parent guide. The documented A3 image is
  `swr.cn-north-4.myhuaweicloud.com/mindspeed/uniagent:v0.1.0-verlv0.9.1-a3-vllm`.
  Drivers/CANN, torch-npu, Megatron, Megatron-Bridge, vLLM-Ascend and `fla_npu` must agree.
- Set `VERL_SOURCE` to the compatible source **inside the training image**. The management machine is
  not a training environment. Each [AgentENV](agentenv/REQUIRED_VERL.txt) /
  [KASandbox](kasandbox/REQUIRED_VERL.txt) version file records the historical reference revision;
  installing that upstream commit alone does not supply the Ascend stack.
- Both workers need the same absolute recipe, Uni-Agent, model and dataset paths on shared storage,
  and the same Python dependencies (`ray`, `psutil`, `pyarrow`, `swebench==4.1.0`, `e2b`, `PyYAML`,
  plus the pinned Uni-Agent checkout's requirements). AgentENV's original vendor SDK was `e2b==2.62.0`.
  Do not copy a vendor directory or an environment from another architecture.
- The model directory contains `config.json` and weights. Sandbox templates contain the task repository,
  matching `base_commit`, test dependencies and a working ARM64 Claude executable or the configured asset source.
- The default 32 prompts × 8 responses need at least 32 training rows. Supply a separate held-out
  validation dataset for evaluation. Repeated rows are only a smoke-test convenience.

The dependency revisions intentionally differ: AgentENV uses Uni-Agent
`02760c3d222366d513845401833caed06be79b47`, KASandbox uses
`1622670bf156c0c11bb1a7317e4eb1f3c3fc7680`, each with its included patch.
`setup.sh` prepares a dedicated checkout, checks the revision and patch applicability, and is repeatable.
It never installs or overwrites the image's verl. Do not replace the pin with current `main` without retesting.

## Configuration and execution contract

1. Copy the recipe's `.env.example` **outside the repository**, edit deployment values, and `chmod 600` it.
   Set `SANDBOX_ENV_FILE` to that absolute path. This is trusted Bash configuration, not an arbitrary data file.
   Use platform secret injection or this private file for `E2B_API_KEY`; never put keys in Hydra arguments.
2. Prepare the dependency with `UNI_AGENT_ROOT=/shared/path bash setup.sh`. Install required Python packages
   in the training image, for example `python -m pip install --no-deps -e "$UNI_AGENT_ROOT"` after its
   dependencies are satisfied. The launcher sets `PYTHONPATH` to this patched checkout on every worker.
3. Establish the PC bridge if required. Preconfigure SSH keys/agent and verify the server's host key on
   the PC and both workers. SSH forwarding must be allowed. A custom SSH port is configuration; these
   scripts do not start another sshd or change the firewall. Allow only the appropriate sandbox network
   to reach the exposed callback/asset ports.
4. Run `bash run_reproduce.sh --dry-run` to render the final training command. It starts no SSH, Ray,
   sandbox or training process and requires neither NPU imports nor credentials. It checks shell-level
   configuration, not dataset contents or the live training configuration schema.
5. Start `bash run_reproduce.sh` on **both workers**, with identical configuration except `VC_TASK_INDEX`
   (`0` on the head, `1` on the worker) and optional `LOCAL_IP`. Supply `VC_WORKER_HOSTS=head,worker` when
   the scheduler does not. The actual launch runs local dependency/data checks before opening tunnels,
   then waits up to ten minutes for both Ray nodes. Additional arguments are Hydra overrides.

`LOCAL_IP`/`GLOO_SOCKET_IFNAME`/`HCCL_SOCKET_IFNAME` are detected by default; set them explicitly on
multi-interface machines. Choose reachable node IPs. The validated topology is 2 × 16; changing model,
parallelism or hardware requires separate capacity and schema validation.

The head submits training and owns API tunnels until it exits. Each Gateway owns its callback SSH
process, routing to its discovered listener rather than assuming it lives on the head. Worker shells
remain alive to maintain node-local API forwarding. End the scheduler job on both nodes when training
finishes. No script kills unrelated Ray/SSH processes; Ray lifetime belongs to the job containers.
Reserve all ports and use isolated containers for concurrent jobs. Private Ray runtime JSON and local
SSH logs may contain deployment information; they are excluded from version control.

## Verification and limits

The bundle is validated with shell/Python/PowerShell syntax checks, clean-checkout patch application,
repeat setup, command rendering and fail-fast checks. These checks do **not** establish NPU training
success on a new platform. Before a long run, validate the generated overrides against the image's
Hydra schema, verify a real sandbox can call its session's model endpoint, and run a short training job.
Success means rollout completion, a finite reward, an actor optimizer update, and advancement to the
next training step. A `200 /health`, running Claude process, or Ray registration alone is insufficient.

Original source provenance is recorded in the individual READMEs. Private datasets, model weights,
tokens, binary assets, logs and machine-specific environment files are not part of these recipes.
