# KASandbox: two-node Ascend training

See [shared prerequisites and execution contract](../README.md) and
[required verl reference](REQUIRED_VERL.txt). This bundle preserves the working two-node Megatron
launch and separate E2B API/client-proxy paths. Source provenance: Uni-Agent deployment revision
`0d4d7cf90a50b78001d605b498fab95c8abc6e77`, including its working launch changes, expressed as a patch
against public ancestor `1622670bf156c0c11bb1a7317e4eb1f3c3fc7680`.

## 1. Prepare source and private configuration

```bash
export UNI_AGENT_ROOT=/workspace/kasandbox-uni-agent
bash setup.sh
# In the compatible training Python environment, after satisfying dependencies:
python -m pip install --no-deps -e "$UNI_AGENT_ROOT"
cp .env.example /secure/kasandbox.env
chmod 600 /secure/kasandbox.env
# Edit the file before proceeding:
export SANDBOX_ENV_FILE=/secure/kasandbox.env
```

The dependency patch includes the E2B provider, NPU/framework integration, SWE task adaptation,
per-Gateway dynamic callback routing, Claude binary upload/NFS support and sandbox lifecycle timing.
It is tied to the public ancestor above, not the current upstream API. SSH host-key checking is
enabled in the exported patch. The optional automatic verl training-timing rewrite from the private
deployment is not part of this launch path; it is not needed to train or connect the sandbox.

## 2. Prepare task data and Claude

Use task-specific templates with `/testbed`, the correct base revision and preinstalled test tools.
The recipe's dataset converter accepts JSON/JSONL/Parquet SWE-Bench rows and an explicit manifest:

```json
{"schema_version": 1, "instances": {
  "example__train-1": {"template": "train-template-id", "split": "train"},
  "example__test-2": {"template": "validation-template-id", "split": "validation"}
}}
```

```bash
python examples/e2b_swe_grpo/prepare_data.py \
  --source /data/swe-bench.json --manifest /data/templates.json --output-dir /data/kasandbox
```

Supply at least 32 real training rows for the default batch; the small manifest above only illustrates
the format. Each row binds `swe_bench` to its own E2B template and includes the benchmark's metadata.
Set `TRAIN_FILE` and `TEST_FILE` to the resulting `train.parquet` and `validation.parquet`.
The converter also supports an explicit `--smoke-only` mode; do not mix smoke data with held-out claims.

Choose `CLAUDE_CODE_BINARY_MODE`:

- `installed` (portable default): provide a working ARM64 `claude` in each template's PATH.
- `upload`: set `CLAUDE_CODE_LOCAL_BINARY` and `CLAUDE_CODE_LOCAL_BINARY_SHA256` on shared training storage.
- `nfs` (original deployment mode): set `CLAUDE_CODE_NFS_SOURCE=host:/export`,
  `CLAUDE_CODE_NFS_RELATIVE_BINARY=VERSION/linux-arm64/claude`, and its SHA256. Guests need NFS tools,
  mount permission, and access to the export. The adapter verifies the existing mount/source and binary;
  it does not configure the NFS server.

No machine-specific NFS address, binary version, token, or template ID is assumed.

## 3. Connect through a Windows PC

Configure `E2B_API_KEY`, the relay SSH target and callback URL in the private file. On a PC that can
reach the sandbox service and relay, run the following in separate PowerShell terminals and keep them open.
Replace the example hostnames. The sandbox service defaults below are API `3000`, client-proxy `3002`;
adjust them to the actual service configuration.

```powershell
$Relay = 'user@relay.example.com'
$SandboxHost = 'sandbox.example.com'
ssh -p 22 -N -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R "127.0.0.1:29191:${SandboxHost}:3000" $Relay
```

```powershell
$Relay = 'user@relay.example.com'
$SandboxHost = 'sandbox.example.com'
ssh -p 22 -N -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R "127.0.0.1:29192:${SandboxHost}:3002" $Relay
```

```powershell
.\scripts\sandbox\start_callback_tunnels.ps1 `
  -RelayTarget user@relay.example.com -PcIp 192.0.2.20 `
  -GatewayCount 8 -CallbackPortBase 29193 -RelayPortBase 29193
```

`192.0.2.20` is a documentation address: replace it with a local PC address reachable from the
actual sandboxes and set `CALLBACK_ADVERTISED_BASE_URL` to that host and base port.
The training containers connect to the relay using `KASANDBOX_SSH_PORT`, which may differ from
the PC's `-SshPort`. The complete path for Gateway `i` is:

```text
sandbox → PC:(29193+i) → relay:(29193+i) → actual Gateway node:dynamic-port
```

Reserve the entire eight-port callback range, plus API/proxy ports. The launcher forwards container
`9191/9192` to relay `29191/29192` on each node. GatewayManager discovers the actual listener and owns
one reverse SSH connection per Gateway, so placement on the worker node is supported.

For direct service reachability, set `KASANDBOX_TUNNEL_MODE=direct`, `E2B_API_URL`, and
`E2B_SANDBOX_URL`. If sandboxes can reach the actual Gateway listeners directly, also set
`CALLBACK_TUNNEL_ENABLED=false`; otherwise retain the callback relay settings.

## 4. Launch

On both workers, using `VC_TASK_INDEX=0` on the head and `1` on the other node:

```bash
export SANDBOX_ENV_FILE=/secure/kasandbox.env
export VC_WORKER_HOSTS=head.example.com,worker.example.com
export VC_TASK_INDEX=0
bash run_reproduce.sh --dry-run
bash run_reproduce.sh trainer.total_training_steps=2 trainer.save_freq=-1
```

Default Ray head/dashboard ports are `6766`/`8260`. Only the head creates a private mode-0600 Ray
runtime JSON and submits training after both nodes register. Recipe/dependency/data paths must exist
at the same absolute locations on both nodes; Ray does not upload the repository or private `.env` file.
The private JSON carries the sandbox credentials to Ray workers and must stay private.
Launch logs are under `logs_reproduce/`, checkpoints/rollouts under `.artifacts/`.

Use the validation criteria in the shared README before a full run. Neither a reachable API nor
successfully mounting the Claude binary proves that a real model session or training update succeeded.
