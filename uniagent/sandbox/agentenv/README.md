# AgentENV: two-node Ascend training

See [shared prerequisites and execution contract](../README.md) and
[required verl reference](REQUIRED_VERL.txt). This recipe retains the working AgentENV task/evaluation
adapter, byte-transparent API relay, per-Gateway callback lifecycle, and the reference GRPO parameters.
Source provenance: deployment revision `1aa377956f51dd8c17d86e396cb515919f4aefe1`, including its working
launch/configuration changes; Uni-Agent revision `01842cc197606629c4591d767012aa6c64a98d50` is represented
as a patch against public base `02760c3d222366d513845401833caed06be79b47`.

## 1. Prepare source and configuration

From this directory on shared training storage:

```bash
export UNI_AGENT_ROOT=/workspace/agentenv-uni-agent
bash setup.sh
# In the compatible training Python environment, after satisfying dependencies:
python -m pip install --no-deps -e "$UNI_AGENT_ROOT"
python -m pip install 'e2b==2.62.0' 'swebench==4.1.0' pyarrow PyYAML psutil
cp .env.example /secure/agentenv.env
chmod 600 /secure/agentenv.env
# Edit /secure/agentenv.env, then:
export SANDBOX_ENV_FILE=/secure/agentenv.env
```

`patches/uni-agent.patch` adds session-owned callback routing, the 60-second Gateway HTTP keep-alive,
and Claude retry/diagnostic behavior from the working source. The small Python adapters beside the
launcher are required at runtime; copying only `run_reproduce.sh` is insufficient.

## 2. Bind data and assets

Existing compatible Parquet files can be used directly. Every row needs `prompt` and
`extra_info.tools_kwargs.task` with `name=agentenv_swe_v2`, `sandbox.provider=e2b`,
`sandbox.sandbox_kwargs.template`, and evaluation `metadata`. Templates and data are deployment assets;
their identifiers must be supplied by the operator, not copied from another platform.
The original migrated format (`sandbox.provider=agentenv`, matching `template_id`/`image`,
`architecture=arm64`, `workdir=/testbed`) is also accepted and normalized by the task runner.

To convert JSON metadata, use `prepare_data.py` with a JSON list of raw metadata rows and a separate
instance-to-template mapping. Example mapping (replace the example identifiers):

```json
{"example__issue-1": {"template": "your-template-id", "workdir": "/testbed"}}
```

```bash
python prepare_data.py --rows /data/train-rows.json --templates /data/train-templates.json --output /data/agentenv/train.parquet
python prepare_data.py --rows /data/validation-rows.json --templates /data/validation-templates.json --output /data/agentenv/validation.parquet
```

Metadata requires `instance_id`, `base_commit`, `problem_statement`, `test_patch`, `FAIL_TO_PASS`,
`PASS_TO_PASS`, and the fields used by `evaluation_adapter.normalize_metadata`. For SWE-Rebench,
provide `source_task=swe_rebench`, the supported pytest `log_parser`, `test_cmd`, `install`, and
`eval_commands`; SWE-Bench needs `source_task=swe_bench`, `repo` and `version`. Other supported
Python/Go/Java tasks need `language` and `install_config.test_cmd`. The converter validates each row.
For a small smoke dataset, `--repeat 8` repeats rows explicitly; do not report that as a held-out benchmark.

Place a trusted ARM64 Claude Code binary at `assets/claude` **on the AgentENV host**, and set its
SHA256 as `CLAUDE_CODE_SHA256` on the training workers. The bridge serves it on the callback base port
minus one (default `40192`); set `AGENTENV_ASSET_URL` accordingly. The task checks the digest before execution.
The sandbox must have a writable `/tmp/uniagent`; the provided task YAML uses that directory consistently.
Java tasks additionally require `AGENTENV_JAVA_M2_ARCHIVE` on shared training storage, an archive with
`repository/` holding the offline Maven dependencies. No binaries or Maven caches are bundled.

## 3. Start the Windows PC bridge

The PC must reach both SSH endpoints. Copy this directory's `callback_proxy.py` and
`connection_diagnostics/callback_http_trace.py` to the AgentENV host, preserving the relative layout;
put the assets beside them. `api_relay.py` must exist in the recipe directory on the relay host.
Both hosts need Python 3.10+; these proxies use only the standard library. Configure network access
to the callback and asset ports externally; the helper does not modify firewall rules.

Run in **PowerShell 7** and keep the terminal open:

```powershell
.\start_agentenv_bridge.ps1 `
  -AgentEnvTarget user@agentenv.example.com -AgentEnvIp agentenv.example.com `
  -RelayTarget user@relay.example.com `
  -AgentEnvWorkDir /opt/agentenv-recipe -RelayWorkDir /shared/recipe/agentenv `
  -GatewayCount 8
```

Set `-AgentEnvSshPort`, `-RelaySshPort` and `-AgentEnvApiPort` when defaults (`22`, `22`, `8000`) differ.
`AgentEnvIp` is a bindable address on the AgentENV host, reachable from its sandbox guests.
The default port mapping is:

| Flow | Ports |
| --- | --- |
| Training API → relay API | `40190` → `40191` |
| Relay API lanes → PC API lanes → AgentENV API | `31090+i` → `40190+i` → `8000` |
| Sandbox callback → AgentENV reverse lane → PC callback lane → relay callback → Gateway | `40193+i` → `31293+i` → `40393+i` → `30193+i` → discovered listener |
| Sandbox binary download | `40192` |

Here `i=0..GATEWAY_COUNT-1`. Keep bridge arguments and training environment variables aligned.
Each lane has its own SSH process to avoid concentrating concurrent API streams in one PC process.
The relay forwards bytes once; it never retries an already-forwarded request. Closing the PC bridge
releases its own listeners and SSH processes.

## 4. Launch on both training workers

```bash
export SANDBOX_ENV_FILE=/secure/agentenv.env
# Set these only if the scheduler does not provide them:
export VC_WORKER_HOSTS=head.example.com,worker.example.com
export VC_TASK_INDEX=0  # 1 on the second worker
bash run_reproduce.sh --dry-run
bash run_reproduce.sh trainer.total_training_steps=2 trainer.save_freq=-1
```

Default Ray head/dashboard ports are `6767`/`8261`. The local compiler and SSH control directories
use `/tmp`; override `AGENTENV_LOCAL_RUNTIME` with another **node-local**, short path if needed.
Logs/checkpoints and task diagnostics go under `artifacts/`. `--dry-run` does not create output files.
The real launch checks both datasets, assets configuration, patched dependency and model path before
opening the API tunnel; it checks `/health` before starting Ray.

If SSH fails, check the chosen port, host keys and forwarding permissions from the actual training
containers. If rollout stalls after sandbox creation, check the complete callback route with the
session-specific URL; reaching the API health endpoint does not validate model callbacks.
