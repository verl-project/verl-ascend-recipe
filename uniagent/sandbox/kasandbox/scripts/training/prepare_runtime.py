"""Write a private Ray runtime JSON using an already installed training environment."""

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
parser.add_argument("--verl-source", required=True, help="Absolute path to compatible installed/shared verl source")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
verl = Path(args.verl_source).resolve()
if not (verl / "verl" / "trainer" / "main_ppo.py").is_file():
    raise SystemExit("verl source is missing; initialize the pinned submodule or specify a validated checkout")
sys.path.insert(0, str(root / "scripts" / "sandbox"))
from environment import worker_environment  # noqa: E402

env = worker_environment()
if os.environ.get("PYTHONWARNINGS"):
    env["PYTHONWARNINGS"] = os.environ["PYTHONWARNINGS"]
env["PYTHONPATH"] = os.pathsep.join(
    filter(None, (os.environ["UNI_AGENT_ROOT"], str(root), str(verl), os.environ.get("PYTHONPATH")))
)
# Training workers must use the same non-secret runtime settings as the launcher.
for key in (
    "VLLM_USE_V1",
    "VLLM_USE_V2_MODEL_RUNNER",
    "TASK_QUEUE_ENABLE",
    "CPU_AFFINITY_CONF",
    "OMP_PROC_BIND",
    "VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE",
    "VLLM_LOGGING_LEVEL",
    "VLLM_LOG_STATS_INTERVAL",
    "HCCL_HOST_SOCKET_PORT_RANGE",
    "HCCL_NPU_SOCKET_PORT_RANGE",
    "TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT",
    "WANDB_MODE",
):
    if os.environ.get(key):
        env[key] = os.environ[key]
runtime = {
    "excludes": ["/.git/", "/.runtime/", "/.artifacts/", "/logs/", "/logs_reproduce/", "/.env", "/.env.*"],
    "env_vars": env,
}
out = Path(args.output).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.fchmod(fd, 0o600)
with os.fdopen(fd, "w") as stream:
    json.dump(runtime, stream, indent=2)
    stream.write("\n")
print(f"Wrote private runtime configuration: {out}")
