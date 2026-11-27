"""Validate local prerequisites without creating sandboxes or contacting Ray."""

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check():
    for name in ("python3", "git", "ssh", "curl", "ray"):
        if not shutil.which(name):
            raise ValueError(f"Missing command: {name}")
    for name in ("torch_npu", "ray", "e2b", "pyarrow", "swebench", "psutil", "yaml"):
        if importlib.util.find_spec(name) is None:
            raise ValueError(f"Missing Python dependency: {name}")
    source = Path(os.environ["UNI_AGENT_ROOT"])
    subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(ROOT / "patches/uni-agent.patch")],
        check=True,
        capture_output=True,
    )
    verl = Path(os.environ["VERL_SOURCE"])
    if not (verl / "verl/trainer/main_ppo.py").is_file():
        raise ValueError("VERL_SOURCE does not contain verl/trainer/main_ppo.py")
    if not (Path(os.environ["MODEL_PATH"]) / "config.json").is_file():
        raise ValueError("MODEL_PATH must contain config.json")
    agentenv = ROOT.name == "agentenv"
    nodes = int(os.environ.get("NNODES", "2"))
    devices = int(os.environ.get("N_GPUS_PER_NODE" if agentenv else "NGPUS_PER_NODE", "16"))
    if (nodes, devices) != (2, 16):
        raise ValueError("These launchers support the validated 2 x 16 NPU topology")
    parallel = 1
    for key, default in (("TRAIN_TP", "4"), ("TRAIN_PP", "2"), ("TRAIN_CP", "4")):
        value = int(os.environ.get(key, default))
        if value < 1:
            raise ValueError(f"{key} must be positive")
        parallel *= value
    if 32 % parallel:
        raise ValueError("TP * PP * CP must divide 32")
    gateways = int(os.environ.get("GATEWAY_COUNT", "8"))
    if gateways < 1:
        raise ValueError("GATEWAY_COUNT must be positive")
    port_names = (
        (("AGENTENV_CALLBACK_PORT", 40193), ("AGENTENV_RELAY_CALLBACK_PORT", 30193))
        if agentenv
        else (("CALLBACK_RELAY_PORT", 29193),)
    )
    for name, default in port_names:
        port = int(os.environ.get(name, str(default)))
        if not 0 < port <= 65536 - gateways:
            raise ValueError(f"{name} + GATEWAY_COUNT exceeds valid port range")
    if agentenv and not re.fullmatch(r"[a-fA-F0-9]{64}", os.environ.get("CLAUDE_CODE_SHA256", "")):
        raise ValueError("CLAUDE_CODE_SHA256 must be a SHA256 hex digest")
    import pyarrow.parquet as pq

    keys = ("TRAIN_DATA", "VAL_DATA") if agentenv else ("TRAIN_FILE", "TEST_FILE")
    for index, key in enumerate(keys):
        path = Path(os.environ[key])
        rows = pq.read_table(path).to_pylist()
        minimum = int(os.environ.get("TRAIN_BATCH_SIZE" if agentenv else "TRAIN_PROMPT_BSZ", "32")) if index == 0 else 1
        if len(rows) < minimum:
            raise ValueError(f"{key} needs at least {minimum} rows")
        for row in rows:
            from migration_adapter import normalize_tools_kwargs

            tools_kwargs = normalize_tools_kwargs(row.get("extra_info", {}).get("tools_kwargs", {}))
            task = (tools_kwargs or {}).get("task", {})
            expected = "agentenv_swe_v2" if agentenv else "swe_bench"
            sandbox = task.get("sandbox", {})
            if (
                task.get("name") != expected
                or sandbox.get("provider") != "e2b"
                or not sandbox.get("sandbox_kwargs", {}).get("template")
            ):
                raise ValueError(f"{key}: each row must bind {expected} to an E2B template")
            if agentenv:
                from evaluation_adapter import normalize_metadata

                metadata = normalize_metadata(task.get("metadata", {}))
                if (
                    metadata["language"] == "java"
                    and not Path(os.environ.get("AGENTENV_JAVA_M2_ARCHIVE", "")).is_file()
                ):
                    raise ValueError("Java tasks require AGENTENV_JAVA_M2_ARCHIVE")
    print("Local preflight passed; live sandbox connectivity is checked during launch")


if __name__ == "__main__":
    try:
        check()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Preflight failed: {exc}") from None
