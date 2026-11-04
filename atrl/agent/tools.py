"""Five sandbox tools used by the ATRL loop."""

from __future__ import annotations

import contextvars
import json
import os
import shlex
import subprocess
from typing import Any

from verl.tools.function_tool import function_tool

workspace_dir: contextvars.ContextVar[str] = contextvars.ContextVar("atrl_workspace", default="/workspace")


def _run(script: str, timeout: int = 30) -> str:
    workdir = workspace_dir.get()
    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=workdir if os.path.isdir(workdir) else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    text = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    if result.returncode != 0:
        return text.strip() or f"command failed with exit {result.returncode}"
    return text.strip()


@function_tool
def str_replace_editor(command: str, path: str, file_text: str = "", old_str: str = "", new_str: str = "") -> str:
    """Create or edit a file in the workspace.

    Args:
        command: One of create, view, str_replace.
        path: File path. Absolute, or relative to the workspace.
        file_text: Full file contents when command is create.
        old_str: Text to replace when command is str_replace.
        new_str: Replacement text when command is str_replace.
    """

    target = path if os.path.isabs(path) else os.path.join(workspace_dir.get(), path)
    if command == "view":
        if not os.path.isfile(target):
            return f"missing file: {target}"
        with open(target, encoding="utf-8") as handle:
            return handle.read()[:8000]
    if command == "create":
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(file_text)
        return f"created {target}"
    if command == "str_replace":
        if not os.path.isfile(target):
            return f"missing file: {target}"
        with open(target, encoding="utf-8") as handle:
            content = handle.read()
        if old_str not in content:
            return "old_str was not found"
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(content.replace(old_str, new_str, 1))
        return f"updated {target}"
    return "command must be create, view, or str_replace"


@function_tool
def list_skills() -> str:
    """List SKILL.md files under the workspace and the image template.

    Args:
    """

    return _run(
        "for root in .skills /opt/triton-agent-template/.skills /opt/triton-agent-template/.claude/skills; do "
        "if [ -d \"$root\" ]; then find \"$root\" -name SKILL.md -print; fi; done"
    ) or "(no skills found)"


@function_tool
def read_skill(path: str) -> str:
    """Read one skill file.

    Args:
        path: Path returned by list_skills, inside the workspace or the template.
    """

    if not path.startswith(("/workspace/", "/opt/triton-agent-template/", "./")):
        return "path must stay inside the workspace or the skill template"
    return _run(f"sed -n '1,400p' {shlex.quote(path)}", timeout=30)


@function_tool
def run_verify() -> str:
    """Run AST check, correctness, and the performance benchmark.

    Args:
    """

    command = os.environ.get("ATRL_VERIFY_COMMAND", "bash /opt/triton-agent-tools/verify_once.sh")
    return _run(command, timeout=int(os.environ.get("TRITON_EVAL_TIMEOUT", "900")))


@function_tool
def submit() -> str:
    """Finish the task after verification.

    Args:
    """

    return "Submitted."


def atrl_tools() -> list[Any]:
    from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY

    names = ["str_replace_editor", "list_skills", "read_skill", "run_verify", "submit"]
    return [FUNCTION_TOOL_REGISTRY[name] for name in names]


def workspace_reward(workdir: str | None = None) -> float:
    """Score verify_result.json plus perf_result.json. Missing files score 0."""

    from atrl.reward import reward_breakdown

    root = workdir or workspace_dir.get()
    verify_path = os.path.join(root, "output", "verify", "verify_result.json")
    perf_path = os.path.join(root, "output", "perf_result.json")
    if not os.path.isfile(verify_path):
        return 0.0
    try:
        with open(verify_path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        if os.path.isfile(perf_path):
            with open(perf_path, encoding="utf-8") as handle:
                metrics["perf_data"] = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return 0.0
    if not isinstance(metrics, dict):
        return 0.0
    return float(reward_breakdown(metrics)["total"])
