"""Dataset-driven AgentENV task, using the upstream E2B provider."""

import os
import re
import shlex
from pathlib import Path

from connection_diagnostics.sandbox_diagnostics import sandbox_with_diagnostics
from evaluation_adapter import benchmark_test_statuses, evaluation_commands, normalize_metadata
from pydantic import Field
from uni_agent.agents.claude_code.agent import ClaudeCodeAgent, ClaudeCodeConfig
from uni_agent.agents.registry import register_agent
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task
from uni_agent.tasks.swe_rebench.task import _GIT_CLEAN_HISTORY

ROOT = Path(__file__).resolve().parent
SCRATCH = "/tmp/uniagent"


class PinnedClaudeConfig(ClaudeCodeConfig):
    name: str = "agentenv_claude"


@register_agent("agentenv_claude")
class PinnedClaude(ClaudeCodeAgent):
    config_model = PinnedClaudeConfig

    async def _ensure_claude(self, sandbox):
        result = await sandbox.exec(["test", "-x", SCRATCH + "/claude"], timeout=30)
        if result.exit_code:
            raise RuntimeError("Pinned Claude Code was not staged")

    def _claude_argv(self, *args, **kwargs):
        argv = super()._claude_argv(*args, **kwargs)
        argv[0] = SCRATCH + "/claude"
        return argv


async def prepare_harness(sandbox):
    """Fetch the pinned ARM64 executable over the AgentENV host's local network."""
    digest = os.environ["CLAUDE_CODE_SHA256"]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("CLAUDE_CODE_SHA256 must be a SHA256 hex digest")
    url = os.environ["AGENTENV_ASSET_URL"].rstrip("/") + "/claude"
    result = await sandbox.exec_shell(
        'if command -v curl >/dev/null; then curl --noproxy "*" -fsS --max-time 180 '
        f"-o {SCRATCH}/claude {shlex.quote(url)}; "
        f"else wget --no-proxy -q -T 180 -O {SCRATCH}/claude {shlex.quote(url)}; fi",
        timeout=200,
    )
    if result.exit_code:
        raise RuntimeError("Cannot download pinned Claude Code: " + result.stderr)
    result = await sandbox.exec_shell(
        f'echo "{digest}  {SCRATCH}/claude" | sha256sum -c - && chmod 755 {SCRATCH}/claude', timeout=60
    )
    if result.exit_code:
        raise RuntimeError("Claude Code checksum mismatch: " + result.stdout + result.stderr)


class Config(TaskConfig):
    name: str = "agentenv_swe_v2"
    run_oracle_solution: bool = False
    eval_timeout: float = Field(default=1200, gt=0)
    eval_build_commands: list[str] = Field(
        default_factory=list,
        description="Explicit optional build commands for patched compiled code; empty for prebuilt templates.",
    )


def passed_tests(output, language):
    output = re.sub(r"\x1b\[[0-9;]*m", "", output).replace("\r", "")
    if language == "python":
        return set(re.findall(r"^PASSED (\S+)", output, re.M))
    if language == "go":
        return set(re.findall(r"^\s*--- PASS: (\S+)", output, re.M))
    if language == "java":
        passed, suite = set(), None
        for line in output.splitlines():
            running = re.match(r"^(?:\[INFO\] )?Running (\S+)", line)
            if running:
                suite = running[1]
            summary = re.search(r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)(.*)", line)
            if summary:
                named = re.search(r" - in (\S+)", summary[5])
                name = named[1] if named else suite
                if name and int(summary[1]) > 0 and all(int(summary[i]) == 0 for i in (2, 3, 4)):
                    passed.add(name)
                suite = None
        return passed
    raise ValueError(f"Unsupported language: {language}")


@register_task("agentenv_swe_v2")
class AgentEnvTask(Task):
    config_model = Config

    async def run(self):
        cfg = self.config
        row = normalize_metadata(cfg.metadata)
        if cfg.sandbox.provider != "e2b" or not cfg.sandbox.sandbox_kwargs.get("template"):
            raise ValueError("Dataset must supply an E2B template ID")
        setup, command = evaluation_commands(row)
        workdir = row["workdir"]
        async with sandbox_with_diagnostics(self.build_sandbox(), ROOT / "artifacts") as sandbox:

            async def checked(argv, **kwargs):
                result = await sandbox.exec(argv, workdir=workdir, **kwargs)
                if result.exit_code:
                    raise RuntimeError(f"{argv[0]} failed: {result.stderr[-2000:]} {result.stdout[-2000:]}")
                return result

            await checked(["bash", "-c", _GIT_CLEAN_HISTORY], env={"BASE_COMMIT": row["base_commit"]}, timeout=600)
            await checked(["mkdir", "-p", SCRATCH, SCRATCH + "/.m2"], timeout=30)
            if row["language"] == "java":
                await sandbox.upload_file(Path(os.environ["AGENTENV_JAVA_M2_ARCHIVE"]), SCRATCH + "/m2.tar.gz")
                await checked(["tar", "-xzf", SCRATCH + "/m2.tar.gz", "-C", SCRATCH + "/.m2"], timeout=120)
            if cfg.run_oracle_solution:
                await sandbox.write_file(SCRATCH + "/solution.patch", row["patch"])
                await checked(["git", "apply", SCRATCH + "/solution.patch"], timeout=30)
                finished = True
            else:
                await prepare_harness(sandbox)
                result = await self.build_agent().run(sandbox=sandbox, messages=cfg.prompt, workdir=workdir)
                finished = result.finished
            await sandbox.write_file(SCRATCH + "/evaluation.patch", row["test_patch"])
            stat = await checked(["git", "apply", "--numstat", SCRATCH + "/evaluation.patch"], timeout=30)
            for line in stat.stdout.splitlines():
                path = line.split("\t", 2)[2]
                if path.startswith("/") or ".." in path.split("/"):
                    raise ValueError("Unsafe evaluation path")
                exists = await sandbox.exec(["git", "cat-file", "-e", row["base_commit"] + ":" + path], workdir=workdir)
                await checked(
                    ["git", "checkout", row["base_commit"], "--", path]
                    if not exists.exit_code
                    else ["rm", "-f", "--", path],
                    timeout=30,
                )
            await checked(["git", "apply", SCRATCH + "/evaluation.patch"], timeout=30)
            if row["language"] == "java":
                command += (
                    f" --offline -Dmaven.repo.local={SCRATCH}/.m2/repository"
                    " -Dmaven.compiler.source=8 -Dmaven.compiler.target=8 -Dstyle.color=never"
                )
            failure_reason = None
            statuses = {}
            if row.get("source_task"):
                # Activate/setup before testing, but preserve the actual test exit code.
                script = "exec 2>&1\nset -eo pipefail\n" + "\n".join(setup + cfg.eval_build_commands) + "\n"
                script += "printf '%s\\n' 'AGENTENV_TEST_OUTPUT_BEGIN'\n"
                script += "set +e\n" + command + "\nstatus=$?\n"
                script += "printf '%s\\n' 'AGENTENV_TEST_OUTPUT_END'\nexit \"$status\"\n"
                await sandbox.write_file(SCRATCH + "/eval.sh", script)
                result = await sandbox.exec(
                    ["bash", SCRATCH + "/eval.sh"],
                    workdir=workdir,
                    timeout=cfg.eval_timeout,
                    env={"HOME": SCRATCH, "TMPDIR": SCRATCH},
                )
                # Merge stderr into stdout inside the script for unittest-style logs.
                output = result.stdout
                begin, end = "AGENTENV_TEST_OUTPUT_BEGIN", "AGENTENV_TEST_OUTPUT_END"
                if begin in output and end in output:
                    output = output.split(begin, 1)[1].split(end, 1)[0]
                    try:
                        statuses = benchmark_test_statuses(output, row)
                    except Exception as exc:
                        failure_reason = "test_log_parse_failed"
                        statuses = {}
                        parse_error = repr(exc)
                    passed = {name for name, status in statuses.items() if status == "PASSED"}
                else:
                    passed = set()
                    failure_reason = "environment_setup_failed" if begin not in output else "test_execution_incomplete"
            else:
                result = await sandbox.exec_shell(
                    command + " 2>&1",
                    workdir=workdir,
                    timeout=cfg.eval_timeout,
                    env={"HOME": SCRATCH, "TMPDIR": SCRATCH, "GOCACHE": SCRATCH + "/go-cache"},
                )
                passed = passed_tests(result.stdout, row["language"])
            required = set(row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]) - {"---NO TEST NAME FOUND YET---"}
            missing = sorted(required - passed)
            resolved = bool(required) and result.exit_code == 0 and not missing and failure_reason is None
            absent = sorted(required - statuses.keys()) if row.get("source_task") else []
            if not resolved and failure_reason is None:
                if not required:
                    failure_reason = "invalid_test_expectations"
                elif absent:
                    failure_reason = "test_results_missing"
                elif missing:
                    failure_reason = "target_tests_failed"
                else:
                    failure_reason = "test_process_failed"
            return TaskResult(
                reward=float(resolved),
                accuracy=float(resolved),
                finished=finished,
                extra_info={
                    "resolved": resolved,
                    "missing_or_failed": missing,
                    "failure_reason": failure_reason,
                    "missing_test_results": absent,
                    "target_test_statuses": {name: statuses.get(name, "MISSING") for name in sorted(required)},
                    "parse_error": parse_error if failure_reason == "test_log_parse_failed" else None,
                    "eval_exit_code": result.exit_code,
                    "eval_output_tail": result.stdout[-8000:],
                },
            )
