"""Prepare benchmark metadata and commands for AgentEnvTask's strict scoring."""

import json
import re
import shlex
from copy import deepcopy


def normalize_metadata(metadata):
    row = deepcopy(metadata)
    for key in ("instance_id", "base_commit", "workdir"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError("Missing task metadata: " + key)
    if not row["workdir"].startswith("/") or ".." in row["workdir"].split("/"):
        raise ValueError("Expected absolute repository workdir")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = row.get(key)
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError("Expected test-name list: " + key)
        row[key] = value
    source = row.get("source_task")
    if source == "swe_rebench":
        # Only the Python pytest parser families are supported by these exports.
        if row.get("log_parser") not in {"parse_log_pytest", "parse_log_pytest_v2"}:
            raise ValueError("Unsupported rebench log parser: " + str(row.get("log_parser")))
        row["language"] = "python"
        row["install_config"] = {k: row.get(k) for k in ("test_cmd", "install", "eval_commands")}
    elif source == "swe_bench":
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
        from swebench.harness.constants.python import MAP_REPO_VERSION_TO_SPECS_PY
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

        repo, version = row["repo"], row["version"]
        if repo not in MAP_REPO_VERSION_TO_SPECS_PY or repo not in MAP_REPO_TO_PARSER:
            raise ValueError("Unsupported Python SWE-bench repository: " + repo)
        row["language"] = "python"
        row["install_config"] = deepcopy(MAP_REPO_VERSION_TO_SPECS[repo][version])
    elif source is not None:
        raise ValueError("Unsupported source_task: " + str(source))
    install = row.get("install_config")
    if isinstance(install, str):
        install = json.loads(install)
    if not isinstance(install, dict) or not install.get("test_cmd"):
        raise ValueError("Missing install_config.test_cmd")
    row["install_config"] = install
    if row.get("language") not in {"python", "go", "java"}:
        raise ValueError("Unsupported task language: " + str(row.get("language")))
    return row


def evaluation_commands(row):
    """Return setup and test commands separately so setup cannot mask test failure."""
    install = row["install_config"]
    command = install["test_cmd"]
    if isinstance(command, list):
        command = " ".join(command)
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Expected nonempty test command")
    setup = []
    if row.get("source_task"):
        from swebench.harness.test_spec.python import get_test_directives

        setup = [
            "if [ -f /opt/miniconda3/bin/activate ]; then "
            "source /opt/miniconda3/bin/activate && conda activate testbed; "
            "elif [ -x /opt/miniconda3/envs/testbed/bin/python ]; then "
            "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH; "
            "else echo 'Missing testbed Python environment' >&2; exit 1; fi"
        ]
        # Use the prebuilt environment; prevent implicit package downloads by test runners.
        setup.append("export PIP_NO_INDEX=1 PIP_NO_DEPS=1 PIP_NO_BUILD_ISOLATION=0")
        extra = install.get("eval_commands") or []
        setup.extend([extra] if isinstance(extra, str) else extra)
        # install/pre_install describe template preparation, not automatic evaluation steps.
        if row["source_task"] == "swe_rebench":
            directives = list(dict.fromkeys(row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]))
            if not directives or any(not v.strip() or v == "---NO TEST NAME FOUND YET---" for v in directives):
                raise ValueError("SWE-rebench requires explicit benchmark test names")
        else:
            directives = get_test_directives(row)
        command += " " + " ".join(shlex.quote(v) for v in directives)
        if row["source_task"] == "swe_bench" and row["repo"] == "sphinx-doc/sphinx":
            # Passed to pytest after tox's '--'; survives resetting tracked tox.ini.
            command += " -rA"
    return setup, command.strip()


def benchmark_test_statuses(output, row):
    output = re.sub(r"\x1b\[[0-9;]*m", "", output).replace("\r", "")
    if row["source_task"] == "swe_bench":
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

        statuses = MAP_REPO_TO_PARSER[row["repo"]](output, None)
    else:
        from uni_agent.tasks.swe_rebench.reward import _LOG_PARSERS

        statuses = _LOG_PARSERS[row["log_parser"]](output)
    return statuses


def benchmark_passed_tests(output, row):
    return {name for name, status in benchmark_test_statuses(output, row).items() if status == "PASSED"}
