#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Workflow parsing and scan collection."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from .config import ST_KIND, UT_KIND, WorkflowConfig, WorkflowInfo, is_ignored_workflow, load_text
from .extractors import (
    expand_pytest_targets,
    extract_bash_target,
    extract_pytest_main_argv,
    extract_pytest_specs,
    extract_torchrun_targets,
    normalize_path_text,
    should_keep_target,
)

WORKFLOW_NAME_RE = re.compile(r"^\s*name:\s*(.+?)\s*$", re.MULTILINE)
JOB_RE = re.compile(r"^(\s{2})([A-Za-z0-9_-]+):\s*$")
STEP_NAME_RE = re.compile(r"^(\s*)-\s+name:\s*(.+?)\s*$")
RUN_RE = re.compile(r"^(\s*)run:\s*(.*)$")
RUNS_ON_ASCEND_RE = re.compile(r"runs-on:\s+.*(?:aarch64|a2|a3)", re.IGNORECASE)
IMAGE_ASCEND_RE = re.compile(r"image:\s+.*ascend", re.IGNORECASE)
ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|\S+)$")
TORCHRUN_RE = re.compile(r"\btorchrun\b")
PYTEST_SELECTION_OPTIONS = {"-k", "-m"}
PYTEST_IGNORED_OPTIONS = {"--ignore", "--ignore-glob"}


def classify_workflow(file_stem: str, content: str) -> str:
    """Classify workflows into cpu, gpu, or npu buckets."""
    stem = file_stem.lower()
    if stem == "cpu_unit_tests":
        return "cpu"
    if stem == "npu_unit_tests" or stem.endswith("_ascend") or "nightly_ascend" in stem:
        return "npu"
    if RUNS_ON_ASCEND_RE.search(content) or IMAGE_ASCEND_RE.search(content):
        return "npu"
    return "gpu"


def workflow_pair_key(file_stem: str) -> str:
    """Map workflow file names into CPU/GPU vs NPU pairing groups."""
    stem = file_stem.lower()
    if stem == "npu_unit_tests":
        return "gpu_unit_tests"
    if stem.endswith("_ascend"):
        return stem[: -len("_ascend")]
    return stem


def normalize_signature(command: str, target: str) -> str:
    """Normalize the whole test command so parity matching keeps env and args aligned."""
    tokens = tokenize_command(command)
    if not tokens:
        return ""

    if command_uses_pytest(tokens):
        return _normalize_pytest_signature(tokens)

    return " ".join(normalize_path_text(token) for token in tokens).strip()


def _normalize_pytest_signature(tokens: list[str]) -> str:
    pytest_idx = tokens.index("pytest")
    prefix_tokens = [token for token in tokens[:pytest_idx] if token not in {"export", "env"}]
    env_prefix = sorted(normalize_path_text(token) for token in prefix_tokens if ENV_ASSIGNMENT_RE.match(token))

    normalized_tokens: list[str] = []
    normalized_tokens.extend(env_prefix)
    normalized_tokens.append("pytest")

    idx = pytest_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in PYTEST_IGNORED_OPTIONS and idx + 1 < len(tokens):
            idx += 2
            continue
        if token.startswith("--ignore=") or token.startswith("--ignore-glob="):
            idx += 1
            continue
        if token in PYTEST_SELECTION_OPTIONS and idx + 1 < len(tokens):
            idx += 2
            continue
        if token.startswith(("-k=", "-m=")):
            idx += 1
            continue
        if token.startswith("-"):
            normalized_tokens.append(normalize_path_text(token))
            idx += 1
            continue
        idx += 1

    return " ".join(normalized_tokens).strip()


def split_shell_commands(command: str) -> list[str]:
    """Split a shell line on common shell separators while respecting simple quote scopes."""
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    idx = 0
    while idx < len(command):
        ch = command[idx]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if not in_single and not in_double:
            separator = ";" if ch == ";" else next((sep for sep in ("&&", "||") if command.startswith(sep, idx)), None)
            if separator:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                idx += len(separator)
                continue

        current.append(ch)
        idx += 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def tokenize_command(command: str) -> list[str]:
    """Tokenize shell text conservatively and fall back when quoting is malformed."""
    try:
        return shlex.split(command, posix=True, comments=True)
    except ValueError:
        return command.split()


def command_uses_torchrun(command: str, tokens: list[str]) -> bool:
    return "torchrun" in tokens or bool(TORCHRUN_RE.search(command))


def command_uses_pytest(tokens: list[str]) -> bool:
    return "pytest" in tokens


def build_case_id(case: dict) -> str:
    """Build a per-occurrence case key that preserves workflow/job/step distinctions."""
    keys = ("case_kind", "command_type", "target", "signature", "workflow_name", "job_name", "step_name")
    return "|".join([*(case[key] for key in keys), str(case["line_number"])])


def build_display_name(case: dict) -> str:
    """Produce the report-facing case name for UT and ST entries."""
    if case["case_kind"] == UT_KIND:
        return case["target"]
    return f"{case['target']} [{case['workflow_name']} / {case['job_name']} / {case['step_name']}]"


def _extract_run_entries(
    lines: list[str], start_idx: int, indent: int, inline: str
) -> tuple[list[tuple[str, int]], int]:
    run_entries: list[tuple[str, int]] = []
    if inline and inline not in {"|", ">"}:
        run_entries.append((inline, start_idx + 1))
    idx = start_idx + 1
    while idx < len(lines):
        next_line = lines[idx]
        next_indent = len(next_line) - len(next_line.lstrip(" "))
        if next_line.strip() == "":
            idx += 1
            continue
        if next_indent <= indent:
            break
        stripped = next_line[indent + 2 :] if len(next_line) >= indent + 2 else next_line.lstrip()
        run_entries.append((stripped, idx + 1))
        idx += 1
    return _merge_shell_continuations(run_entries), idx


def _ends_with_shell_continuation(command: str) -> bool:
    stripped = command.rstrip()
    if not stripped.endswith("\\"):
        return False
    slash_count = 0
    for char in reversed(stripped):
        if char != "\\":
            break
        slash_count += 1
    return slash_count % 2 == 1


def _merge_shell_continuations(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
    merged: list[tuple[str, int]] = []
    pending_parts: list[str] = []
    pending_line = 0

    for command, line_number in entries:
        stripped = command.strip()
        if not stripped and not pending_parts:
            continue
        if not pending_parts:
            pending_line = line_number
        if _ends_with_shell_continuation(stripped):
            pending_parts.append(stripped.rstrip()[:-1].rstrip())
            continue
        pending_parts.append(stripped)
        merged_command = " ".join(part for part in pending_parts if part)
        if merged_command:
            merged.append((merged_command, pending_line))
        pending_parts = []
        pending_line = 0

    if pending_parts:
        merged_command = " ".join(part for part in pending_parts if part)
        if merged_command:
            merged.append((merged_command, pending_line))
    return merged


def _make_command_case(command_type: str, case_kind: str, target: str, command: str, targets: list[str]) -> dict:
    return {
        "command_type": command_type,
        "case_kind": case_kind,
        "target": target,
        "targets": targets,
        "raw_command": command,
        "signature": normalize_signature(command, target),
    }


def _expand_case_record(
    case: dict,
    workflow_info: WorkflowInfo,
    file_name: str,
    job_name: str,
    step_name: str,
    line_number: int,
    workflow_kind: str,
) -> list[dict]:
    expanded_cases: list[dict] = []
    for target in case["targets"]:
        expanded_case = {
            "workflow_name": workflow_info.workflow_name,
            "workflow_path": workflow_info.workflow_path,
            "file_name": file_name,
            "workflow_kind": workflow_kind,
            "pair_key": workflow_info.pair_key,
            "job_name": job_name,
            "step_name": step_name,
            "line_number": line_number,
            "command_type": case["command_type"],
            "case_kind": case["case_kind"],
            "target": target,
            "raw_command": case["raw_command"],
            "signature": case["signature"],
        }
        expanded_case["case_id"] = build_case_id(expanded_case)
        expanded_case["display_name"] = build_display_name(expanded_case)
        expanded_cases.append(expanded_case)
    return expanded_cases


def _pytest_command_cases(
    tokens: list[str],
    pytest_idx: int,
    command: str,
    repo_root: Path,
) -> list[dict]:
    """Build pytest cases for one command, expanding targets to case level."""
    cases: list[dict] = []
    targets, ignore_paths, ignore_globs = extract_pytest_specs(tokens, pytest_idx)
    for target in targets:
        if not should_keep_target(target, "pytest"):
            continue
        normalized_target = normalize_path_text(target)
        if normalized_target in {"tests", "tests/"} or Path(normalized_target).suffix == "":
            target_shape = "pytest-dir"
        elif normalized_target.endswith(".py"):
            target_shape = "pytest-file"
        else:
            target_shape = "pytest-target"

        signature = normalize_signature(command, target)
        case = _make_command_case(
            "pytest",
            UT_KIND,
            target,
            command,
            expand_pytest_targets(target, repo_root, ignore_paths, ignore_globs),
        )
        case["signature"] = f"{signature} {target_shape}" if signature else target_shape
        cases.append(case)
    return cases


def _extract_cases_from_command(
    command: str,
    repo_root: Path,
) -> list[dict]:
    command = command.strip()

    # Workflows that invoke pytest from an inline ``python -c`` script keep the
    # options out of sys.argv; recover that argv list from the source text and
    # treat it as the equivalent command line. Checked before tokenizing because
    # such a script starts with comments and Python statements, not a shell
    # command, and would otherwise be treated as one.
    main_argv = extract_pytest_main_argv(command)
    if main_argv is not None:
        cases = _pytest_command_cases(
            ["pytest", *main_argv],
            0,
            "pytest " + " ".join(shlex.quote(arg) for arg in main_argv),
            repo_root,
        )
        return _dedupe_cases(cases)

    if not command or command.startswith("#"):
        return []

    tokens = tokenize_command(command)
    if not tokens:
        return []

    cases: list[dict] = []
    bash_target = extract_bash_target(tokens)
    if bash_target and should_keep_target(bash_target, "bash"):
        cases.append(_make_command_case("bash", ST_KIND, bash_target, command, [bash_target]))

    if command_uses_torchrun(command, tokens):
        torchrun_idx = tokens.index("torchrun") if "torchrun" in tokens else 0
        for target in extract_torchrun_targets(tokens, torchrun_idx):
            if not should_keep_target(target, "torchrun"):
                continue
            cases.append(_make_command_case("torchrun", ST_KIND, target, command, [target]))
    elif command_uses_pytest(tokens):
        cases.extend(_pytest_command_cases(tokens, tokens.index("pytest"), command, repo_root))

    return _dedupe_cases(cases)


def _dedupe_cases(cases: list[dict]) -> list[dict]:
    """Drop repeated cases that describe the same command and target set."""
    deduped: list[dict] = []
    seen = set()
    for case in cases:
        key = (case["command_type"], tuple(case["targets"]), case["signature"], case["raw_command"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def parse_workflow_content(
    file_name: str,
    workflow_path: str,
    content: str,
    repo_root: Path,
    config: WorkflowConfig,
) -> tuple[WorkflowInfo | None, list[dict]]:
    """Parse one workflow content string and extract test cases from each run block."""
    if is_ignored_workflow(file_name, config):
        return None, []

    file_stem = Path(file_name).stem
    workflow_name = file_stem
    name_match = WORKFLOW_NAME_RE.search(content)
    if name_match:
        workflow_name = name_match.group(1).strip().strip("\"'")
    workflow_kind = classify_workflow(file_stem, content)
    workflow_info = WorkflowInfo(
        workflow_name=workflow_name,
        workflow_path=normalize_path_text(workflow_path),
        file_name=file_name,
        workflow_kind=workflow_kind,
        pair_key=workflow_pair_key(file_stem),
    )

    lines = content.splitlines()
    job_name = "unknown"
    step_name = "(unnamed step)"
    cases: list[dict] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        job_match = JOB_RE.match(line)
        if job_match:
            job_name = job_match.group(2)
        step_match = STEP_NAME_RE.match(line)
        if step_match:
            step_name = step_match.group(2).strip()
        run_match = RUN_RE.match(line)
        if run_match:
            indent = len(run_match.group(1))
            run_entries, next_idx = _extract_run_entries(lines, idx, indent, run_match.group(2).strip())
            block_cases: list[dict] = []
            for raw, line_number in run_entries:
                for command in split_shell_commands(raw):
                    for case in _extract_cases_from_command(command, repo_root):
                        block_cases.extend(
                            _expand_case_record(
                                case,
                                workflow_info,
                                file_name,
                                job_name,
                                step_name,
                                line_number,
                                workflow_kind,
                            )
                        )
            if not block_cases:
                block_cases.extend(
                    _cases_from_stack_script(
                        run_entries, workflow_info, file_name, job_name, step_name, workflow_kind, repo_root
                    )
                )
            cases.extend(block_cases)
            idx = next_idx
            continue
        idx += 1
    return workflow_info, cases


def _cases_from_stack_script(
    run_entries: list[tuple[str, int]],
    workflow_info: WorkflowInfo,
    file_name: str,
    job_name: str,
    step_name: str,
    workflow_kind: str,
    repo_root: Path,
) -> list[dict]:
    """Recover cases from a ``run:`` block whose pytest call spans several lines.

    An inline ``python -c '...'`` script is emitted one line at a time, so no
    single entry holds a complete argv list. Join the block and retry only when
    the per-line pass found nothing, keeping normal parsing untouched.
    """
    joined = "\n".join(raw for raw, _ in run_entries)
    if "pytest.main(" not in joined:
        return []
    line_number = next(
        (line_number for raw, line_number in run_entries if "pytest.main(" in raw),
        run_entries[0][1] if run_entries else 0,
    )
    collected: list[dict] = []
    for case in _extract_cases_from_command(joined, repo_root):
        collected.extend(
            _expand_case_record(case, workflow_info, file_name, job_name, step_name, line_number, workflow_kind)
        )
    return collected


def parse_workflow(path: Path, repo_root: Path, config: WorkflowConfig) -> tuple[WorkflowInfo | None, list[dict]]:
    """Parse one workflow file and extract test cases from each run block."""
    file_name = path.name
    rel_path = normalize_path_text(path.relative_to(repo_root).as_posix())
    return parse_workflow_content(file_name, rel_path, load_text(path), repo_root, config)


def collect_scan_data(repo_root: Path, config: WorkflowConfig) -> tuple[list[WorkflowInfo], list[dict], list[str]]:
    """Scan workflows and return parsed workflow metadata, cases, and ignored paths."""
    workflow_dir = repo_root / ".github" / "workflows"
    workflow_infos: list[WorkflowInfo] = []
    cases: list[dict] = []
    ignored_paths: list[str] = []
    for path in sorted(set(workflow_dir.glob("*.yml")) | set(workflow_dir.glob("*.yaml"))):
        rel_path = normalize_path_text(path.relative_to(repo_root).as_posix())
        if is_ignored_workflow(path.name, config):
            ignored_paths.append(rel_path)
            continue
        workflow_info, workflow_cases = parse_workflow(path, repo_root, config)
        if workflow_info is None:
            ignored_paths.append(rel_path)
            continue
        workflow_infos.append(workflow_info)
        cases.extend(workflow_cases)
    return workflow_infos, cases, ignored_paths


def build_workflow_groups(workflow_infos: list[WorkflowInfo]) -> dict[str, dict[str, list[WorkflowInfo]]]:
    """Group workflows by their CPU/GPU vs NPU pairing key."""
    grouped: dict[str, dict[str, list[WorkflowInfo]]] = {}
    for info in workflow_infos:
        side = "npu" if info.workflow_kind == "npu" else "cpu_gpu"
        grouped.setdefault(info.pair_key, {"cpu_gpu": [], "npu": []})[side].append(info)
    return grouped
