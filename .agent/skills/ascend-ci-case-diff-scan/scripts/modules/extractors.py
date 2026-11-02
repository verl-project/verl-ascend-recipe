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

"""Command target extraction and pytest expansion helpers."""

from __future__ import annotations

import ast
from fnmatch import fnmatch
from pathlib import Path

from .config import load_text

PATH_VALUE_OPTIONS = {"--ignore", "--ignore-glob"}
PYTEST_DIRECTORY_TARGETS = {"tests", "tests/"}
PYTEST_OPTIONS_WITH_VALUE = {
    "-k",
    "-m",
    "--maxfail",
    "--rootdir",
    "--log-level",
    "--capture",
    "--asyncio-mode",
    "--durations",
    # ``--deselect nodeid`` removes a case, so the nodeid it names is not a
    # case the workflow runs and must not be recorded as a target.
    "--deselect",
}


def normalize_path_text(value: str) -> str:
    """Normalize filesystem-like text so matching stays stable across platforms."""
    return value.replace("\\", "/").strip()


def extract_pytest_specs(tokens: list[str], pytest_idx: int) -> tuple[list[str], list[str], list[str]]:
    """Extract pytest targets and ignore options without mistaking option values for paths."""
    targets: list[str] = []
    ignore_paths: list[str] = []
    ignore_globs: list[str] = []
    idx = pytest_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in PATH_VALUE_OPTIONS and idx + 1 < len(tokens):
            value = normalize_path_text(tokens[idx + 1].strip("\"'"))
            if token == "--ignore":
                ignore_paths.append(value)
            else:
                ignore_globs.append(value)
            idx += 2
            continue
        if token in PYTEST_OPTIONS_WITH_VALUE:
            idx += 2
            continue
        if token.startswith("--ignore="):
            ignore_paths.append(normalize_path_text(token.split("=", 1)[1].strip("\"'")))
            idx += 1
            continue
        if token.startswith("--ignore-glob="):
            ignore_globs.append(normalize_path_text(token.split("=", 1)[1].strip("\"'")))
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        normalized = normalize_path_text(token.strip("\"'"))
        if normalized in PYTEST_DIRECTORY_TARGETS or normalized.startswith("tests/"):
            targets.append(normalized)
        idx += 1
    return targets, ignore_paths, ignore_globs


def extract_pytest_main_argv(text: str) -> list[str] | None:
    """Return the argv list passed to an inline ``pytest.main([...])`` call.

    Some workflows cannot pass pytest options through ``sys.argv`` (a launcher
    may parse it during import) and therefore invoke pytest from a ``python -c``
    script. Those cases are invisible to a command-line token scan, so recover
    the argv list from the source text instead. Returns ``None`` when the text
    has no such call or its argument is not a literal list of strings.
    """
    marker = "pytest.main("
    start_marker = text.find(marker)
    if start_marker < 0:
        return None
    start = text.find("[", start_marker)
    if start < 0:
        return None
    literal = _literal_list_at(text, start)
    if literal is None:
        return None
    argv = _parse_string_list(literal)
    return argv or None


def _literal_list_at(text: str, start: int) -> str | None:
    """Return the balanced ``[...]`` slice starting at ``start``, if it closes."""
    depth = 0
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def _parse_string_list(literal: str) -> list[str] | None:
    """Parse a list literal, tolerating whole-line comments inside the brackets."""
    for candidate in (literal, _strip_comment_lines(literal)):
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
    return None


def _strip_comment_lines(literal: str) -> str:
    """Drop whole-line ``#`` comments, which ``literal_eval`` cannot parse."""
    return "\n".join(line for line in literal.splitlines() if not line.strip().startswith("#"))


def extract_torchrun_targets(tokens: list[str], torchrun_idx: int) -> list[str]:
    """Extract test targets from torchrun commands."""
    for idx in range(torchrun_idx + 1, len(tokens) - 1):
        if tokens[idx] == "-m" and tokens[idx + 1] == "pytest":
            return []

    targets: list[str] = []
    idx = torchrun_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("-"):
            if "=" not in token:
                # Only skip the next token when it looks like a value,
                # not another flag and not a test target.
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1]
                    if not nxt.startswith("-") and not nxt.startswith("tests/"):
                        idx += 1
            idx += 1
            continue
        normalized = normalize_path_text(token.strip("\"'"))
        if normalized.startswith("tests/"):
            targets.append(normalized)
        idx += 1
    return targets


def extract_bash_target(tokens: list[str]) -> str | None:
    """Extract bash-invoked test scripts from explicit workflow step commands."""
    for idx, token in enumerate(tokens[:-1]):
        if token == "bash":
            for j in range(idx + 1, len(tokens)):
                target = normalize_path_text(tokens[j].strip("\"'"))
                if target.startswith(("tests/", "examples/")) and target.endswith(".sh"):
                    return target
    return None


def should_keep_target(target: str, command_type: str) -> bool:
    """Drop placeholder paths that do not identify a real test case."""
    normalized_target = normalize_path_text(target)
    return (command_type == "pytest" and normalized_target in PYTEST_DIRECTORY_TARGETS) or (
        normalized_target != "tests/"
    )


def is_ignored_pytest_target(path_text: str, ignore_paths: list[str], ignore_globs: list[str]) -> bool:
    """Check whether a discovered test file should be excluded by pytest ignore options."""
    normalized = normalize_path_text(path_text)
    for ignore_path in ignore_paths:
        candidate = normalize_path_text(ignore_path)
        if normalized == candidate or normalized.startswith(candidate.rstrip("/") + "/"):
            return True
    for pattern in ignore_globs:
        if _is_glob_pattern(pattern):
            if fnmatch(normalized, pattern):
                return True
        else:
            # pytest treats directory-like patterns (no wildcards) as prefix
            # matches: ``--ignore-glob=tests/skip`` excludes all files under
            # ``tests/skip/``.
            candidate = normalize_path_text(pattern)
            if normalized == candidate or normalized.startswith(candidate.rstrip("/") + "/"):
                return True
    return False


def _is_glob_pattern(pattern: str) -> bool:
    """Return True when *pattern* contains ``fnmatch``-style wildcard characters."""
    return any(ch in pattern for ch in ("*", "?", "["))


def extract_test_functions_from_file(path: Path) -> list[str]:
    """Extract unittest-style and pytest-style test functions from one Python file."""
    try:
        module = ast.parse(load_text(path))
    except SyntaxError:
        return []
    return extract_test_functions_from_module(module)


def extract_test_functions_from_text(content: str) -> list[str]:
    """Extract test names from Python source content."""
    try:
        module = ast.parse(content)
    except SyntaxError:
        return []
    return extract_test_functions_from_module(module)


def extract_test_functions_from_module(module: ast.Module) -> list[str]:
    """Extract test names from a parsed Python module."""
    names: list[str] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test_"):
                    names.append(f"{node.name}::{child.name}")
    return sorted(dict.fromkeys(names))


def expand_python_test_file(path_text: str, repo_root: Path) -> list[str]:
    """Expand one Python test file into function-level UT identifiers."""
    path = repo_root / path_text
    if not path.is_file():
        return [path_text]
    functions = extract_test_functions_from_file(path)
    if not functions:
        return [path_text]
    return [f"{path_text}::{function_name}" for function_name in functions]


def expand_pytest_directory(
    target_path: Path,
    repo_root: Path,
    ignore_paths: list[str],
    ignore_globs: list[str],
) -> list[str]:
    """Expand one pytest directory target using pattern-based Python file discovery."""
    expanded: list[str] = []
    for path in sorted(target_path.rglob("test_*.py")):
        if not path.is_file():
            continue
        path_text = normalize_path_text(path.relative_to(repo_root).as_posix())
        if is_ignored_pytest_target(path_text, ignore_paths, ignore_globs):
            continue
        expanded.extend(expand_python_test_file(path_text, repo_root))
    return expanded


def expand_pytest_targets(
    target: str,
    repo_root: Path,
    ignore_paths: list[str],
    ignore_globs: list[str],
) -> list[str]:
    """Expand pytest targets into function-level UT cases."""
    normalized = normalize_path_text(target)
    target_path = repo_root / normalized
    if target_path.is_dir():
        return expand_pytest_directory(
            target_path,
            repo_root,
            ignore_paths,
            ignore_globs,
        )
    if normalized.endswith(".py"):
        return expand_python_test_file(normalized, repo_root)
    return [normalized]
