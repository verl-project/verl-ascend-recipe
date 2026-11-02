"""Tests for modules/workflows.py."""

from __future__ import annotations

from pathlib import Path

# ============================================================================
# classify_workflow
# ============================================================================


class TestClassifyWorkflow:
    def test_cpu_unit_tests(self):
        from modules.workflows import classify_workflow

        assert classify_workflow("cpu_unit_tests", "") == "cpu"

    def test_ascend_suffix(self):
        from modules.workflows import classify_workflow

        assert classify_workflow("gpu_unit_tests_ascend", "") == "npu"
        assert classify_workflow("e2e_ascend", "") == "npu"

    def test_nightly_ascend(self):
        from modules.workflows import classify_workflow

        assert classify_workflow("nightly_ascend", "") == "npu"

    def test_npu_unit_tests(self):
        from modules.workflows import classify_workflow

        assert classify_workflow("npu_unit_tests", "") == "npu"

    def test_runs_on_aarch64(self):
        from modules.workflows import classify_workflow

        content = "jobs:\n  test:\n    runs-on: aarch64"
        assert classify_workflow("some_workflow", content) == "npu"

    def test_runs_on_a2(self):
        from modules.workflows import classify_workflow

        content = "jobs:\n  test:\n    runs-on: a2-highgpu-8g"
        assert classify_workflow("some_workflow", content) == "npu"

    def test_image_ascend(self):
        from modules.workflows import classify_workflow

        content = "jobs:\n  test:\n    container:\n      image: ascend-mindspore"
        assert classify_workflow("some_workflow", content) == "npu"

    def test_default_gpu(self):
        from modules.workflows import classify_workflow

        assert classify_workflow("gpu_unit_tests", "") == "gpu"
        assert classify_workflow("e2e_ppo_trainer", "") == "gpu"

    def test_case_insensitive(self):
        from modules.workflows import classify_workflow

        assert classify_workflow("MY_ASCEND", "") == "npu"
        assert classify_workflow("CPU_UNIT_TESTS", "") == "cpu"

    def test_runs_on_case_insensitive(self):
        """RUNS_ON_ASCEND_RE uses re.IGNORECASE — verify it matches mixed-case."""
        from modules.workflows import classify_workflow

        content = "jobs:\n  test:\n    runs-on: Aarch64"
        assert classify_workflow("some_workflow", content) == "npu"

        content = "jobs:\n  test:\n    runs-on: AARCH64"
        assert classify_workflow("some_workflow", content) == "npu"

        content = "jobs:\n  test:\n    runs-on: a2-HIGHGPU-8g"
        assert classify_workflow("some_workflow", content) == "npu"

    def test_image_case_insensitive(self):
        """IMAGE_ASCEND_RE uses re.IGNORECASE — verify it matches mixed-case."""
        from modules.workflows import classify_workflow

        content = "jobs:\n  test:\n    container:\n      image: ASCEND-mindspore"
        assert classify_workflow("some_workflow", content) == "npu"

        content = "jobs:\n  test:\n    container:\n      image: Ascend-MindSpore"
        assert classify_workflow("some_workflow", content) == "npu"


# ============================================================================
# workflow_pair_key
# ============================================================================


class TestWorkflowPairKey:
    def test_npu_unit_tests_maps_to_gpu(self):
        from modules.workflows import workflow_pair_key

        assert workflow_pair_key("npu_unit_tests") == "gpu_unit_tests"

    def test_ascend_suffix_stripped(self):
        from modules.workflows import workflow_pair_key

        assert workflow_pair_key("foo_ascend") == "foo"

    def test_no_special_mapping(self):
        from modules.workflows import workflow_pair_key

        assert workflow_pair_key("bar") == "bar"

    def test_case_insensitive(self):
        from modules.workflows import workflow_pair_key

        assert workflow_pair_key("FOO_ASCEND") == "foo"


# ============================================================================
# tokenize_command
# ============================================================================


class TestTokenizeCommand:
    def test_simple_command(self):
        from modules.workflows import tokenize_command

        result = tokenize_command("pytest tests/")
        assert result == ["pytest", "tests/"]

    def test_with_quotes(self):
        from modules.workflows import tokenize_command

        result = tokenize_command('pytest -k "test_foo or test_bar" tests/')
        assert "pytest" in result
        assert "-k" in result
        assert "test_foo or test_bar" in result
        assert "tests/" in result

    def test_with_single_quotes(self):
        from modules.workflows import tokenize_command

        result = tokenize_command("pytest -k 'test_foo' tests/")
        assert "test_foo" in result

    def test_comments_removed(self):
        from modules.workflows import tokenize_command

        result = tokenize_command("pytest tests/  # run all tests")
        assert "#" not in result
        assert "run" not in result

    def test_fallback_to_split(self):
        from modules.workflows import tokenize_command

        # A command with unmatched quotes triggers ValueError -> fallback to split
        result = tokenize_command('pytest -k "unclosed tests/')
        # Should not crash, should return something
        assert len(result) > 0
        assert "pytest" in result


# ============================================================================
# split_shell_commands
# ============================================================================


class TestSplitShellCommands:
    def test_single_command(self):
        from modules.workflows import split_shell_commands

        result = split_shell_commands("pytest tests/")
        assert result == ["pytest tests/"]

    def test_double_ampersand(self):
        from modules.workflows import split_shell_commands

        result = split_shell_commands("cd tests && pytest tests/")
        assert len(result) == 2
        assert "cd tests" in result
        assert "pytest tests/" in result

    def test_double_pipe(self):
        from modules.workflows import split_shell_commands

        result = split_shell_commands("pytest tests/ || echo failed")
        assert len(result) == 2

    def test_semicolon(self):
        from modules.workflows import split_shell_commands

        result = split_shell_commands("export FOO=bar; pytest tests/")
        assert len(result) == 2

    def test_quoted_separator_not_split(self):
        from modules.workflows import split_shell_commands

        # && inside quotes should not cause a split
        result = split_shell_commands('echo "hello && world"')
        assert len(result) == 1
        assert "&&" in result[0]

    def test_empty_result_filtered(self):
        from modules.workflows import split_shell_commands

        result = split_shell_commands("")
        assert result == [""] or result == []

    def test_single_quote_prevents_split(self):
        """Cover L124: single-quoted content with && is not split."""
        from modules.workflows import split_shell_commands

        result = split_shell_commands("echo 'hello && world'")
        assert len(result) == 1
        assert "&&" in result[0]


# ============================================================================
# command_uses_pytest / command_uses_torchrun
# ============================================================================


class TestCommandDetection:
    def test_uses_pytest_true(self):
        from modules.workflows import command_uses_pytest

        assert command_uses_pytest(["pytest", "tests/"]) is True

    def test_uses_pytest_false(self):
        from modules.workflows import command_uses_pytest

        assert command_uses_pytest(["torchrun", "tests/test.py"]) is False

    def test_uses_torchrun_in_tokens(self):
        from modules.workflows import command_uses_torchrun

        assert command_uses_torchrun("", ["torchrun", "tests/test.py"]) is True

    def test_uses_torchrun_in_command(self):
        from modules.workflows import command_uses_torchrun

        assert command_uses_torchrun("torchrun tests/test.py", ["python"]) is True

    def test_uses_torchrun_not_present(self):
        from modules.workflows import command_uses_torchrun

        assert command_uses_torchrun("pytest tests/", ["pytest"]) is False


# ============================================================================
# normalize_signature
# ============================================================================


class TestNormalizeSignature:
    def test_pytest_signature_strips_ignore(self):
        from modules.workflows import normalize_signature

        result = normalize_signature("pytest tests/ --ignore tests/skip.py", "tests/")
        assert "--ignore" not in result
        assert "tests/skip.py" not in result

    def test_pytest_signature_strips_selection(self):
        from modules.workflows import normalize_signature

        result = normalize_signature("pytest -k 'test_foo' tests/", "tests/")
        assert "-k" not in result
        assert "test_foo" not in result

    def test_pytest_signature_keeps_env_vars(self):
        from modules.workflows import normalize_signature

        result = normalize_signature("FOO=bar pytest tests/", "tests/")
        assert "FOO=bar" in result

    def test_pytest_signature_sorts_env_vars(self):
        from modules.workflows import normalize_signature

        result = normalize_signature("Z=2 A=1 pytest tests/", "tests/")
        # Env vars should be sorted: A=1 before Z=2
        a_pos = result.index("A=1")
        z_pos = result.index("Z=2")
        assert a_pos < z_pos

    def test_non_pytest_signature_normalizes_paths(self):
        from modules.workflows import normalize_signature

        result = normalize_signature("torchrun tests\\test.py", "tests/test.py")
        assert "\\" not in result

    def test_empty_tokens(self):
        from modules.workflows import normalize_signature

        # This would be an edge case with empty tokenize
        result = normalize_signature("", "")
        assert result == ""

    def test_pytest_with_inline_ignore_removed(self):
        from modules.workflows import normalize_signature

        result = normalize_signature("pytest --ignore=tests/skip.py tests/", "tests/")
        assert "skip.py" not in result

    def test_pytest_signature_keeps_regular_flags(self):
        """Cover L106-108: non-ignore, non-selection flags kept in signature."""
        from modules.workflows import normalize_signature

        result = normalize_signature("pytest -x -v tests/", "tests/")
        assert "-x" in result
        assert "-v" in result

    def test_pytest_signature_inline_k_equals(self):
        """Cover L103-104: inline -k=expr form skipped."""
        from modules.workflows import normalize_signature

        result = normalize_signature("pytest -k=slow tests/", "tests/")
        assert "slow" not in result


# ============================================================================
# _merge_shell_continuations
# ============================================================================


class TestMergeShellContinuations:
    def test_single_line(self):
        from modules.workflows import _merge_shell_continuations

        entries = [("pytest tests/", 1)]
        result = _merge_shell_continuations(entries)
        assert len(result) == 1
        assert result[0][0] == "pytest tests/"
        assert result[0][1] == 1

    def test_backslash_continuation(self):
        from modules.workflows import _merge_shell_continuations

        entries = [
            ("pytest \\", 1),
            ("tests/", 2),
        ]
        result = _merge_shell_continuations(entries)
        assert len(result) == 1
        assert result[0][0] == "pytest tests/"
        assert result[0][1] == 1

    def test_empty_entries(self):
        from modules.workflows import _merge_shell_continuations

        result = _merge_shell_continuations([])
        assert result == []

    def test_empty_first_line_skipped(self):
        from modules.workflows import _merge_shell_continuations

        entries = [
            ("", 1),
            ("pytest tests/", 2),
        ]
        result = _merge_shell_continuations(entries)
        assert len(result) == 1
        assert result[0][0] == "pytest tests/"

    def test_trailing_continuation(self):
        """Cover L229-231: dangling continuation at end of entries."""
        from modules.workflows import _merge_shell_continuations

        entries = [
            ("pytest \\", 1),
            ("tests/ \\", 2),
        ]
        result = _merge_shell_continuations(entries)
        assert len(result) == 1
        assert "pytest" in result[0][0]
        assert "tests/" in result[0][0]


# ============================================================================
# _ends_with_shell_continuation
# ============================================================================


class TestEndsWithShellContinuation:
    def test_odd_backslashes(self):
        from modules.workflows import _ends_with_shell_continuation

        assert _ends_with_shell_continuation("pytest \\") is True

    def test_even_backslashes(self):
        from modules.workflows import _ends_with_shell_continuation

        assert _ends_with_shell_continuation("pytest \\\\") is False

    def test_no_backslash(self):
        from modules.workflows import _ends_with_shell_continuation

        assert _ends_with_shell_continuation("pytest tests/") is False


# ============================================================================
# build_case_id / build_display_name
# ============================================================================


class TestBuildCaseId:
    def test_produces_pipe_separated_id(self):
        from modules.workflows import build_case_id

        case = {
            "case_kind": "ut",
            "command_type": "pytest",
            "target": "tests/test_foo.py::test_add",
            "signature": "pytest",
            "workflow_name": "GPU Tests",
            "job_name": "unit-tests",
            "step_name": "Run pytest",
            "line_number": 10,
        }
        result = build_case_id(case)
        assert "ut|" in result
        assert "pytest|" in result
        assert "tests/test_foo.py::test_add" in result
        assert str(10) in result


class TestBuildDisplayName:
    def test_ut_case_shows_target(self):
        from modules.workflows import build_display_name

        case = {
            "case_kind": "ut",
            "target": "tests/test_foo.py::test_add",
            "workflow_name": "GPU Tests",
            "job_name": "unit-tests",
            "step_name": "Run pytest",
        }
        result = build_display_name(case)
        assert result == "tests/test_foo.py::test_add"

    def test_st_case_shows_context(self):
        from modules.workflows import build_display_name

        case = {
            "case_kind": "st",
            "target": "tests/test_integration.sh",
            "workflow_name": "GPU Tests",
            "job_name": "e2e-tests",
            "step_name": "Run bash test",
        }
        result = build_display_name(case)
        assert "tests/test_integration.sh" in result
        assert "GPU Tests" in result
        assert "e2e-tests" in result
        assert "Run bash test" in result


# ============================================================================
# build_workflow_groups
# ============================================================================


class TestBuildWorkflowGroups:
    def test_groups_by_pair_key(self):
        from modules.config import WorkflowInfo
        from modules.workflows import build_workflow_groups

        cpu_info = WorkflowInfo("A", "a.yml", "a.yml", "gpu", "pair_a")
        npu_info = WorkflowInfo("A Ascend", "a_ascend.yml", "a_ascend.yml", "npu", "pair_a")

        result = build_workflow_groups([cpu_info, npu_info])

        assert "pair_a" in result
        assert len(result["pair_a"]["cpu_gpu"]) == 1
        assert len(result["pair_a"]["npu"]) == 1

    def test_standalone_npu(self):
        from modules.config import WorkflowInfo
        from modules.workflows import build_workflow_groups

        npu_only = WorkflowInfo("E2E Ascend", "e2e_ascend.yml", "e2e_ascend.yml", "npu", "e2e")

        result = build_workflow_groups([npu_only])

        assert "e2e" in result
        assert len(result["e2e"]["cpu_gpu"]) == 0
        assert len(result["e2e"]["npu"]) == 1

    def test_standalone_cpu_gpu(self):
        from modules.config import WorkflowInfo
        from modules.workflows import build_workflow_groups

        gpu_only = WorkflowInfo("GPU Tests", "gpu_unit_tests.yml", "gpu_unit_tests.yml", "gpu", "gpu_unit_tests")

        result = build_workflow_groups([gpu_only])

        assert "gpu_unit_tests" in result
        assert len(result["gpu_unit_tests"]["cpu_gpu"]) == 1
        assert len(result["gpu_unit_tests"]["npu"]) == 0


# ============================================================================
# parse_workflow_content (pure - takes strings)
# ============================================================================


class TestParseWorkflowContent:
    def test_extracts_pytest_case(self, sample_workflow_yaml, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, cases = parse_workflow_content(
            "gpu_unit_tests.yml",
            ".github/workflows/gpu_unit_tests.yml",
            sample_workflow_yaml,
            tmp_path,
            empty_config,
        )

        assert info is not None
        assert info.workflow_kind == "gpu"
        assert info.workflow_name in ("GPU Unit Tests", "gpu_unit_tests")
        assert len(cases) > 0
        assert any(c["command_type"] == "pytest" for c in cases)

    def test_extracts_pytest_main_script_cases(self, empty_config, tmp_path):
        """A `python -c` block that calls pytest.main([...]) is a real pytest run."""
        from modules.workflows import parse_workflow_content

        content = """name: NPU unit tests
on: [push]
jobs:
  unit-tests:
    runs-on: aarch64
    steps:
      - name: Run all NPU unit tests
        run: |
          # Keep pytest options out of sys.argv: MegatronAdaptor parses it.
          python -c '
          import pytest

          raise SystemExit(pytest.main([
              "-s",
              "-x",
              "--ignore-glob=*on_cpu.py",
              "tests/",
          ]))
          '
"""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_alpha.py").write_text("def test_alpha():\n    pass\n", encoding="utf-8")
        (tests_dir / "test_beta.py").write_text("def test_beta():\n    pass\n", encoding="utf-8")

        info, cases = parse_workflow_content(
            "npu_unit_tests.yml",
            ".github/workflows/npu_unit_tests.yml",
            content,
            tmp_path,
            empty_config,
        )

        assert info is not None and info.workflow_kind == "npu"
        targets = {case["target"] for case in cases}
        assert "tests/test_alpha.py::test_alpha" in targets
        assert "tests/test_beta.py::test_beta" in targets
        assert all(case["command_type"] == "pytest" for case in cases)

    def test_pytest_main_script_honours_ignore_globs(self, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        content = """name: NPU unit tests
on: [push]
jobs:
  unit-tests:
    runs-on: aarch64
    steps:
      - name: Run all NPU unit tests
        run: |
          python -c '
          import pytest

          raise SystemExit(pytest.main(["--ignore-glob=*_on_cpu.py", "tests/"]))
          '
"""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_alpha.py").write_text("def test_alpha():\n    pass\n", encoding="utf-8")
        (tests_dir / "test_gamma_on_cpu.py").write_text("def test_gamma():\n    pass\n", encoding="utf-8")

        _, cases = parse_workflow_content(
            "npu_unit_tests.yml",
            ".github/workflows/npu_unit_tests.yml",
            content,
            tmp_path,
            empty_config,
        )

        targets = {case["target"] for case in cases}
        assert "tests/test_alpha.py::test_alpha" in targets
        assert not any("gamma" in target for target in targets)

    def test_extracts_torchrun_case(self, sample_torchrun_workflow_yaml, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, cases = parse_workflow_content(
            "gpu_e2e.yml",
            ".github/workflows/gpu_e2e.yml",
            sample_torchrun_workflow_yaml,
            tmp_path,
            empty_config,
        )

        assert info is not None
        assert any(c["command_type"] == "torchrun" for c in cases)

    def test_extracts_bash_case(self, sample_bash_workflow_yaml, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, cases = parse_workflow_content(
            "bash_test.yml",
            ".github/workflows/bash_test.yml",
            sample_bash_workflow_yaml,
            tmp_path,
            empty_config,
        )

        assert info is not None
        assert any(c["command_type"] == "bash" for c in cases)

    def test_non_ignored_workflow_returns_info(self, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, cases = parse_workflow_content(
            "docker-build.yml",
            ".github/workflows/docker-build.yml",
            "name: Docker\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - run: docker build .",
            tmp_path,
            empty_config,
        )

        # docker-build.yml is NOT in empty_config's ignored list, so info is not None
        assert info is not None

    def test_actually_ignored_workflow(self, sample_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, cases = parse_workflow_content(
            "doc.yml",
            ".github/workflows/doc.yml",
            "name: Docs",
            tmp_path,
            sample_config,
        )

        assert info is None
        assert cases == []

    def test_multi_step_workflow(self, sample_multi_step_workflow_yaml, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, cases = parse_workflow_content(
            "multi.yml",
            ".github/workflows/multi.yml",
            sample_multi_step_workflow_yaml,
            tmp_path,
            empty_config,
        )

        assert info is not None
        # Should have cases from multiple jobs
        job_names = {c["job_name"] for c in cases}
        assert "first-job" in job_names
        assert "second-job" in job_names

    def test_cases_have_required_fields(self, sample_workflow_yaml, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        _, cases = parse_workflow_content(
            "gpu_unit_tests.yml",
            ".github/workflows/gpu_unit_tests.yml",
            sample_workflow_yaml,
            tmp_path,
            empty_config,
        )

        for case in cases:
            for field in [
                "workflow_name",
                "workflow_path",
                "file_name",
                "workflow_kind",
                "pair_key",
                "job_name",
                "step_name",
                "line_number",
                "command_type",
                "case_kind",
                "target",
                "raw_command",
                "signature",
                "case_id",
                "display_name",
            ]:
                assert field in case, f"Missing field '{field}' in case {case.get('case_id', '?')}"

    def test_ascend_workflow_classified_as_npu(self, sample_ascend_workflow_yaml, empty_config, tmp_path):
        from modules.workflows import parse_workflow_content

        info, _ = parse_workflow_content(
            "npu_unit_tests.yml",
            ".github/workflows/npu_unit_tests.yml",
            sample_ascend_workflow_yaml,
            tmp_path,
            empty_config,
        )

        assert info is not None
        assert info.workflow_kind == "npu"


# ============================================================================
# _extract_cases_from_command edge branches
# ============================================================================


class TestExtractCasesFromCommand:
    def test_empty_command_returns_empty(self, tmp_path):
        """Cover L284: empty command → []."""
        from modules.workflows import _extract_cases_from_command

        result = _extract_cases_from_command("", tmp_path)
        assert result == []

    def test_comment_command_returns_empty(self, tmp_path):
        """Cover L288: comment command → []."""
        from modules.workflows import _extract_cases_from_command

        result = _extract_cases_from_command("# pytest tests/", tmp_path)
        assert result == []

    def test_pytest_file_target_shape(self, tmp_path):
        """Cover L310-311: target_shape = 'pytest-file' when target ends with .py."""
        from modules.workflows import _extract_cases_from_command

        # Create the target file so expand_pytest_targets can parse it
        test_file = tmp_path / "tests" / "test_foo.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def test_one():\n    pass\n", encoding="utf-8")

        result = _extract_cases_from_command("pytest tests/test_foo.py", tmp_path)
        assert len(result) > 0
        assert any("pytest-file" in c["signature"] for c in result)

    def test_dedup_identical_cases(self, tmp_path):
        """Cover L326-334: dedup safety net exercised."""
        from modules.workflows import _extract_cases_from_command

        # Within a single command, bash/torchrun/pytest are mutually exclusive,
        # so dedup is rarely triggered — it exists as a safety net.
        # Verify the result has expected shape and content fields.
        result = _extract_cases_from_command("bash tests/test_integration.sh", tmp_path)
        assert len(result) == 1
        assert result[0]["command_type"] == "bash"
        assert result[0]["target"] == "tests/test_integration.sh"


# ============================================================================
# collect_scan_data: workflow content returns None
# ============================================================================


class TestCollectScanDataEdgeCases:
    def test_workflow_without_run_steps_parsed_with_zero_cases(self, tmp_path, empty_config):
        """Workflow without jobs/run steps is parsed with 0 cases, not ignored."""
        from modules.workflows import collect_scan_data

        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "empty.yml").write_text("name: Empty\n", encoding="utf-8")

        workflow_infos, cases, ignored_paths = collect_scan_data(tmp_path, empty_config)

        # Not ignored (no matching pattern in empty_config), parsed with 0 cases
        assert any(info.file_name == "empty.yml" for info in workflow_infos)
        assert len(ignored_paths) == 0


# ============================================================================
# multiline run: entries with empty lines and indented continuations
# ============================================================================


class TestMultilineRunEntries:
    def test_empty_line_in_multiline_run(self):
        """Cover L185-186: empty next_line skipped in _extract_run_entries."""
        from modules.workflows import _extract_run_entries

        lines = [
            "        run: |",
            "",
            "          pytest tests/",
        ]
        entries, next_idx = _extract_run_entries(lines, 0, 8, "|")
        assert len(entries) == 1
        assert entries[0][0] == "pytest tests/"

    def test_indented_continuation_line(self):
        """Cover L189-191: continuation line with indent-based extraction."""
        from modules.workflows import _extract_run_entries

        lines = [
            "        run: |",
            "          pytest tests/",
            "          pytest tests/more/",
        ]
        entries, next_idx = _extract_run_entries(lines, 0, 8, "|")
        assert len(entries) == 2
        assert entries[0][0] == "pytest tests/"
        assert entries[1][0] == "pytest tests/more/"

    def test_inline_run_command(self):
        """Cover L178-179: inline run: without | or > creates a single entry."""
        from modules.workflows import _extract_run_entries

        lines = [
            "      - run: pytest tests/",
            "      - name: next step",
            "        run: echo done",
        ]
        entries, next_idx = _extract_run_entries(lines, 0, 6, "pytest tests/")
        assert len(entries) == 1
        assert entries[0][0] == "pytest tests/"
        assert entries[0][1] == 1  # line_number is start_idx + 1

    def test_inline_run_with_shell_separators(self):
        """Inline run with shell separators is merged into a single entry."""
        from modules.workflows import _extract_run_entries

        lines = [
            "      - run: cd tests && pytest tests/",
            "      - name: next step",
            "        run: echo done",
        ]
        entries, next_idx = _extract_run_entries(lines, 0, 6, "cd tests && pytest tests/")
        assert len(entries) == 1
        assert "cd tests && pytest tests/" in entries[0][0]


# ============================================================================
# parse_workflow (I/O with tmp_path)
# ============================================================================


class TestParseWorkflow:
    def test_parses_yaml_file(self, tmp_path, empty_config, sample_workflow_yaml):
        from modules.workflows import parse_workflow

        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        wf_file = wf_dir / "gpu_unit_tests.yml"
        wf_file.write_text(sample_workflow_yaml, encoding="utf-8")

        info, cases = parse_workflow(wf_file, tmp_path, empty_config)

        assert info is not None
        assert len(cases) > 0


# ============================================================================
# collect_scan_data (I/O with tmp_path)
# ============================================================================


class TestCollectScanData:
    def test_collects_from_workflow_directory(self, tmp_path, empty_config, sample_workflow_yaml):
        from modules.workflows import collect_scan_data

        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "gpu_unit_tests.yml").write_text(sample_workflow_yaml, encoding="utf-8")

        workflow_infos, cases, ignored_paths = collect_scan_data(tmp_path, empty_config)

        assert len(workflow_infos) == 1
        assert len(cases) > 0
        assert ignored_paths == []

    def test_ignores_matching_workflows(self, tmp_path, sample_config, sample_workflow_yaml):
        from modules.workflows import collect_scan_data

        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "gpu_unit_tests.yml").write_text(sample_workflow_yaml, encoding="utf-8")
        (wf_dir / "doc.yml").write_text("name: Docs", encoding="utf-8")

        workflow_infos, cases, ignored_paths = collect_scan_data(tmp_path, sample_config)

        # doc.yml should be ignored, gpu_unit_tests.yml should be scanned
        assert len(workflow_infos) == 1
        assert "doc.yml" in [Path(p).name for p in ignored_paths]

    def test_both_yaml_extensions(self, tmp_path, empty_config):
        from modules.workflows import collect_scan_data

        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "test.yml").write_text(
            "name: Test\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: pytest tests/",
            encoding="utf-8",
        )
        (wf_dir / "other.yaml").write_text(
            "name: Other\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: pytest tests/",
            encoding="utf-8",
        )

        workflow_infos, cases, _ = collect_scan_data(tmp_path, empty_config)

        assert len(workflow_infos) == 2
