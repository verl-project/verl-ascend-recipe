"""Tests for modules/extractors.py."""

from __future__ import annotations

import ast

import pytest

# ============================================================================
# normalize_path_text
# ============================================================================


class TestNormalizePathText:
    def test_backslash_to_slash(self):
        from modules.extractors import normalize_path_text

        assert normalize_path_text("tests\\test_foo.py") == "tests/test_foo.py"

    def test_strip_whitespace(self):
        from modules.extractors import normalize_path_text

        assert normalize_path_text("  tests/test_foo.py  ") == "tests/test_foo.py"

    def test_mixed_separators(self):
        from modules.extractors import normalize_path_text

        assert normalize_path_text("tests\\a/b\\c") == "tests/a/b/c"

    def test_no_change(self):
        from modules.extractors import normalize_path_text

        assert normalize_path_text("tests/test_foo.py") == "tests/test_foo.py"

    def test_empty_string(self):
        from modules.extractors import normalize_path_text

        assert normalize_path_text("") == ""

    def test_only_backslashes(self):
        from modules.extractors import normalize_path_text

        assert normalize_path_text("\\\\") == "//"


# ============================================================================
# extract_pytest_specs
# ============================================================================


class TestExtractPytestSpecs:
    def test_simple_pytest_directory(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(["pytest", "tests/"], 0)
        assert targets == ["tests/"]

    def test_pytest_with_specific_file(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(["pytest", "tests/test_foo.py"], 0)
        assert targets == ["tests/test_foo.py"]

    def test_pytest_with_ignore_option(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(
            ["pytest", "tests/", "--ignore", "tests/skip_this.py"], 0
        )
        assert "tests/skip_this.py" in ignore_paths
        assert "tests/" in targets

    def test_pytest_with_ignore_glob_option(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(
            ["pytest", "tests/", "--ignore-glob", "tests/*/deprecated/*"], 0
        )
        assert "tests/*/deprecated/*" in ignore_globs

    def test_pytest_with_inline_ignore(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(["pytest", "--ignore=tests/skip.py", "tests/"], 0)
        assert "tests/skip.py" in ignore_paths
        assert "tests/" in targets

    def test_pytest_with_inline_ignore_glob(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(["pytest", "--ignore-glob=tests/bad/*", "tests/"], 0)
        assert "tests/bad/*" in ignore_globs

    def test_pytest_with_k_option_skipped(self):
        from modules.extractors import extract_pytest_specs

        targets, _, _ = extract_pytest_specs(["pytest", "-k", "test_foo", "tests/"], 0)
        # -k value should not appear in targets
        assert "test_foo" not in targets
        assert "tests/" in targets

    def test_pytest_with_m_option_skipped(self):
        from modules.extractors import extract_pytest_specs

        targets, _, _ = extract_pytest_specs(["pytest", "-m", "slow", "tests/"], 0)
        assert "slow" not in targets
        assert "tests/" in targets

    def test_pytest_with_maxfail_skipped(self):
        from modules.extractors import extract_pytest_specs

        targets, _, _ = extract_pytest_specs(["pytest", "--maxfail", "5", "tests/"], 0)
        assert "5" not in targets

    def test_pytest_no_targets(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, ignore_globs = extract_pytest_specs(["pytest", "-v"], 0)
        assert targets == []

    def test_pytest_flag_starting_with_dash_not_target(self):
        from modules.extractors import extract_pytest_specs

        targets, _, _ = extract_pytest_specs(["pytest", "-x", "--tb=short", "tests/"], 0)
        assert "-x" not in targets
        assert "--tb=short" not in targets
        assert "tests/" in targets

    def test_quoted_ignore_value(self):
        from modules.extractors import extract_pytest_specs

        targets, ignore_paths, _ = extract_pytest_specs(["pytest", "tests/", "--ignore", '"tests/quoted.py"'], 0)
        assert "tests/quoted.py" in ignore_paths

    def test_deselect_nodeid_is_not_a_target(self):
        from modules.extractors import extract_pytest_specs

        targets, _, _ = extract_pytest_specs(["pytest", "tests/", "--deselect", "tests/test_flaky.py::test_flaky"], 0)
        assert targets == ["tests/"]
        assert "tests/test_flaky.py::test_flaky" not in targets

    def test_inline_deselect_is_not_a_target(self):
        from modules.extractors import extract_pytest_specs

        targets, _, _ = extract_pytest_specs(["pytest", "--deselect=tests/test_flaky.py::test_flaky", "tests/"], 0)
        assert targets == ["tests/"]


# ============================================================================
# extract_pytest_main_argv
# ============================================================================


class TestExtractPytestMainArgv:
    SCRIPT = """\
          # Keep pytest options out of sys.argv: MegatronAdaptor parses it.
          python -c '
          import pytest

          raise SystemExit(pytest.main([
              "-s",
              "-x",
              "--ignore-glob=*on_cpu.py",
              "tests/",
              "-k",
              "not test_slow",
          ]))
          '
    """

    def test_recovers_argv_list(self):
        from modules.extractors import extract_pytest_main_argv

        argv = extract_pytest_main_argv(self.SCRIPT)
        assert argv == [
            "-s",
            "-x",
            "--ignore-glob=*on_cpu.py",
            "tests/",
            "-k",
            "not test_slow",
        ]

    def test_returns_none_without_call(self):
        from modules.extractors import extract_pytest_main_argv

        assert extract_pytest_main_argv("python -m pytest tests/") is None

    def test_returns_none_for_non_literal_argument(self):
        from modules.extractors import extract_pytest_main_argv

        script = "python -c 'pytest.main(build_argv())'"
        assert extract_pytest_main_argv(script) is None

    def test_ignores_comment_lines_inside_list(self):
        from modules.extractors import extract_pytest_main_argv

        script = """python -c '
pytest.main([
    "tests/",
    # keep this comment out of the parse
    "--ignore-glob=*on_cpu.py",
])
'"""
        assert extract_pytest_main_argv(script) == ["tests/", "--ignore-glob=*on_cpu.py"]


# ============================================================================
# extract_torchrun_targets
# ============================================================================


class TestExtractTorchrunTargets:
    def test_extracts_test_target(self):
        from modules.extractors import extract_torchrun_targets

        result = extract_torchrun_targets(["torchrun", "tests/test_trainer.py"], 0)
        assert result == ["tests/test_trainer.py"]

    def test_torchrun_with_m_pytest_returns_empty(self):
        from modules.extractors import extract_torchrun_targets

        # torchrun -m pytest means pytest mode, not torchrun targets
        result = extract_torchrun_targets(["torchrun", "-m", "pytest", "tests/"], 0)
        assert result == []

    def test_torchrun_with_flags_skipped(self):
        from modules.extractors import extract_torchrun_targets

        result = extract_torchrun_targets(["torchrun", "--nproc_per_node", "4", "tests/test_trainer.py"], 0)
        assert result == ["tests/test_trainer.py"]

    def test_no_tests_targets(self):
        from modules.extractors import extract_torchrun_targets

        result = extract_torchrun_targets(["torchrun", "--version"], 0)
        assert result == []


# ============================================================================
# extract_bash_target
# ============================================================================


class TestExtractBashTarget:
    def test_extracts_tests_script(self):
        from modules.extractors import extract_bash_target

        result = extract_bash_target(["bash", "tests/test_integration.sh"])
        assert result == "tests/test_integration.sh"

    def test_extracts_examples_script(self):
        from modules.extractors import extract_bash_target

        result = extract_bash_target(["bash", "examples/test_example.sh"])
        assert result == "examples/test_example.sh"

    def test_non_test_script_returns_none(self):
        from modules.extractors import extract_bash_target

        result = extract_bash_target(["bash", "scripts/setup.sh"])
        assert result is None

    def test_no_bash_returns_none(self):
        from modules.extractors import extract_bash_target

        result = extract_bash_target(["pytest", "tests/"])
        assert result is None

    def test_bash_with_flags(self):
        from modules.extractors import extract_bash_target

        result = extract_bash_target(["bash", "-e", "tests/test.sh"])
        assert result == "tests/test.sh"


# ============================================================================
# should_keep_target
# ============================================================================


class TestShouldKeepTarget:
    def test_keep_pytest_tests_directory(self):
        from modules.extractors import should_keep_target

        assert should_keep_target("tests", "pytest") is True
        assert should_keep_target("tests/", "pytest") is True

    def test_keep_specific_file(self):
        from modules.extractors import should_keep_target

        assert should_keep_target("tests/test_foo.py", "pytest") is True
        assert should_keep_target("tests/test_foo.py", "bash") is True

    def test_drop_tests_directory_for_non_pytest(self):
        from modules.extractors import should_keep_target

        assert should_keep_target("tests/", "bash") is False
        assert should_keep_target("tests/", "torchrun") is False

    def test_drop_tests_for_non_pytest(self):
        from modules.extractors import should_keep_target

        # "tests" (without trailing /) NOT in PYTEST_DIRECTORY_TARGETS
        # and "tests" != "tests/", so second condition is True → keeps
        assert should_keep_target("tests", "bash") is True
        # "tests/" IS in PYTEST_DIRECTORY_TARGETS, so kept only for pytest
        assert should_keep_target("tests/", "bash") is False


# ============================================================================
# is_ignored_pytest_target
# ============================================================================


class TestIsIgnoredPytestTarget:
    def test_exact_path_match(self):
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests/skip.py", ["tests/skip.py"], []) is True

    def test_prefix_match(self):
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests/skip/sub/test.py", ["tests/skip"], []) is True

    def test_glob_match(self):
        from modules.extractors import is_ignored_pytest_target

        assert (
            is_ignored_pytest_target("tests/deprecated/test_old.py", [], ["tests/*/deprecated/*"]) is False
        )  # glob pattern doesn't match this path pattern
        assert is_ignored_pytest_target("tests/deprecated/test_old.py", [], ["tests/deprecated/*"]) is True

    def test_no_match(self):
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests/keep.py", [], []) is False

    def test_normalized_paths(self):
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests\\skip.py", ["tests/skip.py"], []) is True

    def test_directory_glob_match(self):
        """Glob patterns without wildcards match as directory prefixes (pytest behavior)."""
        from modules.extractors import is_ignored_pytest_target

        # Same as --ignore-glob="tests/checkpoint_engine" — should exclude all files under it.
        assert (
            is_ignored_pytest_target("tests/checkpoint_engine/test_correctness.py", [], ["tests/checkpoint_engine"])
            is True
        )
        assert is_ignored_pytest_target("tests/checkpoint_engine/sub/test.py", [], ["tests/checkpoint_engine"]) is True

    def test_directory_glob_with_trailing_slash(self):
        """Glob patterns with trailing slash also match as directory prefixes."""
        from modules.extractors import is_ignored_pytest_target

        # --ignore-glob="tests/models/" should exclude all files under tests/models/
        assert is_ignored_pytest_target("tests/models/test_fsdp.py", [], ["tests/models/"]) is True
        assert is_ignored_pytest_target("tests/models/sub/module.py", [], ["tests/models/"]) is True

    def test_directory_glob_exact_match(self):
        """A directory glob pattern also matches the path exactly (a file at that path)."""
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests/skip.py", [], ["tests/skip.py"]) is True

    def test_directory_glob_no_match(self):
        """Directory glob patterns only match files at or under the named path."""
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests/other/file.py", [], ["tests/checkpoint_engine"]) is False
        assert is_ignored_pytest_target("tests/model_utils.py", [], ["tests/models/"]) is False

    def test_wildcard_glob_still_works(self):
        """Patterns with wildcards still use fnmatch behavior."""
        from modules.extractors import is_ignored_pytest_target

        assert is_ignored_pytest_target("tests/deprecated/test_old.py", [], ["tests/deprecated/*"]) is True
        assert is_ignored_pytest_target("tests/deprecated/test_old.py", [], ["tests/*/deprecated/*"]) is False
        assert is_ignored_pytest_target("tests/special_e2e/foo.py", [], ["tests/special*"]) is True
        assert is_ignored_pytest_target("tests/test_on_npu.py", [], ["*on_npu.py"]) is True

    def test_mixed_ignore_paths_and_globs(self):
        """Both ignore_paths and ignore_globs are checked."""
        from modules.extractors import is_ignored_pytest_target

        # Matched by ignore_paths
        assert is_ignored_pytest_target("tests/skip.py", ["tests/skip.py"], ["tests/other"]) is True
        # Matched by directory-like glob
        assert is_ignored_pytest_target("tests/other/sub/test.py", ["tests/skip.py"], ["tests/other"]) is True
        # Not matched by either
        assert is_ignored_pytest_target("tests/keep.py", ["tests/skip.py"], ["tests/other"]) is False


# ============================================================================
# extract_test_functions_from_text / module
# ============================================================================


class TestExtractTestFunctionsFromText:
    def test_simple_test_function(self):
        from modules.extractors import extract_test_functions_from_text

        code = "def test_simple():\n    pass\n"
        result = extract_test_functions_from_text(code)
        assert "test_simple" in result

    def test_test_class_with_methods(self):
        from modules.extractors import extract_test_functions_from_text

        code = """
class TestMyFeature:
    def test_method_one(self):
        pass
    def test_method_two(self):
        pass
"""
        result = extract_test_functions_from_text(code)
        assert "TestMyFeature::test_method_one" in result
        assert "TestMyFeature::test_method_two" in result

    def test_async_function(self):
        from modules.extractors import extract_test_functions_from_text

        code = "async def test_async():\n    pass\n"
        result = extract_test_functions_from_text(code)
        assert "test_async" in result

    def test_non_test_functions_ignored(self):
        from modules.extractors import extract_test_functions_from_text

        code = "def helper():\n    pass\n"
        result = extract_test_functions_from_text(code)
        assert result == []

    def test_non_test_class_ignored(self):
        from modules.extractors import extract_test_functions_from_text

        code = """
class NotATest:
    def test_method(self):
        pass
"""
        result = extract_test_functions_from_text(code)
        assert result == []

    def test_syntax_error_returns_empty(self):
        from modules.extractors import extract_test_functions_from_text

        result = extract_test_functions_from_text("def broken(")
        assert result == []

    def test_mixed_scenario(self, sample_test_py_content):
        from modules.extractors import extract_test_functions_from_text

        result = extract_test_functions_from_text(sample_test_py_content)
        assert "test_simple" in result
        assert "test_with_fixture" in result
        assert "test_async_function" in result
        assert "TestMyFeature::test_method_one" in result
        assert "TestMyFeature::test_method_two" in result
        assert "TestMyFeature::test_async_method" in result
        assert "TestAnotherFeature::test_another" in result
        # Non-test functions/classes should be excluded
        assert "helper_function" not in result
        assert "NotATestClass::test_looks_like_test" not in result

    def test_results_are_sorted_and_deduped(self):
        from modules.extractors import extract_test_functions_from_text

        code = """
def test_b(): pass
def test_a(): pass
"""
        result = extract_test_functions_from_text(code)
        assert result == ["test_a", "test_b"]


class TestExtractTestFunctionsFromModule:
    def test_extracts_from_parsed_module(self):
        from modules.extractors import extract_test_functions_from_module

        module = ast.parse("def test_x():\n    pass\n")
        result = extract_test_functions_from_module(module)
        assert result == ["test_x"]

    def test_extracts_class_methods(self):
        from modules.extractors import extract_test_functions_from_module

        code = """
class TestX:
    def test_a(self): pass
    async def test_b(self): pass
"""
        module = ast.parse(code)
        result = extract_test_functions_from_module(module)
        assert result == ["TestX::test_a", "TestX::test_b"]


# ============================================================================
# extract_test_functions_from_file (I/O with tmp_path)
# ============================================================================


class TestExtractTestFunctionsFromFile:
    def test_reads_file_and_extracts(self, sample_test_py_file):
        from modules.extractors import extract_test_functions_from_file

        result = extract_test_functions_from_file(sample_test_py_file)
        assert "test_simple" in result

    def test_nonexistent_file_raises(self, tmp_path):
        from modules.extractors import extract_test_functions_from_file

        with pytest.raises(FileNotFoundError):
            extract_test_functions_from_file(tmp_path / "nonexistent.py")

    def test_syntax_error_file_returns_empty(self, tmp_path):
        from modules.extractors import extract_test_functions_from_file

        bad_file = tmp_path / "bad_syntax.py"
        bad_file.write_text("this is not valid python {{{", encoding="utf-8")
        result = extract_test_functions_from_file(bad_file)
        assert result == []

    def test_file_without_test_functions(self, tmp_path):
        from modules.extractors import extract_test_functions_from_file

        helper_file = tmp_path / "helpers.py"
        helper_file.write_text("def helper():\n    pass\n\nCONST = 42\n", encoding="utf-8")
        result = extract_test_functions_from_file(helper_file)
        assert result == []


# ============================================================================
# expand_python_test_file (I/O with tmp_path)
# ============================================================================


class TestExpandPythonTestFile:
    def test_expands_to_function_level(self, sample_test_py_file, tmp_path):
        # Copy to a known relative location
        import shutil

        from modules.extractors import expand_python_test_file

        target = tmp_path / "tests" / "test_sample.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_test_py_file, target)

        result = expand_python_test_file("tests/test_sample.py", tmp_path)
        assert any("::" in r for r in result)

    def test_nonexistent_file_falls_back(self, tmp_path):
        from modules.extractors import expand_python_test_file

        result = expand_python_test_file("tests/nonexistent.py", tmp_path)
        assert result == ["tests/nonexistent.py"]

    def test_file_without_test_functions_falls_back(self, tmp_path):
        """Cover L176: file exists but has no test functions → returns [path_text]."""
        from modules.extractors import expand_python_test_file

        test_file = tmp_path / "tests" / "test_empty.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# no test functions here\nx = 1\n", encoding="utf-8")

        result = expand_python_test_file("tests/test_empty.py", tmp_path)
        assert result == ["tests/test_empty.py"]


# ============================================================================
# expand_pytest_targets
# ============================================================================


class TestExpandPytestTargets:
    def test_directory_target(self, tmp_path):
        from modules.extractors import expand_pytest_targets

        # Create a tests directory with test files
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a.py").write_text("def test_one():\n    pass\n", encoding="utf-8")
        (tests_dir / "test_b.py").write_text(
            "class TestB:\n    def test_method(self):\n        pass\n",
            encoding="utf-8",
        )

        result = expand_pytest_targets("tests", tmp_path, [], [])
        # Should expand directory contents to function level
        assert len(result) > 0
        assert any("test_a.py" in r for r in result)

    def test_python_file_target(self, tmp_path):
        from modules.extractors import expand_pytest_targets

        test_file = tmp_path / "tests" / "test_foo.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("def test_bar():\n    pass\n", encoding="utf-8")

        result = expand_pytest_targets("tests/test_foo.py", tmp_path, [], [])
        assert "tests/test_foo.py::test_bar" in result

    def test_non_py_non_dir_falls_back(self, tmp_path):
        from modules.extractors import expand_pytest_targets

        result = expand_pytest_targets("some/unknown/target", tmp_path, [], [])
        assert result == ["some/unknown/target"]

    def test_with_ignore_paths(self, tmp_path):
        from modules.extractors import expand_pytest_targets

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_keep.py").write_text("def test_keep():\n    pass\n", encoding="utf-8")
        (tests_dir / "test_skip.py").write_text("def test_skip():\n    pass\n", encoding="utf-8")

        result = expand_pytest_targets("tests", tmp_path, ["tests/test_skip.py"], [])
        assert any("test_keep" in r for r in result)
        assert not any("test_skip" in r for r in result)

    def test_with_ignore_globs(self, tmp_path):
        from modules.extractors import expand_pytest_targets

        tests_dir = tmp_path / "tests"
        sub_dir = tests_dir / "deprecated"
        sub_dir.mkdir(parents=True)
        (sub_dir / "test_old.py").write_text("def test_old():\n    pass\n", encoding="utf-8")

        result = expand_pytest_targets("tests", tmp_path, [], ["tests/deprecated/*"])
        assert not any("deprecated" in r for r in result)

    def test_directory_rglob_skips_non_file(self, tmp_path):
        """Cover L190: rglob returns a non-file entry → continue."""
        from modules.extractors import expand_pytest_targets

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        # Create a directory whose name matches test_*.py — rglob will find it
        weird_dir = tests_dir / "test_not_a_file.py"
        weird_dir.mkdir()
        # Also create a real file so we get some results
        (tests_dir / "test_real.py").write_text("def test_real():\n    pass\n", encoding="utf-8")

        result = expand_pytest_targets("tests", tmp_path, [], [])
        # Should only expand the real file, skipping the directory
        assert any("test_real" in r for r in result)
        assert not any("test_not_a_file" in r for r in result)
