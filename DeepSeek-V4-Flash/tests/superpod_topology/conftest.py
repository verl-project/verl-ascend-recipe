"""CPU validation against a supplied patched checkout; no torch/NPU imports."""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--source", required=True, help="Patched VERL repository root")


@pytest.fixture(scope="session")
def source(request):
    return Path(request.config.getoption("--source")).resolve()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def modules(source):
    # Namespace avoids verl.__init__ importing torch. Implementation files are real.
    package = types.ModuleType("topology_validation")
    package.__path__ = [str(source / "verl/single_controller/ray")]
    sys.modules[package.__name__] = package
    plan = load_file(
        "topology_validation.ascend_topology_plan", source / "verl/single_controller/ray/ascend_topology_plan.py"
    )
    topology = load_file(
        "topology_validation.ascend_topology", source / "verl/single_controller/ray/ascend_topology.py"
    )
    dcmi = load_file("topology_validation.dcmi_probe", source / "verl/utils/ascend_dcmi_probe.py")
    return types.SimpleNamespace(plan=plan, topology=topology, dcmi=dcmi)


def definitions(source, path, names, namespace, *, owner=None):
    """Execute unmodified function/class bodies, replacing heavy training dependencies.

    This validates real control flow with fake model servers; it is not an NPU
    training test. Future annotations remove annotation-only import dependencies.
    """
    tree = ast.parse((source / path).read_text())
    body = tree.body if owner is None else next(n for n in tree.body if getattr(n, "name", None) == owner).body
    selected = [n for n in body if getattr(n, "name", None) in names]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(source / path), "exec"), namespace)
    return namespace
