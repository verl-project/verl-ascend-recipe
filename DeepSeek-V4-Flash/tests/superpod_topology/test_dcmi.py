import ast
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

C_SOURCE = r"""
#include <stdio.h>
#include <unistd.h>
struct Spod { unsigned int sdid, size, sp, server, reserve[8]; };
int dcmi_init(void) {
    puts("native DCMI output must not corrupt the result"); fflush(stdout);
#ifdef HANG
    FILE *f = fopen(PID_PATH, "w");
    fprintf(f, "%d\n", getpid()); fflush(f);
    int child = fork();
    if (child == 0) { while (1) sleep(1); }
    fprintf(f, "%d\n", child); fclose(f);
    while (1) sleep(1);
#endif
    return INIT_RET;
}
int dcmi_get_card_list(int *count, int *cards, int cap) { *count=1; cards[0]=0; return 0; }
int dcmi_get_device_num_in_card(int card, int *count) { *count=2; return 0; }
int dcmi_get_device_info(int card, int device, int cmd, unsigned sub, void *data, unsigned *size) {
    if(cmd!=12 || sub!=1 || *size!=48) return 91;
    struct Spod *s = data; s->sp=SP_ID + (device ? MISMATCH : 0); return 0;
}
"""


def build_library(directory, *, sp=6, ret=0, mismatch=0, hang=False):
    source = directory / "fake_dcmi.c"
    library = directory / "libfake_dcmi.so"
    pids = directory / "query.pids"
    source.write_text(C_SOURCE)
    args = [
        "gcc",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(library),
        f"-DSP_ID={sp}",
        f"-DINIT_RET={ret}",
        f"-DMISMATCH={mismatch}",
    ]
    if hang:
        args += ["-DHANG", f'-DPID_PATH="{pids}"']
    subprocess.run(args, check=True, capture_output=True)
    return library, pids


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def wait_for_pids(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text().splitlines()) == 2:
            return [int(p) for p in path.read_text().splitlines()]
        time.sleep(0.05)
    raise AssertionError("DCMI query and child did not start")


def assert_stopped(pids):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.05)
    assert not any(alive(pid) for pid in pids), f"Surviving DCMI query descendants: {pids}"


def test_isolated_dcmi_success(modules, tmp_path):
    library, _ = build_library(tmp_path)
    assert modules.dcmi.query_superpod_id_isolated(str(library), 3) == 6


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"ret": 7}, "init failed: ret=7"),
        ({"mismatch": 1}, "one physical SuperPod"),
        ({"sp": 0xFFFFFFFF}, "Invalid DCMI spod_info"),
    ],
)
def test_dcmi_failures_remain_visible(modules, tmp_path, kwargs, message):
    library, _ = build_library(tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match=message):
        modules.dcmi.query_superpod_id_isolated(str(library), 3)


def test_native_hang_deadline_kills_descendants(modules, tmp_path):
    library, pid_path = build_library(tmp_path, hang=True)
    before = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out.*last stage=dcmi_init"):
        modules.dcmi.query_superpod_id_isolated(str(library), 0.4)
    assert time.monotonic() - before < 4
    assert_stopped(wait_for_pids(pid_path))


def test_caller_sigkill_cleans_native_descendants(source, tmp_path):
    library, pid_path = build_library(tmp_path, hang=True)
    helper = source / "verl/utils/ascend_dcmi_probe.py"
    code = f"import runpy; runpy.run_path({str(helper)!r})['query_superpod_id_isolated']({str(library)!r}, 30)"
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        pids = wait_for_pids(pid_path)
        process.kill()
        process.wait(timeout=3)
        assert_stopped(pids)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def test_warning_filter_is_targeted_and_precedes_verl_import(source):
    tree = ast.parse((source / "verl/single_controller/ray/ascend_topology.py").read_text())
    factory = next(n for n in tree.body if getattr(n, "name", None) == "_make_probe_node")
    probe = next(n for n in factory.body if isinstance(n, ast.FunctionDef))
    prefix = []
    for statement in probe.body:
        if isinstance(statement, ast.ImportFrom) and statement.module.startswith("verl"):
            break
        prefix.append(statement)
    else:
        pytest.fail("Expected VERL import after targeted filter")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        exec(compile(ast.Module(body=prefix, type_ignores=[]), "probe_filter", "exec"), {})
        compile("doc = '\\s'", "/usr/local/lib/torch_npu/op_plugin_docs.py", "exec")
        compile("doc = '\\s'", "/usr/local/lib/another_package/docs.py", "exec")
        warnings.warn("other user warning", UserWarning, stacklevel=1)
        warnings.warn_explicit("other syntax warning", SyntaxWarning, "/usr/local/lib/torch_npu/op_plugin_docs.py", 1)
    assert len(emitted) == 3
    assert any("another_package" in w.filename for w in emitted)
    assert any(str(w.message) == "other user warning" for w in emitted)
    assert any(str(w.message) == "other syntax warning" for w in emitted)
