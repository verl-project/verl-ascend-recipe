# Copyright 2026
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

"""Check installed backend versions and source fingerprints."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

from .vllm_plugin.compatibility import installed_adapter_name

ASCEND_COMMIT = "5ed283a6c0ff5508f060d6203bc7a34e9d38c7bb"
ASCEND_RUNNER_SOURCES = {
    "e4ad64b16d3707189ee54bb73be06b87a04423c1d7189e2dd8122530c605bb50": "public-5ed283a6c",
    "cd0abab15168fa7cb6b094b8f1861975e7c9ab562d29fc0fde9906435f1e96cf": "legacy-telemetry-build",
}
ASCEND_GDN_SHA256 = "d6ec29919268178f5bf6e70e689c1d273d04b1cb1d84dc94efa7bbbc35490816"
VLLM_RUNNER_SHA256 = "6250ca0670c4593aa2f7c4567c9c71d6cb3d8182624a985d4d4e33d5bf054666"
VLLM_SCHEDULER_SHA256 = "9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9"


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations or len(spec.submodule_search_locations) != 1:
        raise ValueError(f"Cannot locate one installed {name} package")
    return Path(next(iter(spec.submodule_search_locations)))


def source_check(path: Path, allowed: set[str] | frozenset[str]) -> dict:
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        return {"path": str(path), "supported": False, "error": str(error)}
    return {"path": str(path), "sha256": actual, "supported": actual in allowed}


_AUDIT_METHODS = {
    "execute_model",
    "propose_draft_token_ids",
    "_bookkeeping_sync",
    "_update_states",
    "_copy_draft_token_ids_to_cpu",
    "_get_draft_token_ids_cpu",
    "sample_tokens",
    "schedule",
}


def audit_source(path: Path) -> dict:
    """Report backend method signatures and source fingerprints."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as error:
        return {"path": str(path), "error": str(error), "methods": []}
    methods = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _AUDIT_METHODS:
                segment = ast.get_source_segment(source, node)
                methods.append(
                    {
                        "class": cls.name,
                        "method": node.name,
                        "line": node.lineno,
                        "arguments": ast.unparse(node.args),
                        "source_sha256": hashlib.sha256(segment.encode("utf-8")).hexdigest(),
                    }
                )
    return {"path": str(path), "methods": methods}


def check_backend(*, targeted_kv: bool = False, audit: bool = False) -> dict:
    adapter, vllm_version, ascend_version = installed_adapter_name()
    checks = []
    try:
        vllm = _package_root("vllm")
        ascend = _package_root("vllm_ascend")
        checks.append(source_check(ascend / "worker/model_runner_v1.py", frozenset(ASCEND_RUNNER_SOURCES)))
        checks.append(source_check(vllm / "v1/worker/gpu_model_runner.py", {VLLM_RUNNER_SHA256}))
        checks.append(source_check(ascend / "ops/gdn.py", {ASCEND_GDN_SHA256}))
        if targeted_kv:
            checks.append(source_check(vllm / "v1/core/sched/scheduler.py", {VLLM_SCHEDULER_SHA256}))
    except (ImportError, ValueError) as error:
        checks.append({"supported": False, "error": str(error)})
    report = {
        "supported": adapter == "v0_23" and all(row["supported"] for row in checks),
        "vllm_version": vllm_version,
        "vllm_ascend_version": ascend_version,
        "ascend_source_commit": ASCEND_COMMIT,
        "profile": ASCEND_RUNNER_SOURCES.get(checks[0].get("sha256", "")),
        "checks": checks,
        "scope": "source/version preflight only; no model, NPU or ABI validation",
    }
    if audit:
        report["upgrade_audit"] = [audit_source(Path(row["path"])) for row in checks if "path" in row]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-kv", action="store_true", help="also verify the optional KV scheduler source")
    parser.add_argument(
        "--audit-source", action="store_true", help="include private-method signatures/hashes for upgrade review"
    )
    args = parser.parse_args()
    report = check_backend(targeted_kv=args.targeted_kv, audit=args.audit_source)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["supported"] else 1)


if __name__ == "__main__":
    main()
