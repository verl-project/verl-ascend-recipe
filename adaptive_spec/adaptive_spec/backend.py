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

"""Source-only runtime preflight; importing the inference backend is unnecessary."""

from __future__ import annotations

import argparse
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


def check_backend(*, targeted_kv: bool = False) -> dict:
    adapter, vllm_version, ascend_version = installed_adapter_name()
    checks = []
    try:
        vllm = _package_root("vllm")
        ascend = _package_root("vllm_ascend")
        checks.append(source_check(ascend / "worker/model_runner_v1.py", frozenset(ASCEND_RUNNER_SOURCES)))
        checks.append(source_check(vllm / "v1/worker/gpu_model_runner.py", {VLLM_RUNNER_SHA256}))
        if targeted_kv:
            checks.append(source_check(vllm / "v1/core/sched/scheduler.py", {VLLM_SCHEDULER_SHA256}))
    except (ImportError, ValueError) as error:
        checks.append({"supported": False, "error": str(error)})
    return {
        "supported": adapter == "v0_23" and all(row["supported"] for row in checks),
        "vllm_version": vllm_version,
        "vllm_ascend_version": ascend_version,
        "ascend_source_commit": ASCEND_COMMIT,
        "profile": ASCEND_RUNNER_SOURCES.get(checks[0].get("sha256", "")),
        "checks": checks,
        "scope": "source/version preflight only; no model, NPU or ABI validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targeted-kv", action="store_true", help="also verify the optional KV scheduler source")
    args = parser.parse_args()
    report = check_backend(targeted_kv=args.targeted_kv)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["supported"] else 1)


if __name__ == "__main__":
    main()
