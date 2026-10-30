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

import hashlib
from types import SimpleNamespace

from adaptive_spec import backend


def test_missing_packages_fail_without_importing_them(monkeypatch):
    imported = []
    monkeypatch.setattr(backend, "installed_adapter_name", lambda: (None, "missing", "missing"))

    def missing(name):
        imported.append(name)
        return None

    monkeypatch.setattr(backend.importlib.util, "find_spec", missing)
    result = backend.check_backend()
    assert not result["supported"]
    assert imported == ["vllm"]
    assert result["checks"][0]["error"]


def test_known_sources_pass_and_one_byte_mutation_is_rejected(tmp_path, monkeypatch):
    paths = [
        "ascend/worker/model_runner_v1.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/core/sched/scheduler.py",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    checksum = hashlib.sha256(b"fixture\n").hexdigest()
    monkeypatch.setattr(backend, "ASCEND_RUNNER_SOURCES", {checksum: "test-fixture"})
    monkeypatch.setattr(backend, "VLLM_RUNNER_SHA256", checksum)
    monkeypatch.setattr(backend, "VLLM_SCHEDULER_SHA256", checksum)
    monkeypatch.setattr(backend, "installed_adapter_name", lambda: ("v0_23", "0.23.0", "0.23.0rc2"))
    monkeypatch.setattr(
        backend.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            submodule_search_locations=[str(tmp_path / ("ascend" if name == "vllm_ascend" else name))]
        ),
    )
    report = backend.check_backend(targeted_kv=True)
    assert report["supported"] and report["profile"] == "test-fixture"
    (tmp_path / paths[0]).write_bytes(b"fixture!\n")
    report = backend.check_backend()
    assert not report["supported"] and not report["checks"][0]["supported"]


def test_source_read_failure_is_reported(tmp_path):
    report = backend.source_check(tmp_path / "absent.py", {"expected"})
    assert not report["supported"] and report["error"]
