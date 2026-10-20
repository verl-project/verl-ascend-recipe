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

"""A probe leader exiting is not evidence that its worker group exited."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from adaptive_spec import calibration


def run_failed_probe(tmp_path, monkeypatch, *, timeout, group_already_gone=False):
    signals = []

    def killpg(pgid, sig):
        assert pgid == 12345
        signals.append(sig)
        if group_already_gone:
            raise ProcessLookupError("probe group already exited")

    class Probe:
        pid = 12345
        returncode = 2

        def __init__(self, *args, **kwargs):
            assert kwargs["start_new_session"] is True
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1 and timeout is not None and timeout == 30:
                raise subprocess.TimeoutExpired("probe", timeout)
            # The leader can exit while a worker still ignores SIGTERM.
            return self.returncode

    monkeypatch.setattr(calibration, "os", SimpleNamespace(name="posix", killpg=killpg))
    monkeypatch.setattr(calibration, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    monkeypatch.setattr(calibration.subprocess, "Popen", Probe)
    plan = {"k_values": [5], "seed": 1, "probe_timeout_seconds": 30 if timeout else 40}
    expected = subprocess.TimeoutExpired if timeout else RuntimeError
    output = tmp_path / "failed"
    with pytest.raises(expected):
        calibration.collect(plan, output, backend="vllm")
    failure = json.loads((output / "failure.json").read_text())
    assert failure["type"] == expected.__name__
    assert not (output / "runtime.json").exists()
    return signals


def test_timeout_kills_remaining_workers_after_leader_exits(tmp_path, monkeypatch):
    assert run_failed_probe(tmp_path, monkeypatch, timeout=True) == [15, 9]


def test_nonzero_probe_exit_also_cleans_worker_group(tmp_path, monkeypatch):
    assert run_failed_probe(tmp_path, monkeypatch, timeout=False) == [15, 9]


def test_exited_group_does_not_mask_original_timeout(tmp_path, monkeypatch):
    assert run_failed_probe(tmp_path, monkeypatch, timeout=True, group_already_gone=True) == [15, 9]
