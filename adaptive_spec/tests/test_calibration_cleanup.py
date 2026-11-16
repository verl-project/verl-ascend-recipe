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

import os
import sys
import types

import pytest

from adaptive_spec import calibration_vllm as collector
from adaptive_spec.vllm_plugin import compatibility


def test_calibration_workers_inherit_cleanup_without_dynamic_control(monkeypatch, tmp_path):
    monkeypatch.delenv("VERL_ADAPTIVE_SPEC_CONFIG", raising=False)
    monkeypatch.delenv("VERL_MTP_KV_PREEMPT_MODE", raising=False)
    monkeypatch.delenv("VERL_ASCEND_CLEANUP_GRACE", raising=False)
    monkeypatch.setenv("VLLM_PLUGINS", "ascend,adaptive_spec_calibration_plugin")
    monkeypatch.setattr(compatibility, "installed_adapter_name", lambda: ("v0_23", "0.23.0", "0.23.0"))
    monkeypatch.setattr(collector, "digest", lambda _: "bank-sha")
    monkeypatch.setattr(collector, "read_bank", lambda _: [])

    def launch(**kwargs):
        assert os.environ["VERL_ASCEND_CLEANUP_GRACE"] == "1"
        plugins = os.environ["VLLM_PLUGINS"].split(",")
        assert plugins.count("adaptive_spec_plugin") == 1
        assert plugins.count("adaptive_spec_calibration_plugin") == 1
        assert not os.environ.get("VERL_ADAPTIVE_SPEC_CONFIG")
        assert kwargs["speculative_config"]["num_speculative_tokens"] == 5
        raise RuntimeError("worker launch reached")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=launch, SamplingParams=object))
    plan = dict(
        states=[],
        warmup_steps=2,
        length_window_tokens=256,
        context_bank=str(tmp_path / "bank.jsonl"),
        context_bank_sha256="bank-sha",
        model="model",
        engine_args={},
    )
    with pytest.raises(RuntimeError, match="worker launch reached"):
        collector.run_probe(plan, tmp_path, 5)
