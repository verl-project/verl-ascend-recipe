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

from types import SimpleNamespace

import pytest

from adaptive_spec.evidence import AcceptanceEvidence, ResponseSummary
from adaptive_spec.exchange import CrossEngineEvidenceExchange
from adaptive_spec.vllm_plugin.async_reconcile_v0_23 import prepare_worker_schedule, wrap_methods


def test_exchange_reads_only_complete_lines_once_and_contains_no_prompt(tmp_path):
    a = CrossEngineEvidenceExchange(tmp_path, source_id="a", poll_decisions=1)
    b = CrossEngineEvidenceExchange(tmp_path, source_id="b", poll_decisions=1)
    e = AcceptanceEvidence(max_depth=3)
    a.publish("hash-only", "r1", ResponseSummary((2.0, 2.0, 0.0), (2.0, 1.0, 0.0), 2))
    original = a.path.read_bytes()
    a.path.write_bytes(original[:-1])
    b.maybe_poll(e)
    assert e.posterior("hash-only", [0.9] * 3).response_count == 0
    a.path.write_bytes(original)
    b.maybe_poll(e)
    b.maybe_poll(e)
    assert e.posterior("hash-only", [0.9] * 3).response_count == 1
    assert e.posterior("new-step-key", [0.9] * 3).response_count == 0
    assert b"prompt_token_ids" not in original


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_exchange_summary_rejected_without_insertion(bad):
    e = AcceptanceEvidence(max_depth=1)
    with pytest.raises(ValueError):
        e.commit_summary("g", "r", exposures=[bad], successes=[0], block_count=1)
    assert e.posterior("g", [0.9]).response_count == 0


def context():
    runner = SimpleNamespace(
        use_async_scheduling=True,
        num_spec_tokens=5,
        _draft_token_ids=SimpleNamespace(shape=(1, 5)),
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        requests={"r": SimpleNamespace(prev_num_draft_len=9)},
        speculative_config=SimpleNamespace(method="mtp"),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1)),
    )
    schedule = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["r"], resumed_req_ids=set()),
        preempted_req_ids=set(),
        finished_req_ids=set(),
        scheduled_spec_decode_tokens={"r": list(range(9))},
        num_scheduled_tokens={"r": 10},
        total_num_scheduled_tokens=10,
    )
    return runner, schedule


def test_worker_local_trim_preserves_scheduler_snapshot():
    r, s = context()
    worker, restore = prepare_worker_schedule(r, s)
    assert worker is not s and worker.num_scheduled_tokens["r"] == 6
    assert worker.total_num_scheduled_tokens == 6 and len(worker.scheduled_spec_decode_tokens["r"]) == 5
    assert s.total_num_scheduled_tokens == 10 and s.num_scheduled_tokens["r"] == 10
    assert restore == {"r": 9}
    s.preempted_req_ids = {"r"}
    worker, restore = prepare_worker_schedule(r, s)
    assert worker is s and restore == {}


def test_optimistic_count_restored_after_update_and_state_cleared():
    r, s = context()
    events = []

    def update(self, schedule):
        self.requests["r"].prev_num_draft_len = len(schedule.scheduled_spec_decode_tokens["r"])
        events.append(self.requests["r"].prev_num_draft_len)
        return 71

    def execute(self, schedule, intermediate_tensors=None):
        assert patched_update(self, schedule) == 71
        assert self.requests["r"].prev_num_draft_len == 9
        return "done"

    patched_execute, patched_update = wrap_methods(execute, update)
    assert patched_execute(r, s) == "done" and events == [5]
    assert r._verl_mtp_async_restore_v023 is None


def test_disabled_plugin_does_not_import_backend(monkeypatch):
    import adaptive_spec.vllm_plugin as plugin

    monkeypatch.delenv("VERL_ADAPTIVE_SPEC_CONFIG", raising=False)
    monkeypatch.delenv("VERL_MTP_KV_PREEMPT_MODE", raising=False)
    monkeypatch.setattr(plugin, "installed_adapter_name", lambda: (None, "missing", "missing"))
    monkeypatch.setattr(plugin.importlib, "import_module", lambda name: (_ for _ in ()).throw(AssertionError(name)))
    assert plugin.adaptive_spec_plugin() is False
