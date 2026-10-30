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

from adaptive_spec.config import AdaptiveSpecConfig
from adaptive_spec.vllm_plugin.compatibility import adapter_name
from adaptive_spec.vllm_plugin.v0_23 import (
    _apply_engine_k,
    _install_pending,
    _logical_draft_count,
    _patch_class,
    _pending_snapshot,
    _sampling_payload,
)


def runner(mode="NONE"):
    return SimpleNamespace(
        drafter=SimpleNamespace(num_speculative_tokens=12, decode_threshold=13, parallel_drafting=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name=mode)),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=12),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        ),
        pcp_size=1,
        dcp_size=1,
        num_spec_tokens=12,
        uniform_decode_query_len=13,
    )


def test_adapter_selection_is_strict():
    assert adapter_name("0.23.0+empty", "0.23.0rc2.dev1") == "v0_23"
    assert adapter_name("0.22.0", "0.23.0") is None
    assert adapter_name("0.23.0", "0.24.0") is None


def test_apply_k_updates_only_eager_engine():
    eager = runner()
    assert _apply_engine_k(eager, 7, require_eager=True)
    assert eager.drafter.num_speculative_tokens == 7
    assert eager.drafter.decode_threshold == 8
    assert eager.num_spec_tokens == 7
    assert eager.uniform_decode_query_len == 8

    graph = runner("FULL")
    with pytest.raises(RuntimeError, match="eager"):
        _apply_engine_k(graph, 7, require_eager=True)
    assert graph.drafter.num_speculative_tokens == 12


def test_adapter_remembers_kmax_and_rejects_overshoot():
    value = runner()
    assert _apply_engine_k(value, 5, require_eager=True)
    try:
        _apply_engine_k(value, 13, require_eager=True)
    except ValueError as error:
        assert "Unsupported" in str(error)
    else:
        raise AssertionError("K above configured Kmax must fail")


def test_sampling_payload_scopes_step_but_excludes_seed():
    params = SimpleNamespace(
        temperature=1.0,
        top_p=0.9,
        seed=8,
        extra_args={"adaptive_spec_rollout_step": 3},
    )
    payload, step = _sampling_payload(params)
    assert payload == {"temperature": 1.0, "top_p": 0.9}
    assert step == 3


def test_logical_draft_count_is_right_censoring_bound():
    assert _logical_draft_count(12, 7) == 7
    assert _logical_draft_count(5, 7) == 5
    assert _logical_draft_count(0, 7) == 0


def test_class_patch_is_idempotent_without_importing_ascend_runner(monkeypatch):
    class FakeRunner:
        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return scheduler_output, intermediate_tensors

        def _bookkeeping_sync(
            self,
            scheduler_output,
            sampler_output=None,
            logits=None,
            hidden_states=None,
            num_scheduled_tokens=0,
            spec_decode_metadata=None,
        ):
            return None, [], {}, [], {}, []

        def propose_draft_token_ids(
            self,
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            aux_hidden_states=None,
            sample_hidden_states=None,
            target_model_batch_desc=None,
        ):
            return valid_sampled_token_ids

    import adaptive_spec.vllm_plugin.async_reconcile_v0_23 as async_mod
    import adaptive_spec.vllm_plugin.mtp_copy_v0_23 as copy_mod

    monkeypatch.setattr(copy_mod, "patch_copy_and_read", lambda cls: True)
    monkeypatch.setattr(async_mod, "patch_class", lambda cls: True)
    config = AdaptiveSpecConfig(enabled=False, phase_map_path=None, model_fingerprint="")
    assert _patch_class(FakeRunner, config)
    first_execute = FakeRunner.execute_model
    assert _patch_class(FakeRunner, config)
    assert FakeRunner.execute_model is first_execute


def test_discarded_rows_are_removed_from_effective_batch():
    pending = {
        "rows": [
            {"context": "r0", "draft_tokens": 7, "discarded": False},
            {"context": "r1", "draft_tokens": 7, "discarded": False},
        ]
    }
    value = SimpleNamespace()
    _install_pending(value, pending, {1})
    assert pending["rows"][0]["discarded"] is False
    assert pending["rows"][1]["discarded"] is True
    assert pending["batch_size"] == 1


def test_pending_snapshot_preserves_runner_row_alignment():
    contexts = ["r0", "r1", "r2"]
    metadata = SimpleNamespace(num_draft_tokens=[7, 3])
    pending = _pending_snapshot(contexts, metadata, 5)
    assert pending is not None
    assert [row["draft_tokens"] for row in pending["rows"]] == [5, 3, 0]


def _patched_runner_for_order_test(monkeypatch):
    import adaptive_spec.vllm_plugin.v0_23 as adapter

    class FakeRunner:
        def _bookkeeping_sync(
            self,
            scheduler_output,
            sampler_output=None,
            logits=None,
            hidden_states=None,
            num_scheduled_tokens=0,
            spec_decode_metadata=None,
        ):
            return None, [], {}, [], {}, [1]

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return scheduler_output, intermediate_tensors

        def propose_draft_token_ids(
            self,
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            aux_hidden_states=None,
            sample_hidden_states=None,
            target_model_batch_desc=None,
        ):
            return SimpleNamespace(shape=(2, self.drafter.num_speculative_tokens))

    import adaptive_spec.vllm_plugin.async_reconcile_v0_23 as async_mod
    import adaptive_spec.vllm_plugin.mtp_copy_v0_23 as copy_mod

    monkeypatch.setattr(copy_mod, "patch_copy_and_read", lambda cls: True)
    monkeypatch.setattr(async_mod, "patch_class", lambda cls: True)
    config = AdaptiveSpecConfig(enabled=False, phase_map_path=None, model_fingerprint="")
    assert _patch_class(FakeRunner, config)
    contexts = [
        SimpleNamespace(rollout_step=0),
        SimpleNamespace(rollout_step=0),
    ]
    controller = SimpleNamespace(
        decide_engine=lambda value: SimpleNamespace(physical_k=7),
    )
    state = {"controller": controller, "rollout_step": 0}
    monkeypatch.setattr(adapter, "_runtime_state", lambda runner, config: state)
    monkeypatch.setattr(adapter, "_request_contexts", lambda runner: contexts)

    def apply_for_test(runner, selected_k, require_eager):
        runner.drafter.num_speculative_tokens = selected_k
        return True

    monkeypatch.setattr(adapter, "_apply_engine_k", apply_for_test)
    value = FakeRunner()
    value.discard_request_indices = SimpleNamespace(np=[])
    value.num_discarded_requests = 0
    value.drafter = SimpleNamespace(num_speculative_tokens=12)
    value.compilation_config = SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"))
    setattr(value, adapter._LAST_DRAFT_K_ATTR, 7)
    return adapter, value, SimpleNamespace(num_draft_tokens=[7, 7])


def _call_propose(runner, metadata):
    return runner.propose_draft_token_ids(
        None,
        None,
        SimpleNamespace(),
        metadata,
        None,
        None,
        0,
        None,
    )


def test_padded_order_reconciles_invalid_rows_after_propose(monkeypatch):
    adapter, runner, metadata = _patched_runner_for_order_test(monkeypatch)
    _call_propose(runner, metadata)
    assert getattr(runner, adapter._CURRENT_ATTR)["rows"][1]["discarded"] is False
    runner._bookkeeping_sync(SimpleNamespace(), spec_decode_metadata=metadata)
    pending = getattr(runner, adapter._PENDING_ATTR)
    assert getattr(runner, adapter._CURRENT_ATTR) is None
    assert pending["batch_size"] == 1
    assert [row["discarded"] for row in pending["rows"]] == [False, True]


def test_nonpadded_order_reconciles_invalid_rows_before_propose(monkeypatch):
    adapter, runner, metadata = _patched_runner_for_order_test(monkeypatch)
    runner._bookkeeping_sync(SimpleNamespace(), spec_decode_metadata=metadata)
    _call_propose(runner, metadata)
    pending = getattr(runner, adapter._PENDING_ATTR)
    assert getattr(runner, adapter._CURRENT_ATTR, None) is None
    assert pending["batch_size"] == 1
    assert [row["discarded"] for row in pending["rows"]] == [False, True]


# These regressions verify attribution across the actual adapter callbacks,
# including the two Ascend bookkeeping orders, not just the clipping helper.
def test_previous_k_binding_across_transitions(monkeypatch):
    for previous, upcoming, raw, emitted, expected in [
        (9, 5, 9, 9, (9, 8)),
        (12, 7, 12, 12, (12, 11)),
        (5, 9, 12, 6, (5, 5)),
        (7, 7, 3, 3, (3, 2)),
    ]:
        adapter, value, metadata = _patched_runner_for_order_test(monkeypatch)
        setattr(value, adapter._LAST_DRAFT_K_ATTR, previous)
        state = adapter._runtime_state(value, None)
        state["controller"].decide_engine = lambda contexts, k=upcoming: SimpleNamespace(physical_k=k)
        observations = []
        state["controller"].observe_round = lambda context, sink=observations, **kw: sink.append(kw)
        metadata.num_draft_tokens = [raw, raw]
        _call_propose(value, metadata)
        assert getattr(value, adapter._LAST_DRAFT_K_ATTR) == upcoming
        value._bookkeeping_sync(SimpleNamespace(), spec_decode_metadata=metadata)
        monkeypatch.setattr(adapter, "_accepted_counts", lambda runner, count, emitted=emitted: [emitted] * count)
        setattr(value, adapter._REQUESTS_ATTR, {})
        adapter._finalize_previous(value, SimpleNamespace(finished_req_ids=[]), state)
        adapter._finalize_previous(value, SimpleNamespace(finished_req_ids=[]), state)
        assert len(observations) == 1  # discarded row ignored; consumed exactly once
        assert (observations[0]["active_k"], observations[0]["accepted_draft_tokens"]) == expected
        assert observations[0]["active_batch_size"] == 1


def test_missing_binding_rejects_positive_draft_before_next_decision(monkeypatch):
    import pytest

    adapter, value, metadata = _patched_runner_for_order_test(monkeypatch)
    delattr(value, adapter._LAST_DRAFT_K_ATTR)
    state = adapter._runtime_state(value, None)
    state["controller"].decide_engine = lambda contexts: (_ for _ in ()).throw(AssertionError("must not decide"))
    with pytest.raises(RuntimeError, match="previous-K binding"):
        _call_propose(value, metadata)


def test_prefill_without_previous_k_binds_first_produced_draft(monkeypatch):
    adapter, value, metadata = _patched_runner_for_order_test(monkeypatch)
    delattr(value, adapter._LAST_DRAFT_K_ATTR)
    metadata.num_draft_tokens = [0, 0]
    _call_propose(value, metadata)
    assert getattr(value, adapter._LAST_DRAFT_K_ATTR) == 7
    value._bookkeeping_sync(SimpleNamespace(), spec_decode_metadata=metadata)
    assert all(row["draft_tokens"] == 0 for row in getattr(value, adapter._PENDING_ATTR)["rows"])


def test_no_metadata_is_not_a_fabricated_observation():
    assert _pending_snapshot(["r0"], None, None) is None


def test_graph_mode_preserves_upstream_without_requiring_evidence_binding(monkeypatch):
    adapter, value, metadata = _patched_runner_for_order_test(monkeypatch)
    delattr(value, adapter._LAST_DRAFT_K_ATTR)
    value.compilation_config.cudagraph_mode.name = "FULL"
    state = adapter._runtime_state(value, None)
    state["controller"].decide_engine = lambda contexts: (_ for _ in ()).throw(AssertionError("must not decide"))
    _call_propose(value, metadata)
    assert not hasattr(value, adapter._LAST_DRAFT_K_ATTR)
    assert getattr(value, adapter._CURRENT_ATTR, None) is None
