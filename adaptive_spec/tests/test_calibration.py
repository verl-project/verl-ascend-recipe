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

import json
from types import SimpleNamespace

import pytest

from adaptive_spec.calibration import collect, main, reduce_records, validate_plan
from adaptive_spec.config import AdaptiveSpecConfig
from adaptive_spec.context_bank import build
from adaptive_spec.vllm_plugin import calibration_v0_23 as observer


def make_plan(tmp_path):
    bank = tmp_path / "bank.jsonl"
    bank.write_text(
        "\n".join(json.dumps({"context_id": f"c{i}", "token_ids": list(range(40))}) for i in range(4)), encoding="utf-8"
    )
    raw = {
        "schema_version": 1,
        "model": "/models/example",
        "model_fingerprint": "example-model",
        "topology_fingerprint": "example-tp2",
        "context_bank": "bank.jsonl",
        "k_values": [5, 9],
        "states": [{"batch_size": 2, "context_length": 16}],
        "repeats": 2,
        "minimum_steps": 2,
        "engine_args": {"max_model_len": 512},
    }
    return validate_plan(raw, tmp_path / "plan.json"), raw


def measurements(plan):
    return [
        {
            "state_index": 0,
            "repeat": repeat,
            "k": k,
            "batch_size": 2,
            "clean": True,
            "context_lengths": [18, 19],
            "accepted_draft_tokens": [k, k - 1],
            "iteration_latency_ns": 1000 if k == 5 else 1600,
        }
        for repeat in range(plan["repeats"])
        for k in plan["k_values"]
        for _ in range(2)
    ]


def test_collection_loads_actual_controller_and_never_overwrites(tmp_path):
    plan, _ = make_plan(tmp_path)
    fixture = tmp_path / "records.jsonl"
    fixture.write_text("\n".join(json.dumps(r) for r in measurements(plan)), encoding="utf-8")
    out = tmp_path / "result"
    collect(plan, out, backend="replay", replay=fixture)
    runtime = json.loads((out / "runtime.json").read_text())
    assert runtime["enabled"] is False
    controller = AdaptiveSpecConfig.from_dict({**runtime, "enabled": True}).build_controller()
    assert controller is not None
    phase = json.loads((out / "phase_map.json").read_text())
    assert phase["collection"]["device_measurement"] is False
    assert phase["cells"][0]["allowed_ks"] == [5, 9]
    assert len(phase["cells"][0]["prior_survival"]) == 9
    assert phase["cells"][0]["measurement"]["exposures_by_depth"] == [16] * 5 + [8] * 4
    with pytest.raises(FileExistsError):
        collect(plan, out, backend="replay", replay=fixture)


@pytest.mark.parametrize(
    "change",
    [
        {"k_values": [5, 11]},
        {"k_values": [0, 5]},
        {"repeats": 0},
        {"engine_args": {"max_model_len": 20}},
        {"engine_args": {"max_model_len": 512, "async_scheduling": True}},
        {"engine_args": {"max_model_len": 512, "max_num_seqs": 1}},
    ],
)
def test_invalid_calibration_plan_is_rejected_before_loading_a_model(tmp_path, change):
    _, raw = make_plan(tmp_path)
    with pytest.raises(ValueError):
        validate_plan({**raw, **change}, tmp_path / "plan.json")


def test_paired_prefixes_repeatable_and_insufficient_contexts_rejected(tmp_path):
    plan, raw = make_plan(tmp_path)
    assert validate_plan(raw, tmp_path / "plan.json")["selections"] == plan["selections"]
    assert len(plan["selections"]) == 2  # Reused by both K workers, not selected separately per K.
    raw["states"] = [{"batch_size": 2, "context_length": 100}]
    with pytest.raises(ValueError, match="natural contexts"):
        validate_plan(raw, tmp_path / "plan.json")


@pytest.mark.parametrize(
    "change",
    [
        {"clean": False},
        {"batch_size": 1},
        {"iteration_latency_ns": float("nan")},
        {"accepted_draft_tokens": [12, 12]},
        {"context_lengths": [1, 2]},
    ],
)
def test_bad_or_out_of_phase_samples_cannot_become_priors(tmp_path, change):
    plan, _ = make_plan(tmp_path)
    rows = measurements(plan)
    rows[0].update(change)
    with pytest.raises(ValueError):
        reduce_records(plan, rows, backend="replay")


def test_missing_deep_k_and_repeats_fail_without_runtime_artifacts(tmp_path):
    plan, _ = make_plan(tmp_path)
    fixture = tmp_path / "partial.jsonl"
    fixture.write_text("\n".join(json.dumps(r) for r in measurements(plan) if r["k"] == 5), encoding="utf-8")
    out = tmp_path / "partial"
    with pytest.raises(ValueError, match="insufficient clean steps"):
        collect(plan, out, backend="replay", replay=fixture)
    assert (out / "failure.json").is_file()
    assert not (out / "runtime.json").exists()


def test_regret_subset_keeps_measured_costs_and_global_prior(tmp_path):
    plan, _ = make_plan(tmp_path)
    rows = measurements(plan)
    for row in rows:
        if row["k"] == 9:
            row["iteration_latency_ns"] = 10000
    cell = reduce_records(plan, rows, backend="replay")["cells"][0]
    assert cell["allowed_ks"] == [5]
    assert cell["fallback_k"] == 5
    assert cell["iteration_latency_ns_by_k"] == {"5": 1000}
    assert len(cell["prior_survival"]) == 9


def test_cli_plan_only_never_imports_or_loads_vllm(tmp_path, monkeypatch):
    _, raw = make_plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(raw))
    out = tmp_path / "planned"
    monkeypatch.setattr("sys.argv", ["collect", "--plan", str(path), "--output-dir", str(out), "--plan-only"])
    main()
    assert (out / "plan.resolved.json").exists()
    assert not (out / "phase_map.json").exists()


def test_observer_excludes_warmup_partial_drafts_and_discarded_rows():
    plan = {"k": 5, "states": [{"batch_size": 2, "context_length": 16}], "warmup_steps": 1, "length_window_tokens": 256}
    snapshot = {
        "state_index": 0,
        "repeat": 0,
        "discarded": False,
        "scheduled_tokens": 12,
        "draft_counts": [5, 5],
        "context_lengths": [17, 18],
        "emitted_counts": [3, 6],
    }
    counters = {}
    assert observer.step_record(plan, snapshot, 100, counters) is None
    assert observer.step_record(plan, snapshot, 100, counters)["accepted_draft_tokens"] == [2, 5]
    assert observer.step_record(plan, {**snapshot, "draft_counts": [5, 0]}, 100, counters) is None
    assert observer.step_record(plan, {**snapshot, "discarded": True}, 100, counters) is None
    assert observer.step_record(plan, {**snapshot, "scheduled_tokens": 100}, 100, counters) is None


def test_separate_target_and_sampling_calls_both_enter_the_cost(tmp_path, monkeypatch):
    # Exercise the real wrapper lifecycle, not just the record reducer.
    monkeypatch.setitem(globals(), "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)))
    ticks = iter([0, 100, 200, 500])
    monkeypatch.setattr(observer.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(observer, "_guard", lambda method, expected: method)
    monkeypatch.setattr(observer, "_is_eager", lambda runner: True)

    class Runner:
        use_async_scheduling = False
        num_spec_tokens = 5
        speculative_config = SimpleNamespace(method="mtp")
        drafter = SimpleNamespace(parallel_drafting=False)
        pcp_size = dcp_size = 1
        vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=0))
        input_batch = SimpleNamespace(req_ids=["a", "b"], num_tokens_no_spec=[17, 18])
        requests = {
            r: SimpleNamespace(
                sampling_params=SimpleNamespace(
                    extra_args={"adaptive_spec_calibration_state": 0, "adaptive_spec_calibration_repeat": 0}
                )
            )
            for r in ["a", "b"]
        }
        num_discarded_requests = 0
        discard_request_indices = SimpleNamespace(np=[])

        def execute_model(self, scheduler_output, intermediate_tensors=None):
            return None

        def sample_tokens(self):
            self._bookkeeping_sync(
                None, num_scheduled_tokens=12, spec_decode_metadata=SimpleNamespace(num_draft_tokens=[5, 5])
            )
            return "sampled"

        def _bookkeeping_sync(self, scheduler_output, num_scheduled_tokens, spec_decode_metadata):
            return None, [[1, 2, 3], [1, 2, 3, 4]], None, ["a", "b"], {}, []

    plan = {"k": 5, "states": [{"batch_size": 2, "context_length": 16}], "warmup_steps": 0, "length_window_tokens": 256}
    observer.patch_class(Runner, plan, tmp_path)
    runner = Runner()
    runner.execute_model(SimpleNamespace(preempted_req_ids=[]))
    assert not list(tmp_path.glob("steps.*.jsonl"))
    assert runner.sample_tokens() == "sampled"
    row = json.loads(next(tmp_path.glob("steps.*.jsonl")).read_text())
    assert row["iteration_latency_ns"] == 400  # 100 target + 300 sampling/draft, excluding inter-call gap.
    assert row["accepted_draft_tokens"] == [2, 3]
    assert runner._verl_calibration_execute_ns is None


def test_calibration_plugin_is_inert_without_offline_environment(monkeypatch):
    monkeypatch.delenv("VERL_ADAPTIVE_SPEC_CALIBRATION_DIR", raising=False)
    assert observer.calibration_plugin() is False


def test_context_export_preserves_exact_tokens_and_source_receipt(tmp_path):
    source = tmp_path / "rollouts.jsonl"
    source.write_text(json.dumps({"context_id": "p1-r0", "prompt_token_ids": [1, 2], "response_token_ids": [3, 4]}))
    out = tmp_path / "contexts.jsonl"
    assert build(source, out) == 1
    assert json.loads(out.read_text())["token_ids"] == [1, 2, 3, 4]
    assert json.loads(out.with_suffix(".jsonl.receipt.json").read_text())["contexts"] == 1
    with pytest.raises(FileExistsError):
        build(source, out)
