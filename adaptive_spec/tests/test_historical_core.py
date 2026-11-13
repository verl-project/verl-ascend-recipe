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

from dataclasses import replace

import pytest

from adaptive_spec import AcceptanceEvidence, AdaptiveSpecController, PhaseMap, RequestContext
from adaptive_spec.config import AdaptiveSpecConfig
from adaptive_spec.evidence import group_key
from adaptive_spec.reconcile import HistoricalEngineReconciler, request_k


def phase_map():
    return PhaseMap.from_dict(
        dict(
            model_fingerprint="m",
            topology_fingerprint="tp2",
            cells=[
                dict(
                    phase_id="short",
                    batch_size=16,
                    context_length=100,
                    allowed_ks=[3],
                    fallback_k=3,
                    iteration_latency_ns_by_k={"3": 4},
                    prior_survival=[0.9] * 7,
                ),
                dict(
                    phase_id="long",
                    batch_size=16,
                    context_length=10000,
                    allowed_ks=[7],
                    fallback_k=7,
                    iteration_latency_ns_by_k={"7": 8},
                    prior_survival=[0.9] * 7,
                ),
            ],
        )
    )


def test_p90_common_phase_is_not_longest_or_per_request():
    ctl = AdaptiveSpecController(phase_map(), model_fingerprint="m")
    requests = [RequestContext(str(i), (1, 2), {}, length) for i, length in enumerate([100] * 15 + [10000])]
    decision = ctl.decide_engine(requests)
    assert decision.physical_k == 3
    assert {r.phase_id for r in decision.request_decisions} == {"short"}
    ctl.reset_step()
    decision = ctl.decide_engine(requests[-2:])
    assert decision.physical_k == 7  # disjoint per-request candidates need no intersection
    assert {r.phase_id for r in decision.request_decisions} == {"long"}


def test_lower_median_confirmation_and_global_ladder():
    rec = HistoricalEngineReconciler([3, 5, 7], confirmations=2, minimum_dwell_rounds=64)
    assert rec.select([3, 7], fallback_k=3) == 3
    assert rec.select([7], fallback_k=7) == 3
    assert rec.select([7], fallback_k=7) == 5  # global intermediate K may be outside current cell
    assert rec.select([7], fallback_k=7) == 5
    for _ in range(63):
        result = rec.select([7], fallback_k=7)
    assert result == 7


def test_score_gain_and_tie():
    assert request_k([1, 2], [1.0, 0.0], {1: 2.0, 2: 2.0 / 1.009}, fallback_k=1) == 1
    assert request_k([1, 2], [1.0, 0.0], {1: 2.0, 2: 2.0 / 1.011}, fallback_k=1) == 2
    assert request_k([1, 2], [1.0, 0.0], {1: 2.0, 2: 2.0}) == 1


def test_censoring_maturity_self_exclusion_and_commit():
    e = AcceptanceEvidence(max_depth=7)
    for _ in range(63):
        e.stage("g", "a", active_k=3, accepted_draft_tokens=3)
    assert e.posterior("g", [0.9] * 7).response_count == 0
    e.stage("g", "a", active_k=3, accepted_draft_tokens=3)
    p = e.posterior("g", [0.9] * 7)
    assert p.response_count == 1 and p.exposed_by_depth == (64.0, 64.0, 64.0, 0.0, 0.0, 0.0, 0.0)
    assert p.survival[3:] == (0.9,) * 4
    assert e.posterior("g", [0.9] * 7, exclude_response_id="a").response_count == 0
    assert e.commit("g", "a") and not e.commit("g", "a")
    assert e.posterior("g", [0.9] * 7) == p
    e.reset_step()
    assert e.posterior("g", [0.9] * 7).response_count == 0


def test_window_and_invalid_observation_have_no_partial_side_effect():
    e = AcceptanceEvidence(max_depth=3, window_responses=2)
    with pytest.raises(ValueError):
        e.stage("g", "invalid", active_k=3, accepted_draft_tokens=4)
    assert not e.commit("g", "invalid")
    for name, accepted in [("a", 3), ("b", 2), ("c", 1)]:
        e.stage("g", name, active_k=3, accepted_draft_tokens=accepted)
        e.commit("g", name)
    assert e.posterior("g", [0.9] * 3).response_count == 2
    with pytest.raises(KeyError):
        e.response_summary("g", "a")


def test_exact_cache_reuses_hash_without_changing_identity(monkeypatch):
    import adaptive_spec.controller as module

    calls = []
    original = module.group_key

    def counted(*args, **kwargs):
        calls.append(kwargs["phase_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "group_key", counted)
    ctl = AdaptiveSpecController(phase_map(), model_fingerprint="m")
    a = RequestContext("a", (1, 2), {"temperature": 1.0, "seed": 3}, 100, 1)
    expected = group_key(
        a.prompt_token_ids, model_fingerprint="m", sampling_params=a.sampling_params, rollout_step=1, phase_id="short"
    )
    assert ctl._key(a, "short") == expected
    assert (
        ctl._key(replace(a, context_length=300, sampling_params={"temperature": 1.0, "seed": 9}), "short") == expected
    )
    assert calls == ["short"]
    assert ctl._key(replace(a, request_id="b"), "short") == expected
    assert ctl._key(a, "long") != expected
    with pytest.raises(ValueError, match="identity changed"):
        ctl._key(replace(a, prompt_token_ids=(1, 3)), "short")
    ctl.discard_request(a)
    assert "a" not in ctl._key_cache and "a" not in ctl._identity_cache
    ctl._key(replace(a, prompt_token_ids=(1, 3)), "short")
    ctl.reset_step()
    assert not ctl._key_cache and not ctl._identity_cache and ctl.reconciler.current_k is None


def test_mutated_nested_sampling_is_not_a_cache_hit():
    ctl = AdaptiveSpecController(phase_map(), model_fingerprint="m")
    sampling = {"stop_token_ids": [1, 2]}
    request = RequestContext("r", (3,), sampling, 100)
    ctl._key(request, "short")
    sampling["stop_token_ids"].append(4)
    with pytest.raises(ValueError, match="identity changed"):
        ctl._key(request, "short")


def test_defaults_and_removed_research_mode():
    c = AdaptiveSpecConfig.from_dict({})
    assert (c.window_responses, c.prior_strength, c.minimum_dwell_rounds, c.minimum_score_gain) == (16, 8.0, 64, 0.01)
    with pytest.raises(ValueError):
        AdaptiveSpecConfig.from_dict({"aggregation_mode": "request_intersection"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_bad_measured_costs_rejected(value):
    cell = phase_map().cells[0]
    with pytest.raises(ValueError):
        type(cell).from_dict(
            dict(
                phase_id="s",
                batch_size=16,
                context_length=100,
                allowed_ks=[3],
                fallback_k=3,
                iteration_latency_ns_by_k={"3": value},
                prior_survival=[0.9] * 7,
            )
        )
