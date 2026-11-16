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

from adaptive_spec.config import AdaptiveSpecConfig
from adaptive_spec.controller import AdaptiveSpecController, RequestContext
from adaptive_spec.phase_map import PhaseMap
from adaptive_spec.reconcile import HistoricalEngineReconciler


def controller(aggregation, **kwargs):
    phase_map = PhaseMap.from_dict(
        {
            "model_fingerprint": "m",
            "topology_fingerprint": "tp2",
            "cells": [
                {
                    "phase_id": "p",
                    "batch_size": 2,
                    "context_length": 1024,
                    "allowed_ks": [5, 7, 9],
                    "fallback_k": 5,
                    "iteration_latency_ns_by_k": {"5": 6, "7": 6.5, "9": 7},
                    "prior_survival": [0.5] * 9,
                }
            ],
        }
    )
    ctl = AdaptiveSpecController(
        phase_map,
        model_fingerprint="m",
        reconciler=HistoricalEngineReconciler(
            [5, 7, 9],
            engine_aggregation=aggregation,
            confirmations=1,
            minimum_dwell_rounds=0,
            one_rung_per_switch=False,
        ),
        **kwargs,
    )
    a = RequestContext("a", (1, 2), {"temperature": 1.0}, 1024)
    b = replace(a, request_id="b")
    for _ in range(64):
        ctl.observe_round(a, active_batch_size=2, active_k=9, accepted_draft_tokens=0)
        ctl.observe_round(b, active_batch_size=2, active_k=9, accepted_draft_tokens=9)
    return ctl, a, b


def test_max_uses_actual_sibling_posteriors_and_keeps_common_phase():
    low, a, b = controller("lower_median")
    high, _, _ = controller("max")
    lower = low.decide_engine([a, b])
    maximum = high.decide_engine([a, b])
    assert [d.selected_k for d in maximum.request_decisions] == [9, 5]
    assert maximum.request_decisions == lower.request_decisions
    assert {d.phase_id for d in maximum.request_decisions} == {"p"}
    assert (lower.physical_k, maximum.physical_k) == (5, 9)


def test_max_keeps_two_confirmations_dwell_and_global_rungs():
    rec = HistoricalEngineReconciler([5, 7, 9], engine_aggregation="max")
    assert rec.select([5, 5], fallback_k=5) == 5
    assert rec.select([5, 9], fallback_k=5) == 5
    assert rec.select([5, 9], fallback_k=5) == 7
    for _ in range(63):
        assert rec.select([5, 9], fallback_k=5) == 7
    assert rec.select([5, 9], fallback_k=5) == 9


def test_drain_arms_after_saturation_debounces_and_does_not_reenter():
    ctl, a, b = controller(
        "max",
        phase_stage_control="drain_triggered",
        phase_stage_max_batch_size=2,
        phase_stage_below_max_confirm_steps=4,
    )
    for _ in range(8):
        d = ctl.decide_engine([a])
        assert (d.phase_stage, d.physical_k) == ("warmup", 5)
    assert ctl.decide_engine([a, b]).phase_stage == "saturated"
    for _ in range(3):
        assert ctl.decide_engine([a]).physical_k == 5
    # A full batch resets a transient below-capacity streak.
    assert ctl.decide_engine([a, b]).phase_stage == "saturated"
    for _ in range(3):
        assert ctl.decide_engine([a]).phase_stage == "saturated"
    d = ctl.decide_engine([a])
    assert (d.phase_stage, d.physical_k) == ("drain", 9)
    d = ctl.decide_engine([a, b])
    assert (d.phase_stage, d.physical_k) == ("drain", 9)
    ctl.reset_step()
    d = ctl.decide_engine([a])
    assert (d.phase_stage, d.physical_k) == ("warmup", 5)


def test_new_options_preserve_omitted_field_behavior():
    c = AdaptiveSpecConfig.from_dict({})
    assert (c.engine_aggregation, c.phase_stage_control, c.tp_control_transport) == ("lower_median", "off", "object")
    assert (c.confirmations, c.minimum_dwell_rounds) == (2, 64)


@pytest.mark.parametrize(
    "raw",
    [
        {"engine_aggregation": "unknown"},
        {"tp_control_transport": "auto"},
        {"phase_stage_control": "unknown"},
        {"phase_stage_max_batch_size": 0},
        {"phase_stage_below_max_confirm_steps": 0},
    ],
)
def test_invalid_modes_are_rejected_even_while_disabled(raw):
    with pytest.raises(ValueError):
        AdaptiveSpecConfig.from_dict(raw)


@pytest.mark.parametrize("values", [[5, 11], [True, 5], [5.0, 9]])
def test_invalid_request_recommendations_are_not_hidden_by_max(values):
    rec = HistoricalEngineReconciler([5, 7, 9], engine_aggregation="max")
    with pytest.raises(ValueError):
        rec.select(values, fallback_k=5)
