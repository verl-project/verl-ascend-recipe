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
from collections import deque
from types import SimpleNamespace

import pytest

from adaptive_spec.config import AdaptiveSpecConfig
from adaptive_spec.controller import RequestContext
from adaptive_spec.evidence import ResponseSummary
from adaptive_spec.exchange import CrossEngineEvidenceExchange
from adaptive_spec.vllm_plugin import v0_23 as adapter


def states(tmp_path, monkeypatch):
    packets = deque()

    class Group:
        world_size = 2

        def __init__(self, rank):
            self.rank_in_group = rank

        def broadcast_object(self, packet, src):
            assert src == 0
            if self.rank_in_group == 0:
                packets.append(packet)
                return packet
            assert packet is None
            return packets.popleft()

    phase = tmp_path / "phase.json"
    phase.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": "model",
                "topology_fingerprint": "tp2",
                "cells": [
                    {
                        "phase_id": "p",
                        "batch_size": 1,
                        "context_length": 1024,
                        "allowed_ks": [5, 9],
                        "iteration_latency_ns_by_k": {"5": 6.0, "9": 7.0},
                        "prior_survival": [0.8] * 9,
                        "fallback_k": 5,
                    }
                ],
            }
        )
    )
    config = AdaptiveSpecConfig.from_dict(
        {
            "enabled": True,
            "phase_map_path": str(phase),
            "model_fingerprint": "model",
            "topology_fingerprint": "tp2",
            "evidence_exchange_directory": str(tmp_path / "shared"),
        }
    )
    runners = [SimpleNamespace(rank=rank) for rank in range(2)]
    monkeypatch.setattr(adapter, "_tensor_parallel_group", lambda runner: Group(runner.rank))
    state = [adapter._runtime_state(runner, config) for runner in runners]
    request = RequestContext("active", (1, 2), {"temperature": 1.0}, 1024, 0)
    return runners, state, request


def test_shared_evidence_arrival_cannot_diverge_tp_depth(tmp_path, monkeypatch):
    _, (leader, follower), request = states(tmp_path, monkeypatch)
    assert follower["controller"] is None
    for _ in range(63):
        assert adapter._select_engine_k(leader, [request]) == 5
        assert adapter._select_engine_k(follower, [request]) == 5
    # A completed sibling arrives between the two ranks' iteration-64 calls.
    assert adapter._select_engine_k(leader, [request]) == 5
    writer = CrossEngineEvidenceExchange(tmp_path / "shared", source_id="other-engine")
    key = leader["controller"]._key(request, "p")
    writer.publish(key, "sibling", ResponseSummary((100.0,) * 9, (100.0,) * 9, 100))
    assert adapter._select_engine_k(follower, [request]) == 5
    # Default polling is 64 and confirmations=2: neither rank switches early.
    for _ in range(64):
        assert adapter._select_engine_k(leader, [request]) == 5
        assert adapter._select_engine_k(follower, [request]) == 5
    assert adapter._select_engine_k(leader, [request]) == 9
    assert adapter._select_engine_k(follower, [request]) == 9


def test_only_leader_observes_and_publishes_completed_response(tmp_path, monkeypatch):
    runners, state, request = states(tmp_path, monkeypatch)
    calls = []

    def counts(runner, count):
        assert runner.rank == 0 and count == 1
        calls.append(runner.rank)
        return [6]

    monkeypatch.setattr(adapter, "_accepted_counts", counts)
    for runner, worker_state in zip(runners, state, strict=True):
        setattr(runner, adapter._REQUESTS_ATTR, {request.request_id: request})
        setattr(
            runner,
            adapter._PENDING_ATTR,
            {"rows": [{"context": request, "draft_tokens": 5, "discarded": False}], "batch_size": 1},
        )
        adapter._finalize_previous(runner, SimpleNamespace(finished_req_ids=[request.request_id]), worker_state)
        assert getattr(runner, adapter._PENDING_ATTR) is None
        assert not getattr(runner, adapter._REQUESTS_ATTR)
    assert calls == [0]
    files = list((tmp_path / "shared").glob("source-*.jsonl"))
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == 1


def test_policy_failure_is_broadcast_before_all_ranks_raise(tmp_path, monkeypatch):
    _, (leader, follower), request = states(tmp_path, monkeypatch)

    def fail(contexts):
        raise ValueError("bad phase input")

    monkeypatch.setattr(leader["controller"], "decide_engine", fail)
    for state in [leader, follower]:
        with pytest.raises(RuntimeError, match="TP leader decision failed: ValueError: bad phase input"):
            adapter._select_engine_k(state, [request])


def test_out_of_order_tp_packet_is_rejected(tmp_path, monkeypatch):
    _, (leader, follower), request = states(tmp_path, monkeypatch)
    adapter._select_engine_k(leader, [request])
    follower["decision_round"] += 1
    with pytest.raises(RuntimeError, match="out-of-order"):
        adapter._select_engine_k(follower, [request])
