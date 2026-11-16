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

import struct
import sys
from collections import deque
from types import SimpleNamespace

import pytest

from adaptive_spec.vllm_plugin import v0_23 as adapter
from adaptive_spec.vllm_plugin.tp_control import PACKET_BYTES, decode_packet, encode_packet


def states(monkeypatch):
    packets = deque()
    world = SimpleNamespace(rank=0, backend="gloo", broadcasts=0)

    def broadcast(tensor, *, src, group):
        assert src == 4 and group == "existing-tp-cpu"
        world.broadcasts += 1
        if world.rank == 0:
            packets.append(bytes(tensor))
        else:
            tensor[:] = packets.popleft()

    dist = SimpleNamespace(
        get_world_size=lambda group: 2,
        get_rank=lambda group: world.rank,
        get_backend=lambda group: world.backend,
        broadcast=broadcast,
    )
    torch = SimpleNamespace(distributed=dist, uint8="uint8", frombuffer=lambda b, dtype: memoryview(b))
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = []

    def decide(contexts):
        calls.append("leader")
        return SimpleNamespace(physical_k=9 if len(calls) > 1 else 5)

    workers = [
        dict(
            controller=SimpleNamespace(decide_engine=decide) if rank == 0 else None,
            tp_group=SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group="existing-tp-cpu", ranks=[4, 5]),
            decision_round=0,
            tp_control_transport="packed",
        )
        for rank in range(2)
    ]
    return workers, world, calls


def test_packed_decisions_use_one_collective_and_reuse_buffer(monkeypatch):
    workers, world, calls = states(monkeypatch)
    buffers = []
    for expected in [5, 9, 9]:
        for rank, worker in enumerate(workers):
            world.rank = rank
            assert adapter._select_engine_k(worker, []) == expected
        current = [id(worker["packed_tp_control"].tensor) for worker in workers]
        if buffers:
            assert current == buffers
        buffers = current
    assert calls == ["leader"] * 3 and world.broadcasts == 6
    assert [w["decision_round"] for w in workers] == [3, 3]


def test_long_leader_error_reaches_both_ranks_without_pickle(monkeypatch):
    workers, world, _ = states(monkeypatch)

    def fail(contexts):
        raise ValueError("bad policy " * 10000)

    workers[0]["controller"].decide_engine = fail
    for rank, worker in enumerate(workers):
        world.rank = rank
        with pytest.raises(RuntimeError, match="TP leader decision failed: ValueError"):
            adapter._select_engine_k(worker, [])
    assert world.broadcasts == 2


def test_packed_requires_existing_gloo_group_without_object_fallback(monkeypatch):
    workers, world, _ = states(monkeypatch)
    world.backend = "hccl"
    with pytest.raises(RuntimeError, match="matching TP CPU Gloo"):
        adapter._select_engine_k(workers[0], [])
    assert world.broadcasts == 0


def test_malformed_or_oversized_packets_are_rejected():
    with pytest.raises(ValueError, match="bounded"):
        encode_packet((0, None, "x" * PACKET_BYTES))
    wire = bytearray(PACKET_BYTES)
    with pytest.raises(RuntimeError, match="invalid length"):
        decode_packet(wire)
    payload = b'{"wrong": 1}'
    struct.pack_into("<I", wire, 0, len(payload))
    wire[4 : 4 + len(payload)] = payload
    with pytest.raises(RuntimeError, match="invalid shape"):
        decode_packet(wire)
