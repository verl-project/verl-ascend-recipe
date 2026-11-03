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

"""One bounded, reusable CPU tensor per decision on the existing TP Gloo group."""

from __future__ import annotations

import json
import struct

PACKET_BYTES = 8192


def encode_packet(packet: tuple) -> bytes:
    payload = json.dumps(packet, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    if len(payload) > PACKET_BYTES - 4:
        raise ValueError("AdaptiveSpec TP control packet exceeds the bounded buffer")
    return payload


def decode_packet(wire: bytearray) -> tuple:
    size = struct.unpack_from("<I", wire)[0]
    if not 0 < size <= len(wire) - 4:
        raise RuntimeError("AdaptiveSpec TP control packet has an invalid length")
    packet = json.loads(wire[4 : 4 + size])
    if not isinstance(packet, list) or len(packet) != 3:
        raise RuntimeError("AdaptiveSpec TP control packet has an invalid shape")
    return tuple(packet)


class PackedTPControl:
    """Broadcast the leader decision through the existing TP CPU group."""

    def __init__(self, group):
        import torch
        from torch import distributed as dist

        self.rank = int(group.rank_in_group)
        self.size = int(group.world_size)
        self.group = group.cpu_group
        ranks = tuple(group.ranks)
        if (
            self.size < 2
            or not 0 <= self.rank < self.size
            or len(ranks) != self.size
            or len(set(ranks)) != self.size
            or dist.get_world_size(self.group) != self.size
            or dist.get_rank(self.group) != self.rank
            or str(dist.get_backend(self.group)).lower() != "gloo"
        ):
            raise RuntimeError("AdaptiveSpec packed control requires the existing matching TP CPU Gloo group")
        self.source_rank = int(ranks[0])
        self.dist = dist
        self.wire = bytearray(PACKET_BYTES)
        self.tensor = torch.frombuffer(self.wire, dtype=torch.uint8)

    def broadcast(self, packet):
        if self.rank == 0:
            payload = encode_packet(packet)
            struct.pack_into("<I", self.wire, 0, len(payload))
            self.wire[4 : 4 + len(payload)] = payload
        elif packet is not None:
            raise RuntimeError("Only the TP leader can publish a control packet")
        self.dist.broadcast(self.tensor, src=self.source_rank, group=self.group)
        return decode_packet(self.wire)
