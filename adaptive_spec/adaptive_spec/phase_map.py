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

"""Offline phase-map schema and nearest-state lookup."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def _int_float_map(value: Mapping[str | int, Any], name: str) -> dict[int, float]:
    result = {int(key): float(item) for key, item in value.items()}
    if not result or any(key < 1 or not math.isfinite(item) or item <= 0 for key, item in result.items()):
        raise ValueError(f"{name} must contain positive K/cost pairs")
    return result


@dataclass(frozen=True)
class PhaseCell:
    """A measured deployment state.

    Survival values are indexed from draft depth one. They are priors, while
    iteration_latency_ns_by_k is the measured cost of one draft/verify
    iteration for each candidate K.
    """

    phase_id: str
    batch_size: int
    context_length: int
    allowed_ks: tuple[int, ...]
    iteration_latency_ns_by_k: Mapping[int, float]
    prior_survival: tuple[float, ...]
    fallback_k: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PhaseCell:
        allowed = tuple(sorted({int(value) for value in raw["allowed_ks"]}))
        costs = _int_float_map(raw["iteration_latency_ns_by_k"], "iteration_latency_ns_by_k")
        prior = tuple(float(value) for value in raw["prior_survival"])
        fallback = int(raw["fallback_k"])
        if not allowed or allowed[0] < 1:
            raise ValueError("allowed_ks must contain positive depths")
        if not set(allowed).issubset(costs):
            raise ValueError("latency grid must cover every allowed K")
        if fallback not in allowed:
            raise ValueError("fallback_k must be an allowed K")
        if len(prior) < max(allowed):
            raise ValueError("prior_survival must cover the deepest allowed K")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in prior):
            raise ValueError("prior survival probabilities must be in [0, 1]")
        if any(right > left for left, right in zip(prior, prior[1:], strict=False)):
            raise ValueError("prior survival probabilities must be non-increasing")
        return cls(
            phase_id=str(raw["phase_id"]),
            batch_size=max(1, int(raw["batch_size"])),
            context_length=max(1, int(raw["context_length"])),
            allowed_ks=allowed,
            iteration_latency_ns_by_k=costs,
            prior_survival=prior,
            fallback_k=fallback,
        )


class PhaseMap:
    """Frozen deployment-specific phase map."""

    def __init__(
        self,
        cells: Sequence[PhaseCell],
        *,
        model_fingerprint: str,
        topology_fingerprint: str,
        batch_scale: float = 8.0,
        log2_context_scale: float = 1.0,
    ) -> None:
        if not cells:
            raise ValueError("phase map requires at least one cell")
        if len({cell.phase_id for cell in cells}) != len(cells):
            raise ValueError("phase_id values must be unique")
        self.cells = tuple(cells)
        self.model_fingerprint = str(model_fingerprint)
        self.topology_fingerprint = str(topology_fingerprint)
        self.batch_scale = max(1.0, float(batch_scale))
        self.log2_context_scale = max(0.01, float(log2_context_scale))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PhaseMap:
        if int(raw.get("schema_version", 1)) != 1:
            raise ValueError("unsupported adaptive-spec phase-map schema")
        return cls(
            [PhaseCell.from_dict(cell) for cell in raw["cells"]],
            model_fingerprint=str(raw["model_fingerprint"]),
            topology_fingerprint=str(raw["topology_fingerprint"]),
            batch_scale=float(raw.get("batch_scale", 8.0)),
            log2_context_scale=float(raw.get("log2_context_scale", 1.0)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> PhaseMap:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("phase-map payload must be a JSON object")
        return cls.from_dict(raw)

    def cell_for(self, batch_size: int, context_length: int) -> PhaseCell:
        batch_size = max(1, int(batch_size))
        context_length = max(1, int(context_length))
        return min(
            self.cells,
            key=lambda cell: (
                abs(batch_size - cell.batch_size) / self.batch_scale
                + abs(math.log2(context_length) - math.log2(cell.context_length)) / self.log2_context_scale,
                cell.batch_size,
                cell.context_length,
                cell.phase_id,
            ),
        )
