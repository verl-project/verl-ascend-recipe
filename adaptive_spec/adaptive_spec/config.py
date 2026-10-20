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

"""Runtime configuration with explicit fail-closed fallbacks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .controller import AdaptiveSpecController
from .evidence import AcceptanceEvidence
from .exchange import CrossEngineEvidenceExchange
from .phase_map import PhaseMap
from .reconcile import HistoricalEngineReconciler


@dataclass(frozen=True)
class AdaptiveSpecConfig:
    enabled: bool
    phase_map_path: str | None
    model_fingerprint: str
    topology_fingerprint: str = ""
    require_eager: bool = True
    window_responses: int = 16
    minimum_responses: int = 1
    minimum_live_blocks: int = 64
    prior_strength: float = 8.0
    response_decay: float = 1.0
    confirmations: int = 2
    minimum_dwell_rounds: int = 64
    one_rung_per_switch: bool = True
    minimum_score_gain: float = 0.01
    evidence_exchange_directory: str | None = None
    evidence_exchange_poll_decisions: int = 64

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AdaptiveSpecConfig:
        if int(raw.get("schema_version", 1)) != 1:
            raise ValueError("unsupported adaptive-spec config schema")
        if raw.get("aggregation_mode", "engine_p90") != "engine_p90":
            raise ValueError("only the historical engine_p90 algorithm is supported")
        if not raw.get("require_eager", True):
            raise ValueError("dynamic MTP requires eager execution")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            phase_map_path=(str(raw["phase_map_path"]) if raw.get("phase_map_path") else None),
            model_fingerprint=str(raw.get("model_fingerprint", "")),
            topology_fingerprint=str(raw.get("topology_fingerprint", "")),
            require_eager=bool(raw.get("require_eager", True)),
            window_responses=int(raw.get("window_responses", 16)),
            minimum_responses=int(raw.get("minimum_responses", 1)),
            minimum_live_blocks=int(raw.get("minimum_live_blocks", 64)),
            prior_strength=float(raw.get("prior_strength", 8.0)),
            response_decay=float(raw.get("response_decay", 1.0)),
            confirmations=int(raw.get("confirmations", 2)),
            minimum_dwell_rounds=int(raw.get("minimum_dwell_rounds", 64)),
            one_rung_per_switch=bool(raw.get("one_rung_per_switch", True)),
            minimum_score_gain=float(raw.get("minimum_score_gain", 0.01)),
            evidence_exchange_directory=(
                str(raw["evidence_exchange_directory"]) if raw.get("evidence_exchange_directory") else None
            ),
            evidence_exchange_poll_decisions=int(raw.get("evidence_exchange_poll_decisions", 64)),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> AdaptiveSpecConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("adaptive-spec config must be a JSON object")
        return cls.from_dict(raw)

    def build_controller(self) -> AdaptiveSpecController | None:
        """Return None when the plugin must preserve the upstream path."""

        if not self.enabled or self.phase_map_path is None:
            return None
        if not self.model_fingerprint:
            raise ValueError("enabled adaptive-spec requires model_fingerprint")
        if not self.topology_fingerprint:
            raise ValueError("enabled adaptive-spec requires topology_fingerprint")
        phase_map = PhaseMap.from_path(self.phase_map_path)
        candidates = sorted({k for cell in phase_map.cells for k in cell.allowed_ks})
        if 11 in candidates:
            raise ValueError("K11 is excluded from the verified Ascend backend scope")
        return AdaptiveSpecController(
            phase_map,
            model_fingerprint=self.model_fingerprint,
            topology_fingerprint=self.topology_fingerprint,
            minimum_score_gain=self.minimum_score_gain,
            evidence=AcceptanceEvidence(
                max_depth=max(candidates),
                window_responses=self.window_responses,
                minimum_responses=self.minimum_responses,
                minimum_live_blocks=self.minimum_live_blocks,
                prior_strength=self.prior_strength,
                response_decay=self.response_decay,
            ),
            exchange=(
                CrossEngineEvidenceExchange(
                    self.evidence_exchange_directory,
                    poll_decisions=self.evidence_exchange_poll_decisions,
                )
                if self.evidence_exchange_directory
                else None
            ),
            reconciler=HistoricalEngineReconciler(
                candidates,
                confirmations=self.confirmations,
                minimum_dwell_rounds=self.minimum_dwell_rounds,
                one_rung_per_switch=self.one_rung_per_switch,
            ),
        )
