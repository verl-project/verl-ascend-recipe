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

"""Phase prior + same-group posterior + Engine reconciliation."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .evidence import AcceptanceEvidence, Posterior, group_key
from .exchange import CrossEngineEvidenceExchange
from .phase_map import PhaseCell, PhaseMap
from .reconcile import HistoricalEngineReconciler, request_k


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: Mapping[str, Any]
    context_length: int
    rollout_step: int = 0


@dataclass(frozen=True)
class RequestDecision:
    request_id: str
    phase_id: str
    selected_k: int
    fallback_k: int
    used_posterior: bool
    evidence_responses: int
    evidence_blocks: int


@dataclass(frozen=True)
class EngineDecision:
    physical_k: int
    request_decisions: tuple[RequestDecision, ...]
    fallback_reason: str | None = None


class AdaptiveSpecController:
    """Reference implementation of the adaptive-spec decision loop."""

    def __init__(
        self,
        phase_map: PhaseMap,
        *,
        model_fingerprint: str,
        topology_fingerprint: str | None = None,
        evidence: AcceptanceEvidence | None = None,
        exchange: CrossEngineEvidenceExchange | None = None,
        reconciler: HistoricalEngineReconciler | None = None,
        minimum_score_gain: float = 0.01,
    ) -> None:
        if model_fingerprint != phase_map.model_fingerprint:
            raise ValueError("runtime model does not match the phase map")
        if topology_fingerprint is not None and topology_fingerprint != phase_map.topology_fingerprint:
            raise ValueError("runtime topology does not match the phase map")
        candidates = sorted({k for cell in phase_map.cells for k in cell.allowed_ks})
        if not math.isfinite(minimum_score_gain) or minimum_score_gain < 0:
            raise ValueError("minimum_score_gain cannot be negative")
        if any(len(cell.prior_survival) < max(candidates) for cell in phase_map.cells):
            raise ValueError("every phase prior must cover global evidence depth")
        self.minimum_score_gain = float(minimum_score_gain)
        self.phase_map = phase_map
        self.model_fingerprint = str(model_fingerprint)
        self.evidence = evidence or AcceptanceEvidence(max_depth=max(candidates))
        self.exchange = exchange
        self.reconciler = reconciler or HistoricalEngineReconciler(candidates)
        self._request_keys: dict[str, set[str]] = {}
        self._identity_cache = {}
        self._key_cache = {}

    def _key(self, request: RequestContext, phase_id: str) -> str:
        sampling = {k: v for k, v in request.sampling_params.items() if k not in {"seed", "request_id"}}
        identity = (request.prompt_token_ids, sampling, request.rollout_step)
        previous = self._identity_cache.get(request.request_id)
        if previous is None:
            self._identity_cache[request.request_id] = deepcopy(identity)
        elif previous != identity:
            raise ValueError("request identity changed before finish/discard")
        keys = self._key_cache.setdefault(request.request_id, {})
        if phase_id not in keys:
            keys[phase_id] = group_key(
                request.prompt_token_ids,
                model_fingerprint=self.model_fingerprint,
                sampling_params=sampling,
                rollout_step=request.rollout_step,
                phase_id=phase_id,
            )
        self._request_keys.setdefault(request.request_id, set()).add(keys[phase_id])
        return keys[phase_id]

    def observe_round(
        self, request: RequestContext, *, active_batch_size: int, active_k: int, accepted_draft_tokens: int
    ) -> None:
        cell = self.phase_map.cell_for(active_batch_size, request.context_length)
        self.evidence.stage(
            self._key(request, cell.phase_id),
            request.request_id,
            active_k=active_k,
            accepted_draft_tokens=accepted_draft_tokens,
        )

    def finish_request(self, request: RequestContext) -> bool:
        self._identity_cache.pop(request.request_id, None)
        self._key_cache.pop(request.request_id, None)
        keys = self._request_keys.pop(request.request_id, set())
        committed = False
        for key in keys:
            accepted = self.evidence.commit(key, request.request_id)
            committed = accepted or committed
            if accepted and self.exchange is not None:
                self.exchange.publish(key, request.request_id, self.evidence.response_summary(key, request.request_id))
        return committed

    def discard_request(self, request: RequestContext) -> None:
        self._identity_cache.pop(request.request_id, None)
        self._key_cache.pop(request.request_id, None)
        keys = self._request_keys.pop(request.request_id, set())
        for key in keys:
            self.evidence.discard(key, request.request_id)

    def _request_decision(
        self, request: RequestContext, *, active_batch_size: int, phase_override: PhaseCell | None = None
    ) -> tuple[RequestDecision, PhaseCell]:
        cell = phase_override or self.phase_map.cell_for(active_batch_size, request.context_length)
        posterior: Posterior = self.evidence.posterior(
            self._key(request, cell.phase_id), cell.prior_survival, exclude_response_id=request.request_id
        )
        use_posterior = posterior.response_count >= self.evidence.minimum_responses
        selected = (
            request_k(
                cell.allowed_ks,
                posterior.survival,
                cell.iteration_latency_ns_by_k,
                fallback_k=cell.fallback_k,
                minimum_score_gain=self.minimum_score_gain,
            )
            if use_posterior
            else cell.fallback_k
        )
        return (
            RequestDecision(
                request_id=request.request_id,
                phase_id=cell.phase_id,
                selected_k=selected,
                fallback_k=cell.fallback_k,
                used_posterior=use_posterior,
                evidence_responses=posterior.response_count,
                evidence_blocks=posterior.block_count,
            ),
            cell,
        )

    def decide_engine(self, requests: Sequence[RequestContext]) -> EngineDecision:
        if self.exchange is not None:
            self.exchange.maybe_poll(self.evidence)
        if not requests:
            if self.reconciler.current_k is None:
                raise ValueError("an empty Engine has no initial K")
            return EngineDecision(self.reconciler.current_k, (), "empty_engine")
        if len({r.rollout_step for r in requests}) != 1:
            raise ValueError("one Engine batch cannot mix rollout steps")
        lengths = sorted(r.context_length for r in requests if r.context_length > 0)
        length = lengths[round(0.9 * (len(lengths) - 1))] if lengths else 0
        cell = self.phase_map.cell_for(len(requests), length)
        pairs = [self._request_decision(r, active_batch_size=len(requests), phase_override=cell) for r in requests]
        decisions = tuple(item[0] for item in pairs)
        physical_k = self.reconciler.select([d.selected_k for d in decisions], fallback_k=cell.fallback_k)
        return EngineDecision(physical_k, decisions)

    def reset_step(self) -> None:
        self.evidence.reset_step()
        self._request_keys.clear()
        self._identity_cache.clear()
        self._key_cache.clear()
        self.reconciler.reset()
