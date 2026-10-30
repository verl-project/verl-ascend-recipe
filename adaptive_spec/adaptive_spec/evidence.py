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

"""Bounded same-group acceptance evidence with right censoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_EXCLUDED_SAMPLING_KEYS = frozenset({"seed", "request_id"})


def sampling_fingerprint(params: Mapping[str, Any]) -> str:
    """Fingerprint sampling semantics while deliberately excluding the seed."""

    canonical = {str(key): params[key] for key in sorted(params) if str(key) not in _EXCLUDED_SAMPLING_KEYS}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def group_key(
    prompt_token_ids: Sequence[int],
    *,
    model_fingerprint: str,
    sampling_params: Mapping[str, Any],
    rollout_step: int,
    phase_id: str,
) -> str:
    """Build an exact same-prompt key scoped to one step and phase."""

    payload = {
        "prompt_token_ids": [int(value) for value in prompt_token_ids],
        "model_fingerprint": str(model_fingerprint),
        "sampling_fingerprint": sampling_fingerprint(sampling_params),
        "rollout_step": int(rollout_step),
        "phase_id": str(phase_id),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class AcceptanceObservation:
    active_k: int
    accepted_draft_tokens: int

    def __post_init__(self) -> None:
        if self.active_k < 1:
            raise ValueError("active_k must be positive")
        if not 0 <= self.accepted_draft_tokens <= self.active_k:
            raise ValueError("accepted_draft_tokens must be in [0, active_k]")


@dataclass
class _ResponseEvidence:
    exposures: list[float] = field(default_factory=list)
    successes: list[float] = field(default_factory=list)
    block_count: int = 0


@dataclass(frozen=True)
class ResponseSummary:
    exposures: tuple[float, ...]
    successes: tuple[float, ...]
    block_count: int


@dataclass(frozen=True)
class Posterior:
    survival: tuple[float, ...]
    response_count: int
    block_count: int
    exposed_by_depth: tuple[float, ...]


class AcceptanceEvidence:
    """Keep bounded evidence for exact same-prompt response groups.

    A block generated with depth K exposes only positions 1..K. Full
    acceptance at K therefore says nothing about K+1, which is the right
    censoring rule required by heterogeneous draft depths.
    """

    def __init__(
        self,
        *,
        max_depth: int,
        window_responses: int = 16,
        minimum_responses: int = 1,
        minimum_live_blocks: int = 64,
        prior_strength: float = 8.0,
        response_decay: float = 1.0,
    ) -> None:
        if max_depth < 1 or window_responses < 1 or minimum_responses < 0:
            raise ValueError("invalid evidence bounds")
        if minimum_live_blocks < 1 or not math.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("invalid evidence calibration")
        if not 0 < response_decay <= 1:
            raise ValueError("response_decay must be in (0, 1]")
        self.max_depth = int(max_depth)
        self.window_responses = int(window_responses)
        self.minimum_responses = int(minimum_responses)
        self.minimum_live_blocks = int(minimum_live_blocks)
        self.prior_strength = float(prior_strength)
        self.response_decay = float(response_decay)
        self._live: dict[str, OrderedDict[str, _ResponseEvidence]] = {}
        self._committed: dict[str, deque[tuple[str, ResponseSummary]]] = {}

    def stage(
        self,
        key: str,
        response_id: str,
        *,
        active_k: int,
        accepted_draft_tokens: int,
    ) -> None:
        observation = AcceptanceObservation(int(active_k), int(accepted_draft_tokens))
        item = self._live.setdefault(str(key), OrderedDict()).setdefault(str(response_id), _ResponseEvidence())
        if not item.exposures:
            item.exposures = [0.0] * self.max_depth
            item.successes = [0.0] * self.max_depth
        exposed = min(observation.active_k, self.max_depth)
        accepted = min(observation.accepted_draft_tokens, exposed)
        for index in range(exposed):
            item.exposures[index] += 1.0
        for index in range(accepted):
            item.successes[index] += 1.0
        item.block_count += 1

    def _summarize(self, item: _ResponseEvidence) -> ResponseSummary:
        return ResponseSummary(tuple(item.exposures), tuple(item.successes), item.block_count)

    def _history(self, key: str) -> deque[tuple[str, ResponseSummary]]:
        return self._committed.setdefault(str(key), deque(maxlen=self.window_responses))

    def commit(self, key: str, response_id: str) -> bool:
        key = str(key)
        response_id = str(response_id)
        live = self._live.get(key)
        if not live or response_id not in live:
            return False
        item = live.pop(response_id)
        if not live:
            self._live.pop(key, None)
        summary = self._summarize(item)
        return self.commit_summary(
            key,
            response_id,
            exposures=summary.exposures,
            successes=summary.successes,
            block_count=summary.block_count,
        )

    def commit_summary(
        self,
        key: str,
        response_id: str,
        *,
        exposures: Sequence[float],
        successes: Sequence[float],
        block_count: int,
    ) -> bool:
        key = str(key)
        response_id = str(response_id)
        if len(exposures) != self.max_depth or len(successes) != self.max_depth:
            raise ValueError("response summary depth does not match evidence")
        exposed = tuple(float(value) for value in exposures)
        accepted = tuple(float(value) for value in successes)
        if int(block_count) < 0:
            raise ValueError("response summary block_count must be non-negative")
        if any(
            not math.isfinite(exposure)
            or not math.isfinite(success)
            or exposure < 0
            or success < 0
            or success > exposure
            for exposure, success in zip(exposed, accepted, strict=False)
        ):
            raise ValueError("invalid response summary counts")
        history = self._history(key)
        if any(existing_id == response_id for existing_id, _ in history):
            return False
        history.append((response_id, ResponseSummary(exposed, accepted, int(block_count))))
        return True

    def response_summary(self, key: str, response_id: str) -> ResponseSummary:
        for existing_id, summary in reversed(self._committed.get(str(key), ())):
            if existing_id == str(response_id):
                return summary
        raise KeyError(f"committed response summary is unavailable: {response_id}")

    def discard(self, key: str, response_id: str) -> None:
        live = self._live.get(str(key))
        if live is None:
            return
        live.pop(str(response_id), None)
        if not live:
            self._live.pop(str(key), None)

    def posterior(
        self,
        key: str,
        prior_survival: Sequence[float],
        *,
        exclude_response_id: str | None = None,
    ) -> Posterior:
        if len(prior_survival) < self.max_depth:
            raise ValueError("prior does not cover max_depth")
        key = str(key)
        responses: list[tuple[ResponseSummary, float]] = []
        committed = list(self._committed.get(key, ()))
        for age, (response_id, summary) in enumerate(reversed(committed)):
            if response_id != exclude_response_id:
                responses.append((summary, self.response_decay**age))
        for response_id, item in self._live.get(key, {}).items():
            if response_id != exclude_response_id and item.block_count >= self.minimum_live_blocks:
                responses.append((self._summarize(item), 1.0))

        exposures = [0.0] * self.max_depth
        successes = [0.0] * self.max_depth
        block_count = 0
        for summary, weight in responses:
            block_count += summary.block_count
            for index in range(self.max_depth):
                exposures[index] += summary.exposures[index] * weight
                successes[index] += summary.successes[index] * weight

        posterior = []
        for index in range(self.max_depth):
            prior = float(prior_survival[index])
            denominator = self.prior_strength + exposures[index]
            value = prior if denominator == 0 else (self.prior_strength * prior + successes[index]) / denominator
            posterior.append(min(1.0, max(0.0, value)))

        for index in range(1, len(posterior)):
            posterior[index] = min(posterior[index], posterior[index - 1])
        return Posterior(
            survival=tuple(posterior),
            response_count=len(responses),
            block_count=block_count,
            exposed_by_depth=tuple(exposures),
        )

    def reset_step(self) -> None:
        """Drop all request evidence at a training-step boundary."""

        self._live.clear()
        self._committed.clear()
