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

"""Request scoring and Engine-level physical-K reconciliation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def request_k(
    allowed_ks: Sequence[int],
    survival: Sequence[float],
    latency_ns_by_k: Mapping[int, float],
    *,
    fallback_k: int | None = None,
    minimum_score_gain: float = 0.01,
) -> int:
    """Maximize expected verified tokens per measured iteration cost."""
    candidates = sorted({int(value) for value in allowed_ks})
    if not candidates:
        raise ValueError("allowed_ks cannot be empty")

    def score(k: int) -> float:
        if k not in latency_ns_by_k or latency_ns_by_k[k] <= 0:
            raise ValueError(f"missing positive latency for K={k}")
        return (1.0 + sum(float(value) for value in survival[:k])) / float(latency_ns_by_k[k])

    selected = max(candidates, key=lambda k: (score(k), -k))
    if fallback_k is not None:
        if fallback_k not in candidates or minimum_score_gain < 0:
            raise ValueError("invalid fallback or score gain")
        if score(selected) < score(fallback_k) * (1.0 + minimum_score_gain):
            return fallback_k
    return selected


def lower_median(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot reconcile an empty request set")
    ordered = sorted(int(value) for value in values)
    return ordered[(len(ordered) - 1) // 2]


class HistoricalEngineReconciler:
    """Historical direction confirmation and dwell on the global candidate ladder."""

    def __init__(self, candidate_ks, *, confirmations=2, minimum_dwell_rounds=64, one_rung_per_switch=True):
        self.candidate_ks = tuple(sorted(set(int(k) for k in candidate_ks)))
        if not self.candidate_ks or self.candidate_ks[0] < 1:
            raise ValueError("positive backend candidate depths required")
        if confirmations < 1 or minimum_dwell_rounds < 0:
            raise ValueError("invalid stabilization settings")
        self.confirmations = int(confirmations)
        self.minimum_dwell_rounds = int(minimum_dwell_rounds)
        self.one_rung_per_switch = bool(one_rung_per_switch)
        self.current_k = None
        self.pending_direction = 0
        self.pending_count = 0
        self.round_index = 0
        self.last_switch_round = None

    def clear_pending(self):
        self.pending_direction = 0
        self.pending_count = 0

    def select(self, recommendations, *, fallback_k):
        allowed = self.candidate_ks
        target = lower_median(recommendations) if recommendations else int(fallback_k)
        if not allowed or target not in allowed:
            raise ValueError("target is outside the declared candidate set")
        if self.current_k is None:
            self.current_k = target
            self.clear_pending()
            self.round_index += 1
            return target
        if self.current_k not in self.candidate_ks:
            raise ValueError("current K is outside the backend candidate set")
        direction = (target > self.current_k) - (target < self.current_k)
        if direction == 0:
            self.pending_count = 0
        elif direction == self.pending_direction:
            self.pending_count += 1
        else:
            self.pending_count = 1
        self.pending_direction = direction
        ready = self.last_switch_round is None or self.round_index - self.last_switch_round >= self.minimum_dwell_rounds
        if direction and self.pending_count >= self.confirmations and ready:
            self.current_k = allowed[allowed.index(self.current_k) + direction] if self.one_rung_per_switch else target
            self.last_switch_round = self.round_index
            self.pending_count = 0
        self.round_index += 1
        return self.current_k

    def reset(self):
        self.current_k = None
        self.clear_pending()
        self.round_index = 0
        self.last_switch_round = None
