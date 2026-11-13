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

"""Adaptive speculative-depth control for grouped RL rollouts."""

from .config import AdaptiveSpecConfig
from .controller import AdaptiveSpecController, EngineDecision, RequestContext
from .evidence import AcceptanceEvidence, group_key, sampling_fingerprint
from .exchange import CrossEngineEvidenceExchange
from .phase_map import PhaseCell, PhaseMap
from .reconcile import HistoricalEngineReconciler
from .scheduler import ElasticInterleaveRouter

__all__ = [
    "AcceptanceEvidence",
    "AdaptiveSpecConfig",
    "AdaptiveSpecController",
    "CrossEngineEvidenceExchange",
    "EngineDecision",
    "HistoricalEngineReconciler",
    "ElasticInterleaveRouter",
    "PhaseCell",
    "PhaseMap",
    "RequestContext",
    "group_key",
    "sampling_fingerprint",
]
