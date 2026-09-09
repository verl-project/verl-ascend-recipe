# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""L4 partial-batch helper.

Spec §每层合同 row L4:
    | N prompts | ≤N DataProto | BatchMostlyFailed | gather(return_exceptions) + min_ok_ratio |

This module exposes a pure helper that partitions `asyncio.gather(..., return_exceptions=True)`
results into ok outputs and failures, applying the `min_ok_ratio` policy.

Kept Ray-free / DataProto-free so it can be unit-tested in isolation.
"""

from __future__ import annotations

from typing import Any

from verl.workers.rollout.fault_tolerance.exceptions import (
    AllServersFailed,
    BatchMostlyFailed,
    ServerUnavailable,
    is_transient_fault,
)

# Prompt-level fault exceptions that L4 should absorb into the partial batch.
# - AllServersFailed: L3 retry budget exhausted (L3 already wrapped L2 faults)
# - ServerUnavailable: L2 fault when no L3 wrapper sits between L2 and L4
#   (e.g. one_step_off_policy uses plain LLMServerClient, not FullyLLMServerClient).
_PROMPT_FAULT_TYPES: tuple = (AllServersFailed, ServerUnavailable)


def filter_partial_batch(
    results: list[Any],
    min_ok_ratio: float,
) -> tuple[list[Any], list[int]]:
    """Partition gather results into (ok_outputs, ok_indices).

    Rules:
      * Prompt-level fault exceptions (`AllServersFailed`, `ServerUnavailable`)
        are counted toward the failure ratio.
      * Any other `BaseException` is re-raised immediately — programming errors
        and unexpected runtime failures must surface, not be silently absorbed.
      * If `len(ok)/len(results) < min_ok_ratio`, raise `BatchMostlyFailed`.
      * Empty batch (`len(results)==0`) is vacuously ok; returns `([], [])`.

    Returns:
        (ok_outputs, ok_indices) — outputs that succeeded, with their indices into `results`.
    """
    ok_outputs: list[Any] = []
    ok_indices: list[int] = []
    fault_count = 0
    for i, r in enumerate(results):
        if isinstance(r, _PROMPT_FAULT_TYPES):
            fault_count += 1
            continue
        if is_transient_fault(r):
            fault_count += 1
            continue
        if isinstance(r, BaseException):
            raise r
        ok_outputs.append(r)
        ok_indices.append(i)

    total = len(results)
    if total == 0:
        return ok_outputs, ok_indices

    ok_ratio = len(ok_outputs) / total
    if ok_ratio < min_ok_ratio:
        raise BatchMostlyFailed(
            f"batch ok ratio {ok_ratio:.3f} below threshold {min_ok_ratio:.3f} "
            f"({len(ok_outputs)}/{total} ok, {fault_count} faulted)"
        )
    return ok_outputs, ok_indices
