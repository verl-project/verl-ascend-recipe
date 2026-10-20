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
"""VLLMProgressCheckPoint: per-request progress capture + resume bookkeeping.

The checkpoint is constructed in ``FullyLLMServerClient.generate`` (driver side)
and serialized via ``ProgressContext`` to ``vLLMHttpServer.generate`` (rollout
replica process) where ``ingest`` is called inside the ``async for`` streaming
loop. Flushes are fire-and-forget ``store.save.remote(payload)`` calls; the
store actor serializes writes per ``recovery_id``.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.workers.rollout.fault_tolerance.progress.types import (
    CheckPointPayLoad,
    LoadResult,
    ModelVersionPolicy,
    ResumeOutcome,
    ResumeResult,
)

logger = logging.getLogger(__name__)


@dataclass
class VLLMProgressCheckPoint:
    """Single-request progress checkpoint.

    Identity / prompt-context fields are immutable for the lifetime of one
    ``attempt_id``. Cumulative state grows monotonically with ``ingest``.
    On fault, ``create_or_resume`` allocates a new ``attempt_id`` and seeds
    ``cumulative_*`` from the latest persisted attempt (RESUMED) or empty
    (DEGRADED_FRESH).
    """

    # ----- identity (immutable) -----
    run_id: str
    recovery_id: str
    attempt_id: int
    # ----- prompt context (immutable) -----
    prompt_token_ids: list[int]
    sampling_params: dict
    model_weight_version: Optional[str]
    # ----- resume bookkeeping (immutable) -----
    inherited_prefix_len: int
    base_attempt_id: int
    # ----- cumulative state (mutable, grows with ingest) -----
    cumulative_token_ids: list[int] = field(default_factory=list)
    cumulative_log_probs: Optional[list[float]] = None
    cumulative_routed_experts: Optional[Any] = None
    num_preempted: int = 0
    finished: bool = False
    finished_reason: Optional[str] = None
    # ----- flush config & bookkeeping -----
    flush_token_interval: int = 64
    _tokens_since_last_flush: int = 0
    _store_ref: Any = None  # RolloutProgressStoreActor ActorHandle
    # ----- metadata -----
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = 1

    # ------------------------------------------------------------------ store
    def attach_store(self, store: Any) -> None:
        """Inject the persistence target; called by create_or_resume."""
        self._store_ref = store

    # ------------------------------------------------------------------ ingest
    def ingest(self, request_output: Any) -> None:
        """Consume a vLLM ``RequestOutput`` from the streaming loop.

        vLLM emits cumulative ``token_ids`` per output chunk, so we replace
        (not extend) the cumulative list and compute the delta for flush
        accounting. Idempotent: repeated ingestion of the same chunk is a
        no-op because ``len(cumulative)`` already covers it.
        """
        output = request_output.outputs[0]
        all_ids: list[int] = list(output.token_ids)

        inherited = self.cumulative_token_ids[: self.inherited_prefix_len]
        prev_new = len(self.cumulative_token_ids) - self.inherited_prefix_len
        new_count = len(all_ids) - prev_new

        # vLLM shouldn't shrink token_ids, but be defensive.
        if new_count <= 0:
            return

        self.cumulative_token_ids = inherited + all_ids

        # logprobs: list[dict[token_id, Logprob]]; rebuild cumulative list.
        logprobs = getattr(output, "logprobs", None)
        if logprobs is not None and len(logprobs) > 0:
            rebuilt: list[float] = []
            for i, lp_dict in enumerate(logprobs):
                if i >= len(self.cumulative_token_ids):
                    break
                tid = all_ids[i]
                if lp_dict and tid in lp_dict:
                    rebuilt.append(float(lp_dict[tid].logprob))
                else:
                    rebuilt.append(0.0)
            if self.cumulative_log_probs is not None:
                self.cumulative_log_probs = list(self.cumulative_log_probs[: self.inherited_prefix_len]) + rebuilt
            else:
                self.cumulative_log_probs = rebuilt

        # routed_experts: cumulative per-token tensor (optional).
        routed = getattr(output, "routed_experts", None)
        if routed is not None:
            if self.cumulative_routed_experts is not None and self.inherited_prefix_len > 0:
                import torch

                inherited_routed = self.cumulative_routed_experts[: self.inherited_prefix_len]
                self.cumulative_routed_experts = torch.cat([inherited_routed, routed], dim=0)
            else:
                self.cumulative_routed_experts = routed

        self.finished = bool(getattr(output, "finished", False))
        self.finished_reason = getattr(output, "finish_reason", None)
        np_obj = getattr(output, "num_preempted", None)
        if np_obj is not None:
            self.num_preempted = int(np_obj)

        self.updated_at = time.time()
        self._tokens_since_last_flush += new_count

        if 0 < self.flush_token_interval <= self._tokens_since_last_flush:
            self._maybe_flush()

    # ------------------------------------------------------------------ flush
    def _maybe_flush(self, force: bool = False) -> None:
        if self._store_ref is None:
            return
        if not force and self._tokens_since_last_flush < self.flush_token_interval:
            return
        payload = self.to_payload()
        try:
            # Fire-and-forget: do not await. The store actor serializes writes
            # per recovery_id; a dropped ref does not cancel the queued call.
            self._store_ref.save.remote(payload)
        except Exception as e:  # actor dead / serialization error
            logger.warning(
                "VLLMProgressCheckPoint flush failed (run=%s rid=%s attempt=%d): %s",
                self.run_id,
                self.recovery_id,
                self.attempt_id,
                e,
            )
        self._tokens_since_last_flush = 0

    # ------------------------------------------------------------------ state
    def is_terminal(self) -> bool:
        return self.finished

    def resume_prefix_token_ids(self) -> list[int]:
        """Return ``prompt_token_ids + cumulative_token_ids``."""
        return list(self.prompt_token_ids) + list(self.cumulative_token_ids)

    def remaining_max_tokens(self) -> int:
        """Return ``max(0, original_max_tokens - len(cumulative_token_ids))``.

        Fresh scenario (cumulative empty) returns the original max_tokens —
        i.e. degrades to a plain prompt retry.
        """
        original = self._original_max_tokens()
        if original is None:
            # No max_tokens in sampling_params: unbounded; return large number.
            return 2**31 - 1
        return max(0, original - len(self.cumulative_token_ids))

    def _original_max_tokens(self) -> Optional[int]:
        if not self.sampling_params:
            return None
        for key in ("max_tokens", "max_new_tokens"):
            if key in self.sampling_params:
                try:
                    return int(self.sampling_params[key])
                except (TypeError, ValueError):
                    return None
        return None

    # ------------------------------------------------------------------ payload
    def to_payload(self) -> CheckPointPayLoad:
        return CheckPointPayLoad(
            run_id=self.run_id,
            recovery_id=self.recovery_id,
            attempt_id=self.attempt_id,
            prompt_token_ids=list(self.prompt_token_ids),
            sampling_params=copy.deepcopy(self.sampling_params),
            model_weight_version=self.model_weight_version,
            inherited_prefix_len=self.inherited_prefix_len,
            cumulative_token_ids=list(self.cumulative_token_ids),
            cumulative_log_probs=list(self.cumulative_log_probs) if self.cumulative_log_probs is not None else None,
            cumulative_routed_experts=self.cumulative_routed_experts,
            num_preempted=self.num_preempted,
            finished=self.finished,
            finished_reason=self.finished_reason,
            created_at=self.created_at,
            updated_at=self.updated_at,
            schema_version=self.schema_version,
        )

    @classmethod
    def from_payload(
        cls,
        payload: CheckPointPayLoad,
        flush_token_interval: int,
    ) -> VLLMProgressCheckPoint:
        """Reconstruct a checkpoint from a payload (internal helper).

        ``create_or_resume`` is the public entry; it mixes caller-supplied
        prompt/sampling with payload cumulative state.
        """
        return cls(
            run_id=payload.run_id,
            recovery_id=payload.recovery_id,
            attempt_id=payload.attempt_id,
            prompt_token_ids=list(payload.prompt_token_ids),
            sampling_params=copy.deepcopy(payload.sampling_params),
            model_weight_version=payload.model_weight_version,
            inherited_prefix_len=payload.inherited_prefix_len,
            base_attempt_id=payload.attempt_id,
            cumulative_token_ids=list(payload.cumulative_token_ids),
            cumulative_log_probs=list(payload.cumulative_log_probs)
            if payload.cumulative_log_probs is not None
            else None,
            cumulative_routed_experts=payload.cumulative_routed_experts,
            num_preempted=payload.num_preempted,
            finished=payload.finished,
            finished_reason=payload.finished_reason,
            flush_token_interval=flush_token_interval,
            created_at=payload.created_at,
            updated_at=payload.updated_at,
            schema_version=payload.schema_version,
        )

    def validate_integrity(self) -> None:
        """Lightweight in-memory integrity check; raises on inconsistency."""
        if self.inherited_prefix_len > len(self.cumulative_token_ids):
            raise ValueError(
                f"inherited_prefix_len {self.inherited_prefix_len} > cumulative {len(self.cumulative_token_ids)}"
            )
        if self.cumulative_log_probs is not None and len(self.cumulative_log_probs) != len(self.cumulative_token_ids):
            raise ValueError(
                f"log_probs len {len(self.cumulative_log_probs)} != tokens len {len(self.cumulative_token_ids)}"
            )

    # ------------------------------------------------------------------ create_or_resume
    @classmethod
    async def create_or_resume(
        cls,
        store: Any,
        run_id: str,
        recovery_id: str,
        prompt_token_ids: list[int],
        sampling_params: dict,
        model_weight_version: Optional[str],
        flush_token_interval: int,
        model_version_policy: ModelVersionPolicy,
    ) -> ResumeResult:
        """Entry point called by ``FullyLLMServerClient.generate``.

        1. ``store.load_latest(run_id, recovery_id, model_weight_version, policy)``
        2. On hit (payload not None): new ``attempt_id = payload.attempt_id + 1``,
           seed cumulative from payload, ``mark_superseded`` old attempt,
           return ``RESUMED``.
        3. On miss (failure not None): new ``attempt_id = prev+1 or 1``, empty
           cumulative, return ``DEGRADED_FRESH`` — ``resume_prefix_token_ids``
           equals the original prompt, ``remaining_max_tokens`` equals the
           original max_tokens, i.e. degrades to a plain prompt retry.
        """
        result: LoadResult = await store.load_latest.remote(
            run_id=run_id,
            recovery_id=recovery_id,
            requested_model_version=model_weight_version,
            policy=model_version_policy,
        )

        if result.payload is not None:
            payload = result.payload
            # Optional integrity: verify the caller's prompt matches the
            # persisted attempt (cheap sha8 compare would require disk-side
            # hash; here we trust load_latest's id/alignment checks).
            cp = cls(
                run_id=run_id,
                recovery_id=recovery_id,
                attempt_id=payload.attempt_id + 1,
                prompt_token_ids=list(prompt_token_ids),
                sampling_params=copy.deepcopy(sampling_params),
                model_weight_version=model_weight_version,
                inherited_prefix_len=len(payload.cumulative_token_ids),
                base_attempt_id=payload.attempt_id,
                cumulative_token_ids=list(payload.cumulative_token_ids),
                cumulative_log_probs=list(payload.cumulative_log_probs)
                if payload.cumulative_log_probs is not None
                else None,
                cumulative_routed_experts=payload.cumulative_routed_experts,
                num_preempted=payload.num_preempted,
                finished=False,
                finished_reason=None,
                flush_token_interval=flush_token_interval,
            )
            cp.attach_store(store)
            try:
                await store.mark_superseded.remote(
                    run_id=run_id,
                    recovery_id=recovery_id,
                    attempt_id=payload.attempt_id,
                )
            except Exception as e:
                logger.warning(
                    "mark_superseded failed (run=%s rid=%s attempt=%d): %s",
                    run_id,
                    recovery_id,
                    payload.attempt_id,
                    e,
                )
            return ResumeResult(
                checkpoint=cp,
                outcome=ResumeOutcome.RESUMED,
                attempt_id=cp.attempt_id,
                inherited_prefix_len=cp.inherited_prefix_len,
            )

        # DEGRADED_FRESH
        prev_aid = result.prev_attempt_id
        new_aid = (prev_aid + 1) if prev_aid is not None else 1
        cp = cls(
            run_id=run_id,
            recovery_id=recovery_id,
            attempt_id=new_aid,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=copy.deepcopy(sampling_params),
            model_weight_version=model_weight_version,
            inherited_prefix_len=0,
            base_attempt_id=new_aid,
            cumulative_token_ids=[],
            cumulative_log_probs=None,
            cumulative_routed_experts=None,
            num_preempted=0,
            finished=False,
            finished_reason=None,
            flush_token_interval=flush_token_interval,
        )
        cp.attach_store(store)
        logger.info(
            "[progress] run=%s rid=%s attempt=%d DEGRADED_FRESH(prev=%s) reason=%s",
            run_id,
            recovery_id,
            new_aid,
            prev_aid,
            result.detail,
        )
        return ResumeResult(
            checkpoint=cp,
            outcome=ResumeOutcome.DEGRADED_FRESH,
            attempt_id=new_aid,
            inherited_prefix_len=0,
            failure_detail=result.detail,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def prompt_sha8(prompt_token_ids: list[int]) -> str:
        return hashlib.sha256(",".join(str(t) for t in prompt_token_ids).encode()).hexdigest()[:8]
