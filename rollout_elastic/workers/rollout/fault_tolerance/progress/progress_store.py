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
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import pickle
import shutil
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from verl.workers.rollout.fault_tolerance.progress.types import (
    CheckPointPayLoad,
    GCStats,
    LoadFailure,
    LoadResult,
    ModelVersionPolicy,
    ProgressConfig,
)

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
TOKENS_BIN = "tokens.bin"
LOGPROBS_BIN = "logprobs.bin"
ROUTING_BIN = "routing.bin"
PREFLIGHTS_PROBE = "_preflights_probe"
GC_STATS_FILE = "_gc_stats.json"

STATUS_STREAMING = "STREAMING"
STATUS_TERMINAL_OK = "TERMINAL_OK"
STATUS_TERMINAL_FAULT = "TERMINAL_FAULT"
STATUS_SUPERSEDED = "SUPERSEDED"

SUPPORTED_SCHEMA_VERSION = 1

FIRST_ATTEMPT_ID = 1


def _safe_component(name: str) -> str:
    return urllib.parse.quote(name, safe="._-")


class PreflightError(RuntimeError):
    """Raised when the persist root cannot be created / written / read."""


class WriteRejectedError(RuntimeError):
    """Raised when a checkpoint snapshot cannot be written durably."""


@dataclass
class _AttemptRecord:
    run_id: str
    recovery_id: str
    attempt_id: int
    dir_path: str
    durable_offset: int
    model_weight_version: Optional[str]
    finished: bool
    finished_reason: Optional[str]
    status: str
    updated_at: float
    gc_eligible_at: Optional[float]


@dataclass
class _RecoverySlot:
    run_id: str
    recovery_id: str
    attempts: dict[int, _AttemptRecord]
    latest_attempt_id: int = 0


class RolloutProgressStore:
    def __init__(self, config: Optional[ProgressConfig] = None) -> None:
        self._config = config or ProgressConfig()
        self._root = Path(self._config.persist_root)
        self._index: dict[tuple[str, str], _RecoverySlot] = {}
        self._write_queue: asyncio.Queue[CheckPointPayLoad] = asyncio.Queue()
        self._gc_heap: list[_AttemptRecord] = []
        self._revision: int = 0
        self._lock = asyncio.Lock()
        self._stats = GCStats()
        self._write_failed = 0
        self._saved_count = 0
        self._dropped_pending = 0
        self._pending: dict[tuple[str, str], int] = {}
        self._tasks: list[asyncio.Task] = []
        self._started = False

    async def init(self, config: Optional[ProgressConfig] = None) -> None:
        if self._started:
            return
        if config is not None:
            self._config = config
            self._root = Path(self._config.persist_root)
        self.preflight_dir()
        self._index = {}
        self._gc_heap = []
        self._revision = 0
        self._rebuild_index_from_disk()
        self._started = True
        self._tasks.append(asyncio.create_task(self._write_loop(), name="progress-write-loop"))
        self._tasks.append(asyncio.create_task(self._periodic_gc_loop(), name="progress-gc_loop"))

    def preflight_dir(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            probe = self._root / PREFLIGHTS_PROBE
            probe.write_bytes(b"probe")
            if probe.read_bytes() != b"probe":
                raise PreflightError(f"preflight probe readback mismatch: {probe}")
            probe.unlink(missing_ok=True)
        except OSError as e:
            raise PreflightError(f"preflight dir check failed at {self._root}: {e}") from e

    async def shutdown(self) -> None:
        self._started = False
        # 停止 GC 与写循环，先清空待处理队列，保证最后一批进度落盘。
        for t in self._tasks:
            t.cancel()
        while not self._write_queue.empty():
            payload = self._write_queue.get_nowait()
            key = (payload.run_id, payload.recovery_id)
            try:
                await self._do_save(payload)
            except asyncio.CancelledError:
                self._write_queue.put_nowait(payload)
                raise
            except Exception:
                self._write_failed += 1
                logger.exception(
                    "[progress-store] shutdown drain failed for %s/%s/%s",
                    payload.run_id,
                    payload.recovery_id,
                    payload.attempt_id,
                )
            finally:
                self._clear_pending(key)
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    def _clear_pending(self, key: tuple) -> None:
        self._pending[key] = max(0, self._pending.get(key, 0) - 1)

    async def save(self, payload: CheckPointPayLoad) -> None:
        self._ensure_started()
        key = (payload.run_id, payload.recovery_id)
        cap = self._config.max_pending_writes_per_recovery
        if cap and self._pending.get(key, 0) >= cap:
            self._dropped_pending += 1
            logger.warning(
                "[progress-store] dropping save for %s/%s (pending=%d >= cap=%d)",
                payload.run_id,
                payload.recovery_id,
                self._pending.get(key, 0),
                cap,
            )
            return
        self._pending[key] = self._pending.get(key, 0) + 1
        self._write_queue.put_nowait(payload)

    async def load_latest(
        self,
        run_id: str,
        recovery_id: str,
        requested_model_version: Optional[str],
        policy: ModelVersionPolicy,
    ) -> LoadResult:
        self._ensure_started()
        async with self._lock:
            slot = self._index.get((run_id, recovery_id))
            if slot is None or not slot.attempts:
                return LoadResult(None, LoadFailure.NOT_FOUND, "no slot", None)
            aid = slot.latest_attempt_id
            record = slot.attempts[aid]
        dir_path = Path(record.dir_path)

        try:
            manifest = await self._read_json(dir_path / MANIFEST)
        except (OSError, json.JSONDecodeError) as e:
            return LoadResult(None, LoadFailure.CORRUPTED, f"read manifest failed: {e}", aid)
        if manifest is None:
            return LoadResult(None, LoadFailure.CORRUPTED, "no manifest", aid)
        if manifest.get("run_id") != run_id:
            return LoadResult(None, LoadFailure.RUN_ID_MISMATCH, "run_id mismatch", aid)
        if manifest.get("recovery_id") != recovery_id:
            return LoadResult(None, LoadFailure.CORRUPTED, "recovery_id mismatch", aid)
        if manifest.get("attempt_id") != aid:
            return LoadResult(None, LoadFailure.ATTEMPT_STALE, "attempt_id mismatch", aid)
        if int(manifest.get("schema_version", 1)) > SUPPORTED_SCHEMA_VERSION:
            return LoadResult(None, LoadFailure.CORRUPTED, "unsupported schema_version", aid)

        tokens_path = dir_path / TOKENS_BIN
        if not tokens_path.exists():
            return LoadResult(None, LoadFailure.CORRUPTED, "tokens.bin missing", aid)
        try:
            tokens = self._read_array(tokens_path, np.int32)
            logprobs = self._read_array(dir_path / LOGPROBS_BIN, np.float32)
            routing = self._read_optional_pickle(dir_path / ROUTING_BIN)
        except (OSError, ValueError, pickle.UnpicklingError) as e:
            return LoadResult(None, LoadFailure.CORRUPTED, f"shard read failed: {e}", aid)
        if tokens is None or len(tokens) == 0:
            return LoadResult(None, LoadFailure.CORRUPTED, "tokens.bin empty", aid)
        if not self._validate_alignment(manifest, tokens, logprobs, routing):
            return LoadResult(None, LoadFailure.LENGTH_MISMATCH, "shard length mismatch", aid)
        if not policy.check(manifest.get("model_weight_version"), requested_model_version):
            return LoadResult(None, LoadFailure.MODEL_VERSION_BLOCKED, "model_version blocked", aid)

        payload = self._assemble_payload(manifest, tokens, logprobs, routing)
        return LoadResult(payload, None, "ok", aid)

    async def mark_superseded(self, run_id: str, recovery_id: str, attempt_id: int) -> None:
        self._ensure_started()
        async with self._lock:
            slot = self._index.get((run_id, recovery_id))
            if slot is None:
                return
            rec = slot.attempts.get(attempt_id)
            if rec is None:
                return
            self._mark_gc_eligible(rec, STATUS_SUPERSEDED)

    async def periodic_collect(self) -> GCStats:
        self._ensure_started()
        now = time.time()
        async with self._lock:
            due = [r for r in self._gc_heap if r.gc_eligible_at is not None and r.gc_eligible_at <= now]
            self._gc_heap = [r for r in self._gc_heap if r not in due]

        for rec in due:
            ok, bytes_freed = await self._delete_attempt_dir(rec)
            if ok:
                self._stats.collected_count += 1
                self._stats.collected_bytes += bytes_freed
                async with self._lock:
                    slot = self._index.get((rec.run_id, rec.recovery_id))
                    if slot is not None:
                        slot.attempts.pop(rec.attempt_id, None)
                        if not slot.attempts:
                            self._index.pop((rec.run_id, rec.recovery_id), None)
            else:
                self._stats.failed_count += 1
                self._stats.failed_paths.append(rec.dir_path)
                rec.gc_eligible_at = now + self._config.gc_retry_backoff_s
                async with self._lock:
                    self._gc_heap.append(rec)

        orphan = await self._clean_orphan_tmp()
        self._stats.orphan_tmp_cleaned += orphan
        self._stats.last_collect_at = now
        self._write_gc_stats()
        return copy.deepcopy(self._stats)

    async def get_stats(self) -> dict:
        async with self._lock:
            return {
                "write_failed": self._write_failed,
                "saved_count": self._saved_count,
                "dropped_pending": self._dropped_pending,
                "revision": self._revision,
                "recoveries": len(self._index),
                "gc": copy.deepcopy(self._stats),
            }

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("ProgressStore not started, RolloutProgressStore.init() must be called first")

    # write path
    async def _write_loop(self) -> None:
        while True:
            payload = await self._write_queue.get()
            key = (payload.run_id, payload.recovery_id)
            try:
                await self._do_save(payload)
            except asyncio.CancelledError:
                self._write_queue.put_nowait(payload)
                raise
            except Exception:
                self._write_failed += 1
                logger.exception(
                    "[progress-store] save failed for %s/%s/%s", payload.run_id, payload.recovery_id, payload.attempt_id
                )
            finally:
                self._clear_pending(key)

    async def _do_save(self, payload: CheckPointPayLoad) -> None:
        slot = self._index.get((payload.run_id, payload.recovery_id))
        if slot is not None and payload.attempt_id < slot.latest_attempt_id:
            logger.warning(
                "[progress-store] dropping stale attempt %s/%s/%s, latest = %s",
                payload.run_id,
                payload.recovery_id,
                payload.attempt_id,
                slot.latest_attempt_id,
            )
            return
        attempt_dir = (
            self._root
            / _safe_component(payload.run_id)
            / _safe_component(payload.recovery_id)
            / str(payload.attempt_id)
        )
        tmp_dir = Path(f"{attempt_dir}.tmp")
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)
            await self._write_shard(
                tmp_dir / TOKENS_BIN, np.asarray(payload.cumulative_token_ids, dtype=np.int32).tobytes()
            )
            if payload.cumulative_log_probs is not None:
                await self._write_shard(
                    tmp_dir / LOGPROBS_BIN, np.asarray(payload.cumulative_log_probs, dtype=np.float32).tobytes()
                )
            if payload.cumulative_routed_experts is not None:
                await self._write_shard(tmp_dir / ROUTING_BIN, pickle.dumps(payload.cumulative_routed_experts))
            manifest = self._build_manifest(payload)
            await self._write_shard(tmp_dir / MANIFEST, json.dumps(manifest).encode("utf-8"))
            if attempt_dir.exists():
                stale_dir = Path(f"{attempt_dir}.old")
                shutil.rmtree(stale_dir, ignore_errors=True)
                os.replace(attempt_dir, stale_dir)
                try:
                    os.replace(tmp_dir, attempt_dir)
                except OSError:
                    os.replace(stale_dir, attempt_dir)
                    raise
                shutil.rmtree(stale_dir, ignore_errors=True)
            else:
                os.replace(tmp_dir, attempt_dir)
        except (OSError, asyncio.TimeoutError) as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise WriteRejectedError(str(e)) from e
        self._commit_revision(payload, durable_offset=len(payload.cumulative_token_ids))

    async def _write_shard(self, path: Path, data: bytes) -> None:
        timeout = self._config.write_timeout_s
        loop = asyncio.get_running_loop()

        def _write() -> None:
            with open(path, "wb") as f:
                f.write(data)

        await asyncio.wait_for(loop.run_in_executor(None, _write), timeout=timeout)

    def _build_manifest(self, payload: CheckPointPayLoad) -> dict:
        lp = payload.cumulative_log_probs
        re = payload.cumulative_routed_experts
        return {
            "run_id": payload.run_id,
            "recovery_id": payload.recovery_id,
            "attempt_id": payload.attempt_id,
            "schema_version": payload.schema_version,
            "prompt_token_ids": payload.prompt_token_ids,
            "sampling_params": payload.sampling_params,
            "model_weight_version": payload.model_weight_version,
            "inherited_prefix_len": payload.inherited_prefix_len,
            "len_tokens": len(payload.cumulative_token_ids),
            "has_log_probs": lp is not None,
            "len_log_probs": len(lp) if lp is not None else 0,
            "has_routed_experts": re is not None,
            "len_routed_experts": len(re) if re is not None else 0,
            "finished": payload.finished,
            "finished_reason": payload.finished_reason,
            "num_preempted": payload.num_preempted,
            "durable_offset": len(payload.cumulative_token_ids),
            "created_at": payload.created_at,
            "updated_at": payload.updated_at,
        }

    def _commit_revision(self, payload: CheckPointPayLoad, durable_offset: int) -> None:
        key = (payload.run_id, payload.recovery_id)
        slot = self._index.get(key)
        if slot is None:
            slot = _RecoverySlot(payload.run_id, payload.recovery_id, {})
            self._index[key] = slot
        # 若已有更新的 attempt，把之前最新的一批标记为可回收（被替代）。
        prev = slot.attempts.get(slot.latest_attempt_id)
        if prev is not None and slot.latest_attempt_id != payload.attempt_id:
            self._mark_gc_eligible(prev, STATUS_SUPERSEDED)
        rec = _AttemptRecord(
            run_id=payload.run_id,
            recovery_id=payload.recovery_id,
            attempt_id=payload.attempt_id,
            dir_path=str(
                self._root
                / _safe_component(payload.run_id)
                / _safe_component(payload.recovery_id)
                / str(payload.attempt_id)
            ),
            durable_offset=durable_offset,
            model_weight_version=payload.model_weight_version,
            finished=payload.finished,
            finished_reason=payload.finished_reason,
            status=STATUS_TERMINAL_OK if payload.finished else STATUS_STREAMING,
            updated_at=payload.updated_at,
            gc_eligible_at=None,
        )
        slot.attempts[payload.attempt_id] = rec
        slot.latest_attempt_id = payload.attempt_id
        self._revision += 1
        self._saved_count += 1

    def _mark_gc_eligible(self, rec: _AttemptRecord, status: str) -> None:
        # 幂等：已经可回收就不再重复入堆。
        if rec.gc_eligible_at is not None:
            return
        rec.status = status
        rec.gc_eligible_at = time.time() + self._config.gc_delay_s
        if rec not in self._gc_heap:
            self._gc_heap.append(rec)

    async def _read_json(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _read_array(self, path: Path, dtype):
        if not path.exists():
            return None
        return np.fromfile(path, dtype=dtype)

    def _read_optional_pickle(self, path: Path):
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _validate_alignment(self, manifest: dict, tokens, logprobs, routing) -> bool:
        if tokens is None or len(tokens) == 0:
            return False
        if len(tokens) != manifest.get("len_tokens"):
            return False
        if manifest.get("has_log_probs"):
            if logprobs is None or len(logprobs) != manifest.get("len_log_probs"):
                return False
        if manifest.get("has_routed_experts"):
            if routing is None or len(routing) != manifest.get("len_routed_experts"):
                return False
        return True

    def _to_list(self, value):
        if value is None:
            return None
        return value.tolist() if isinstance(value, np.ndarray) else value

    def _assemble_payload(self, manifest: dict, tokens, logprobs, routing) -> CheckPointPayLoad:
        return CheckPointPayLoad(
            run_id=manifest["run_id"],
            recovery_id=manifest["recovery_id"],
            attempt_id=int(manifest["attempt_id"]),
            prompt_token_ids=manifest.get("prompt_token_ids", []),
            sampling_params=manifest.get("sampling_params", {}),
            model_weight_version=manifest.get("model_weight_version"),
            inherited_prefix_len=int(manifest.get("inherited_prefix_len", 0)),
            cumulative_token_ids=self._to_list(tokens) or [],
            cumulative_log_probs=self._to_list(logprobs),
            cumulative_routed_experts=self._to_list(routing),
            num_preempted=int(manifest.get("num_preempted", 0)),
            finished=bool(manifest.get("finished", False)),
            finished_reason=manifest.get("finished_reason"),
            created_at=float(manifest.get("created_at", 0)),
            updated_at=float(manifest.get("updated_at", 0)),
            schema_version=int(manifest.get("schema_version", 1)),
        )

    def _rebuild_index_from_disk(self) -> None:
        if not self._root.exists():
            return
        for run_dir in self._root.iterdir():
            if not run_dir.is_dir() or run_dir.name.endswith(".tmp"):
                continue
            for rec_dir in run_dir.iterdir():
                if not rec_dir.is_dir() or rec_dir.name.endswith(".tmp"):
                    continue
                for attempt_dir in rec_dir.iterdir():
                    if not attempt_dir.is_dir() or attempt_dir.name.endswith(".tmp"):
                        continue
                    try:
                        attempt_id = int(attempt_dir.name)
                    except ValueError:
                        continue
                    try:
                        with open(attempt_dir / MANIFEST, encoding="utf-8") as f:
                            manifest = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        logger.warning("[progress-store] skip unreadable manifest %s", attempt_dir / MANIFEST)
                        continue
                    if not isinstance(manifest, dict):
                        continue
                    run_id = manifest.get("run_id")
                    recovery_id = manifest.get("recovery_id")
                    if not run_id or not recovery_id or manifest.get("attempt_id") != attempt_id:
                        logger.warning(
                            "[progress-store] skip inconsistent manifest %s",
                            attempt_dir / MANIFEST,
                        )
                        continue
                    finished = bool(manifest.get("finished", False))
                    rec = _AttemptRecord(
                        run_id=run_id,
                        recovery_id=recovery_id,
                        attempt_id=manifest["attempt_id"],
                        dir_path=str(attempt_dir),
                        durable_offset=int(manifest.get("durable_offset", manifest.get("len_tokens", 0))),
                        model_weight_version=manifest.get("model_weight_version"),
                        finished=finished,
                        finished_reason=manifest.get("finished_reason"),
                        status=STATUS_TERMINAL_OK if finished else STATUS_STREAMING,
                        updated_at=manifest.get("updated_at"),
                        gc_eligible_at=None,
                    )
                    key = (run_id, recovery_id)
                    slot = self._index.get(key)
                    if slot is None:
                        slot = _RecoverySlot(run_id=run_id, recovery_id=recovery_id, attempts={})
                        self._index[key] = slot
                    slot.attempts[attempt_id] = rec
                    slot.latest_attempt_id = max(slot.latest_attempt_id, attempt_id)
                    if finished:
                        rec.gc_eligible_at = rec.updated_at + self._config.gc_delay_s
                        self._gc_heap.append(rec)
        if self._index:
            logger.info("[progress-store] rebuild index from disk :%d recoveries", len(self._index))

    async def _delete_attempt_dir(self, rec: _AttemptRecord) -> tuple[bool, int]:
        path = Path(rec.dir_path)
        if not path.exists():
            return True, 0

        def _rm() -> int:
            total = 0
            if path.exists():
                for f in path.rglob("*"):
                    try:
                        if f.is_file():
                            total += f.stat().st_size
                    except OSError:
                        pass
                shutil.rmtree(path)
            return total

        try:
            freed = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _rm),
                timeout=max(10.0, self._config.write_timeout_s),
            )
            return True, freed
        except (OSError, asyncio.TimeoutError):
            return False, 0

    async def _clean_orphan_tmp(self) -> int:
        def _scan() -> int:
            removed = 0
            for pattern in ("*.tmp", "*.old"):
                for p in self._root.rglob(pattern):
                    try:
                        shutil.rmtree(p)
                        removed += 1
                    except OSError:
                        pass
            return removed

        return await asyncio.get_running_loop().run_in_executor(None, _scan)

    def _write_gc_stats(self) -> None:
        try:
            data = {
                "collected_count": self._stats.collected_count,
                "collected_bytes": self._stats.collected_bytes,
                "failed_count": self._stats.failed_count,
                "orphan_tmp_cleaned": self._stats.orphan_tmp_cleaned,
                "last_collect_at": self._stats.last_collect_at,
            }
            (self._root / GC_STATS_FILE).write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.warning("[progress-store] failed to write gc stats: %s", e)

    async def _periodic_gc_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._config.gc_period_s)
                await self.periodic_collect()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[progress-store] periodic collect failed")
