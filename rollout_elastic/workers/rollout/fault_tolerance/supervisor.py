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
"""Supervisor — periodic heartbeat + dead replica detection (P2 §横向系统合同).

Spec contract:
    "独立 Ray actor，心跳 → 连续 K 次 miss 标 dead → 调 lb.mark_failed +
     checkpoint_mgr.remove_replicas →（可选）后台 task spawn replacement。
     探针绕 LB 计数；spawn 不阻塞心跳循环。"

S9 scope: HeartbeatTracker (pure state machine) + Supervisor (async probe loop
+ K-miss transition + on_dead callback). Replacement spawn is S11; this module
only signals "replica X is dead" via the injected `on_dead` handler.

Design:
  - HeartbeatTracker is a plain dataclass with no I/O, so it's unit-testable
    in isolation.
  - Supervisor.probe_fn is injected (an awaitable callable), so the bypass-LB
    property is enforced by *who provides the probe* — not by Supervisor itself.
  - Once a replica transitions to dead, it is never re-probed (INV-6: dead
    replicas don't receive new traffic, including probes).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class HeartbeatTracker:
    """Per-replica consecutive-miss state.

    The transition rule: `miss_count >= miss_threshold` → dead. Any successful
    probe resets the miss count to 0 (so K must be *consecutive*).
    """

    miss_threshold: int
    miss_counts: dict[str, int] = field(default_factory=dict)
    dead: set[str] = field(default_factory=set)

    def record_ok(self, replica_id: str) -> None:
        """A successful probe — reset consecutive-miss counter (no-op if already dead)."""
        if replica_id in self.dead:
            return
        self.miss_counts[replica_id] = 0

    def record_miss(self, replica_id: str) -> bool:
        """A failed probe — increment counter; return True iff this transitions to dead.

        Already-dead replicas return False without further mutation (idempotent).
        """
        if replica_id in self.dead:
            return False
        self.miss_counts[replica_id] = self.miss_counts.get(replica_id, 0) + 1
        if self.miss_counts[replica_id] >= self.miss_threshold:
            self.dead.add(replica_id)
            return True
        return False

    def is_dead(self, replica_id: str) -> bool:
        return replica_id in self.dead


class Supervisor:
    """Async heartbeat loop over a fixed set of rollout replicas.

    The probe (`probe_fn`) is callable taking the replica handle and returning
    a coroutine that resolves to a bool (ok) — or raises (counts as miss).
    The dead-transition handler (`on_dead`) is awaited once per replica when
    it crosses the K-miss threshold. Subsequent probes of that replica are
    skipped.

    The loop body is `gather` over all alive replicas — so a slow probe on
    one replica doesn't delay probes on others (within an iteration).
    """

    def __init__(
        self,
        *,
        replicas: dict[str, Any],
        probe_fn: Callable[[Any], Awaitable[bool]],
        on_dead: Callable[[str], Awaitable[None]],
        promote_fn: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        interval_s: float,
        miss_threshold: int,
        probe_timeout_s: float,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        if miss_threshold < 1:
            raise ValueError(f"miss_threshold must be >= 1, got {miss_threshold}")
        if probe_timeout_s <= 0:
            raise ValueError(f"probe_timeout_s must be > 0, got {probe_timeout_s}")

        self.replicas: dict[str, Any] = dict(replicas)
        self.probe_fn = probe_fn
        self.on_dead = on_dead
        self.promote_fn = promote_fn
        self.interval_s = interval_s
        self.probe_timeout_s = probe_timeout_s
        self.tracker = HeartbeatTracker(miss_threshold=miss_threshold)
        self._log = logger or logging.getLogger(__name__)
        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task[None]] = None
        # Serial dead-event consumer: probes add ids to _pending_dead and signal
        # a queue sentinel. The consumer processes a snapshot one-at-a-time.
        self._dead_queue: Optional[asyncio.Queue[None]] = None
        self._consumer_task: Optional[asyncio.Task[None]] = None
        self._pending_dead: set[str] = set()
        self._admission_lock: Optional[asyncio.Lock] = None

    def _get_admission_lock(self) -> asyncio.Lock:
        if self._admission_lock is None:
            self._admission_lock = asyncio.Lock()
        return self._admission_lock

    async def _probe_one(self, replica_id: str, handle: Any) -> None:
        try:
            ok = bool(await asyncio.wait_for(self.probe_fn(handle), timeout=self.probe_timeout_s))
        except Exception as e:
            self._log.debug("[FT] heartbeat probe failed for %s: %r", replica_id, e)
            ok = False
        if ok:
            self.tracker.record_ok(replica_id)
            return
        transitioned = self.tracker.record_miss(replica_id)
        if transitioned:
            self._log.warning(
                "[FT] supervisor: replica %s declared dead after %d consecutive misses",
                replica_id,
                self.tracker.miss_threshold,
            )
            # Enqueue for serial dispatch; do NOT await on_dead here (would stall ticks)
            if replica_id not in self._pending_dead and self._dead_queue is not None:
                self._pending_dead.add(replica_id)
                self._dead_queue.put_nowait(None)

    async def _consumer_loop(self) -> None:
        """Serial consumer of dead events. Runs on_dead one-at-a-time so LB.remove +
        CKE.callback don't race or run concurrently for different replicas."""
        assert self._dead_queue is not None and self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._dead_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            while True:
                try:
                    self._dead_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            for replica_id in list(self._pending_dead):
                try:
                    # LB add and remove share this lock.  If death is observed
                    # during add, remove necessarily runs after add completes.
                    async with self._get_admission_lock():
                        await self.on_dead(replica_id)
                except Exception as e:
                    self._log.error(
                        "[FT] supervisor consumer: on_dead handler for %s raised: %r",
                        replica_id,
                        e,
                    )
                finally:
                    self._pending_dead.discard(replica_id)

    async def tick(self) -> None:
        """One pass over all alive replicas. Public for testing without the loop."""
        alive = [(rid, h) for rid, h in self.replicas.items() if not self.tracker.is_dead(rid)]
        if not alive:
            return
        await asyncio.gather(*(self._probe_one(rid, h) for rid, h in alive))

    async def _heartbeat_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass  # expected — interval elapsed, run another tick

    def start(self) -> None:
        """Start the heartbeat loop + dead-event consumer as background asyncio tasks."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("Supervisor already running")
        self._stop_event = asyncio.Event()
        self._dead_queue = asyncio.Queue()
        self._pending_dead = set()
        self._admission_lock = asyncio.Lock()
        self._task = asyncio.create_task(self._heartbeat_loop())
        self._consumer_task = asyncio.create_task(self._consumer_loop())

    async def stop(self, drain_timeout_s: float = 5.0) -> None:
        """Stop the heartbeat loop and wait (bounded) for the current tick.

        If the in-flight tick doesn't return within ``drain_timeout_s``, the
        task is cancelled so a wedged probe / on_dead handler can't block
        shutdown indefinitely.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        for task_attr in ("_task", "_consumer_task"):
            task = getattr(self, task_attr)
            if task is None:
                continue
            try:
                await asyncio.wait_for(task, timeout=drain_timeout_s)
            except asyncio.TimeoutError:
                self._log.warning(
                    "[FT] supervisor: %s did not drain in %.1fs; cancelling",
                    task_attr,
                    drain_timeout_s,
                )
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            setattr(self, task_attr, None)

    def add_replica(self, replica_id: str, handle: Any) -> None:
        """Register a new replica with the Supervisor at runtime.

        Used after a successful replacement spawn (S11) so the new replica is
        included in subsequent ticks. If the id was previously marked dead,
        the dead flag and miss count are cleared (resurrection).
        """
        self.replicas[replica_id] = handle
        self.tracker.dead.discard(replica_id)
        self.tracker.miss_counts[replica_id] = 0

    async def promote_replica(self, replica_id: str, servers: dict[str, Any]) -> bool:
        """Admit a synced replica only while it remains alive.

        The same loop serializes this operation with the dead-event consumer,
        so a concurrent death either rejects admission or removes it afterward.
        """
        if self.promote_fn is None:
            return False
        async with self._get_admission_lock():
            if replica_id not in self.replicas or self.tracker.is_dead(replica_id):
                return False
            await self.promote_fn(servers)
            return not self.tracker.is_dead(replica_id)

    def report_failure(self, replica_id: str, source: str = "unknown") -> bool:
        """Queue an idempotent immediate-death report from another control path."""
        if replica_id not in self.replicas:
            self._log.info("[FT] supervisor: ignoring failure for unknown replica %s", replica_id)
            return False
        if self.tracker.is_dead(replica_id) or replica_id in self._pending_dead:
            return False
        self.tracker.dead.add(replica_id)
        self.tracker.miss_counts[replica_id] = self.tracker.miss_threshold
        self._log.warning(
            "[FT] supervisor: replica %s reported dead by %s",
            replica_id,
            source,
        )
        if self._dead_queue is not None:
            self._pending_dead.add(replica_id)
            self._dead_queue.put_nowait(None)
        return True

    def remove_replica(self, replica_id: str) -> None:
        """Drop a replica from the Supervisor's probe set.

        Idempotent: a missing id is silently ignored.
        """
        self.replicas.pop(replica_id, None)
        self.tracker.miss_counts.pop(replica_id, None)
        self.tracker.dead.discard(replica_id)


def make_on_dead(
    *,
    lb_handle: Any,
    replica_to_server_ids: Callable[[str], list[str]],
    ckpt_mgr_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    spawner: Optional[Callable[[str], Awaitable[Any]]] = None,
    on_spawn_success: Optional[Callable[[str, Any], Awaitable[None]]] = None,
    logger: Optional[logging.Logger] = None,
) -> Callable[[str], Awaitable[None]]:
    """Build the default ``on_dead`` handler wired to LB + (optional) CKE + spawn.

    Control-plane contract: remove servers from LB admission, notify the
    checkpoint manager, then optionally spawn a replacement in the background.

    Three steps:

    1. LB.remove_servers for every server in the dead replica (awaited; LB
       unavailability is logged, not raised).
    2. (optional) checkpoint-manager callback so the next group rebuild skips it.
    3. (optional) spawn replacement via background asyncio.create_task —
       MUST not block. Idempotent per replica_id: if a spawn for `r0` is
       already in flight, a second on_dead call for `r0` skips step 3.

    Spawn caveats (validated via /codex:rescue):
    - Spawn runs on the same event loop as the Supervisor tick (Ray async
      actors are single-threaded). The spawner MUST reach `await` points and
      MUST NOT call `ray.get`; otherwise it blocks heartbeats.
    - Resource teardown (GPU release of the dead replica), NCCL communicator
      rebuild, and rank reassignment are NOT handled here — they are runtime
      concerns gated behind `FaultToleranceConfig.replace_dead_replicas`
      (default False). Producers should construct `spawner=None` until those
      are validated end-to-end on a GPU cluster (S12 chaos).
    - The created task has an explicit done callback that consumes
      `task.exception()` so silent failures don't hide as
      "Task exception was never retrieved" at GC time.

    Args:
        lb_handle: Ray actor handle for the GlobalRequestLoadBalancer.
        replica_to_server_ids: Maps `replica_id` → list of server IDs (str).
        ckpt_mgr_handle: Optional Ray actor handle for CheckpointEngineManager.
        replica_by_id: Maps `replica_id` → replica object for
            `ckpt_mgr.remove_replicas`. Required iff `ckpt_mgr_handle` is set.
        spawner: Optional `async (dead_replica_id) -> new_replica_obj`. When
            present, on_dead schedules a background spawn task. Step 3 is
            skipped if None.
        on_spawn_success: Optional `async (dead_id, new_replica) -> None`.
            Invoked on the background task after `spawner` returns
            successfully — typically used to register the new replica with
            the Supervisor (`supervisor.add_replica(...)`) and/or LB
            (`lb.add_servers(...)`).
        logger: Logger for error reporting (defaults to module logger).
    """
    log = logger or logging.getLogger(__name__)
    inflight_spawns: set[str] = set()

    def _spawn_done(replica_id: str, task: asyncio.Task[Any]) -> None:
        """Cleanup + explicit exception capture for the background spawn task."""
        inflight_spawns.discard(replica_id)
        if task.cancelled():
            log.warning("[FT] spawn task for %s was cancelled", replica_id)
            return
        exc = task.exception()
        if exc is not None:
            log.error("[FT] spawn task for %s raised: %r", replica_id, exc)

    async def _do_spawn(spawner_fn: Callable[[str], Awaitable[Any]], replica_id: str) -> None:
        new_replica = await spawner_fn(replica_id)
        if on_spawn_success is not None:
            await on_spawn_success(replica_id, new_replica)

    async def _on_dead(replica_id: str) -> None:
        # Step 1: LB.remove_servers — atomic, awaited (replaces fire-and-forget mark_failed
        # so happens-before guarantee holds: next acquire sees _dead updated).
        server_ids = replica_to_server_ids(replica_id)
        if server_ids:
            try:
                await asyncio.wait_for(
                    lb_handle.remove_servers.remote(server_ids=server_ids),
                    timeout=10.0,
                )
            except Exception as e:
                log.error(
                    "[FT] on_dead: lb.remove_servers(%s) for replica %s raised: %r",
                    server_ids,
                    replica_id,
                    e,
                )
        # Step 2: CKE local callback (atomic remove + flag set, see CheckpointEngineManager.on_replica_dead)
        if ckpt_mgr_callback is not None:
            try:
                await ckpt_mgr_callback(replica_id)
            except Exception as e:
                log.error(
                    "[FT] on_dead: ckpt_mgr_callback(%s) raised: %r",
                    replica_id,
                    e,
                )
        # Step 3: background spawn (optional, idempotent per replica_id)
        if spawner is not None:
            if replica_id in inflight_spawns:
                log.info(
                    "[FT] on_dead: spawn for %s already in flight; skipping duplicate",
                    replica_id,
                )
                return
            inflight_spawns.add(replica_id)
            task = asyncio.create_task(_do_spawn(spawner, replica_id))
            task.add_done_callback(lambda t, rid=replica_id: _spawn_done(rid, t))

    return _on_dead
