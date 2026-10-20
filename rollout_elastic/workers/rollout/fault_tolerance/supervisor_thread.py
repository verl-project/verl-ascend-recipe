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
"""ThreadedSupervisor — run a Supervisor on its own thread + asyncio loop.

The Supervisor's heartbeat tasks need a *running* asyncio loop. If we attach
them to the trainer actor's main event loop, any synchronous blocking call on
that loop (e.g. ``ray.get`` inside ``_fit_update_weights``) starves the
heartbeat — the heartbeat task never gets scheduled and dead replicas are
never declared. Running on a dedicated thread with a dedicated loop
guarantees the heartbeat ticks regardless of what the trainer is doing.

Boundary: ``start()`` and ``stop()`` are plain sync methods. The trainer
calls them from its main thread without awaiting.

Inside the child loop:
  - ObjectRef awaits work fine (Ray ObjectRefs have no event-loop affinity;
    awaiting an ``actor.method.remote()`` from any loop is supported).
  - The ``on_dead`` handler runs on this loop. Its calls into shared CKE
    state (``CKE.on_replica_dead``) acquire a ``threading.Lock`` to serialise
    with main-thread reads/clears of ``membership_changed``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from .supervisor import Supervisor


class ThreadedSupervisor:
    """Sync wrapper that hosts a ``Supervisor`` on its own thread + loop."""

    def __init__(
        self,
        supervisor: Supervisor,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._supervisor = supervisor
        self._log = logger or logging.getLogger(__name__)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None

    @property
    def supervisor(self) -> Supervisor:
        return self._supervisor

    @property
    def is_running(self) -> bool:
        """Whether the dedicated heartbeat thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def report_failure(self, replica_id: str, source: str = "unknown") -> None:
        """Thread-safe, non-blocking bridge for CKE sync-stage failures."""
        loop = self._loop
        if loop is None or not loop.is_running():
            self._log.debug(
                "[FT] ThreadedSupervisor.report_failure ignored before loop start: %s (%s)",
                replica_id,
                source,
            )
            return
        loop.call_soon_threadsafe(self._supervisor.report_failure, replica_id, source)

    async def promote_replica(
        self,
        replica_id: str,
        servers: dict[str, object],
        attempt_id: int,
        target_version: int,
    ) -> bool:
        """Run serving admission on the Supervisor loop and await its decision."""
        del attempt_id, target_version
        loop = self._loop
        if loop is None or not loop.is_running():
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._supervisor.promote_replica(replica_id, servers),
            loop,
        )
        return bool(await asyncio.wrap_future(future))

    def start(self, ready_timeout_s: float = 10.0) -> None:
        if self._thread is not None:
            raise RuntimeError("ThreadedSupervisor already started")
        self._ready.clear()
        self._start_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="ft-supervisor",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=ready_timeout_s):
            raise RuntimeError(f"ThreadedSupervisor: child loop did not become ready in {ready_timeout_s}s")
        if self._start_error is not None:
            raise self._start_error
        self._log.warning("[FT] ThreadedSupervisor: started on thread %s", self._thread.name)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop

        async def _bring_up() -> None:
            self._supervisor.start()

        try:
            try:
                loop.run_until_complete(_bring_up())
            except BaseException as e:
                self._start_error = e
                self._ready.set()
                return
            self._ready.set()
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        thread = self._thread
        loop = self._loop
        if thread is None:
            return
        if loop is not None and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._supervisor.stop(drain_timeout_s=drain_timeout_s),
                    loop,
                )
                fut.result(timeout=drain_timeout_s + 5.0)
            except Exception as e:
                self._log.warning("[FT] ThreadedSupervisor.stop: supervisor.stop raised %r", e)
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        thread.join(timeout=drain_timeout_s + 10.0)
        if thread.is_alive():
            self._log.warning("[FT] ThreadedSupervisor.stop: thread did not exit within deadline")
        self._thread = None
