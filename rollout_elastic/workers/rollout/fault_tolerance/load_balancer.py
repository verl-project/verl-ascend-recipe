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
"""Plain-Python load-balancer state machine for elastic rollout.

``_LoadBalancerCore`` is a new class kept in the recipe (it does not exist in
verl@dfc01f85). It is wrapped by verl's native ``GlobalRequestLoadBalancer``
Ray actor via the ``patch.llm_server`` decorators, so the state machine is
unit-testable without ``ray.init()``.
"""

from __future__ import annotations

from cachetools import LRUCache

DEFAULT_ROUTING_CACHE_SIZE = 10000


class _LoadBalancerCore:
    """Plain-Python state machine for load balancing.

    Wrapped by `GlobalRequestLoadBalancer` (Ray actor) for remote access.
    Splitting the logic from the Ray decorator makes it unit-testable without
    `ray.init()`, while keeping the Ray actor as a thin forwarder.

    Per spec §横向系统 LB:
      - state: `_server` / `_inflight` / `_dead` / `_sticky`
      - all write methods idempotent
      - `acquire_server` skips dead, clears stale sticky entries
      - `enable_fault_tolerance` toggles lenient semantics for `release_server`
        (idempotent decrement vs. original strict ValueError) — when False,
        behavior is bit-exact identical to the pre-FT implementation
    """

    def __init__(
        self,
        servers: dict,
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        enable_fault_tolerance: bool = False,
    ) -> None:
        if not servers:
            raise ValueError("server must be non-empty")

        self._server: dict = dict(servers)
        self._inflight: dict[str, int] = {sid: 0 for sid in self._server}
        self._dead: set[str] = set()
        self._sticky: LRUCache = LRUCache(maxsize=max_cache_size)
        self._ft = enable_fault_tolerance

    # ----- helpers -----

    def _alive_ids(self) -> list[str]:
        """Return server_ids that are not in the dead set."""
        return [sid for sid in self._inflight if sid not in self._dead]

    # ----- core API (sync, matching the public Ray actor surface) -----

    def acquire_server(self, request_id: str) -> str:
        # Request-level sticky (multi-turn). If cached server is dead, evict and re-pick.
        if request_id in self._sticky:
            sid = self._sticky[request_id]
            if sid in self._dead:
                del self._sticky[request_id]
            else:
                self._inflight[sid] += 1
                return sid

        alive = self._alive_ids()
        if not alive:
            from verl.workers.rollout.fault_tolerance import AllServersFailed

            raise AllServersFailed("no alive servers in pool")

        sid = min(alive, key=lambda s: self._inflight[s])
        self._sticky[request_id] = sid
        self._inflight[sid] += 1
        return sid

    def release_server(self, server_id: str) -> None:
        if self._ft:
            # Lenient: never raise; cap at 0 to keep INV-3 (no negative inflight).
            if server_id in self._inflight:
                self._inflight[server_id] = max(0, self._inflight[server_id] - 1)
            return
        # Bit-exact original behavior when FT is off.
        if server_id not in self._inflight:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        self._inflight[server_id] -= 1

    def mark_failed(self, server_id: str) -> None:
        """Mark a server as dead; subsequent acquires skip it. Idempotent."""
        if server_id in self._server or server_id in self._inflight:
            self._dead.add(server_id)
        # If server_id is unknown entirely, silently no-op (idempotent on unknown ids).

    def add_servers(self, servers: dict) -> None:
        """Register new servers. Idempotent on existing ids. Resurrect dead ids."""
        for sid, handle in servers.items():
            self._server[sid] = handle
            if sid in self._dead:
                self._dead.discard(sid)
                self._inflight[sid] = 0
            elif sid not in self._inflight:
                self._inflight[sid] = 0
            # If sid already exists and alive, leave inflight count untouched (idempotent).

    def remove_servers(self, server_ids: list) -> None:
        """Drop servers from routing. Adds to dead; clears their sticky entries."""
        ids = set(server_ids)
        for sid in ids:
            if sid in self._server:
                self._dead.add(sid)
                # Keep _inflight entry so callers' release_server still finds it
                # (idempotency / no spurious ValueError mid-flight on FT=False).
        # Clear sticky entries pointing to removed servers.
        stale = [rid for rid, sid in self._sticky.items() if sid in ids]
        for rid in stale:
            del self._sticky[rid]

    def get_server_handle(self, server_id: str):
        """Return the Ray actor handle for `server_id`, or None if unknown."""
        return self._server.get(server_id)
