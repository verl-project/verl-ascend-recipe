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

"""Optional prompt-credit routing with a work-conserving escape."""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import defaultdict
from typing import Any


class ElasticInterleaveRouter:
    """Interleave siblings, then relax the credit if server capacity is idle."""

    def __init__(self, servers: dict[str, Any], config: dict[str, Any]):
        self._servers = dict(servers)
        self._inflight_requests = {key: 0 for key in servers}
        self._max_inflight_per_server = int(config.get("max_inflight_per_server", 0))
        self._prompt_credit = int(config.get("prompt_credit", 1))
        self._elastic_delay_seconds = float(config.get("elastic_delay_ms", 20.0)) / 1000
        self._group_namespace = str(config.get("group_namespace", "adaptive-spec-rollout"))
        if self._max_inflight_per_server < 1 or self._prompt_credit < 1:
            raise ValueError("capacity and prompt_credit must be positive")
        if not math.isfinite(self._elastic_delay_seconds) or self._elastic_delay_seconds < 0:
            raise ValueError("elastic_delay_ms must be finite and non-negative")
        if config.get("affinity_mode", "free") != "free":
            raise ValueError("only free routing is supported")
        # Active accounting must never be evicted by an LRU cache.
        self._request_to_server: dict[str, str] = {}
        self._request_to_group: dict[str, str] = {}
        self._pending: list[dict[str, Any]] = []
        self._group_inflight: dict[str, int] = defaultdict(int)
        self._group_last_dispatch: dict[str, int] = defaultdict(lambda: -1)
        self._arrival_sequence = 0
        self._dispatch_sequence = 0
        self._elastic_escape_task: asyncio.Task | None = None

    def _group_key(self, request_id, prompt_ids, routing_group_id):
        digest = hashlib.sha256()
        digest.update(self._group_namespace.encode())
        digest.update(b"\x00")
        if routing_group_id is not None:
            digest.update(b"explicit\x00")
            digest.update(str(routing_group_id).encode())
        elif prompt_ids is not None:
            digest.update(b"tokens\x00")
            for token_id in prompt_ids:
                digest.update(int(token_id).to_bytes(8, "big", signed=True))
        else:
            digest.update(b"request\x00")
            digest.update(request_id.encode())
        return digest.hexdigest()

    def require_acquire_fields(self):
        return ["prompt_ids", "routing_group_id"]

    def require_release_fields(self):
        return ["request_id"]

    def _available_servers(self):
        return [key for key, count in self._inflight_requests.items() if count < self._max_inflight_per_server]

    def _forget_idle_group(self, group):
        if not self._group_inflight[group] and not any(p["group_id"] == group for p in self._pending):
            self._group_inflight.pop(group, None)
            self._group_last_dispatch.pop(group, None)

    def _dispatch_pending(self, *, allow_elastic=False):
        while self._pending and self._available_servers():
            regular = [
                i for i, p in enumerate(self._pending) if self._group_inflight[p["group_id"]] < self._prompt_credit
            ]
            if not regular and not allow_elastic:
                if self._elastic_escape_task is None or self._elastic_escape_task.done():
                    self._elastic_escape_task = asyncio.create_task(self._elastic_escape_after_delay())
                return
            candidates = regular if regular else range(len(self._pending))
            index = min(
                candidates,
                key=lambda i: (
                    self._group_inflight[self._pending[i]["group_id"]],
                    self._group_last_dispatch[self._pending[i]["group_id"]],
                    self._pending[i]["arrival_sequence"],
                ),
            )
            pending = self._pending.pop(index)
            if pending["future"].done():
                self._forget_idle_group(pending["group_id"])
                continue
            server = min(self._available_servers(), key=self._inflight_requests.get)
            request, group = pending["request_id"], pending["group_id"]
            self._request_to_server[request] = server
            self._request_to_group[request] = group
            self._inflight_requests[server] += 1
            self._group_inflight[group] += 1
            self._group_last_dispatch[group] = self._dispatch_sequence
            self._dispatch_sequence += 1
            pending["future"].set_result((server, self._servers[server]))

    async def _elastic_escape_after_delay(self):
        await asyncio.sleep(self._elastic_delay_seconds)
        self._elastic_escape_task = None
        self._dispatch_pending(allow_elastic=True)

    async def acquire_server(self, request_id, prompt_ids=None, routing_group_id=None):
        if not self._servers:
            raise RuntimeError("No available servers in load balancer")
        if request_id in self._request_to_server or any(p["request_id"] == request_id for p in self._pending):
            raise ValueError("duplicate active request_id")
        group = self._group_key(request_id, prompt_ids, routing_group_id)
        future = asyncio.get_running_loop().create_future()
        self._pending.append(
            dict(request_id=request_id, group_id=group, arrival_sequence=self._arrival_sequence, future=future)
        )
        self._arrival_sequence += 1
        self._dispatch_pending()
        try:
            return await future
        except asyncio.CancelledError:
            self._pending = [p for p in self._pending if p["future"] is not future]
            assigned = self._request_to_server.get(request_id)
            if assigned is not None:
                self.release_server(assigned, request_id)
            self._forget_idle_group(group)
            self._dispatch_pending()
            raise

    def release_server(self, server_id, request_id=None):
        if request_id is None:
            raise ValueError("Elastic release requires request_id")
        if self._request_to_server.get(request_id) != server_id:
            return
        self._request_to_server.pop(request_id)
        group = self._request_to_group.pop(request_id)
        self._inflight_requests[server_id] -= 1
        self._group_inflight[group] -= 1
        self._forget_idle_group(group)
        self._dispatch_pending()

    def add_servers(self, servers):
        for key, handle in servers.items():
            self._servers[key] = handle
            self._inflight_requests.setdefault(key, 0)
        self._dispatch_pending()

    def remove_servers(self, server_ids):
        if any(self._inflight_requests.get(key, 0) for key in server_ids):
            raise ValueError("Cannot remove busy server")
        for key in server_ids:
            self._servers.pop(key, None)
            self._inflight_requests.pop(key, None)

    def get_all_servers(self):
        return list(self._servers)

    def clear_sticky_cache(self):
        # This router has no sticky cache; never erase active accounting.
        return {"cleared_entries": 0, "server_loads": dict(self._inflight_requests)}

    def get_total_inflight(self):
        return sum(self._inflight_requests.values())

    def get_status(self):
        return dict(
            servers=dict(self._inflight_requests),
            total_inflight=self.get_total_inflight(),
            pending=len(self._pending),
            prompt_credit=self._prompt_credit,
            max_inflight_per_server=self._max_inflight_per_server,
            affinity_mode="free",
            elastic_delay_ms=self._elastic_delay_seconds * 1000,
        )
