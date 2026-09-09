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
"""Protocol classes that the real Ray actors and the test fakes must both implement.

Per spec §6.4 mock 自洽约束: mypy --strict + isinstance(...,  Protocol)
catches drift between mock and real component API surfaces.
"""

from __future__ import annotations

from typing import Any, Awaitable, Protocol, runtime_checkable


@runtime_checkable
class RemoteCallable(Protocol):
    """What `actor.method` looks like in Ray (and in our fakes)."""

    def remote(self, *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


@runtime_checkable
class ServerProtocol(Protocol):
    """vLLMHttpServer Ray actor (and FakeServer) must expose this."""

    generate: RemoteCallable
    abort_all_requests: RemoteCallable


@runtime_checkable
class LoadBalancerProtocol(Protocol):
    """GlobalRequestLoadBalancer (and FakeLoadBalancer) must expose this."""

    acquire_server: RemoteCallable
    release_server: RemoteCallable
    mark_failed: RemoteCallable
    add_servers: RemoteCallable
    remove_servers: RemoteCallable
    set_fault_tolerance: RemoteCallable
