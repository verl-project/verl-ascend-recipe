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

import asyncio

import pytest

from adaptive_spec.scheduler import ElasticInterleaveRouter


def router(**kwargs):
    return ElasticInterleaveRouter(
        {"s0": 0, "s1": 1}, dict(max_inflight_per_server=1, prompt_credit=1, elastic_delay_ms=2, **kwargs)
    )


def test_interleave_release_and_duplicate_accounting():
    async def run():
        r = router()
        server, _ = await r.acquire_server("a", [1])
        same = asyncio.create_task(r.acquire_server("b", [1]))
        await asyncio.sleep(0)
        other, _ = await r.acquire_server("c", [2])
        assert server != other and not same.done()
        r.release_server(other, "a")  # incorrect release must not erase the assignment
        assert r.get_total_inflight() == 2
        with pytest.raises(ValueError, match="duplicate"):
            await r.acquire_server("a", [1])
        r.clear_sticky_cache()
        r.add_servers({server: 0})
        assert r.get_total_inflight() == 2
        r.release_server(server, "a")
        assigned, _ = await same
        r.release_server(assigned, "b")
        r.release_server(other, "c")
        r.release_server(other, "c")
        assert r.get_total_inflight() == 0 and not r._request_to_group and not r._group_inflight

    asyncio.run(run())


def test_elastic_escape_fills_idle_capacity():
    async def run():
        r = router()
        await r.acquire_server("a", [1])
        server, _ = await asyncio.wait_for(r.acquire_server("b", [1]), 0.5)
        assert r.get_total_inflight() == 2
        r.release_server(server, "b")

    asyncio.run(run())


def test_cancel_pending_and_already_dispatched_does_not_leak():
    async def run():
        r = router()
        first, _ = await r.acquire_server("a", [1])
        pending = asyncio.create_task(r.acquire_server("b", [1]))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not r._pending and r.get_total_inflight() == 1
        task = asyncio.create_task(r.acquire_server("c", [1]))
        await asyncio.sleep(0)
        r.release_server(first, "a")  # resolves future; cancel before acquire resumes
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert r.get_total_inflight() == 0 and not r._request_to_server

    asyncio.run(run())


def test_busy_removal_is_atomic():
    async def run():
        r = router()
        await r.acquire_server("a", [1])
        with pytest.raises(ValueError):
            r.remove_servers(["s1", "s0"])
        assert r.get_all_servers() == ["s0", "s1"]
        with pytest.raises(ValueError, match="request_id"):
            r.release_server("s0")

    asyncio.run(run())
