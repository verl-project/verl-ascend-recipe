from __future__ import annotations

import asyncio

import pytest

from dapo_predictor.length_scheduler.event_loop import event_driven_map


def test_event_loop_refills_on_completion_and_preserves_order() -> None:
    async def exercise() -> tuple[list[int], int, list[tuple[str, int]]]:
        running = 0
        maximum_running = 0
        events = []
        release_first = asyncio.Event()

        async def submit(item: int) -> int:
            nonlocal running, maximum_running
            running += 1
            maximum_running = max(maximum_running, running)
            events.append(("start", item))
            if item == 0:
                await release_first.wait()
            elif item == 1:
                await asyncio.sleep(0)
            else:
                # Item 2 can only start after item 1 frees a slot. Releasing
                # item 0 here makes the refill assertion deterministic on
                # event loops with coarse timer granularity (notably Windows).
                release_first.set()
                await asyncio.sleep(0)
            running -= 1
            events.append(("done", item))
            return item * 10

        values = await event_driven_map([0, 1, 2], submit, max_concurrency=2)
        return values, maximum_running, events

    values, maximum_running, events = asyncio.run(exercise())
    assert values == [0, 10, 20]
    assert maximum_running == 2
    assert events.index(("start", 2)) < events.index(("done", 0))


def test_event_loop_rejects_invalid_concurrency() -> None:
    async def submit(item: int) -> int:
        return item

    with pytest.raises(ValueError, match="positive"):
        asyncio.run(event_driven_map([1], submit, max_concurrency=0))


def test_event_loop_preserves_none_results() -> None:
    async def submit(item: int) -> None:
        del item
        return None

    assert asyncio.run(event_driven_map([1, 2], submit, max_concurrency=1)) == [None, None]


def test_event_loop_propagates_failure() -> None:
    async def submit(item: int) -> int:
        if item == 1:
            raise RuntimeError("boom")
        await asyncio.sleep(0.05)
        return item

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(event_driven_map([0, 1, 2], submit, max_concurrency=2))
