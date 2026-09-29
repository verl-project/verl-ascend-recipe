"""Small async primitive used by the recipe-side EPWS manager."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


async def event_driven_map(
    items: Sequence[_Item],
    submit: Callable[[_Item], Awaitable[_Result]],
    *,
    max_concurrency: int,
) -> list[_Result]:
    """Run at most ``max_concurrency`` items and refill on each completion.

    Results preserve the supplied item order. The caller determines admission
    priority by ordering ``items`` before this function is called.
    """

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if not items:
        return []

    results: list[_Result | None] = [None] * len(items)
    active: dict[asyncio.Task[_Result], int] = {}
    next_index = 0

    def refill() -> None:
        nonlocal next_index
        while next_index < len(items) and len(active) < max_concurrency:
            index = next_index
            next_index += 1
            active[asyncio.create_task(submit(items[index]))] = index

    refill()
    try:
        while active:
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = active.pop(task)
                results[index] = task.result()
            refill()
    except BaseException:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        raise

    return [result for result in results if result is not None]
