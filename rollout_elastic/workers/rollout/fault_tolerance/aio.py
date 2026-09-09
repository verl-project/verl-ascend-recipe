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
"""Asyncio helpers for fault-tolerant best-effort cleanup operations.

`lenient_gather` is used wherever a list of best-effort coroutines (e.g. abort
all requests across replicas before weight sync) should not topple the calling
op when one replica is dead. The contract is: run all concurrently, log per-task
failures, and never raise.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any, Awaitable, Optional


async def lenient_gather(
    awaitables: Iterable[Awaitable[Any]],
    *,
    op_name: str,
    logger: Optional[logging.Logger] = None,
) -> list[Any]:
    """Run awaitables concurrently; log exceptions but never raise.

    Unlike `asyncio.gather(*, return_exceptions=False)`, this neither raises on
    the first exception nor cancels still-running siblings — every task gets to
    complete so the post-condition (e.g. "best-effort cleanup attempted on all
    replicas") is satisfied.

    Args:
        awaitables: Coroutines / futures to run.
        op_name: Short string included in log lines (e.g. "abort_all_requests").
        logger: Logger to receive WARNING-level per-failure messages. If None,
                a module logger is used.

    Returns:
        The raw list of results — successes are values, failures are exceptions
        (the same shape as `gather(return_exceptions=True)`).
    """
    log = logger or logging.getLogger(__name__)
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            log.warning("[FT] %s task #%d failed: %r", op_name, i, r)
    return results
