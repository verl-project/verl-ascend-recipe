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

"""Source transform for targeted KV-pressure preemption in vLLM V1.

The installed scheduler preempts trailing requests until a KV-hungry request
can grow. Under sustained long-context pressure that can repeatedly evict all
other runnable work. This transform preempts only the request whose allocation
failed and continues scheduling the remaining runnable requests. It preserves
the upstream ``not preempted_reqs`` guard around waiting/resumed work: a request
preempted in the current async scheduler step must not be resumed in that same
step, because it may still be present in ``prev_step_scheduled_req_ids``.
"""

from __future__ import annotations

_RUNNING_START = "            # Schedule newly needed KV blocks for the request.\n"
_RUNNING_END = "            # Schedule the request.\n"

_TARGETED_BLOCK = """            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

            if new_blocks is None:
                # Keep KV pressure local: evict only the request that could not
                # grow, then let the remaining runnable requests make progress.
                self.running.pop(req_index)
                self._preempt_request(request, scheduled_timestamp)
                preempted_reqs.append(request)
                continue

"""


def patch_scheduler_source(source: str) -> str:
    """Patch one known vLLM Scheduler.schedule implementation.

    Fail closed if the installed source no longer matches the audited control
    flow, preventing silent application to a different scheduler revision.
    """

    if source.count(_RUNNING_START) != 1 or source.count(_RUNNING_END) != 1:
        raise RuntimeError("unexpected vLLM scheduler allocation markers")

    prefix, remainder = source.split(_RUNNING_START, 1)
    old_running_block, suffix = remainder.split(_RUNNING_END, 1)
    required_fragments = (
        "while True:",
        "self.running.pop()",
        "self._preempt_request(preempted_req, scheduled_timestamp)",
        "if preempted_req == request:",
        "if new_blocks is None:",
    )
    missing = [item for item in required_fragments if item not in old_running_block]
    if missing:
        raise RuntimeError(f"unexpected vLLM preemption block; missing {missing}")

    patched = prefix + _TARGETED_BLOCK + _RUNNING_END + suffix
    return patched
