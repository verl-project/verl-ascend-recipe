# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ATRL agent loop: five tools, nudges, and structured compaction."""

from __future__ import annotations

import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from ..compaction import ConversationCompactor, next_step_section
from ..reward import combined_segment_rewards
from .tools import atrl_tools, workspace_dir, workspace_reward

_RESUME = (
    "[Context was compacted. Continue from the checkpoint's Current State and execute "
    "the Next Step now.\n\nNext Step from checkpoint:\n{next_step}]"
)


@register("atrl")
class AtrlAgentLoop(ToolAgentLoop):
    """Stock tool loop plus ATRL compaction and segment rewards."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        tools = atrl_tools()
        self.tools = {tool.name: tool for tool in tools}
        self.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tools
        ]
        self._compactor = ConversationCompactor()
        self._segments: list[dict[str, Any]] = []
        self._boundary_rewards: list[float] = []

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs: Any) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        workdir = str(extra.get("workspace_dir") or os.environ.get("ATRL_WORKSPACE", "/workspace"))
        token = workspace_dir.set(workdir)
        self._segments = []
        self._boundary_rewards = []
        self._workdir = workdir
        try:
            output = await super().run(sampling_params, priority=priority, **kwargs)
        finally:
            workspace_dir.reset(token)
        return self._attach_segments(output)

    async def _handle_processing_tools_state(self, agent_data: Any) -> AgentState:
        names = [call.name for call in agent_data.tool_calls]
        turns_left = max((self.max_assistant_turns or 0) - agent_data.assistant_turns, 0)
        self._nudge(agent_data, names, turns_left)
        state = await super()._handle_processing_tools_state(agent_data)
        if "submit" in names:
            return AgentState.TERMINATED
        if "run_verify" in names and state != AgentState.TERMINATED:
            await self._maybe_compact(agent_data, turns_left)
        return state

    def _nudge(self, agent_data: Any, names: list[str], turns_left: int) -> None:
        impl_margin = int(os.environ.get("ATRL_IMPL_NUDGE_MARGIN", "12"))
        verify_margin = int(os.environ.get("ATRL_VERIFY_NUDGE_MARGIN", "8"))
        has_impl = any(message.get("name") == "str_replace_editor" for message in agent_data.messages)
        has_verified = any(message.get("name") == "run_verify" for message in agent_data.messages)
        if not has_impl and "str_replace_editor" not in names and turns_left <= impl_margin:
            agent_data.messages.append(
                {
                    "role": "user",
                    "content": f"IMPORTANT: {turns_left} turns left. Create the implementation, then call run_verify.",
                }
            )
        elif has_impl and not has_verified and "run_verify" not in names and turns_left <= verify_margin:
            agent_data.messages.append(
                {
                    "role": "user",
                    "content": f"IMPORTANT: call run_verify now. ({turns_left} turns left.)",
                }
            )

    async def _maybe_compact(self, agent_data: Any, turns_left: int) -> None:
        min_remaining = int(os.environ.get("TRITON_COMPACTION_MIN_REMAINING_TURNS", "3"))
        self._compactor.notify_step()
        usage = (agent_data.extra_fields.get("prompt_tokens"), None) if agent_data.extra_fields else None
        if not self._compactor.should_compact(agent_data.messages, usage):
            return
        if turns_left < min_remaining:
            return
        mask = list(agent_data.response_mask or [])
        if not mask:
            return

        async def query(messages: list[dict[str, Any]]) -> str:
            prompt_ids = await self.ct_build_initial_tokens(messages, tools=self.tool_schemas)
            generated = await self.server_manager.generate(
                request_id=f"{agent_data.request_id}-compact-{len(self._segments)}",
                prompt_ids=prompt_ids,
                sampling_params={"max_tokens": 1024, "temperature": 0.2},
            )
            return self.tokenizer.decode(generated.token_ids, skip_special_tokens=True)

        rewritten = await self._compactor.compact(agent_data.messages, query)
        if rewritten is None:
            return
        self._segments.append(
            {
                "ids": list(agent_data.prompt_ids[-len(mask) :]),
                "mask": mask,
                "logprobs": list(agent_data.response_logprobs or []),
            }
        )
        self._boundary_rewards.append(workspace_reward(self._workdir))
        checkpoint = rewritten[-1]["content"]
        agent_data.messages = [
            *rewritten,
            {"role": "user", "content": _RESUME.format(next_step=next_step_section(checkpoint) or "Continue.")},
        ]
        agent_data.prompt_ids = await self.ct_build_initial_tokens(agent_data.messages, tools=self.tool_schemas)
        agent_data.response_mask = []
        agent_data.response_logprobs = []

    def _attach_segments(self, output: AgentLoopOutput) -> AgentLoopOutput:
        pieces = list(self._segments)
        if output.response_ids:
            pieces.append(
                {
                    "ids": list(output.response_ids),
                    "mask": list(output.response_mask),
                    "logprobs": list(output.response_logprobs or []),
                }
            )
        if not pieces:
            return output
        ids: list[int] = []
        mask: list[int] = []
        logprobs: list[float] = []
        bounds = []
        cursor = 0
        for piece in pieces:
            start = cursor
            ids.extend(piece["ids"])
            mask.extend(piece["mask"])
            logprobs.extend(piece["logprobs"])
            cursor += len(piece["ids"])
            bounds.append(
                {
                    "response_start": start,
                    "response_end": cursor,
                    "num_segments_in_rollout": len(pieces),
                }
            )
        final = workspace_reward(self._workdir)
        output.response_ids = ids[: self.response_length]
        output.response_mask = mask[: self.response_length]
        if output.response_logprobs is not None or any(piece["logprobs"] for piece in pieces):
            output.response_logprobs = logprobs[: self.response_length]
        output.reward_score = final
        output.extra_fields["segment_boundaries"] = bounds
        output.extra_fields["combined_rewards"] = combined_segment_rewards(self._boundary_rewards, final)
        output.extra_fields["boundary_rewards"] = list(self._boundary_rewards)
        return output
