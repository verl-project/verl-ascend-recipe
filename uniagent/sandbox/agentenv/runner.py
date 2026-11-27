"""Deployment entry point for the fresh uni-agent task runner."""

from migration_adapter import normalize_tools_kwargs
from task import AgentEnvTask as AgentEnvTask  # Import registers the task and Claude agent.
from uni_agent.framework.task_runner import run_task as native_run_task


async def run_task(**kwargs):
    kwargs["tools_kwargs"] = normalize_tools_kwargs(kwargs.get("tools_kwargs"))
    return await native_run_task(**kwargs)
