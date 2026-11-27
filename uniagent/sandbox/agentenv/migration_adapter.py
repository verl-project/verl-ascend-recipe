"""Route migrated parquet rows through the deployment AgentEnvTask."""

from copy import deepcopy


def normalize_tools_kwargs(tools_kwargs):
    result = deepcopy(tools_kwargs)
    task = (result or {}).get("task", {})
    sandbox = task.get("sandbox", {})
    if sandbox.get("provider") != "agentenv":
        return result
    source = task.get("name")
    if source not in {"swe_bench", "swe_rebench"}:
        raise ValueError("Unsupported migrated task: " + str(source))
    template = sandbox.get("template_id")
    if not template or sandbox.get("image") != template:
        raise ValueError("Missing or inconsistent AgentENV template ID")
    if sandbox.get("workdir") != "/testbed" or sandbox.get("architecture") != "arm64":
        raise ValueError("Expected migrated ARM64 /testbed template")
    task["name"] = "agentenv_swe_v2"
    task["metadata"]["source_task"] = source
    task["metadata"]["workdir"] = sandbox["workdir"]
    task["sandbox"] = {"provider": "e2b", "sandbox_kwargs": {"template": template, "user": "root"}}
    return result
