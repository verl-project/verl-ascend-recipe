# Backend upgrades

AdaptiveSpec's policy is implemented independently of vLLM and verl. Integration uses two required hooks and one optional hook.

| Hook | Current interface | Where to adapt an interface change |
|---|---|---|
| vLLM plugin, required | `vllm.general_plugins` loads `adaptive_spec_plugin` | Plugin entry point and version adapter |
| verl manager / worker, required | `agent_loop_manager_class` loads `AdaptiveSpecAgentLoopManager` | `verl_integration.py`: manager construction, `_run_agent_loop`, and sampling metadata |
| verl router, when Elastic is enabled | `router_config_path` loads `ElasticInterleaveRouter` | `scheduler.py`: request IDs, acquire/release, and router parameters |

Inside the vLLM plugin, runner hooks set K before drafting, associate verification counts with the original request and preceding draft, copy and read variable-width draft buffers, and keep K consistent across asynchronous execution and TP ranks. The relevant methods include `execute_model`, `_bookkeeping_sync`, `propose_draft_token_ids`, draft copy/read, and `_update_states`, plus the TP CPU communication group.

| Upstream change | Required work | Expected effort |
|---|---|---|
| verl reward, loss, optimizer, or logging changes; integration interfaces unchanged | No AdaptiveSpec changes | None |
| Local changes to verl manager / router interfaces | Update the integration file and check metadata and routing | Low |
| vLLM / Ascend operator updates; runner interfaces and semantics unchanged | Check source fingerprints, run checks, and recollect the cost table | Low |
| Local changes to draft layout, verification counts, asynchronous execution, or TP interfaces | Update the version adapter and run CPU/NPU checks and cost collection | Medium |
| Runner / scheduler execution architecture changes | Add an adapter, reconnect state and draft transfer, and run integration regression checks | High |

The current adapter targets vLLM 0.23.0 and the [specified Ascend source](BACKEND.md). For an upgrade, run `adaptive-spec-check-backend --audit-source` to inspect these dependencies. Enable the adapter after version and source fingerprint checks pass.

Integration uses the general plugin mechanism and hooks methods in the process. It does not rewrite installed vLLM or vLLM-Ascend source files.
