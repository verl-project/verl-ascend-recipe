# Runtime upgrades

The recipe is pinned to verl commit `bcb638649a50e58494a8ddd92085ad1174f674b8`. The recorded device stack is vLLM 0.19.1 / vLLM-Ascend 0.19.1rc1. A backend-only upgrade does not require editing vLLM in this repository; it requires a compatible verl/runtime combination and checks of the contracts below.

## Integration contracts and expected work

| Integration | Dependencies to check | Files | Typical work |
| --- | --- | --- | --- |
| Manager registration | `rollout.agent.agent_loop_manager_class`, worker handles, server handles | `config.py`, `main_dapo_predictor_reorder.py` | None if the extension contract is unchanged; low for configuration/import changes |
| Bounded dispatch | `AgentLoopManager.generate_sequences`, awaitable worker RPCs, DataProto slicing/concat, per-row metrics | `epws_manager.py` | Low for field/method changes; medium if worker batching or RPC semantics change |
| Actor representation | FSDP worker/actor classes, hidden-state outputs, padding, dynamic batches, Ulysses/SP gather | `predictor_worker.py` | Medium for hidden-state layout, sharding or offload changes |
| Trainer lifecycle | DAPO fit order, reward/filtering, actor updates, checkpoint manager and dataloader resume | `predictor_dapo_trainer.py` | Medium for local API changes; high for training-loop or checkpoint orchestration redesign |
| Predictor math | PyTorch CPU tensor operations, checkpoint schema and projection/tap identity | `length_scheduler/` | No backend-specific changes if these inputs remain unchanged; re-fit/revalidate for a changed model or feature schema |

Inference server selection belongs to verl's `GlobalRequestLoadBalancer`. EWS has no backend logits, KV-cache or decode-progress hook to port. It does inherit the actor forward and rollout worker contracts; absence of a direct vLLM import is not evidence that every backend revision is compatible.

## Upgrade procedure

1. Read the target verl and rollout-backend release requirements. Select matching PyTorch/torch-npu, CANN, vLLM and vLLM-Ascend builds; do not upgrade just one package inside a shared container.
2. Create an isolated checkout/environment. Compare the pinned contracts above with the target source before changing `REQUIRED_VERL.txt`.
3. Run the CPU suites and configuration composition in [VALIDATION.md](VALIDATION.md). Keep D1 output dtype, grouped row order, response-mask labels and checkpoint behavior covered.
4. On allocated devices, execute the baseline, cold-start, activated D1, save/resume and optional D2 admission/fallback paths through the documented launcher.
5. Check output/reward alignment, the global in-flight bound, generation token count, predictor overhead and normal shutdown. Measure both rollout and full-step time.
6. Update the tested-version table and installer pin only after validation on that combination. If an actor/model/tokenizer/tap schema changes, collect matching D2 features and validate its checkpoint again.

## Deployment boundaries

The current worker extension supports FSDP/FSDP2. Megatron and other actor strategies are rejected when this predictor is enabled. Disabling the predictor retains the upstream worker/manager selection.

D2 quality monitoring identifies text prompts by token IDs. Multimodal inputs use FCFS rather than sharing evidence across images with identical text. The generic distribution API can be reused by another project, but that project must define its own features, provenance and validation data.
