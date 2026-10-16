"""Recipe-side AgentLoopManager with event-driven waiting-pool admission."""

from __future__ import annotations

import math
from typing import Any

from verl.experimental.agent_loop import AgentLoopManager as VerlAgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await

from .length_scheduler.event_loop import event_driven_map
from .length_scheduler.lifecycle import ActivationConfig, PredictorLifecycle
from .length_scheduler.scheduler import EPWSWaitingPool
from .length_scheduler.types import WaitingRequest

_PREDICTED_WORK_KEY = "epws_predicted_work"
_PREDICTOR_ACTIVE_KEY = "epws_predictor_active"


def _as_list(values: Any) -> list[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _predicted_work(prompts: DataProto) -> list[float | None]:
    values = None
    if prompts.batch is not None and _PREDICTED_WORK_KEY in prompts.batch:
        values = prompts.batch[_PREDICTED_WORK_KEY]
    elif _PREDICTED_WORK_KEY in prompts.non_tensor_batch:
        values = prompts.non_tensor_batch[_PREDICTED_WORK_KEY]
    if values is None:
        return [None] * len(prompts)

    rows = _as_list(values)
    if len(rows) != len(prompts):
        raise ValueError(f"{_PREDICTED_WORK_KEY} has {len(rows)} rows for batch size {len(prompts)}")
    normalized = []
    for value in rows:
        value = float(value)
        normalized.append(value if math.isfinite(value) and value >= 0 else None)
    return normalized


def _prompt_ids(prompts: DataProto) -> list[str]:
    for key in ("uid", "index"):
        if key in prompts.non_tensor_batch:
            values = _as_list(prompts.non_tensor_batch[key])
            if len(values) == len(prompts):
                return [str(value) for value in values]
    return [f"prompt-{index}" for index in range(len(prompts))]


class EPWSAgentLoopManager(VerlAgentLoopManager):
    """Refill a bounded global request window whenever one request completes.

    This class uses verl's documented ``agent_loop_manager_class`` extension
    point. Server selection remains delegated to verl's existing
    ``GlobalRequestLoadBalancer``; EPWS controls only request admission.
    """

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        if len(prompts) == 0:
            return prompts

        predictor_active = bool(prompts.meta_info.get(_PREDICTOR_ACTIVE_KEY, False))
        lifecycle = PredictorLifecycle(ActivationConfig(min_epoch=0, min_samples=0))
        lifecycle.update(epoch=0, samples=0, ready=predictor_active)
        pool = EPWSWaitingPool(lifecycle)

        prompt_ids = _prompt_ids(prompts)
        work = _predicted_work(prompts)
        pool.add(
            WaitingRequest(
                request_id=str(index),
                arrival_index=index,
                prompt_id=prompt_ids[index],
                predicted_work=work[index],
            )
            for index in range(len(prompts))
        )

        admission_order = []
        reasons = []
        while len(pool):
            decision = pool.pop_next()
            admission_order.append(int(decision.request_id))
            reasons.append(decision.reason)

        cfg = self.config.trainer.get("predictor_reorder", {}).get("epws", {})
        slots_per_server = int(cfg.get("slots_per_server", 8))
        if slots_per_server < 1:
            raise ValueError("trainer.predictor_reorder.epws.slots_per_server must be positive")
        default_concurrency = max(len(self.server_handles) * slots_per_server, 1)
        configured_concurrency = cfg.get("max_concurrent_requests")
        max_concurrency = default_concurrency if configured_concurrency is None else int(configured_concurrency)
        if max_concurrency < 1:
            raise ValueError("trainer.predictor_reorder.epws.max_concurrent_requests must be positive")

        worker_loads = [0] * len(self.agent_loop_workers)

        async def submit(index: int) -> tuple[int, DataProto]:
            worker_index = min(range(len(worker_loads)), key=lambda worker: (worker_loads[worker], worker))
            worker_loads[worker_index] += 1
            try:
                output = await self.agent_loop_workers[worker_index].generate_sequences.remote(
                    prompts[index : index + 1]
                )
                return index, output
            finally:
                worker_loads[worker_index] -= 1

        completed = await event_driven_map(
            admission_order,
            submit,
            max_concurrency=min(max_concurrency, len(prompts)),
        )
        outputs = [output for _, output in sorted(completed, key=lambda row: row[0])]
        output = DataProto.concat(outputs)

        metrics = [row.meta_info.pop("metrics") for row in outputs]
        timing = self._performance_metrics(metrics, output)
        timing["epws/max_concurrent_requests"] = max_concurrency
        timing["epws/predicted_long_first"] = sum(reason == "predicted_long_first" for reason in reasons)
        timing["epws/fcfs_fallback"] = sum(reason == "fcfs_fallback" for reason in reasons)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output
