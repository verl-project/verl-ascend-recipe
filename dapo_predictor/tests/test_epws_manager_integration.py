from __future__ import annotations

# ruff: noqa: E402 -- integration imports intentionally follow importorskip
import asyncio
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verl")

from dapo_predictor.epws_manager import EPWSAgentLoopManager
from verl.protocol import DataProto


class _RemoteGenerate:
    def __init__(self, starts: list[int]):
        self.starts = starts

    async def remote(self, batch: DataProto) -> DataProto:
        row_id = int(batch.batch["row_id"][0])
        self.starts.append(row_id)
        await asyncio.sleep(0)
        batch.meta_info = {"metrics": []}
        return batch


class _FakeWorker:
    def __init__(self, starts: list[int]):
        self.generate_sequences = _RemoteGenerate(starts)


def test_manager_admits_long_first_and_restores_original_batch_order() -> None:
    starts: list[int] = []
    manager = EPWSAgentLoopManager.__new__(EPWSAgentLoopManager)
    manager.config = SimpleNamespace(
        trainer={"predictor_reorder": {"epws": {"slots_per_server": 1, "max_concurrent_requests": 1}}}
    )
    manager.server_handles = [object()]
    manager.agent_loop_workers = [_FakeWorker(starts), _FakeWorker(starts)]
    manager._performance_metrics = MethodType(lambda self, metrics, output: {}, manager)

    prompts = DataProto.from_dict(
        tensors={
            "row_id": torch.tensor([0, 1, 2]),
            "epws_predicted_work": torch.tensor([10.0, 100.0, 50.0]),
        },
        non_tensors={"uid": np.asarray(["p0", "p1", "p2"], dtype=object)},
        meta_info={"epws_predictor_active": True},
    )

    output = manager.generate_sequences(prompts)

    assert starts == [1, 2, 0]
    assert output.batch["row_id"].tolist() == [0, 1, 2]
    assert output.meta_info["timing"]["epws/predicted_long_first"] == 3
    assert output.meta_info["timing"]["epws/fcfs_fallback"] == 0
