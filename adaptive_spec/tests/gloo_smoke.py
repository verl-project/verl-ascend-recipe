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

"""Optional real PyTorch/Gloo check: python tests/gloo_smoke.py --output /absolute/out."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


def worker(rank, rendezvous, output):
    import torch.distributed as dist

    from adaptive_spec.reconcile import HistoricalEngineReconciler
    from adaptive_spec.vllm_plugin.v0_23 import _select_engine_k

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        rec = HistoricalEngineReconciler([5, 7, 9], engine_aggregation="max")

        def decide(contexts):
            if contexts == ["fail"]:
                raise ValueError("injected leader failure")
            return SimpleNamespace(physical_k=rec.select(contexts, fallback_k=5))

        state = dict(
            controller=SimpleNamespace(decide_engine=decide) if rank == 0 else None,
            tp_group=SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=dist.group.WORLD, ranks=[0, 1]),
            decision_round=0,
            tp_control_transport="packed",
        )
        values = []
        tensor_id = None
        for desired in [5] * 4 + [9] * 140 + [5] * 140:
            values.append(_select_engine_k(state, [5, desired]))
            current = id(state["packed_tp_control"].tensor)
            if tensor_id is not None:
                assert current == tensor_id
            tensor_id = current
        assert set(values) == {5, 7, 9} and values[0] == values[-1] == 5
        try:
            _select_engine_k(state, ["fail"])
        except RuntimeError as error:
            assert "injected leader failure" in str(error)
            propagated = True
        else:
            raise AssertionError("leader error was not propagated")
        Path(output, f"rank_{rank}.json").write_text(
            json.dumps(dict(rank=rank, decisions=values, error_propagated=propagated, buffer_reused=True)),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


def main():
    import torch
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        rendezvous = (Path(temp) / "rendezvous").as_uri()
        mp.spawn(worker, args=(rendezvous, str(args.output)), nprocs=2, join=True)
    ranks = [json.loads((args.output / f"rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert ranks[0]["decisions"] == ranks[1]["decisions"]
    report = dict(
        torch=torch.__version__,
        ranks=2,
        decisions=len(ranks[0]["decisions"]),
        agreement=True,
        error_propagated=all(r["error_propagated"] for r in ranks),
        buffer_reused=True,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
