"""Local Ray 2.48 smoke test with custom NPU resources and a fake DCMI library.

No real NPU kernels, model loading, HCCL, or weight synchronization are exercised.
Run in its own process: python ray_smoke.py --source /path/to/patched/verl
"""

import argparse
import importlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from test_dcmi import assert_stopped, build_library, wait_for_pids


def main(source):
    import ray
    from ray.util import placement_group_table
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    report = {"ray": ray.__version__, "python": sys.version.split()[0], "checks": []}
    with tempfile.TemporaryDirectory(prefix="verl-sp-ray-") as temporary:
        root = Path(temporary)
        runtime = root / "runtime"
        # Make only the patch's real modules importable in workers. Empty package
        # initializers deliberately replace torch/model imports for this CPU test.
        for relative in ["verl", "verl/single_controller", "verl/single_controller/ray", "verl/utils"]:
            directory = runtime / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("")
        for relative in [
            "verl/single_controller/ray/ascend_topology.py",
            "verl/single_controller/ray/ascend_topology_plan.py",
            "verl/utils/ascend_dcmi.py",
            "verl/utils/ascend_dcmi_probe.py",
        ]:
            shutil.copy2(source / relative, runtime / relative)
        sys.path.insert(0, str(runtime))
        topo = importlib.import_module("verl.single_controller.ray.ascend_topology")
        plan = importlib.import_module("verl.single_controller.ray.ascend_topology_plan")
        good_dir, hang_dir = root / "good", root / "hang"
        good_dir.mkdir()
        hang_dir.mkdir()
        good_lib, _ = build_library(good_dir)
        hang_lib, pids_file = build_library(hang_dir, hang=True)
        ray.init(
            num_cpus=6,
            resources={"NPU": 2},
            include_dashboard=False,
            object_store_memory=80 * 1024 * 1024,
            _temp_dir=str(root / "ray"),
            runtime_env={"env_vars": {"PYTHONPATH": str(runtime)}},
        )
        try:
            assert ray.get(topo._probe_node.remote(str(good_lib), 3), timeout=20) == 6
            report["checks"].append("Ray probe imports and isolated DCMI result")
            cfg = {
                "trainer": {
                    "device": "npu",
                    "nnodes": 1,
                    "n_gpus_per_node": 2,
                    "ascend_topology": {
                        "enabled": True,
                        "dcmi_library": str(good_lib),
                        "probe_timeout_s": 3,
                        "placement_timeout_s": 15,
                        "allocation_retries": 0,
                        "sp_plan": [{"sp": 6, "nodes": 1}],
                    },
                },
                "actor_rollout_ref": {
                    "rollout": {"name": "vllm", "n_gpus_per_node": 2, "tensor_model_parallel_size": 2}
                },
            }
            topology = topo.prepare_topology(cfg, {"global_pool": [2]}, mode="sync")
            request = plan.PoolRequest("global_pool", (2,), 3)
            again = topo.topology_from_config(cfg)
            assert again.name == topology.name
            assert again.allocator == topology.allocator
            report["checks"].append("public prepare_topology config/manifest path and job-scoped allocator reuse")
            pgs = topology.create_placement_groups(request.schemes(), "global_pool")
            assert pgs == topology.create_placement_groups(request.schemes(), "global_pool")
            info = placement_group_table(pgs[0])
            node_id = ray.get_runtime_context().get_node_id()
            assert info["state"] == "CREATED" and set(info["bundles_to_node_id"].values()) == {node_id}

            @ray.remote(num_cpus=0, resources={"NPU": 1})
            def where():
                return ray.get_runtime_context().get_node_id()

            refs = [
                where.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pgs[0], placement_group_bundle_index=i
                    )
                ).remote()
                for i in range(2)
            ]
            assert ray.get(refs, timeout=20) == [node_id, node_id]
            report["checks"].append("reserved PGs, idempotent claims, exact bundle node placement")
            try:
                topology.create_placement_groups([[{"NPU": 1, "CPU": 3}]], "global_pool")
                raise AssertionError("Changed demand unexpectedly accepted")
            except ray.exceptions.RayTaskError as exc:
                assert "differs" in str(exc)
            report["checks"].append("frozen manifest rejects changed demand")

            probe = topo._probe_node.remote(str(hang_lib), 30)
            pids = wait_for_pids(pids_file)
            ray.cancel(probe, force=True)
            try:
                ray.get(probe, timeout=10)
                raise AssertionError("Cancelled probe returned normally")
            except (ray.exceptions.TaskCancelledError, ray.exceptions.WorkerCrashedError):
                pass
            assert_stopped(pids)
            report["checks"].append("Ray force cancellation kills hung DCMI query and forked child")
            # PG lifetime is job-scoped. Test explicit rollback while the driver
            # remains alive; killing only the allocator is not a job shutdown.
            ray.get(topology.allocator._rollback.remote(), timeout=15)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and placement_group_table(pgs[0]).get("state") not in (None, "REMOVED"):
                time.sleep(0.1)
            assert placement_group_table(pgs[0]).get("state") in (None, "REMOVED")
            ray.kill(topology.allocator, no_restart=True)
            report["checks"].append("explicit allocator rollback releases all owned PGs")
        finally:
            ray.shutdown()
        sys.path.remove(str(runtime))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve())
