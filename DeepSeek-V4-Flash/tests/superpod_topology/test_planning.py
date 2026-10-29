import itertools
import random

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def config(mode="sync", dynamic=False, validate=False, standalone_nodes=8):
    return OmegaConf.create(
        {
            "trainer": {"nnodes": 8, "n_gpus_per_node": 16, "device": "npu", "ascend_topology": {"enabled": True}},
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "nnodes": standalone_nodes,
                    "n_gpus_per_node": 16,
                    "tensor_model_parallel_size": 8,
                    "data_parallel_size": 8,
                    "pipeline_model_parallel_size": 1,
                }
            },
            "rollout": {"nnodes": standalone_nodes, "n_gpus_per_node": 16},
            "async_training": {"use_dynamic_resource_scheduling": dynamic, "use_trainer_do_validate": validate},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "distillation": {"enabled": False},
        }
    )


def spec(mode):
    return {"trainer_pool" if mode == "fully_async" else "global_pool": [16] * 8}


def cluster(sizes, npus=16, cpus=48):
    nodes, available = [], {}
    for sp, count in enumerate(sizes):
        for _ in range(count):
            index = len(nodes) + 1
            node = {
                "id": str(index),
                "ip": f"10.0.{index // 250}.{index % 250 + 1}",
                "sp": sp,
                "resource": f"node:10.0.{index // 250}.{index % 250 + 1}",
            }
            nodes.append(node)
            available[node["id"]] = {"NPU": npus, "CPU": cpus, node["resource"]: 1}
    return nodes, available


@pytest.mark.parametrize(
    "mode,dynamic,validate,names,groups",
    [
        ("v0", False, False, ["global_pool"], [(4, 4)]),
        ("sync", False, False, ["global_pool"], [(4, 4)]),
        ("colocate_async", False, False, ["global_pool"], [(4, 4)]),
        ("separate_async", False, False, ["global_pool", "rollout_pool_2", "rollout_pool_3"], [(4, 4), (4,), (4,)]),
        ("fully_async", False, False, ["trainer_pool", "rollout_pool_0", "rollout_pool_1"], [(1,) * 8, (4,), (4,)]),
        ("fully_async", True, False, ["trainer_pool", "rollout_pool_2", "rollout_pool_3"], [(4, 4), (4,), (4,)]),
        ("fully_async", False, True, ["trainer_pool", "rollout_pool_2", "rollout_pool_3"], [(4, 4), (4,), (4,)]),
        ("fully_async", True, True, ["trainer_pool", "rollout_pool_2", "rollout_pool_3"], [(4, 4), (4,), (4,)]),
    ],
)
def test_mode_manifest(modules, mode, dynamic, validate, names, groups):
    requests = modules.plan.requests_from_config(config(mode, dynamic, validate), spec(mode), mode=mode)
    assert [r.name for r in requests] == names
    assert [r.groups for r in requests] == groups
    assert sum(len(r.processes) for r in requests) == (8 if len(names) == 1 else 16)
    assert [r.cpus_per_process for r in requests] == [3] + [2] * (len(names) - 1)


@pytest.mark.parametrize(
    "mode,dynamic,validate,legal",
    [
        ("fully_async", True, False, True),
        ("fully_async", True, True, True),
        ("fully_async", False, True, False),
        ("fully_async", False, False, False),
        ("separate_async", False, False, False),
    ],
)
def test_zero_standalone(modules, mode, dynamic, validate, legal):
    cfg = config(mode, dynamic, validate, standalone_nodes=0)
    if not legal:
        with pytest.raises(ValueError, match="Zero standalone"):
            modules.plan.requests_from_config(cfg, spec(mode), mode=mode)
    else:
        req = modules.plan.requests_from_config(cfg, spec(mode), mode=mode)
        assert len(req) == 1 and req[0].groups == (4, 4)


@pytest.mark.parametrize(
    "mode,dedicated", [("sync", False), ("sync", True), ("separate_async", True), ("fully_async", True)]
)
def test_reward_and_teachers(modules, mode, dedicated):
    cfg = config(mode)
    cfg.reward.reward_model = {
        "enable": True,
        "enable_resource_pool": dedicated,
        "nnodes": 8,
        "n_gpus_per_node": 16,
        "rollout": {"name": "vllm", "tensor_model_parallel_size": 32},
    }
    cfg.distillation = {
        "enabled": True,
        "nnodes": 5,
        "n_gpus_per_node": 16,
        "teacher_models": {
            "teacher_model": {},
            "a": {"num_replicas": 1, "inference": {"name": "vllm", "tensor_model_parallel_size": 32}},
            "b": {"num_replicas": 1, "inference": {"name": "sglang", "tensor_model_parallel_size": 48}},
        },
    }
    specs = spec(mode)
    if dedicated and mode != "fully_async":
        specs["reward_pool"] = [16] * 8
    req = {r.name: r for r in modules.plan.requests_from_config(cfg, specs, mode=mode)}
    assert req["teacher_pool"].groups == (2, 3)
    if mode == "fully_async":
        assert all(req[f"rollout_pool_reward_{i}"].groups == (2,) for i in range(4))
    elif dedicated:
        assert req["reward_pool"].groups == (2,) * 4
    else:
        assert req["global_pool"].groups == (4, 4)


def test_small_standalone_replicas_count_physical_nodes(modules):
    cfg = config("fully_async")
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    cfg.actor_rollout_ref.rollout.data_parallel_size = 8
    req = modules.plan.requests_from_config(cfg, spec("fully_async"), mode="fully_async")
    assert sum(len(r.processes) for r in req) == 24  # 8 trainer + 16 independently reserved 8-NPU replicas
    with pytest.raises(ValueError, match="requires 24 physical"):
        modules.plan.plan_nodes(*cluster([30]), req, sp_plan=((0, 16),))


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda c: c.actor_rollout_ref.rollout.update({"tensor_model_parallel_size": 3}), "divisible by per-node"),
        (lambda c: c.actor_rollout_ref.rollout.update({"n_gpus_per_node": 8}), "same n_gpus"),
        (lambda c: c.actor_rollout_ref.rollout.update({"disaggregation": {"enabled": True}}), "PD backends"),
        (lambda c: c.actor_rollout_ref.rollout.update({"name": "trtllm"}), "vllm/sglang"),
    ],
)
def test_reject_unsupported_before_reservation(modules, change, match):
    cfg = config()
    change(cfg)
    with pytest.raises(ValueError, match=match):
        modules.plan.requests_from_config(cfg, spec("sync"))


def test_auto_compaction_and_exact_plan(modules):
    p = modules.plan
    nodes, available = cluster([12, 8, 4])
    req = [p.PoolRequest("a", (16,) * 8, groups=(4, 4))]
    chosen = p.plan_nodes(nodes, available, req)
    assert {n["sp"] for n in chosen["a"]} == {1}  # exact-fitting 8-node block
    chosen = p.plan_nodes(nodes, available, req, sp_plan=((2, 4), (0, 4)))
    assert [n["sp"] for n in chosen["a"]] == [2] * 4 + [0] * 4
    with pytest.raises(RuntimeError, match="No feasible"):
        p.plan_nodes(nodes, available, req, sp_plan=((2, 3), (0, 5)))


def test_joint_capacity_matching_and_heterogeneous_order(modules):
    p = modules.plan
    nodes, available = cluster([3, 2], cpus=32)
    for node in nodes[3:]:
        available[node["id"]]["CPU"] = 48
    req = [p.PoolRequest("rollout", (16,) * 3, 2, (3,)), p.PoolRequest("trainer", (16,) * 2, 3, (2,))]
    chosen = p.plan_nodes(nodes, available, req)
    assert {n["sp"] for n in chosen["trainer"]} == {1}
    req = [p.PoolRequest("teacher", (16,) * 5, 2, (2, 3))]
    chosen = p.plan_nodes(nodes, available, req)["teacher"]
    assert [n["sp"] for n in chosen] == [1, 1, 0, 0, 0]  # cannot globally sort heterogeneous groups


@pytest.mark.parametrize("missing", ["CPU", "NPU", "resource"])
def test_busy_nodes_not_eligible(modules, missing):
    nodes, available = cluster([2])
    available[nodes[0]["id"]][nodes[0]["resource"] if missing == "resource" else missing] = 0
    with pytest.raises(RuntimeError, match="No feasible"):
        modules.plan.plan_nodes(nodes, available, [modules.plan.PoolRequest("a", (16, 16))])


def test_random_small_plans_against_exhaustive_oracle(modules):
    p = modules.plan
    rng = random.Random(20261028)
    for _ in range(80):
        sizes = [rng.randint(1, 3) for _ in range(3)]
        nodes, available = cluster(sizes, npus=1, cpus=3)
        groups = (rng.randint(1, 2), rng.randint(1, 2))
        total = sum(groups)
        feasible = []
        for assignment in itertools.permutations(range(len(nodes)), total):
            if any(
                len({nodes[i]["sp"] for i in assignment[start : start + size]}) > 1
                for start, size in [(0, groups[0]), (groups[0], groups[1])]
            ):
                continue
            used = {nodes[i]["sp"] for i in assignment}
            feasible.append((len(used), sum(sizes[sp] for sp in used)))
        req = [p.PoolRequest("a", (1,) * total, groups=groups)]
        if not feasible:
            with pytest.raises(RuntimeError):
                p.plan_nodes(nodes, available, req)
        else:
            selected = p.plan_nodes(nodes, available, req)["a"]
            used = {n["sp"] for n in selected}
            assert (len(used), sum(sizes[sp] for sp in used)) == min(feasible)
            assert len({n["id"] for n in selected}) == total


def test_timeouts_do_not_relax_constraints(modules, monkeypatch, caplog):
    p = modules.plan
    with monkeypatch.context() as m:
        times = iter([0, 100])
        m.setattr(p.time, "monotonic", lambda: next(times))
        with pytest.raises(TimeoutError):
            p.plan_nodes(*cluster([3, 2]), [p.PoolRequest("a", (16, 16))], timeout_s=1)
    with monkeypatch.context() as m:
        # Seed max-flow: deadline, solve, match, two flow iterations. Timeout at optimization.
        times = iter([0] * 5 + [100] * 100)
        m.setattr(p.time, "monotonic", lambda: next(times))
        result = p.plan_nodes(*cluster([3, 2]), [p.PoolRequest("a", (16, 16))], timeout_s=1)
        assert len(result["a"]) == 2
        assert "best feasible plan" in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        "[{sp:2, nodes:8}]",
        ["{sp:2, nodes:8}"],
        ["{sp:2, nodes:8},{sp:6, nodes:8}"],
        ["sp:2", "nodes:8"],
        {"sp": 2, "nodes": 8},
        False,
        0,
    ],
)
def test_sp_plan_bad_shape_diagnostic(modules, bad):
    with pytest.raises(ValueError, match="got") as exc:
        modules.topology.TopologyPolicy.from_config({"sp_plan": bad})
    assert 'Pass one complete shell argument: "trainer.ascend_topology.sp_plan=' in str(exc.value)


@pytest.mark.parametrize(
    "bad",
    [
        [{"sp": True, "nodes": 8}],
        [{"sp": 2, "nodes": 0}],
        [{"sp": -1, "nodes": 8}],
        [{"sp": 2, "nodes": 8}, {"sp": 2, "nodes": 8}],
    ],
)
def test_sp_plan_bad_value(modules, bad):
    with pytest.raises(ValueError):
        modules.topology.TopologyPolicy.from_config({"sp_plan": bad})


def test_removed_layout_and_disabled_path(modules, monkeypatch):
    with pytest.raises(ValueError, match="removed"):
        modules.topology.TopologyPolicy.from_config({"sp_layout": []})
    monkeypatch.setattr(modules.topology, "topology_from_config", lambda _: pytest.fail("must not initialize"))
    assert modules.topology.prepare_topology({}, {}, mode="unsupported") is None


def test_replica_alignment_rejected_before_pg(modules):
    with pytest.raises(ValueError, match="not node aligned"):
        modules.plan.pool_request("teacher", (4, 4, 4), [(0, 3), (3, 7), (7, 12)])
    cfg = config()
    cfg.trainer.nnodes = 6
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 3
    # 96 is divisible by replica world 24, but RolloutReplica rejects 24 % 16.
    with pytest.raises(ValueError, match="divisible by per-node"):
        modules.plan.requests_from_config(cfg, {"global_pool": [16] * 6})


def test_invalid_mode_and_unmerged_fully_async_dimensions(modules):
    with pytest.raises(ValueError, match="Unsupported"):
        modules.plan.requests_from_config(config(), spec("sync"), mode="typo")
    cfg = config("fully_async")
    cfg.actor_rollout_ref.rollout.nnodes = 0
    with pytest.raises(ValueError, match="copy top-level rollout"):
        modules.plan.requests_from_config(cfg, spec("fully_async"), mode="fully_async")


@pytest.mark.parametrize("mode", ["separate_async", "fully_async"])
def test_async_reward_requires_dedicated_resources(modules, mode):
    cfg = config(mode)
    cfg.reward.reward_model.enable = True
    with pytest.raises(ValueError, match="enable_resource_pool=true"):
        modules.plan.requests_from_config(cfg, spec(mode), mode=mode)


@pytest.mark.parametrize("fully_async", [False, True])
def test_real_hydra_configuration(source, modules, fully_async):
    directory = source / ("verl/experimental/fully_async_policy/config" if fully_async else "verl/trainer/config")
    overrides = [
        "trainer.device=npu",
        "trainer.nnodes=8",
        "trainer.n_gpus_per_node=16",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=8",
        "actor_rollout_ref.rollout.data_parallel_size=8",
        "trainer.ascend_topology.enabled=true",
        "trainer.ascend_topology.sp_plan=[{sp:2, nodes:8}, {sp:6, nodes:8}]"
        if fully_async
        else "trainer.ascend_topology.sp_plan=[{sp:2, nodes:8}]",
    ]
    if fully_async:
        overrides += [
            f"hydra.searchpath=[file://{source}/verl/trainer/config]",
            "rollout.nnodes=8",
            "rollout.n_gpus_per_node=16",
        ]
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        cfg = compose(config_name="fully_async_ppo_trainer" if fully_async else "ppo_trainer", overrides=overrides)
    if fully_async:
        cfg.actor_rollout_ref.rollout.nnodes = cfg.rollout.nnodes
        cfg.actor_rollout_ref.rollout.n_gpus_per_node = cfg.rollout.n_gpus_per_node
    policy = modules.topology.TopologyPolicy.from_config(cfg.trainer.ascend_topology)
    mode = "fully_async" if fully_async else "sync"
    req = modules.plan.requests_from_config(cfg, spec(mode), mode=mode)
    assert sum(c for _, c in policy.sp_plan) == sum(len(r.processes) for r in req)
