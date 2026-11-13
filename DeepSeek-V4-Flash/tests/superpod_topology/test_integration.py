import asyncio
import logging
import sys
import types
from dataclasses import dataclass, field
from enum import IntEnum
from unittest.mock import Mock

import numpy as np
import pytest
from conftest import definitions
from omegaconf import OmegaConf, open_dict
from test_planning import cluster, config


class Role(IntEnum):
    Actor = 1
    ActorRollout = 2
    ActorRolloutRef = 3
    Critic = 4
    RefPolicy = 5
    Rollout = 6
    RewardModel = 7
    TeacherModel = 8


class PG:
    def __init__(self, index, bundles, node=None):
        self.id = types.SimpleNamespace(hex=lambda: str(index))
        self.bundle_count = len(bundles)
        self.bundle_specs = bundles
        self.node = node

    def ready(self):
        return self


class Reservations:
    """A planner-backed fake allocator used to verify consumers of real methods."""

    def __init__(self, plan, requests):
        self.name, self.policy = "job", "policy"
        self.requests = {r.name: r for r in requests}
        self.selected = plan.plan_nodes(*cluster([8, 8, 8, 8]), requests)
        self.pgs = {
            r.name: [
                PG(f"{r.name}:{i}", bundle, node)
                for i, (bundle, node) in enumerate(zip(r.schemes(), self.selected[r.name], strict=True))
            ]
            for r in requests
        }
        self.order = {p.id.hex(): i for i, p in enumerate(p for group in self.pgs.values() for p in group)}
        self.claimed = []

    def create_placement_groups(self, schemes, pool_name, detached=False):
        assert not detached
        assert schemes == self.requests[pool_name].schemes()
        self.claimed.append(pool_name)
        return self.pgs[pool_name]

    def sort_placement_groups(self, pgs):
        return sorted(pgs, key=lambda p: self.order[p.id.hex()])


@pytest.fixture
def resource_types(source):
    env = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "np": np,
        "logger": logging.getLogger(__name__),
        "get_device_name": lambda: "npu",
        "get_platform": lambda: types.SimpleNamespace(device_name="npu", ray_resource_name=lambda: "NPU"),
        "get_random_string": lambda **_: "random",
    }
    definitions(source, "verl/single_controller/base/worker_group.py", {"ResourcePool"}, env)
    definitions(
        source,
        "verl/single_controller/ray/base.py",
        {"RayResourcePool", "SubRayResourcePool", "ResourcePoolManager", "split_resource_pool", "merge_resource_pool"},
        env,
    )
    return env


def alias(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "mode,dynamic,validate,standalone_nodes",
    [
        ("v0", False, False, 8),
        ("sync", False, False, 8),
        ("colocate_async", False, False, 8),
        ("separate_async", False, False, 8),
        ("fully_async", False, False, 8),
        ("fully_async", True, False, 8),
        ("fully_async", False, True, 8),
        ("fully_async", True, True, 0),
    ],
)
def test_real_entrypoints_and_rollout_consumers(
    source, modules, resource_types, monkeypatch, mode, dynamic, validate, standalone_nodes
):
    cfg = config(mode, dynamic, validate, standalone_nodes)
    cfg.actor_rollout_ref.model = {}
    cfg.reward.reward_model.update({"nnodes": 1, "n_gpus_per_node": 16})
    cfg.actor_rollout_ref.rollout.update({"prometheus": {"enable": False}, "disable_log_stats": True})
    env = dict(resource_types)
    env.update(
        {
            "Role": Role,
            "ray": types.SimpleNamespace(remote=lambda cls: cls),
            "ActorRolloutRefWorker": object,
            "TrainingWorker": object,
            "need_reference_policy": lambda _: False,
            "need_critic": lambda _: False,
            "is_distillation_enabled": lambda _: False,
        }
    )
    calls = []

    def prepare(config, specs, *, mode):
        reservations = Reservations(modules.plan, modules.plan.requests_from_config(config, specs, mode=mode))
        calls.append((mode, reservations))
        return reservations

    alias(
        monkeypatch,
        "verl.single_controller.ray.ascend_topology",
        prepare_topology=prepare,
        topology_from_config=lambda _: calls[0][1],
    )
    alias(monkeypatch, "verl.trainer.ppo.ray_trainer", ResourcePoolManager=env["ResourcePoolManager"])
    if mode == "v0":
        definitions(source, "verl/trainer/main_ppo_v0.py", {"init_resource_pool_mgr"}, env, owner="BaseTaskRunner")
        manager = env["init_resource_pool_mgr"](types.SimpleNamespace(mapping={}), cfg)
    elif mode == "fully_async":
        definitions(source, "verl/experimental/separation/utils.py", {"create_resource_pool_manager"}, env)
        manager = env["create_resource_pool_manager"](cfg, [Role.ActorRollout if dynamic or validate else Role.Actor])
    else:
        definitions(source, "verl/trainer/ppo/v1/trainer_base.py", {"_init_resource_pool_mgr"}, env, owner="PPOTrainer")
        trainer = types.SimpleNamespace(config=cfg, trainer_mode=mode)
        env["_init_resource_pool_mgr"](trainer)
        manager = trainer.resource_pool_manager
    assert len(calls) == 1 and calls[0][0] == mode
    reservations = calls[0][1]
    manager.create_resource_pool()
    training_name = "trainer_pool" if mode == "fully_async" else "global_pool"
    training_pool = manager.resource_pool_dict[training_name]
    training_pgs = training_pool.get_placement_groups(device_name="npu")
    worker_group = types.SimpleNamespace(
        world_size=128, workers=[n for pg in training_pgs for n in [pg.node] * pg.bundle_count]
    )
    replica_env = dict(resource_types)

    class WorkerGroup:
        def __init__(self, resource_pool, **kwargs):
            self.workers = [
                p.node for p in resource_pool.get_placement_groups(device_name="npu") for _ in range(p.bundle_count)
            ]

    replica_env.update(
        RayWorkerGroup=WorkerGroup, RolloutMode=types.SimpleNamespace(STANDALONE="standalone", HYBRID="hybrid")
    )
    definitions(
        source,
        "verl/workers/rollout/replica.py",
        {"prepare_standalone", "init_standalone", "init_hybrid"},
        replica_env,
        owner="RolloutReplica",
    )

    class Replica:
        def __init__(self, replica_rank, config, model_config, gpus_per_node):
            self.replica_rank = replica_rank
            self.world_size = modules.plan.replica_world_size(config)
            self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
            self.nnodes = self.world_size // self.gpus_per_replica_node
            self.is_reward_model = self.is_teacher_model = False
            self.name_suffix = ""
            self.resource_pool = None

        def get_ray_class_with_init_args(self):
            return None

        async def launch_servers(self):
            assert len(self.workers) == self.world_size
            assert len({n["sp"] for n in self.workers}) == 1
            self._server_handle = self.replica_rank
            self._server_address = str(self.replica_rank)

    for name in ("prepare_standalone", "init_standalone", "init_hybrid"):
        setattr(Replica, name, replica_env[name])
    llm_env = {"asyncio": asyncio, "RLInsightLogger": types.SimpleNamespace(enabled=lambda: False)}
    definitions(
        source, "verl/workers/rollout/llm_server.py", {"_initialize_llm_servers"}, llm_env, owner="LLMServerManager"
    )

    class LLM:
        _initialize_llm_servers = llm_env["_initialize_llm_servers"]

        def __init__(self, wg, start=0):
            self.config, self.rollout_config = cfg, cfg.actor_rollout_ref.rollout
            self.worker_group, self.start_rank = wg, start
            self.rollout_replica_class, self.model_config = Replica, None

    hybrid = mode != "fully_async" or dynamic or validate
    if mode == "fully_async":
        # Execute the actual two-phase hybrid/standalone method, including rank offsets.
        async_env = {"LLMServerManager": LLM, "open_dict": open_dict, "time": __import__("time")}
        tree = __import__("ast").parse(
            (source / "verl/experimental/fully_async_policy/fully_async_rollouter.py").read_text()
        )
        cls = next(n for n in tree.body if getattr(n, "name", None) == "FullyAsyncLLMServerManager")
        cls.body = [n for n in cls.body if getattr(n, "name", None) == "_initialize_llm_servers"]
        exec(
            compile(__import__("ast").Module(body=[cls], type_ignores=[]), "fully_async_rollouter.py", "exec"),
            async_env,
        )
        llm = async_env["FullyAsyncLLMServerManager"](worker_group if hybrid else None)
        llm.hybrid_replicas, llm.prometheus_server_addresses = {}, []
        asyncio.run(llm._initialize_llm_servers())
        assert len(llm.hybrid_replicas) == (2 if hybrid else 0)
    else:
        llm = LLM(worker_group)
        asyncio.run(llm._initialize_llm_servers())
        if mode == "separate_async":
            llm = LLM(None, start=len(llm.rollout_replicas))
            asyncio.run(llm._initialize_llm_servers())
    assert set(reservations.claimed) == set(reservations.requests)
    assert sum(len(p) for p in reservations.pgs.values()) == (
        8 if mode in ("v0", "sync", "colocate_async") or standalone_nodes == 0 else 16
    )


def test_subpool_offsets_and_merge_order(modules, resource_types):
    env = resource_types
    req = [modules.plan.PoolRequest("teacher_pool", (16,) * 5, groups=(2, 3)), modules.plan.PoolRequest("other", (16,))]
    topology = Reservations(modules.plan, req)
    pool = env["RayResourcePool"]([16] * 5, name_prefix="teacher_pool", max_colocate_count=3, topology=topology)
    a, b = env["split_resource_pool"](pool, [32, 48])
    nested = env["split_resource_pool"](b, 16)
    assert [p.start_bundle_index for p in nested] == [32, 48, 64]
    assert all(p.topology is topology and p.pgs is pool.pgs for p in nested)
    other = env["RayResourcePool"]([16], name_prefix="other", max_colocate_count=3, topology=topology)
    merged = env["merge_resource_pool"](other, pool)
    assert merged.pgs == pool.pgs + other.pgs
    other.topology = None
    with pytest.raises(ValueError, match="Cannot merge"):
        env["merge_resource_pool"](other, pool)


def test_worker_rank_uses_planned_order(source, modules, resource_types):
    env = resource_types
    topology = Reservations(modules.plan, [modules.plan.PoolRequest("a", (2,) * 2)])
    pool = env["RayResourcePool"]([2, 2], name_prefix="a", max_colocate_count=3, topology=topology)
    pool.get_placement_groups(device_name="npu")
    pool.pgs.reverse()
    definitions(source, "verl/single_controller/ray/base.py", {"_init_with_resource_pool"}, env, owner="RayWorkerGroup")
    calls = []
    worker = types.SimpleNamespace(
        device_name="npu",
        name_prefix="test",
        _ray_master_port_range=None,
        _get_master_addr_port=lambda *a, **kw: None,
        _create_worker=lambda **kw: calls.append(kw),
    )
    env["_init_with_resource_pool"](worker, pool, None, True, False)
    assert [c["rank"] for c in calls] == list(range(4))
    assert [c["pg"].id.hex() for c in calls] == ["a:0", "a:0", "a:1", "a:1"]


def test_disabled_resource_path_uses_existing_checks(resource_types, monkeypatch):
    env = resource_types
    manager = env["ResourcePoolManager"]({"a": [2]}, {})
    check = Mock()
    monkeypatch.setattr(manager, "_check_resource_available", check)
    manager.create_resource_pool()
    check.assert_called_once()
    assert manager.resource_pool_dict["a"].topology is None


def test_allocator_failure_rollback_retry_and_freeze(modules, monkeypatch):
    m = modules.topology
    policy = m.TopologyPolicy(allocation_retries=1, placement_timeout_s=1)
    allocator = m._TopologyAllocator(policy)
    nodes, resources = cluster([2], npus=1, cpus=3)
    monkeypatch.setattr(allocator, "_discover", lambda: nodes)
    states, created, removed = {}, [], []

    def new_pg(bundles, **kwargs):
        # Replanning must not begin until all previous reservations are gone.
        if len(created) == 2:
            assert len(removed) == 2
        node = next(n for n in nodes if n["resource"] in bundles[0])
        pg = PG(len(created), bundles, node)
        created.append(pg)
        states[pg.id.hex()] = "CREATED"
        return pg

    def table(pg):
        return {"state": states[pg.id.hex()], "bundles_to_node_id": {i: pg.node["id"] for i in range(pg.bundle_count)}}

    def remove(pg):
        removed.append(pg)
        states[pg.id.hex()] = "REMOVED"

    waits = []

    def get(refs, **kwargs):
        waits.append(refs)
        if len(waits) == 1:
            raise TimeoutError("reservation race")
        return refs

    monkeypatch.setattr(m, "placement_group", new_pg)
    monkeypatch.setattr(m, "placement_group_table", table)
    monkeypatch.setattr(m, "remove_placement_group", remove)
    monkeypatch.setattr(m.ray, "get", get)
    monkeypatch.setattr(m.ray._private.state, "available_resources_per_node", lambda: resources)
    requests = (modules.plan.PoolRequest("a", (1, 1)),)
    allocator.prepare(policy, requests)
    assert len(created) == 4 and len(removed) == 2
    assert allocator.claim(policy, "a", requests[0].schemes()) == created[2:]
    allocator.prepare(policy, requests)
    assert len(created) == 4
    with pytest.raises(ValueError, match="manifest"):
        allocator.prepare(policy, (modules.plan.PoolRequest("b", (1,)),))
    with pytest.raises(ValueError, match="Unplanned"):
        allocator.claim(policy, "bad", requests[0].schemes())
    with pytest.raises(ValueError, match="differs"):
        allocator.claim(policy, "a", [[{"NPU": 1, "CPU": 2}]])
    states[created[-1].id.hex()] = "PENDING"
    with pytest.raises(RuntimeError, match="lost its selected node"):
        allocator.claim(policy, "a", requests[0].schemes())


def test_incomplete_rollback_never_replans(modules, monkeypatch):
    m = modules.topology
    allocator = m._TopologyAllocator(m.TopologyPolicy(placement_timeout_s=0.01))
    nodes, resources = cluster([1], npus=1, cpus=3)
    discover = Mock(return_value=nodes)
    monkeypatch.setattr(allocator, "_discover", discover)
    pg = PG(0, [{"NPU": 1, "CPU": 3}], nodes[0])
    monkeypatch.setattr(m, "placement_group", lambda *a, **kw: pg)
    monkeypatch.setattr(m.ray, "get", Mock(side_effect=TimeoutError()))
    monkeypatch.setattr(m.ray._private.state, "available_resources_per_node", lambda: resources)
    monkeypatch.setattr(m, "placement_group_table", lambda _: {"state": "PENDING"})
    monkeypatch.setattr(m, "remove_placement_group", lambda _: None)
    with pytest.raises(RuntimeError, match="rollback not confirmed"):
        allocator.prepare(allocator.policy, [modules.plan.PoolRequest("a", (1,))])
    discover.assert_called_once()
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        allocator.prepare(allocator.policy, [modules.plan.PoolRequest("a", (1,))])


@pytest.fixture
def inference_replica(source, modules, resource_types):
    env = dict(resource_types)
    definitions(
        source,
        "verl/single_controller/ray/base.py",
        {"_init_with_resource_pool", "_init_with_subresource_pool"},
        env,
        owner="RayWorkerGroup",
    )

    class WorkerGroup:
        def __init__(self, resource_pool, **kwargs):
            self.device_name, self.name_prefix = "npu", "test"
            self._ray_master_port_range = None
            self._get_master_addr_port = lambda *a, **kw: None
            self.workers = []
            self._create_worker = lambda **kw: self.workers.append(kw["pg"].node)
            method = (
                "_init_with_subresource_pool"
                if hasattr(resource_pool, "start_bundle_index")
                else "_init_with_resource_pool"
            )
            env[method](self, resource_pool, None, False, False)

    env.update(
        RayWorkerGroup=WorkerGroup, RolloutMode=types.SimpleNamespace(STANDALONE="standalone", COLOCATED="colocated")
    )
    definitions(
        source,
        "verl/workers/rollout/replica.py",
        {"prepare_standalone", "init_standalone", "init_colocated"},
        env,
        owner="RolloutReplica",
    )

    class Replica:
        def __init__(
            self,
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model=False,
            is_teacher_model=False,
            name_suffix="",
        ):
            self.replica_rank = replica_rank
            self.world_size = modules.plan.replica_world_size(config)
            self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
            self.nnodes = self.world_size // self.gpus_per_replica_node
            self.is_reward_model, self.is_teacher_model = is_reward_model, is_teacher_model
            self.name_suffix = "_" + name_suffix if name_suffix else ""
            self.resource_pool = None

        def get_ray_class_with_init_args(self):
            return None

        def rollout_worker_use_gpu(self):
            return True

        async def launch_servers(self):
            assert len(self.workers) == self.world_size
            assert len({node["sp"] for node in self.workers}) == 1
            self._server_handle, self._server_address = self.replica_rank, str(self.replica_rank)

    for name in ("prepare_standalone", "init_standalone", "init_colocated"):
        setattr(Replica, name, env[name])
    return Replica


def run_all(coroutines):
    async def gather():
        return await asyncio.gather(*coroutines)

    return asyncio.run(gather())


@pytest.mark.parametrize("standalone", [False, True])
def test_reward_manager_actual_pool_consumption(source, modules, resource_types, inference_replica, standalone):
    config = OmegaConf.create(
        {
            "nnodes": 4,
            "n_gpus_per_node": 16,
            "model_path": "unused",
            "rollout": {
                "name": "vllm",
                "tensor_model_parallel_size": 16,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 2,
            },
        }
    )
    req = (
        modules.plan.standalone_requests(config.rollout, 4, 16, prefix="rollout_pool_reward_")
        if standalone
        else [modules.plan.PoolRequest("reward_pool", (16,) * 4, groups=(2, 2))]
    )
    topology = Reservations(modules.plan, req)
    pool = (
        None
        if standalone
        else resource_types["RayResourcePool"](
            [16] * 4, name_prefix="reward_pool", max_colocate_count=3, topology=topology
        )
    )
    env = dict(resource_types)
    env.update(
        get_rollout_replica_class=lambda _: inference_replica,
        HFModelConfig=lambda **kw: types.SimpleNamespace(get_processor=lambda: None),
    )
    definitions(
        source,
        "verl/experimental/reward_loop/reward_model.py",
        {"_initialize_llm_servers"},
        env,
        owner="RewardModelManager",
    )
    manager = types.SimpleNamespace(config=config, resource_pool=pool, topology=topology, _run_all=run_all)
    env["_initialize_llm_servers"](manager)
    assert len(manager.rollout_replicas) == 2
    assert set(topology.claimed) == {r.name for r in req}


def test_multi_teacher_actual_nested_splits(source, modules, resource_types, inference_replica):
    req = [modules.plan.PoolRequest("teacher_pool", (16,) * 5, groups=(2, 3))]
    topology = Reservations(modules.plan, req)
    chosen = modules.plan.plan_nodes(*cluster([3, 2]), req)["teacher_pool"]
    for pg, node in zip(topology.pgs["teacher_pool"], chosen, strict=True):
        pg.node = node
    pool = resource_types["RayResourcePool"](
        [16] * 5, name_prefix="teacher_pool", max_colocate_count=3, topology=topology
    )
    env = dict(resource_types)
    env.update(OmegaConf=OmegaConf, get_rollout_replica_class=lambda _: inference_replica, _run_all=run_all)
    definitions(
        source,
        "verl/experimental/teacher_loop/teacher_model.py",
        {"_initialize_llm_servers", "_validate_replica_node_alignment"},
        env,
        owner="TeacherModelManager",
    )

    class TeacherManager:
        _initialize_llm_servers = env["_initialize_llm_servers"]
        _validate_replica_node_alignment = env["_validate_replica_node_alignment"]

        def __init__(self, distillation_config, teacher_model_config, resource_pool):
            self.distillation_config, self.teacher_model_config, self.resource_pool = (
                distillation_config,
                teacher_model_config,
                resource_pool,
            )
            self._initialize_llm_servers()
            self.load_balancer_handle = None

    env["TeacherModelManager"] = TeacherManager
    definitions(
        source,
        "verl/experimental/teacher_loop/teacher_model.py",
        {"_initialize_teacher_model_managers"},
        env,
        owner="MultiTeacherModelManager",
    )
    teachers = {
        key: types.SimpleNamespace(
            key=key,
            model_path="unused",
            num_replicas=1,
            per_replica_world_size=world,
            world_size=world,
            inference=OmegaConf.create({"name": "vllm", "tensor_model_parallel_size": world}),
        )
        for key, world in [("a", 32), ("b", 48)]
    }
    manager = types.SimpleNamespace(
        distillation_config=types.SimpleNamespace(n_gpus_per_node=16, teacher_models=teachers),
        resource_pool=pool,
        teacher_model_managers={},
        server_addresses={},
        server_handles={},
        load_balancer_handle={},
    )
    env["_initialize_teacher_model_managers"](manager)
    assert [m.resource_pool.start_bundle_index for m in manager.teacher_model_managers.values()] == [0, 32]
    assert {n["sp"] for n in manager.teacher_model_managers["a"].rollout_replicas[0].workers} == {1}
    assert {n["sp"] for n in manager.teacher_model_managers["b"].rollout_replicas[0].workers} == {0}
