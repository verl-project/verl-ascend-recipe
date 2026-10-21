"""Test production actor patching and rebinding with lightweight native classes.

Ray is real; training dependencies and native constructors are stand-ins. The
tests cover patch installation and RPC dispatch, not training or recovery.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import logging
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch as mock_patch

PATCH_DIR = Path(__file__).parents[1] / "patch"


def _build_patched_actors(ray):
    spec = importlib.util.spec_from_file_location("actor_test_core", PATCH_DIR / "_core.py")
    assert spec is not None and spec.loader is not None
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)

    class ConfigAccess:
        @staticmethod
        def select(config, path, default=None):
            value = config
            for part in path.split("."):
                value = getattr(value, part, default)
            return value

    class Separate:
        def __init__(self, config):
            self.config = config

        def _init_resource_pools(self):
            return "native-resource-pools"

    class NativeRollouter(Separate):
        def __init__(self, config):
            super().__init__(config)
            self.llm_server_manager = SimpleNamespace(global_load_balancer="native-lb")

        async def _init_async_rollout_manager(self):
            return "native-manager"

        async def fit(self):
            return "native-fit"

        def ft_initialized(self):
            return hasattr(self, "_ft_supervisor"), hasattr(self, "_trainer_handle")

    class NativeTrainer(Separate):
        def _setup_checkpoint_manager(self, rollouter):
            return "native-checkpoint", rollouter

        async def _get_samples_from_queue(self):
            return "native-samples"

    class NativeTaskRunner:
        def _initialize_components(self, config):
            return "native-main"

    class FakeResourcePool:
        """@wrap target for RayResourcePool.get_placement_groups."""

        name_prefix = "trainer_pool"
        _store = [2]
        max_colocate_count = 3
        use_gpu = True
        accelerator_type = None
        detached = False
        pgs = None

        def get_placement_groups(self, *args, **kwargs):
            return "native-pgs"

    original = {
        "FullyAsyncRollouter": ray.remote(num_cpus=10, max_concurrency=100)(NativeRollouter),
        "FullyAsyncTrainer": ray.remote(num_cpus=10)(NativeTrainer),
        "FullyAsyncTaskRunner": ray.remote(num_cpus=1)(NativeTaskRunner),
    }
    main_module = SimpleNamespace(**original)
    rollouter_module = SimpleNamespace(FullyAsyncRollouter=original["FullyAsyncRollouter"])
    trainer_module = SimpleNamespace(FullyAsyncTrainer=original["FullyAsyncTrainer"])
    namespace = dict(
        original,
        __name__="fully_async_patch_test",
        ray=ray,
        OmegaConf=ConfigAccess,
        SeparateRayPPOTrainer=Separate,
        RayResourcePool=FakeResourcePool,
        fully_async_main=main_module,
        fully_async_rollouter=rollouter_module,
        fully_async_trainer=trainer_module,
        add=core.add,
        patch=core.patch,
        wrap=core.wrap,
        unwrap_ray_remote=core.unwrap_ray_remote,
        logging=logging,
        logger=logging.getLogger("fully_async_patch_test"),
    )
    tree = ast.parse((PATCH_DIR / "experimental.py").read_text(encoding="utf-8"))
    nodes = []
    finalizing = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "FullyAsyncRollouter" for target in node.targets
        ):
            finalizing = True
        if finalizing:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_FtNonInferenceRemote":
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "_sep_trainer_init"
            or node.name.startswith(("_rollouter_", "_async_trainer_", "_async_main_", "_ft_"))
        ):
            nodes.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(PATCH_DIR / "experimental.py"), "exec"), namespace)
    return namespace, original


class FullyAsyncRayPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("ray") is None:
            raise unittest.SkipTest("Ray is not installed")
        import ray

        cls.ray = ray
        cls.started_here = not ray.is_initialized()
        if cls.started_here:
            # Single-node test cluster: label the local node (which is also
            # the head) so the strict non-inference placement constraint has
            # a matching scheduling target. In production only training
            # workers carry the label; the head never does and is therefore
            # excluded by the hard constraint.
            try:
                ray.init(
                    num_cpus=1,
                    include_dashboard=False,
                    log_to_driver=False,
                    labels={"verl.io/role": "trainer"},
                )
            except TypeError:  # older Ray without the labels kwarg
                ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
        try:
            cls.patched, cls.original = _build_patched_actors(ray)
        except Exception:
            if cls.started_here:
                ray.shutdown()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.started_here:
            cls.ray.shutdown()

    def test_creation_aliases_options_and_method_tables(self):
        options = {
            "FullyAsyncRollouter": {"num_cpus": 10, "max_concurrency": 100},
            "FullyAsyncTrainer": {"num_cpus": 10},
            "FullyAsyncTaskRunner": {"num_cpus": 1},
        }
        methods = {
            "FullyAsyncRollouter": (
                "get_load_balancer",
                "init_ft_supervisor",
                "report_sync_failure",
                "promote_synced_replica",
            ),
            "FullyAsyncTrainer": (
                "_get_samples_from_queue",
                "_on_replica_dead_from_supervisor",
                "_on_replica_added_from_supervisor",
            ),
            "FullyAsyncTaskRunner": ("_initialize_components",),
        }
        for name, actor_options in options.items():
            with self.subTest(actor=name):
                actor = self.patched[name]
                original = self.original[name]
                self.assertIsNot(actor, original)
                # fully_async_main may hold a placement proxy; unwrap it.
                rebound = getattr(self.patched["fully_async_main"], name)
                self.assertIs(getattr(rebound, "_actor_cls", rebound), actor)
                self.assertEqual(actor._default_options, actor_options)
                self.assertIs(actor.__ray_actor_class__, original.__ray_actor_class__)
                self.assertIsNot(
                    actor.__ray_metadata__.method_meta,
                    original.__ray_metadata__.method_meta,
                )
                self.assertNotEqual(
                    actor.__ray_metadata__.actor_creation_function_descriptor.function_id,
                    original.__ray_metadata__.actor_creation_function_descriptor.function_id,
                )
                self.assertLessEqual(
                    set(methods[name]),
                    actor.__ray_metadata__.method_meta.methods.keys(),
                )
        self.assertIs(
            self.patched["fully_async_rollouter"].FullyAsyncRollouter,
            self.patched["FullyAsyncRollouter"],
        )
        self.assertIs(
            self.patched["fully_async_trainer"].FullyAsyncTrainer,
            self.patched["FullyAsyncTrainer"],
        )

    def test_new_actors_execute_constructor_and_added_rpc(self):
        for ft_enabled in (None, False, True):
            with self.subTest(ft_enabled=ft_enabled):
                config = SimpleNamespace()
                if ft_enabled is not None:
                    config.async_training = SimpleNamespace(fault_tolerance=SimpleNamespace(enabled=ft_enabled))
                actors = []
                try:
                    rollouter = self.patched["FullyAsyncRollouter"].options(num_cpus=0).remote(config)
                    actors.append(rollouter)
                    self.assertEqual(
                        self.ray.get(rollouter.ft_initialized.remote(), timeout=20),
                        (bool(ft_enabled), bool(ft_enabled)),
                    )
                    self.assertEqual(
                        self.ray.get(rollouter.get_load_balancer.remote(), timeout=20),
                        "native-lb",
                    )
                    self.assertIsNone(self.ray.get(rollouter.report_sync_failure.remote("replica"), timeout=20))
                    self.assertFalse(
                        self.ray.get(
                            rollouter.promote_synced_replica.remote("replica", {}, 1, 1),
                            timeout=20,
                        )
                    )
                    if not ft_enabled:
                        self.assertEqual(
                            self.ray.get(rollouter.fit.remote(), timeout=20),
                            "native-fit",
                        )
                        self.assertEqual(
                            self.ray.get(
                                rollouter._init_async_rollout_manager.remote(),
                                timeout=20,
                            ),
                            "native-manager",
                        )
                        trainer = self.patched["FullyAsyncTrainer"].options(num_cpus=0).remote(config)
                        runner = self.patched["FullyAsyncTaskRunner"].options(num_cpus=0).remote()
                        actors.extend([trainer, runner])
                        self.assertEqual(
                            self.ray.get(
                                trainer._setup_checkpoint_manager.remote("rollouter"),
                                timeout=20,
                            ),
                            ("native-checkpoint", "rollouter"),
                        )
                        self.assertEqual(
                            self.ray.get(trainer._get_samples_from_queue.remote(), timeout=20),
                            "native-samples",
                        )
                        self.assertEqual(
                            self.ray.get(
                                runner._initialize_components.remote(config),
                                timeout=20,
                            ),
                            "native-main",
                        )
                finally:
                    for actor in actors:
                        self.ray.kill(actor)

    def test_downstream_can_derive_from_rebound_plain_classes(self):
        for name in (
            "FullyAsyncRollouter",
            "FullyAsyncTrainer",
            "FullyAsyncTaskRunner",
        ):
            with self.subTest(actor=name):

                class Downstream(self.patched[name].__ray_actor_class__):
                    pass

                actor = self.ray.remote(Downstream)
                self.assertEqual(
                    actor.__ray_actor_class__.__bases__,
                    (self.patched[name].__ray_actor_class__,),
                )
                self.assertLessEqual(
                    self.patched[name].__ray_metadata__.method_meta.methods.keys(),
                    actor.__ray_metadata__.method_meta.methods.keys(),
                )


class _FakeBatch:
    """Minimal full_batch stand-in: the filter only inspects len()."""

    def __init__(self, rows: int):
        self.rows = rows

    def __len__(self):
        return self.rows


class FullyAsyncTrainerSampleFilterTests(unittest.TestCase):
    """FT trainer drops partial rollout samples (rows < rollout.n) and keeps
    waiting instead of assembling a batch that breaks dp divisibility."""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("ray") is None:
            raise unittest.SkipTest("Ray is not installed")
        import ray

        cls.ray = ray
        cls.patched, _ = _build_patched_actors(ray)
        cls.trainer_cls = cls.patched["FullyAsyncTrainer"].__ray_actor_class__
        try:
            import verl.experimental.fully_async_policy.detach_utils  # noqa: F401

            cls._verl_stubs = None
        except Exception:
            # No verl: stub the import chain so the patched method's local
            # import resolves against the fake module injected per-test.
            cls._verl_stubs = {
                "verl": ModuleType("verl"),
                "verl.experimental": ModuleType("verl.experimental"),
                "verl.experimental.fully_async_policy": ModuleType("verl.experimental.fully_async_policy"),
                "verl.experimental.fully_async_policy.detach_utils": ModuleType(
                    "verl.experimental.fully_async_policy.detach_utils"
                ),
            }

    def _make_trainer(self, samples, required=2):
        config = SimpleNamespace(
            async_training=SimpleNamespace(fault_tolerance=SimpleNamespace(enabled=True)),
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=2)),
            trainer=SimpleNamespace(balance_batch=True),
        )
        trainer = self.trainer_cls(config)
        trainer.required_samples = required
        trainer.tokenizer = "tokenizer"
        trainer._balance_batch = Mock(name="balance_batch")

        queue = iter(samples)

        class _Queue:
            async def get_sample(self):
                try:
                    return next(queue), 0
                except StopIteration:
                    return None, 0

        trainer.message_queue_client = _Queue()
        return trainer

    def _run(self, trainer):
        assemble_calls = []

        def _assemble(samples, *args):
            assemble_calls.append(list(samples))
            return SimpleNamespace(meta_info={})

        module = ModuleType("verl.experimental.fully_async_policy.detach_utils")
        module.assemble_batch_from_rollout_samples = Mock(side_effect=_assemble)
        overrides = dict(self._verl_stubs) if self._verl_stubs else {}
        overrides["verl.experimental.fully_async_policy.detach_utils"] = module
        with mock_patch.dict(sys.modules, overrides):
            result = asyncio.run(trainer._get_samples_from_queue())
        return result, module.assemble_batch_from_rollout_samples

    def _sample(self, sample_id, rows):
        return self.ray.cloudpickle.dumps(SimpleNamespace(full_batch=_FakeBatch(rows), sample_id=sample_id))

    def test_partial_samples_dropped_and_collection_continues(self):
        trainer = self._make_trainer(
            [
                self._sample("partial", 1),
                self._sample("c1", 2),
                self._sample("c2", 2),
            ]
        )

        (epoch, batch), assemble = self._run(trainer)

        self.assertEqual(epoch, 0)
        self.assertIn("fully_async/total_wait_time", batch.meta_info)
        assemble.assert_called_once()
        collected = assemble.call_args[0][0]
        self.assertEqual([s.sample_id for s in collected], ["c1", "c2"])
        self.assertEqual(assemble.call_args[0][2], trainer.config)
        self.assertEqual(assemble.call_args[0][3], trainer._balance_batch)

    def test_termination_signal_returns_none_when_insufficient(self):
        trainer = self._make_trainer(
            [
                self._sample("partial", 1),
                self._sample("c1", 2),
            ]
        )

        result, assemble = self._run(trainer)

        self.assertEqual(result, (None, None))
        assemble.assert_not_called()

    def test_fault_tolerance_disabled_delegates_to_native(self):
        trainer = self.trainer_cls(SimpleNamespace())

        self.assertEqual(asyncio.run(trainer._get_samples_from_queue()), "native-samples")

    def test_rollout_pg_label_selector(self):
        selector_fn = self.patched["_ft_rollout_pg_label_selector"]
        trainer_selector_fn = self.patched["_ft_trainer_pg_label_selector"]

        # No cached config (run_ppo never ran): stay native.
        saved = self.patched.get("_ft_label_cfg")
        self.patched["_ft_label_cfg"] = None
        self.assertIsNone(selector_fn("trainer_pool"))
        self.assertIsNone(selector_fn("rollout_pool_0abc"))  # warns once, stays native
        self.assertIsNone(trainer_selector_fn("rollout_pool_0abc"))
        self.assertIsNone(trainer_selector_fn("trainer_pool"))  # warns once, stays native

        # Cached config: standalone-rollout pools get the rollout selector,
        # every other pool (training GPU workers, CPU coordination pools)
        # gets the trainer selector, so no training bundle can occupy a
        # rollout pod.
        self.patched["_ft_label_cfg"] = {"key": "verl.io/role", "trainer": "trainer", "rollout": "rollout"}
        self.assertIsNone(selector_fn("trainer_pool"))
        self.assertIsNone(selector_fn("reward_pool"))
        self.assertIsNone(selector_fn("teacher_pool"))
        self.assertEqual(selector_fn("rollout_pool_0abc"), {"verl.io/role": "rollout"})
        self.assertEqual(selector_fn("rollout_reward_pool_0abc"), {"verl.io/role": "rollout"})
        self.assertEqual(selector_fn("rollout_teacher_pool_0abc"), {"verl.io/role": "rollout"})
        self.assertIsNone(trainer_selector_fn("rollout_pool_0abc"))
        self.assertEqual(trainer_selector_fn("trainer_pool"), {"verl.io/role": "trainer"})
        self.assertEqual(trainer_selector_fn("reward_pool"), {"verl.io/role": "trainer"})
        self.assertEqual(trainer_selector_fn("teacher_pool"), {"verl.io/role": "trainer"})

        # Empty value disables the corresponding constraint.
        self.patched["_ft_label_cfg"] = {"key": "verl.io/role", "trainer": "trainer", "rollout": ""}
        self.assertIsNone(selector_fn("rollout_pool_0abc"))
        self.patched["_ft_label_cfg"] = {"key": "verl.io/role", "trainer": "", "rollout": "rollout"}
        self.assertIsNone(trainer_selector_fn("trainer_pool"))
        self.patched["_ft_label_cfg"] = saved

    def test_rollout_pg_created_with_label_selector(self):
        try:
            import importlib

            # importlib (not `import ray.util.placement_group as ...`): the
            # module attribute on ray.util is shadowed by a function of the
            # same name.
            pg_module = importlib.import_module("ray.util.placement_group")
            import verl.single_controller.ray.base as ray_base
        except ImportError:
            self.skipTest("verl.single_controller is not importable")

        pool_cls = self.patched["RayResourcePool"]
        captured = {}

        class FakePG:
            def __init__(self, ready_ref, **kwargs):
                self.ready_ref = ready_ref
                captured.update(kwargs)

            def ready(self):
                return self.ready_ref

        def fake_placement_group(**kwargs):
            return FakePG(ready_ref=self.ray.put(1), **kwargs)

        saved = self.patched.get("_ft_label_cfg")
        saved_nodes = self.ray.nodes
        self.patched["_ft_label_cfg"] = {"key": "verl.io/role", "trainer": "trainer", "rollout": "rollout"}

        # The strict whitelist check must see alive nodes carrying the role
        # labels (a real cluster would; stand two in here).
        self.ray.nodes = lambda: [
            {"Alive": True, "NodeManagerAddress": "10.0.0.1", "Labels": {"verl.io/role": "trainer"}},
            {"Alive": True, "NodeManagerAddress": "10.0.0.2", "Labels": {"verl.io/role": "rollout"}},
        ]
        orig_create, orig_sort = pg_module.placement_group, ray_base.sort_placement_group_by_node_ip
        pg_module.placement_group = fake_placement_group
        ray_base.sort_placement_group_by_node_ip = lambda pgs: pgs
        try:
            # Trainer pool: every bundle pinned to trainer nodes (a training
            # bundle must never occupy a rollout pod).
            trainer_pool = pool_cls()
            trainer_pool.name_prefix = "trainer_pool"
            trainer_pool.get_placement_groups()
            self.assertEqual(
                captured["bundle_label_selector"], [{"verl.io/role": "trainer"}] * len(captured["bundles"])
            )
            self.assertEqual(captured["bundles"], [{"CPU": 3, "GPU": 1}] * 2)
            self.assertTrue(captured["name"].startswith("trainer_pool"))

            captured.clear()

            # Standalone rollout pool: every bundle carries the rollout selector.
            rollout_pool = pool_cls()
            rollout_pool.name_prefix = "rollout_pool_0abc"
            rollout_pool.get_placement_groups()
            self.assertEqual(
                captured["bundle_label_selector"], [{"verl.io/role": "rollout"}] * len(captured["bundles"])
            )
            self.assertEqual(captured["bundles"], [{"CPU": 3, "GPU": 1}] * 2)
            self.assertEqual(captured["strategy"], "STRICT_PACK")
            self.assertTrue(captured["name"].startswith("rollout_pool_0abc"))
            self.assertEqual(len(rollout_pool.pgs), 1)
            self.assertIsInstance(rollout_pool.pgs[0], FakePG)
        finally:
            pg_module.placement_group = orig_create
            ray_base.sort_placement_group_by_node_ip = orig_sort
            self.ray.nodes = saved_nodes
            self.patched["_ft_label_cfg"] = saved


class FullyAsyncLauncherTests(unittest.TestCase):
    def _subprocess_env(self):
        if importlib.util.find_spec("verl") is None:
            self.skipTest("verl is not installed")
        env = os.environ.copy()
        env["VERL_USE_EXTERNAL_MODULES"] = "rollout_elastic.patch"
        return env

    def test_launcher_calls_canonical_hydra_main(self):
        module_name = "verl.experimental.fully_async_policy.fully_async_main"
        module = ModuleType(module_name)
        module.main = Mock()
        with mock_patch.dict(sys.modules, {module_name: module}):
            runpy.run_path(str(PATCH_DIR.parent / "fully_async_main.py"), run_name="__main__")
        module.main.assert_called_once_with()

    def test_real_launcher_prints_canonical_hydra_config(self):
        result = subprocess.run(
            [sys.executable, "-m", "rollout_elastic.fully_async_main", "--cfg", "job"],
            cwd=PATCH_DIR.parents[1],
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, msg=output)
        self.assertIn("async_training", output)

    def test_repeated_install_preserves_canonical_actor_identities(self):
        code = """
from rollout_elastic.patch import install
from verl.experimental.fully_async_policy import fully_async_main

def actors():
    return (
        fully_async_main.FullyAsyncRollouter,
        fully_async_main.FullyAsyncTrainer,
        fully_async_main.FullyAsyncTaskRunner,
    )

before = actors()
install()
after_once = actors()
install()
after_twice = actors()
assert all(left is right for left, right in zip(before, after_once))
assert all(left is right for left, right in zip(after_once, after_twice))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PATCH_DIR.parents[1],
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
