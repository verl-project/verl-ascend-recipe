"""Exercise FT-off dispatch without importing the training/hardware dependencies.

Only the named production functions are loaded; their bodies run unchanged.
The native methods are spies so any accidental entry into FT code fails early.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock


def load_functions(filename, names, namespace, *, keep_decorators=False):
    path = Path(__file__).parents[1] / "patch" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    if not keep_decorators:
        for node in functions:
            node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class ConfigAccess:
    @staticmethod
    def select(config, path, default=None):
        value = config
        for part in path.split("."):
            value = getattr(value, part, default)
        return value


def config_with_ft(enabled):
    if enabled is None:
        return SimpleNamespace()
    return SimpleNamespace(async_training=SimpleNamespace(fault_tolerance=SimpleNamespace(enabled=enabled)))


class TestNativeDispatch(unittest.TestCase):
    def test_client_dispatch_preserves_ft_body_and_trace_when_enabled(self):
        namespace = {"functools": functools}
        load_functions("llm_server.py", {"_native_when_ft_disabled"}, namespace)
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                native_result, ft_result = object(), object()
                native = AsyncMock(return_value=native_result)
                traced_ft = AsyncMock(return_value=ft_result)

                class Client:
                    generate = native

                    def _ft_enabled(self):
                        return bool(self.enabled)

                Client.generate = namespace["_native_when_ft_disabled"](Client)(traced_ft)
                client = Client()
                client.enabled = enabled
                result = asyncio.run(client.generate("request", prompt_ids=[1]))
                if enabled:
                    self.assertIs(result, ft_result)
                    traced_ft.assert_awaited_once_with(client, "request", prompt_ids=[1])
                    native.assert_not_called()
                else:
                    self.assertIs(result, native_result)
                    native.assert_awaited_once_with(client, "request", prompt_ids=[1])
                    traced_ft.assert_not_called()

    def test_fully_async_methods_delegate_without_using_ft_components(self):
        namespace = {"OmegaConf": ConfigAccess}
        cases = (
            ("_rollouter_init_async_rollout_manager", "_orig__init_async_rollout_manager", True, False),
            ("_rollouter_fit", "_orig_fit", True, False),
            ("_async_trainer_setup_checkpoint_manager", "_orig__setup_checkpoint_manager", False, True),
            ("_async_main_initialize_components", "_orig__initialize_components", False, True),
        )
        load_functions("experimental.py", {name for name, *_ in cases}, namespace)
        for enabled in (None, False):
            for name, original_name, is_async, has_arg in cases:
                with self.subTest(enabled=enabled, name=name):
                    config = config_with_ft(enabled)
                    result = object()
                    original = AsyncMock(return_value=result) if is_async else Mock(return_value=result)
                    obj = SimpleNamespace(config=config, **{original_name: original})
                    args = (
                        (config,) if name == "_async_main_initialize_components" else ((object(),) if has_arg else ())
                    )
                    value = namespace[name](obj, *args)
                    if is_async:
                        value = asyncio.run(value)
                    self.assertIs(value, result)
                    original.assert_called_once_with(*args)

    def test_rollouter_constructor_preserves_native_state_when_ft_off(self):
        namespace = {"OmegaConf": ConfigAccess}
        load_functions("experimental.py", {"_rollouter_init"}, namespace)
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                obj = SimpleNamespace()
                config = config_with_ft(enabled)

                def native(instance, given_config):
                    instance.config = given_config

                namespace["_rollouter_init"](native, obj, config)
                self.assertEqual(hasattr(obj, "_ft_supervisor"), enabled is True)
                self.assertEqual(hasattr(obj, "_trainer_handle"), enabled is True)

    def test_fully_client_native_super_call_does_not_reenter_child(self):
        spec = importlib.util.spec_from_file_location(
            "native_dispatch_core", Path(__file__).parents[1] / "patch" / "_core.py"
        )
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        native = AsyncMock(return_value=object())
        traces = []

        def trace(fn):
            @functools.wraps(fn)
            async def traced(*args, **kwargs):
                traces.append(fn.__name__)
                return await fn(*args, **kwargs)

            return traced

        class BaseClient:
            def _ft_enabled(self):
                return False

            async def generate(self, *args, **kwargs):
                return await native(self, *args, **kwargs)

        class FullyClient(BaseClient):
            async def generate(self, *args, **kwargs):
                return await super().generate(*args, **kwargs)

        namespace = dict(
            LLMServerClient=BaseClient,
            FullyLLMServerClient=FullyClient,
            functools=functools,
            patch=core.patch,
            rollout_trace_op=trace,
        )
        load_functions(
            "llm_server.py",
            {"_native_when_ft_disabled", "generate", "_fully_generate"},
            namespace,
            keep_decorators=True,
        )
        client = FullyClient()
        kwargs = dict(prompt_ids=[1], sampling_params={"max_tokens": 2}, image_data=None, video_data=None)
        self.assertIs(asyncio.run(client.generate("request", **kwargs)), native.return_value)
        native.assert_awaited_once_with(client, "request", **kwargs)
        self.assertEqual(traces, [])


_FT_EXCEPTIONS_MODULE = "verl.workers.rollout.fault_tolerance.exceptions"


def _prime_ft_exceptions_module():
    """Expose the recipe's ``is_transient_fault`` under its production import path.

    ``_manager_clear_kv_cache`` lazily imports
    ``verl.workers.rollout.fault_tolerance.exceptions``. Registering the recipe
    module in ``sys.modules`` under that full dotted name short-circuits the
    parent-package walk, so the test stays hermetic (no verl install needed).
    """
    if _FT_EXCEPTIONS_MODULE in sys.modules:
        return
    path = Path(__file__).parents[1] / "workers" / "rollout" / "fault_tolerance" / "exceptions.py"
    spec = importlib.util.spec_from_file_location(_FT_EXCEPTIONS_MODULE, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_FT_EXCEPTIONS_MODULE] = module
    spec.loader.exec_module(module)


def _make_replica(address, clear_result=None):
    """Stub RolloutReplica; ``clear_result`` as an exception is raised on clear."""
    calls = []

    async def clear_kv_cache():
        calls.append(address)
        if isinstance(clear_result, BaseException):
            raise clear_result
        return clear_result

    return SimpleNamespace(_server_address=address, clear_kv_cache=clear_kv_cache), calls


class TestClearKvCacheIsolation(unittest.TestCase):
    """LLMServerManager.clear_kv_cache: FT control-plane fault isolation matrix.

    The one-step trainer clears KV cache right after every weight sync while
    the Manager still lists dead replicas for ``spawn_replacement``. Transient
    rollout faults must be isolated through the trainer-wired callback so the
    step survives on the N-1 tier; everything else must propagate.
    """

    @classmethod
    def setUpClass(cls):
        _prime_ft_exceptions_module()
        namespace = {"asyncio": asyncio, "Any": Any, "logger": Mock()}
        load_functions("llm_server.py", {"_manager_clear_kv_cache"}, namespace)
        # staticmethod: a plain function class attribute would bind `self` and
        # shift every call's arguments by one.
        cls.clear_kv_cache = staticmethod(namespace["_manager_clear_kv_cache"])

    @staticmethod
    def _make_manager(replicas, *, ft_enabled=True, callback=None, native=None):
        manager = SimpleNamespace(rollout_replicas=replicas, _ft_enabled=lambda: ft_enabled)
        if native is not None:
            manager._orig_clear_kv_cache = native
        if callback is not None:
            manager._ft_clear_kv_fault_callback = callback
        return manager

    def test_ft_disabled_delegates_to_native(self):
        native = AsyncMock(return_value="native")
        ok, ok_calls = _make_replica("80.0.0.1")
        dead, dead_calls = _make_replica("80.0.0.2", ConnectionError("down"))
        manager = self._make_manager([ok, dead], ft_enabled=False, native=native)
        self.assertIs(asyncio.run(self.clear_kv_cache(manager)), "native")
        # No args: on a real manager `_orig_clear_kv_cache` is the bound native
        # method; SimpleNamespace doesn't bind, so the mock sees zero args.
        native.assert_awaited_once_with()
        self.assertEqual(ok_calls, [])
        self.assertEqual(dead_calls, [])

    def test_missing_callback_delegates_to_native(self):
        # FT on but the trainer wiring never ran (e.g. fully-async manager):
        # behavior must stay exactly native, not silently "succeed".
        native = AsyncMock(return_value="native")
        ok, ok_calls = _make_replica("80.0.0.1")
        manager = self._make_manager([ok], ft_enabled=True, native=native)
        self.assertIs(asyncio.run(self.clear_kv_cache(manager)), "native")
        native.assert_awaited_once_with()
        self.assertEqual(ok_calls, [])

    def test_transient_fault_isolated_and_survivor_continues(self):
        ok, ok_calls = _make_replica("80.0.0.1")
        dead, dead_calls = _make_replica("80.0.0.2", ConnectionError("down"))
        isolated = []

        async def callback(replica_id):
            isolated.append(replica_id)

        manager = self._make_manager([ok, dead], ft_enabled=True, callback=callback)
        self.assertIsNone(asyncio.run(self.clear_kv_cache(manager)))
        self.assertEqual(isolated, ["80.0.0.2"])
        self.assertEqual(ok_calls, ["80.0.0.1"])
        self.assertEqual(dead_calls, ["80.0.0.2"])
        # Snapshot semantics: the Manager list stays untouched —
        # spawn_replacement must still find the dead replica object.
        self.assertEqual(len(manager.rollout_replicas), 2)

    def test_non_transient_propagates_without_isolation(self):
        ok, ok_calls = _make_replica("80.0.0.1")
        buggy, _ = _make_replica("80.0.0.2", ValueError("ordinary bug"))
        isolated = []

        async def callback(replica_id):
            isolated.append(replica_id)

        manager = self._make_manager([ok, buggy], ft_enabled=True, callback=callback)
        with self.assertRaises(ValueError):
            asyncio.run(self.clear_kv_cache(manager))
        self.assertEqual(isolated, [])
        # gather starts every clear concurrently, so the survivor was still
        # attempted before the fatal error propagated — that's fine; the
        # invariant under test is that isolation never ran for a non-fault.

    def test_callback_failure_reraises_original_fault(self):
        dead, _ = _make_replica("80.0.0.2", ConnectionError("down"))

        async def callback(replica_id):
            raise RuntimeError(f"supervisor not running: {replica_id}")

        manager = self._make_manager([dead], ft_enabled=True, callback=callback)
        with self.assertRaises(ConnectionError) as ctx:
            asyncio.run(self.clear_kv_cache(manager))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_all_failed_is_not_a_success(self):
        dead1, calls1 = _make_replica("80.0.0.1", ConnectionError("down-1"))
        dead2, calls2 = _make_replica("80.0.0.2", ConnectionError("down-2"))
        isolated = []

        async def callback(replica_id):
            isolated.append(replica_id)

        manager = self._make_manager([dead1, dead2], ft_enabled=True, callback=callback)
        with self.assertRaises(ConnectionError):
            asyncio.run(self.clear_kv_cache(manager))
        self.assertEqual(isolated, ["80.0.0.1", "80.0.0.2"])
        self.assertEqual(calls1, ["80.0.0.1"])
        self.assertEqual(calls2, ["80.0.0.2"])

    def test_cancelled_child_propagates(self):
        dead, _ = _make_replica("80.0.0.1", asyncio.CancelledError())
        manager = self._make_manager([dead], ft_enabled=True, callback=AsyncMock())
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.clear_kv_cache(manager))

    def test_production_function_is_patched_with_auto_await(self):
        tree = ast.parse((Path(__file__).parents[1] / "patch" / "llm_server.py").read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_manager_clear_kv_cache"
        )
        # @patch must be outermost (class attr becomes the auto_await-wrapped
        # coroutine, preserving the native sync/async dual call convention).
        self.assertEqual(ast.unparse(node.decorator_list[0]), "patch(LLMServerManager, 'clear_kv_cache')")
        self.assertEqual(ast.unparse(node.decorator_list[1]), "auto_await")


if __name__ == "__main__":
    unittest.main()
