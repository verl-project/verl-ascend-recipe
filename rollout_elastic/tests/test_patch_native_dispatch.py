"""Exercise FT-off dispatch without importing the training/hardware dependencies.

Only the named production functions are loaded; their bodies run unchanged.
The native methods are spies so any accidental entry into FT code fails early.
"""

from __future__ import annotations

import ast
import asyncio
import functools
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
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
            FullyAsyncLLMServerClient=FullyClient,
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


class TestClearKvCacheIsolation(unittest.TestCase):
    """Protect the target v0.9.0 API surface from the obsolete manager hook."""

    def test_manager_clear_hook_is_not_bound(self):
        tree = ast.parse((Path(__file__).parents[1] / "patch" / "llm_server.py").read_text(encoding="utf-8"))
        function_names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertNotIn("_manager_clear_kv_cache", function_names)
        clear_patch_targets = {
            ast.unparse(node.decorator_list[0])
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.decorator_list
            and ast.unparse(node.decorator_list[0]).startswith("patch(LLMServerManager")
        }
        self.assertNotIn("patch(LLMServerManager, 'clear_kv_cache')", clear_patch_targets)

    def test_supervisor_keeps_checkpoint_wiring_without_clear_callback(self):
        tree = ast.parse((Path(__file__).parents[1] / "patch" / "experimental.py").read_text(encoding="utf-8"))
        source = ast.unparse(tree)
        self.assertNotIn("_ft_clear_kv_fault_callback", source)
        self.assertIn("set_sync_failure_reporter", source)
        self.assertIn("set_replica_promotion_reporter", source)

    def test_valid_server_clear_path_remains(self):
        source = (Path(__file__).parents[1] / "patch" / "vllm.py").read_text(encoding="utf-8")
        self.assertIn("server_handle.clear_kv_cache.remote()", source)

    def test_vllm_generate_override_keeps_v09_request_surface(self):
        """The recipe override must accept every native server request field."""
        path = Path(__file__).parents[1] / "patch" / "vllm.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        generate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "generate"
        )
        names = {arg.arg for arg in generate.args.args + generate.args.kwonlyargs}
        self.assertTrue(
            {
                "image_data",
                "video_data",
                "audio_data",
                "mm_processor_kwargs",
                "priority",
                "kv_transfer_params",
                "progress_ctx",
            }.issubset(names)
        )


if __name__ == "__main__":
    unittest.main()
