"""Regression tests for the recipe's decorator-based patch primitives."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_patch_core():
    core_path = Path(__file__).parents[1] / "patch" / "_core.py"
    spec = importlib.util.spec_from_file_location("rollout_elastic_test_patch_core", core_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PatchCoreTests(unittest.TestCase):
    def test_patch_treats_parent_and_child_implementations_independently(self):
        core = _load_patch_core()

        class Parent:
            def value(self):
                return "parent-native"

        class Child(Parent):
            def value(self):
                return "child-native"

        parent_native = Parent.__dict__["value"]
        child_native = Child.__dict__["value"]

        @core.patch(Parent, "value")
        def parent_value(self):
            return "parent-patched"

        @core.patch(Child, "value")
        def child_value(self):
            return "child-patched"

        self.assertEqual(Parent().value(), "parent-patched")
        self.assertEqual(Child().value(), "child-patched")
        self.assertIs(Parent.__dict__["_orig_value"], parent_native)
        self.assertIs(Child.__dict__["_orig_value"], child_native)

    def test_wrap_treats_parent_and_child_implementations_independently(self):
        core = _load_patch_core()

        class Parent:
            def value(self):
                return "parent-native"

        class Child(Parent):
            def value(self):
                return "child-native"

        parent_native = Parent.__dict__["value"]
        child_native = Child.__dict__["value"]

        @core.wrap(Parent, "value")
        def wrap_parent(original, self):
            return f"parent-wrap:{original(self)}"

        @core.wrap(Child, "value")
        def wrap_child(original, self):
            return f"child-wrap:{original(self)}"

        self.assertEqual(Parent().value(), "parent-wrap:parent-native")
        self.assertEqual(Child().value(), "child-wrap:child-native")
        self.assertIs(Parent.__dict__["_orig_value"], parent_native)
        self.assertIs(Child.__dict__["_orig_value"], child_native)

    def test_patch_is_idempotent_and_keeps_the_first_native_backup(self):
        core = _load_patch_core()

        class Target:
            def value(self):
                return "native"

        native = Target.__dict__["value"]

        @core.patch(Target, "value")
        def first_patch(self):
            return "first"

        @core.patch(Target, "value")
        def second_patch(self):
            return "second"

        self.assertEqual(Target().value(), "first")
        self.assertIs(Target.__dict__["_orig_value"], native)

    def test_wrap_is_idempotent_and_keeps_the_first_native_backup(self):
        core = _load_patch_core()

        class Target:
            def value(self):
                return "native"

        native = Target.__dict__["value"]

        @core.wrap(Target, "value")
        def first_wrap(original, self):
            return f"first:{original(self)}"

        @core.wrap(Target, "value")
        def second_wrap(original, self):
            return f"second:{original(self)}"

        self.assertEqual(Target().value(), "first:native")
        self.assertIs(Target.__dict__["_orig_value"], native)

    def test_add_preserves_a_method_inherited_from_the_parent(self):
        core = _load_patch_core()

        class Parent:
            def value(self):
                return "parent-native"

        class Child(Parent):
            pass

        @core.add(Child, "value")
        def added_value(self):
            return "recipe-added"

        self.assertEqual(Child().value(), "parent-native")
        self.assertNotIn("value", Child.__dict__)


class RayPatchCoreTests(unittest.TestCase):
    def test_redecorated_actor_exposes_all_patched_methods(self):
        if importlib.util.find_spec("ray") is None:
            self.skipTest("Ray is not installed")
        import ray

        if ray.__version__ != "2.55.1":
            self.skipTest("This test characterizes Ray 2.55.1 actor metadata")

        core = _load_patch_core()

        class PlainActor:
            def __init__(self, prefix):
                self.prefix = prefix

            def sync_value(self, suffix):
                return f"native-sync:{self.prefix}:{suffix}"

            async def async_value(self, suffix):
                return f"native-async:{self.prefix}:{suffix}"

        old_actor_class = ray.remote(num_cpus=0.01)(PlainActor)

        @core.wrap(old_actor_class, "__init__")
        def wrap_init(original, self, prefix):
            original(self, f"wrapped-{prefix}")

        @core.patch(old_actor_class, "sync_value")
        def sync_value(self, suffix):
            return f"patched-sync:{self.prefix}:{suffix}"

        @core.patch(old_actor_class, "async_value")
        async def async_value(self, suffix):
            return f"patched-async:{self.prefix}:{suffix}"

        @core.add(old_actor_class, "added_value")
        def added_value(self, suffix):
            return f"added:{self.prefix}:{suffix}"

        patched_actor_class = ray.remote(num_cpus=0.01)(PlainActor)
        self.assertIsNot(
            patched_actor_class.__ray_metadata__.method_meta,
            old_actor_class.__ray_metadata__.method_meta,
        )
        modified_class = patched_actor_class.__ray_metadata__.modified_class
        self.assertIs(modified_class.__ray_actor_class__, PlainActor)
        self.assertEqual(modified_class.__mro__[1:], PlainActor.__mro__)

        started_here = not ray.is_initialized()
        if started_here:
            ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)

        actors = []
        try:
            old_actor = old_actor_class.remote("old")
            actors.append(old_actor)
            self.assertFalse(hasattr(old_actor, "added_value"))

            actor = patched_actor_class.remote("value")
            actors.append(actor)
            self.assertEqual(
                ray.get(actor.sync_value.remote("x"), timeout=15),
                "patched-sync:wrapped-value:x",
            )
            self.assertEqual(
                ray.get(actor.async_value.remote("y"), timeout=15),
                "patched-async:wrapped-value:y",
            )
            self.assertEqual(
                ray.get(actor.added_value.remote("z"), timeout=15),
                "added:wrapped-value:z",
            )
        finally:
            for actor in actors:
                ray.kill(actor)
            if started_here:
                ray.shutdown()


if __name__ == "__main__":
    unittest.main()
