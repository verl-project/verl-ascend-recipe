# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Decorator-based patch primitives.

rollout_elastic extends verl by *decorating* native classes/methods at import
time instead of shipping modified copies of verl sources or `.patch` files.
Three primitives are provided:

- ``@patch(Cls, "method")``: replace ``Cls.method`` with the decorated function.
  The original implementation is preserved as ``Cls._orig_method``.
- ``@add(Cls, "method")``: attach a brand-new method to ``Cls`` (only if the
  native class does not already define it). This is how new methods that belong
  to a native class are kept in the recipe.
- ``@wrap(Cls, "method")``: wrap ``Cls.method`` so the decorated function
  receives the original implementation as its first argument.

All patchers are idempotent: a method that has already been patched (its
``_orig_<name>`` slot exists on that class) is left untouched on re-import.

For ``@ray.remote``-decorated classes, these helpers only modify the underlying
plain class. The area module must then recreate the ActorClass from that class
and rebind its consumers before any actor is started. Existing Ray wrappers and
method metadata cannot be updated by changing the original class alone.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional


def unwrap_ray_remote(cls: type) -> type:
    """Unwrap a ``@ray.remote``-decorated class to the underlying plain class.

    ``@ray.remote class Foo`` rebinds ``Foo`` to an ``ActorClass`` object whose
    methods cannot be ``setattr``-ed directly. Ray keeps the original class at
    ``ActorClass.__ray_actor_class__``. Patching it does not refresh the existing
    ActorClass; callers must reconstruct and publish the actor after patching.
    """
    return getattr(cls, "__ray_actor_class__", cls)


def _mark_patched(cls: type, name: str) -> None:
    setattr(cls, f"_orig_{name}", getattr(cls, name))


def _is_patched(cls: type, name: str) -> bool:
    return f"_orig_{name}" in cls.__dict__


def patch(cls: type, name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator: replace ``cls.<name>`` (default: function name) with the decorated fn.

    Example::

        @patch(GlobalRequestLoadBalancer, "acquire_server")
        def acquire_server(self, request_id: str) -> str:
            ...

    The native implementation is kept as ``cls._orig_<name>`` for delegation.
    """

    def decorator(fn: Callable) -> Callable:
        method_name = name or fn.__name__
        target = unwrap_ray_remote(cls)
        if not _is_patched(target, method_name):
            _mark_patched(target, method_name)
            setattr(target, method_name, fn)
        return fn

    return decorator


def add(cls: type, name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator: attach the decorated fn to ``cls`` as a new method.

    No-op if the class already defines ``<name>`` (native method wins), so this
    is safe to use for methods that may land in verl in the future.
    """

    def decorator(fn: Callable) -> Callable:
        method_name = name or fn.__name__
        target = unwrap_ray_remote(cls)
        if not hasattr(target, method_name):
            setattr(target, method_name, fn)
        return fn

    return decorator


def wrap(cls: type, name: str) -> Callable[[Callable], Callable]:
    """Decorator: wrap ``cls.<name>``.

    The decorated fn receives the original implementation as its first
    argument::

        @wrap(LLMServerManager, "__init__")
        def __init__(orig, self, *args, **kwargs):
            orig(self, *args, **kwargs)
            self.run_id = ...
    """

    def decorator(fn: Callable) -> Callable:
        target = unwrap_ray_remote(cls)
        if _is_patched(target, name):
            return fn
        original = getattr(target, name)

        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(original, *args, **kwargs)

        _mark_patched(target, name)
        setattr(target, name, wrapped)
        return fn

    return decorator


def patch_module_function(module: Any, name: str) -> Callable[[Callable], Callable]:
    """Decorator: replace a module-level function ``module.<name>``.

    Example::

        from verl import utils as verl_utils
        import verl.utils.attention_utils as attention_utils

        @patch_module_function(attention_utils, "_get_attention_functions")
        def _get_attention_functions() -> ...:
            ...
    """

    def decorator(fn: Callable) -> Callable:
        if not hasattr(module, f"_orig_{name}"):
            setattr(module, f"_orig_{name}", getattr(module, name))
            setattr(module, name, fn)
        return fn

    return decorator
