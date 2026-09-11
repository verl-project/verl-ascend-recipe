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
``_orig_<name>`` slot exists) is left untouched on re-import.

``@ray.remote``-decorated classes are handled transparently: the decorators
unwrap ``ActorClass`` to its ``__ray_actor_class__`` before patching, so the
same call site works for plain and Ray-actor classes.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional


def unwrap_ray_remote(cls: type) -> type:
    """Unwrap a ``@ray.remote``-decorated class to the underlying plain class.

    ``@ray.remote class Foo`` rebinds ``Foo`` to an ``ActorClass`` object whose
    methods cannot be ``setattr``-ed directly. Ray keeps the original class at
    ``ActorClass.__ray_actor_class__``, so we patch that instead — the actor
    serializes the same class object when ``Foo.remote()`` is called.
    """
    return getattr(cls, "__ray_actor_class__", cls)


def _mark_patched(cls: type, name: str) -> None:
    setattr(cls, f"_orig_{name}", getattr(cls, name))


def _is_patched(cls: type, name: str) -> bool:
    return hasattr(cls, f"_orig_{name}")


def _ray_method_meta(target: type) -> Optional[Any]:
    """Return Ray's per-method metadata table of ``target`` (``None`` if absent).

    ``@ray.remote`` actors keep their per-method options (signatures,
    num_returns, retry config, ...) in ``__ray_metadata__.method_meta``; a
    method must be registered there before it can be called via
    ``actor.<method>.remote()``.
    """
    meta = getattr(target, "__ray_metadata__", None)
    if meta is None:
        return None
    return getattr(meta, "method_meta", None)


def _register_actor_method(target: type, name: str, method: Callable) -> None:
    """Register ``method`` in Ray's method metadata for ``target``.

    ``@patch``/``@add`` only ``setattr`` the underlying plain class, which is
    invisible to Ray's actor dispatch. Declaring the method in ``method_meta``
    with default options lets ``actor.<name>.remote()`` resolve and execute it.
    Idempotent: an existing entry is left untouched.
    """
    method_meta = _ray_method_meta(target)
    if method_meta is None or name in method_meta.methods:
        return
    from ray._common.signature import extract_signature

    method_meta.methods[name] = method
    method_meta.signatures[name] = extract_signature(method, ignore_first=True)
    method_meta.decorators[name] = None
    method_meta.method_is_generator[name] = False
    method_meta.num_returns[name] = None
    method_meta.max_task_retries[name] = 0
    method_meta.retry_exceptions[name] = False
    method_meta.generator_backpressure_num_objects[name] = -1
    method_meta.enable_task_events[name] = False


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
            _register_actor_method(cls, method_name, fn)
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
            _register_actor_method(cls, method_name, fn)
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
