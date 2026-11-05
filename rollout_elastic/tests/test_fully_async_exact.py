"""Regression tests for selected-server versions in fully-async recovery."""

import ast
import asyncio
import copy
import hashlib
import logging
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).parents[1]
ns = dict(
    dataclass=dataclass,
    field=field,
    Any=Any,
    Optional=Optional,
    copy=copy,
    hashlib=hashlib,
    logging=logging,
    time=time,
    logger=logging.getLogger(__name__),
)
for filename in ("types.py", "progress_checkpoint.py"):
    path = ROOT / "workers/rollout/fault_tolerance/progress" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if filename == "types.py":
        from enum import Enum, auto

        ns.update(Enum=Enum, auto=auto)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
CP = ns["VLLMProgressCheckPoint"]


class ClientRecoveryTests(unittest.TestCase):
    def run_client(self, checkpoints, outputs):
        import sys
        from unittest.mock import AsyncMock, Mock, patch
        from uuid import uuid4

        from test_patch_native_dispatch import load_functions

        class Output:
            def __init__(
                self,
                token_ids,
                log_probs=None,
                num_preempted=0,
                routed_experts=None,
                stop_reason=None,
                extra_fields=None,
            ):
                self.token_ids = token_ids
                self.log_probs = log_probs or []
                self.num_preempted = num_preempted
                self.routed_experts = routed_experts
                self.stop_reason = stop_reason
                self.extra_fields = extra_fields or {}

        class Unavailable(Exception):
            def __init__(self, server_id, cause=None):
                self.server_id, self.cause = server_id, cause

        factory = AsyncMock(side_effect=checkpoints)
        fake = SimpleNamespace(
            AllServersFailed=RuntimeError,
            ServerUnavailable=Unavailable,
            is_transient_fault=lambda e: isinstance(e, ConnectionError),
            ProgressContext=ns["ProgressContext"],
            VLLMProgressCheckPoint=SimpleNamespace(create_or_resume=factory),
        )
        scope = dict(
            asyncio=asyncio,
            TokenOutput=Output,
            uuid4=uuid4,
            normalize_token_ids=list,
            logger=logging.getLogger(__name__),
        )
        load_functions("llm_server.py", {"_fully_generate", "_generate_once"}, scope)
        client = SimpleNamespace(
            _ft_enabled=lambda: True,
            _progress_enabled=lambda: True,
            _ft_max_request_retries=lambda: 4,
            _generation_budget_key=lambda: "max_tokens",
            _resolve_original_max_tokens=lambda p: 3,
            _ft_call_timeout_s=lambda: 1,
            _flush_token_interval=lambda: 64,
            _model_version_policy=lambda: ns["ModelVersionPolicy"](mode="exact"),
            _progress_store=object(),
            _run_id="r",
            config=SimpleNamespace(async_training=SimpleNamespace(partial_rollout=True)),
            _release_server=Mock(),
            _mark_server_failed=AsyncMock(),
        )
        server = SimpleNamespace(get_global_steps=SimpleNamespace(remote=AsyncMock(return_value=10)))
        client._acquire_server = AsyncMock(return_value=("s", server))
        client._call_server = AsyncMock(
            side_effect=[Unavailable("<version-change>") if o is None else Output(**o) for o in outputs]
        )
        client._generate_once = lambda *a, **k: scope["_generate_once"](client, *a, **k)
        with patch.dict(sys.modules, {"verl.workers.rollout.fault_tolerance": fake}):
            result = asyncio.run(scope["_fully_generate"](client, "req", prompt_ids=[1], sampling_params={}))
        self.assertEqual(client._release_server.call_count, client._acquire_server.await_count)
        return result, client, factory

    def checkpoint_result(self, ids):
        cp = CP(
            run_id="r",
            recovery_id="req",
            attempt_id=2,
            prompt_token_ids=[1],
            sampling_params={"max_tokens": 3},
            model_weight_version=10,
            inherited_prefix_len=len(ids),
            base_attempt_id=1,
            cumulative_token_ids=ids,
        )
        return SimpleNamespace(checkpoint=cp)

    def test_store_failure_discards_partial_answer_and_recovery_identity(self):
        result, client, factory = self.run_client(
            [self.checkpoint_result([]), ConnectionError(), self.checkpoint_result([])],
            [
                dict(token_ids=[2], stop_reason="abort", extra_fields={"global_steps": 9}),
                dict(token_ids=[3], stop_reason="abort", extra_fields={"global_steps": 10}),
                dict(token_ids=[4, 5, 6], stop_reason="length", extra_fields={"global_steps": 10}),
            ],
        )
        self.assertEqual(result.token_ids, [4, 5, 6])
        calls = client._call_server.call_args_list
        self.assertEqual(calls[1].kwargs["prompt_ids"], [1])
        self.assertEqual(calls[1].kwargs["sampling_params"]["max_tokens"], 3)
        self.assertEqual(result.extra_fields["min_global_steps"], 10)
        self.assertNotEqual(
            factory.call_args_list[1].kwargs["recovery_id"], factory.call_args_list[2].kwargs["recovery_id"]
        )
        client._mark_server_failed.assert_not_called()

    def test_resume_uses_selected_server_version(self):
        result, client, factory = self.run_client(
            [self.checkpoint_result([2])],
            [dict(token_ids=[3, 4], stop_reason="length", extra_fields={"global_steps": 10})],
        )
        self.assertEqual(factory.call_args.kwargs["model_weight_version"], "10")
        self.assertEqual(client._call_server.call_args.kwargs["prompt_ids"], [1, 2])
        self.assertEqual(result.token_ids, [2, 3, 4])

    def test_exhausted_checkpoint_releases_without_generation(self):
        result, client, _ = self.run_client([self.checkpoint_result([2, 3, 4])], [])
        client._call_server.assert_not_called()
        self.assertEqual(result.token_ids, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
