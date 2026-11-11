"""Dependency-free tests of the production checkpoint version bookkeeping."""

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


class VersionRangeTests(unittest.TestCase):
    def checkpoint(self):
        return CP(
            run_id="r",
            recovery_id="q",
            attempt_id=1,
            prompt_token_ids=[1],
            sampling_params={"max_tokens": 3},
            model_weight_version=None,
            inherited_prefix_len=0,
            base_attempt_id=1,
        )

    def chunk(self, ids):
        return SimpleNamespace(outputs=[SimpleNamespace(token_ids=ids)])

    def test_only_new_tokens_contribute_and_zero_is_valid(self):
        cp = self.checkpoint()
        cp.ingest(self.chunk([]), 99)
        self.assertIsNone(cp.min_global_steps)
        cp.ingest(self.chunk([2]), 0)
        cp.ingest(self.chunk([2]), 99)
        self.assertEqual((cp.min_global_steps, cp.max_global_steps), (0, 0))
        cp.ingest(self.chunk([2, 3]), 4)
        self.assertEqual((cp.min_global_steps, cp.max_global_steps), (0, 4))

    def test_payload_round_trip_and_continuation(self):
        cp = self.checkpoint()
        cp.ingest(self.chunk([2]), 10)
        cp = CP.from_payload(cp.to_payload(), 64)
        cp.inherited_prefix_len = 1
        cp.ingest(self.chunk([3]), 11)
        self.assertEqual(cp.cumulative_token_ids, [2, 3])
        self.assertEqual((cp.min_global_steps, cp.max_global_steps), (10, 11))
        cp = CP.from_payload(cp.to_payload(), 64)
        self.assertEqual((cp.min_global_steps, cp.max_global_steps), (10, 11))

    def test_resume_inherits_range_and_fresh_retry_discards_it(self):
        from unittest.mock import AsyncMock

        cp = self.checkpoint()
        cp.ingest(self.chunk([2]), 10)
        payload = cp.to_payload()
        store = SimpleNamespace(
            load_latest=SimpleNamespace(remote=AsyncMock()), mark_superseded=SimpleNamespace(remote=AsyncMock())
        )

        async def resume(value):
            store.load_latest.remote.return_value = value
            return await CP.create_or_resume(
                store=store,
                run_id="r",
                recovery_id="q",
                prompt_token_ids=[1],
                sampling_params={"max_tokens": 3},
                model_weight_version=11,
                flush_token_interval=64,
                model_version_policy=ns["ModelVersionPolicy"](mode="relaxed"),
            )

        result = asyncio.run(resume(ns["LoadResult"](payload, None, "hit")))
        self.assertEqual(result.checkpoint.cumulative_token_ids, [2])
        self.assertEqual((result.checkpoint.min_global_steps, result.checkpoint.max_global_steps), (10, 10))
        result = asyncio.run(resume(ns["LoadResult"](None, ns["LoadFailure"].NOT_FOUND, "miss", 1)))
        self.assertEqual(result.checkpoint.cumulative_token_ids, [])
        self.assertIsNone(result.checkpoint.min_global_steps)
        self.assertIsNone(result.checkpoint.max_global_steps)

    def test_old_payload_does_not_invent_range(self):
        payload = self.checkpoint().to_payload()
        del payload.min_global_steps
        del payload.max_global_steps
        cp = CP.from_payload(payload, 64)
        self.assertIsNone(cp.min_global_steps)
        self.assertIsNone(cp.max_global_steps)


class DiskRecoveryTests(unittest.TestCase):
    def test_rebuild_index_and_restore_range_from_manifest(self):
        import json
        import tempfile

        path = ROOT / "workers/rollout/fault_tolerance/progress/progress_store.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        store_ns = dict(
            ns,
            json=json,
            Path=Path,
            MANIFEST="manifest.json",
            STATUS_TERMINAL_OK="TERMINAL_OK",
            STATUS_STREAMING="STREAMING",
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), store_ns)
        Store = store_ns["RolloutProgressStore"]
        with tempfile.TemporaryDirectory() as tmp:
            for with_range in (False, True):
                manifest = dict(
                    run_id="r",
                    recovery_id="q",
                    attempt_id=1,
                    model_weight_version=11,
                    inherited_prefix_len=1,
                    finished=False,
                    updated_at=123.0,
                    len_tokens=2,
                )
                if with_range:
                    manifest.update(min_global_steps=10, max_global_steps=11)
                directory = Path(tmp) / "r" / "q" / "1"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                store = Store.__new__(Store)
                store._root = Path(tmp)
                store._index = {}
                store._gc_heap = []
                store._config = SimpleNamespace(gc_delay_s=30)
                store._rebuild_index_from_disk()
                record = store._index[("r", "q")].attempts[1]
                self.assertEqual(record.durable_offset, 2)
                self.assertEqual(record.model_weight_version, 11)
                store._to_list = lambda value: value
                payload = store._assemble_payload(manifest, [2, 3], None, None)
                self.assertEqual(payload.min_global_steps, 10 if with_range else None)
                self.assertEqual(payload.max_global_steps, 11 if with_range else None)


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
            _model_version_policy=lambda: ns["ModelVersionPolicy"](),
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
            min_global_steps=10 if ids else None,
            max_global_steps=10 if ids else None,
        )
        return SimpleNamespace(checkpoint=cp)

    def test_store_failure_discards_partial_answer_and_budget(self):
        result, client, factory = self.run_client(
            [self.checkpoint_result([]), ConnectionError(), self.checkpoint_result([])],
            [
                dict(token_ids=[2], stop_reason="abort", extra_fields={"global_steps": 10}),
                dict(token_ids=[3], stop_reason="abort", extra_fields={"global_steps": 10}),
                dict(token_ids=[4, 5, 6], stop_reason="length", extra_fields={"global_steps": 10}),
            ],
        )
        self.assertEqual(result.token_ids, [4, 5, 6])
        calls = client._call_server.call_args_list
        self.assertEqual(calls[1].kwargs["prompt_ids"], [1])
        self.assertEqual(calls[1].kwargs["sampling_params"]["max_tokens"], 3)
        self.assertNotEqual(
            factory.call_args_list[1].kwargs["recovery_id"], factory.call_args_list[2].kwargs["recovery_id"]
        )

    def test_resume_uses_selected_server_version(self):
        result, client, factory = self.run_client(
            [self.checkpoint_result([2])],
            [dict(token_ids=[3, 4], stop_reason="length", extra_fields={"global_steps": 10})],
        )
        self.assertEqual(factory.call_args.kwargs["model_weight_version"], "10")
        self.assertEqual(client._call_server.call_args.kwargs["prompt_ids"], [1, 2])
        self.assertEqual(result.token_ids, [2, 3, 4])

    def test_version_race_retries_without_isolating_healthy_server(self):
        result, client, _ = self.run_client(
            [self.checkpoint_result([2]), self.checkpoint_result([2])],
            [None, dict(token_ids=[3, 4], stop_reason="length", extra_fields={"global_steps": 10})],
        )
        self.assertEqual(result.token_ids, [2, 3, 4])
        self.assertEqual(result.extra_fields["llm_generate_retries"], 1)
        client._mark_server_failed.assert_not_called()

    def test_exhausted_checkpoint_releases_without_generation(self):
        result, client, _ = self.run_client([self.checkpoint_result([2, 3, 4])], [])
        client._call_server.assert_not_called()
        self.assertEqual(result.token_ids, [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
