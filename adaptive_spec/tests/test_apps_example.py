# Copyright 2026
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

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("apps_example", Path(__file__).parents[1] / "examples/apps.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def source(tmp_path, id_field="id"):
    rows = [
        {
            id_field: n,
            "question": f"Question {n}",
            "input_output": json.dumps({"inputs": [f"secret-{n}"], "outputs": [str(n)], "fn_name": "solve"}),
            "starter_code": "def solve(value):\n    pass",
            "difficulty": "introductory",
        }
        for n in range(20)
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("id_field", ["id", "problem_id"])
def test_prepare_repeats_selection_and_keeps_answers_out_of_prompts(tmp_path, id_field):
    path = source(tmp_path, id_field)
    outputs = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for output in outputs:
        example.prepare(SimpleNamespace(input=path, output=output, split="train", limit=7, seed=123))
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    rows = example.read_rows(outputs[0])
    assert len(rows) == 7
    assert [r["row_index"] for r in rows] == sorted(r["row_index"] for r in rows)
    for row in rows:
        assert "secret-" not in row["messages"][0]["content"]
        assert row["test_cases"]["inputs"] == [f"secret-{row['row_index']}"]
        assert row["id"] == f"apps-train-{row['row_index']}"
    receipt = json.loads(outputs[0].with_suffix(".selection.json").read_text())
    assert receipt["source_rows"] == 20
    assert receipt["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_zero_limit_uses_the_whole_split(tmp_path):
    path = source(tmp_path)
    output = tmp_path / "all.jsonl"
    example.prepare(SimpleNamespace(input=path, output=output, split="test", limit=0, seed=42))
    rows = example.read_rows(output)
    assert [row["row_index"] for row in rows] == list(range(20))
    assert len({row["id"] for row in rows}) == 20


@pytest.mark.parametrize("limit", [-1, 21])
def test_prepare_rejects_invalid_selection_size(tmp_path, limit):
    with pytest.raises(ValueError):
        example.prepare(
            SimpleNamespace(input=source(tmp_path), output=tmp_path / "out", split="test", limit=limit, seed=1)
        )


def test_paired_response_seeds_are_unique_across_the_full_split():
    seeds = [example.response_seed(20261103, row, reply, 8) for row in range(5000) for reply in range(8)]
    assert len(set(seeds)) == 40000
    assert example.response_seed(20261103, 32, 0, 8) == seeds[32 * 8]
    with pytest.raises(ValueError):
        example.response_seed(2**32 - 1, 1, 0, 8)


def test_export_keeps_generated_tokens_and_removes_padding():
    assert example.generated_tokens([8, 9, 0, 0], [1, 1, 0, 0]) == [8, 9]
    with pytest.raises(ValueError):
        example.generated_tokens([8, 9, 0], [1, 0, 1])


def test_score_rejects_partial_rollouts_before_running_code(tmp_path):
    data = tmp_path / "problems.jsonl"
    data.write_text(json.dumps({"id": "q1", "test_cases": {}}) + "\n")
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps({"id": "q1", "response_index": 0, "text": "pass"}) + "\n")
    (tmp_path / "summary.json").write_text(
        json.dumps({"input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "responses_per_prompt": 2})
    )
    with pytest.raises(ValueError, match="every expected response"):
        example.score(
            SimpleNamespace(input=data, responses=responses, output=tmp_path / "scores.json", workers=1, timeout=5)
        )


@pytest.mark.parametrize("results,correct", [([True, True], True), ([-2, True], False), ([], False)])
def test_score_preserves_function_tests_and_rejects_error_codes(monkeypatch, results, correct):
    module = ModuleType("verl.utils.reward_score.prime_code.utils")
    cases = {"inputs": [[1], [2]], "outputs": [2, 3], "fn_name": "solve"}

    def check(in_outs, generation, timeout, debug):
        assert in_outs == cases
        assert generation.strip() == "def solve(x): return x + 1"
        return results, []

    module.check_correctness = check
    monkeypatch.setitem(sys.modules, "verl.utils.reward_score.prime_code.utils", module)
    row = example.score_one(
        (
            {"id": "q1", "response_index": 0, "text": "```python\ndef solve(x): return x + 1\n```"},
            {"test_cases": cases},
            5,
        )
    )
    assert row["scorable"] is True
    assert row["correct"] is correct


def test_engine_shutdown_cancels_pending_output_reader_first(monkeypatch):
    import asyncio

    async def run():
        reader = asyncio.create_task(asyncio.sleep(3600))
        await asyncio.sleep(0)
        module = ModuleType("vllm.utils.async_utils")
        module.cancel_task_threadsafe = lambda task: task.cancel() if task and not task.done() else None
        monkeypatch.setitem(sys.modules, module.__name__, module)

        def shutdown(*, timeout):
            assert reader.cancelling() == 1
            assert timeout == 60

        example.close_rollout_engine(SimpleNamespace(engine=SimpleNamespace(output_handler=reader, shutdown=shutdown)))
        with pytest.raises(asyncio.CancelledError):
            await reader

    asyncio.run(run())


def test_engine_shutdown_propagates_cleanup_failure(monkeypatch):
    module = ModuleType("vllm.utils.async_utils")
    module.cancel_task_threadsafe = lambda task: None
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def shutdown(*, timeout):
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        example.close_rollout_engine(SimpleNamespace(engine=SimpleNamespace(shutdown=shutdown)))
