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

from types import SimpleNamespace

import pytest

from adaptive_spec.vllm_plugin import v0_23
from adaptive_spec.vllm_plugin.compatibility import adapter_name
from adaptive_spec.vllm_plugin.kv_source_transform import patch_scheduler_source
from adaptive_spec.vllm_plugin.kv_v0_23 import patch_scheduler_class


def test_raw_counts_not_mamba_counts():
    class Buffer:
        def __getitem__(self, key):
            return self

        def tolist(self):
            return [10, 6]

    calls = []
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(disable_padded_drafter_batch=False),
        valid_sampled_token_count_event=SimpleNamespace(synchronize=lambda: calls.append("sync")),
        valid_sampled_token_count_cpu=Buffer(),
        input_batch=SimpleNamespace(prev_sampled_token_ids=SimpleNamespace(shape=(2, 1))),
        num_accepted_tokens_cpu=[1, 1],
    )
    assert v0_23._accepted_counts(runner, 2) == [10, 6] and calls == ["sync"]
    del runner.valid_sampled_token_count_cpu
    with pytest.raises(RuntimeError, match="raw sampled-token"):
        v0_23._accepted_counts(runner, 2)


def test_unreviewed_version_and_source_fail_closed():
    assert adapter_name("0.24.0", "0.24.0rc1") is None

    class Scheduler:
        def schedule(self):
            return None

    original = Scheduler.schedule
    with pytest.raises(RuntimeError, match="unaudited"):
        patch_scheduler_class(Scheduler, "targeted")
    assert Scheduler.schedule is original
    with pytest.raises(RuntimeError):
        patch_scheduler_source("class Scheduler: pass")
