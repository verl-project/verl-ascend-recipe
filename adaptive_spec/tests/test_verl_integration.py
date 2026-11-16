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

from adaptive_spec.step_scope import sampling_params_with_step


def test_step_scope_preserves_sampling_and_existing_extra_args():
    original = {
        "temperature": 1.0,
        "extra_args": {"existing": "value"},
    }
    scoped = sampling_params_with_step(original, 7)
    assert scoped["temperature"] == 1.0
    assert scoped["extra_args"] == {
        "existing": "value",
        "adaptive_spec_rollout_step": 7,
    }
    assert original == {
        "temperature": 1.0,
        "extra_args": {"existing": "value"},
    }
