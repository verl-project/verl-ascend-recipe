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

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_phase_map.py"
SPEC = importlib.util.spec_from_file_location("convert_phase_map", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def source():
    return {
        "model_fingerprint": "model-a",
        "payload_sha256": "frozen-payload",
        "states": [
            {
                "phase_id": "p",
                "b": 8,
                "l": 4096,
                "allowed_ks": [5, 7, 11],
                "regret10_min_k": 7,
                "iteration_latency_ns_by_k": {
                    "5": 6.0,
                    "7": 7.0,
                    "11": 10.0,
                },
                "prior_survival_by_depth": {str(depth): max(0.05, 1.0 - depth * 0.08) for depth in range(1, 12)},
            }
        ],
    }


def test_converter_preserves_measured_fallback_and_excludes_backend_boundary():
    converted = MODULE.convert(
        source(),
        topology_fingerprint="tp2",
        excluded_ks={11},
    )
    cell = converted["cells"][0]
    assert cell["allowed_ks"] == [5, 7]
    assert cell["fallback_k"] == 7
    assert converted["topology_fingerprint"] == "tp2"


def test_converter_fails_when_exclusion_removes_every_safe_depth():
    value = source()
    value["states"][0]["allowed_ks"] = [11]
    try:
        MODULE.convert(value, topology_fingerprint="tp2", excluded_ks={11})
    except ValueError as error:
        assert "no safe K" in str(error)
    else:
        raise AssertionError("empty safe grid must fail")


def test_mixed_candidate_depth_cells_preserve_full_measured_prior():
    from copy import deepcopy

    from adaptive_spec.controller import AdaptiveSpecController, RequestContext
    from adaptive_spec.phase_map import PhaseMap

    value = source()
    shallow = deepcopy(value["states"][0])
    shallow.update(phase_id="shallow", l=8192, allowed_ks=[5], regret10_min_k=5)
    value["states"].append(shallow)
    converted = MODULE.convert(value, topology_fingerprint="tp2", excluded_ks={11})
    assert len(converted["cells"][1]["prior_survival"]) == 7
    assert converted["cells"][1]["allowed_ks"] == [5]
    ctl = AdaptiveSpecController(PhaseMap.from_dict(converted), model_fingerprint="model-a")
    requests = [RequestContext(str(i), (1, 2), {}, 8192) for i in range(8)]
    assert ctl.decide_engine(requests).physical_k == 5
