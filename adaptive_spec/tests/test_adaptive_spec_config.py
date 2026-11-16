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

import json

from adaptive_spec.config import AdaptiveSpecConfig


def test_disabled_config_is_upstream_noop():
    config = AdaptiveSpecConfig.from_dict(
        {
            "schema_version": 1,
            "enabled": False,
            "model_fingerprint": "",
        }
    )
    assert config.build_controller() is None


def test_missing_phase_map_is_upstream_noop():
    config = AdaptiveSpecConfig.from_dict(
        {
            "schema_version": 1,
            "enabled": True,
            "model_fingerprint": "model-a",
        }
    )
    assert config.build_controller() is None


def test_enabled_config_builds_controller(tmp_path):
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": "model-a",
                "topology_fingerprint": "tp2",
                "cells": [
                    {
                        "phase_id": "p",
                        "batch_size": 1,
                        "context_length": 1024,
                        "allowed_ks": [5],
                        "iteration_latency_ns_by_k": {"5": 1.0},
                        "prior_survival": [0.9, 0.8, 0.7, 0.6, 0.5],
                        "fallback_k": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = AdaptiveSpecConfig.from_dict(
        {
            "schema_version": 1,
            "enabled": True,
            "phase_map_path": str(phase_path),
            "model_fingerprint": "model-a",
            "topology_fingerprint": "tp2",
            "evidence_exchange_directory": str(tmp_path / "exchange"),
            "evidence_exchange_poll_decisions": 8,
        }
    )
    controller = config.build_controller()
    assert controller is not None
    assert controller.exchange is not None
    assert controller.exchange.poll_decisions == 8


def test_enabled_config_rejects_topology_mismatch(tmp_path):
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_fingerprint": "model-a",
                "topology_fingerprint": "tp2",
                "cells": [
                    {
                        "phase_id": "p",
                        "batch_size": 1,
                        "context_length": 1024,
                        "allowed_ks": [5],
                        "iteration_latency_ns_by_k": {"5": 1.0},
                        "prior_survival": [0.9, 0.8, 0.7, 0.6, 0.5],
                        "fallback_k": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = AdaptiveSpecConfig.from_dict(
        {
            "schema_version": 1,
            "enabled": True,
            "phase_map_path": str(phase_path),
            "model_fingerprint": "model-a",
            "topology_fingerprint": "tp4",
        }
    )
    try:
        config.build_controller()
    except ValueError as error:
        assert "topology" in str(error)
    else:
        raise AssertionError("mismatched topology must fail closed")
