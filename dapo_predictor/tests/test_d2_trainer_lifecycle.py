"""Execute the real trainer methods on CPU without initializing Ray/actors."""

from __future__ import annotations
import __future__

import ast
import hashlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dapo_predictor.length_scheduler.checkpoint import (
    DISTRIBUTION_BACKEND,
    LINEAR_BACKEND,
    build_lifecycle_state,
    validate_lifecycle_state,
)
from dapo_predictor.length_scheduler.d2_quality import D2QualityConfig, D2QualityLifecycle

torch = pytest.importorskip("torch")


class Proto:
    def __init__(self, batch, meta_info=None):
        self.batch = batch
        self.meta_info = meta_info or {}


@pytest.fixture
def trainer_type():
    # Importing a real FSDP worker would initialize distributed framework imports.
    # Compile the actual trainer class unchanged, supplying only its dependencies.
    source = Path(__file__).parents[1] / "predictor_dapo_trainer.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    base = type("Base", (), {"_load_checkpoint": lambda self: None, "_save_checkpoint": lambda self: None})
    namespace = dict(
        RayDAPOTrainer=base,
        D2QualityConfig=D2QualityConfig,
        D2QualityLifecycle=D2QualityLifecycle,
        DISTRIBUTION_BACKEND=DISTRIBUTION_BACKEND,
        LINEAR_BACKEND=LINEAR_BACKEND,
        hashlib=hashlib,
        json=json,
        warnings=warnings,
        torch=torch,
        np=np,
        build_lifecycle_state=build_lifecycle_state,
        validate_lifecycle_state=validate_lifecycle_state,
    )
    import os
    import uuid

    namespace.update(os=os, uuid=uuid)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return namespace[node.name]


def trainer(trainer_type, *, backend="censored_lognormal", total_epochs=30, lifecycle=None):
    value = trainer_type()
    value.config = SimpleNamespace(
        trainer=SimpleNamespace(
            get=lambda key, default=None: {
                "predictor_reorder": {
                    "enable": True,
                    "backend": backend,
                    "scheduler": "epws",
                    "activation": {"min_epoch": 10, "min_samples": 0},
                    "d2_lifecycle": lifecycle or {},
                }
            }.get(key, default),
            total_epochs=total_epochs,
        ),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(response_length=100)),
    )
    value.actor_rollout_wg = SimpleNamespace(load_d2_candidate=Mock())
    value._predictor_observed_rollouts = 5000
    value._predictor_ready = False
    value._init_d2_quality()
    return value


def test_d1_does_not_enter_new_lifecycle(trainer_type):
    value = trainer(trainer_type, backend=LINEAR_BACKEND)
    assert value._d2_quality is None
    value._predictor_ready = True
    assert value._predictor_active(10)
    assert not value._predictor_active(9)
    value._d2_epoch_boundary(10)
    value.actor_rollout_wg.load_d2_candidate.assert_not_called()


def test_twenty_is_warning_not_disable_gate(trainer_type):
    with pytest.warns(UserWarning, match="recommendation, not a gate"):
        value = trainer(trainer_type, total_epochs=15)
    assert value._d2_quality is not None


def test_explicit_legacy_d2_bypasses_new_quality_lifecycle(trainer_type):
    value = trainer(trainer_type, lifecycle={"enable": False})
    value._predictor_ready = True
    assert value._d2_quality is None
    assert value._predictor_active(10)


def test_pre_rollout_forecasts_are_observed_before_filtering(trainer_type):
    value = trainer(trainer_type)
    value._d2_quality.status = "active"
    predictions = Proto(
        {
            "input_ids": torch.tensor([[0, 7], [0, 7], [0, 8]]),
            "attention_mask": torch.tensor([[0, 1], [0, 1], [0, 1]]),
            "d2_expected": torch.tensor([10.0, 10.0, 20.0]),
            "d2_median": torch.tensor([8.0, 8.0, 18.0]),
            "d2_p90": torch.tensor([16.0, 16.0, 30.0]),
        },
        {"epws_predictor_active": True},
    )
    value._observe_d2_rollout(predictions, Proto({"response_mask": torch.tensor([[1, 1], [1, 1], [1, 0]])}), 10)
    metrics = value._d2_quality.window.metrics(value._d2_quality.config)
    assert metrics["distinct_prompts"] == 2
    assert metrics["rollouts"] == 3
    source = (Path(__file__).parents[1] / "predictor_dapo_trainer.py").read_text()
    assert source.index("self._observe_d2_rollout(d2_forecasts") < source.index(
        "if not self.config.algorithm.filter_groups.enable"
    )


def test_missing_forecasts_does_not_recompute_posthoc(trainer_type):
    value = trainer(trainer_type)
    value._d2_quality.status = "active"
    value._observe_d2_rollout(Proto({}, {"epws_predictor_active": False}), Proto({}), 10)
    assert value._d2_quality.window.groups == {}
    value._d2_quality.last_boundary = 10
    value._d2_epoch_boundary(15)
    assert not value._predictor_active(15)


def test_mask_fallback_and_cap_censoring(trainer_type):
    value = trainer(trainer_type)
    value._d2_quality.status = "active"
    forecasts = Proto(
        {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
            "d2_expected": torch.tensor([100.0]),
            "d2_median": torch.tensor([80.0]),
            "d2_p90": torch.tensor([130.0]),
        },
        {"epws_predictor_active": True},
    )
    completed = Proto({"responses": torch.zeros(1, 100), "attention_mask": torch.ones(1, 101)})
    value._observe_d2_rollout(forecasts, completed, 10)
    assert value._d2_quality.window.metrics(value._d2_quality.config)["censored_rollouts"] == 1


def test_worker_rpc_is_frozen_load_only_and_d1_update_is_preserved():
    source = (Path(__file__).parents[1] / "predictor_worker.py").read_text()
    module = ast.parse(source)
    load = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == "load_d2_candidate")
    text = ast.get_source_segment(source, load)
    assert text.count("file_sha256(path)") == 2
    assert "from_checkpoint(path)" in text
    assert "optimizer" not in text and "backward" not in text
    assert "Dispatch.ONE_TO_ALL" in ast.unparse(load.decorator_list[0])


def test_real_trainer_admission_and_resume_load_the_same_hash(trainer_type, tmp_path):
    from dapo_predictor.length_scheduler.d2_quality import make_validation_report

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"immutable-head")
    rows = [
        dict(
            prompt_id=str(i),
            epoch=10,
            observed=(i + 1) * 100,
            expected=(i + 1) * 100,
            median=(i + 1) * 100,
            p90=(i + 1) * 150,
        )
        for i in range(40)
    ]
    report = make_validation_report(rows, checkpoint=checkpoint, trained_through_epoch=9, validation_epoch=10)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    original = trainer(trainer_type, lifecycle={"candidate_report": str(path)})
    original._d2_epoch_boundary(10)
    assert original._predictor_active(10)
    original.actor_rollout_wg.load_d2_candidate.assert_called_once_with(str(checkpoint), report["checkpoint_sha256"])
    original.config.trainer.default_local_dir = str(tmp_path)
    original.global_steps = 3
    (tmp_path / "global_step_3").mkdir()
    original._d2_rl_epoch = 11
    original._save_checkpoint()
    restored = trainer(trainer_type, lifecycle={"candidate_report": str(path)})
    restored.config.trainer.default_local_dir = str(tmp_path)
    restored.global_steps = 3
    restored._load_checkpoint()
    assert restored._d2_rl_epoch == 11
    assert restored._predictor_active(11)
    restored.actor_rollout_wg.load_d2_candidate.assert_called_once_with(str(checkpoint), report["checkpoint_sha256"])
    saved_path = tmp_path / "global_step_3" / "predictor_state.json"
    saved = json.loads(saved_path.read_text())
    corruptions = [dict(backend=LINEAR_BACKEND), dict(ready=False), dict(d2_rl_epoch=9), dict(d2_rl_epoch=11.5)]
    for changed in corruptions:
        saved_path.write_text(json.dumps(saved | changed))
        invalid = trainer(trainer_type, lifecycle={"candidate_report": str(path)})
        invalid.config.trainer.default_local_dir = str(tmp_path)
        invalid.global_steps = 3
        with pytest.raises(ValueError):
            invalid._load_checkpoint()
        invalid.actor_rollout_wg.load_d2_candidate.assert_not_called()
        assert not invalid._predictor_ready


def test_multimodal_rows_cannot_silently_share_text_identity(trainer_type):
    value = trainer(trainer_type)
    value._d2_quality.status = "active"
    value._predictor_ready = True
    batch = Proto({})
    batch.non_tensor_batch = {"multi_modal_data": [{"image": "image-A"}, {"image": "image-B"}]}
    value.actor_rollout_wg.compute_predictor_score = Mock(side_effect=AssertionError("must not forward"))
    with pytest.warns(UserWarning, match="text prompt identities only"):
        output = value._attach_epws_predictions(batch, 10)
    assert output.meta_info["epws_predictor_active"] is False
    assert value._d2_quality.status == "suspended"
    value.actor_rollout_wg.compute_predictor_score.assert_not_called()


def test_candidate_rpc_failure_never_leaves_ready_latched(trainer_type, tmp_path):
    from dapo_predictor.length_scheduler.d2_quality import make_validation_report

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"frozen-head")
    rows = [
        dict(
            prompt_id=str(i),
            epoch=10,
            observed=(i + 1) * 100,
            expected=(i + 1) * 100,
            median=(i + 1) * 100,
            p90=(i + 1) * 150,
        )
        for i in range(40)
    ]
    report = make_validation_report(rows, checkpoint=checkpoint, trained_through_epoch=9, validation_epoch=10)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    value = trainer(trainer_type, lifecycle={"candidate_report": str(path)})
    value.actor_rollout_wg.load_d2_candidate.side_effect = RuntimeError("rank rejected file")
    with pytest.raises(RuntimeError, match="rank rejected"):
        value._d2_epoch_boundary(10)
    assert not value._predictor_ready
    assert not value._predictor_active(10)
    assert value._d2_quality.status == "suspended"


def test_bad_numeric_observation_suspends_immediately(trainer_type):
    value = trainer(trainer_type)
    value._d2_quality.status = "active"
    forecasts = Proto(
        {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
            "d2_expected": torch.tensor([float("nan")]),
            "d2_median": torch.tensor([80.0]),
            "d2_p90": torch.tensor([130.0]),
        },
        {"epws_predictor_active": True},
    )
    with pytest.warns(UserWarning, match="falls back to FCFS"):
        value._observe_d2_rollout(forecasts, Proto({"response_mask": torch.ones(1, 2)}), 10)
    assert not value._predictor_active(10)


def test_real_worker_loads_and_runs_frozen_d2_on_cpu(tmp_path):
    import math

    from dapo_predictor.length_scheduler.checkpoint import file_sha256
    from dapo_predictor.length_scheduler.distribution import GatedDistributionHead
    from dapo_predictor.length_scheduler.predictor import CensoredLogNormalPredictor

    head = GatedDistributionHead(projection_dim=2, tap_count=1, sigma_floor=0.05)
    checkpoint = tmp_path / "head.pt"
    torch.save(
        {
            "feature_keys": ["layer_0"],
            "projection_dim": 2,
            "projection_bank": {
                "layer_0": {
                    "center": torch.zeros(3),
                    "components": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
                    "scale": torch.ones(2),
                }
            },
            "standardized_sigma_floor": 0.05,
            "head_state_dict": head.state_dict(),
            "log_length_center": math.log1p(100.0),
            "log_length_scale": 0.5,
        },
        checkpoint,
    )
    source = Path(__file__).parents[1] / "predictor_worker.py"
    module = ast.parse(source.read_text())
    load = next(n for n in ast.walk(module) if isinstance(n, ast.FunctionDef) and n.name == "load_d2_candidate")
    load.decorator_list = []
    namespace = dict(file_sha256=file_sha256, CensoredLogNormalPredictor=CensoredLogNormalPredictor, torch=torch)
    exec(compile(ast.Module(body=[load], type_ignores=[]), str(source), "exec"), namespace)
    worker = SimpleNamespace(_predictor_cfg=lambda: {"backend": DISTRIBUTION_BACKEND}, actor=SimpleNamespace())
    digest = file_sha256(checkpoint)
    namespace["load_d2_candidate"](worker, str(checkpoint), digest)
    model = worker.actor.distribution_predictor
    before = {k: v.clone() for k, v in model.prior.head.state_dict().items()}
    predictions = model.predict_distribution_batch({"layer_0": torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])})
    assert len(predictions) == 2
    assert all(row.p90 >= row.median >= 0 for row in predictions)
    assert all(torch.equal(v, model.prior.head.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError, match="SHA-256"):
        namespace["load_d2_candidate"](worker, str(checkpoint), "0" * 64)
    assert worker.actor.distribution_predictor is model
