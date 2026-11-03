"""Exercise predictor worker methods without starting distributed actors."""

from __future__ import annotations

# ruff: noqa: E402 -- optional dependencies are checked before importing the recipe.
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
Timer = pytest.importorskip("codetiming").Timer

from dapo_predictor.length_scheduler.calibration import MonotoneCalibration, fit_pava


class _TensorBatch(dict):
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        return type(self)({name: value[key] for name, value in self.items()})


class _Proto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = _TensorBatch(batch or {})
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None):
        return cls(tensors, non_tensors, meta_info)

    def to(self, device):
        self.batch = _TensorBatch({key: value.to(device) for key, value in self.batch.items()})
        return self


def _identity_decorator(*args, **kwargs):
    return lambda function: function


@pytest.fixture
def worker():
    path = Path(__file__).parents[1] / "predictor_worker.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    namespace = {
        "torch": torch,
        "np": pytest.importorskip("numpy"),
        "nn": torch.nn,
        "DataProto": _Proto,
        "DataParallelPPOActor": object,
        "AsyncActorRolloutRefWorker": object,
        "register": _identity_decorator,
        "Dispatch": SimpleNamespace(ONE_TO_ALL="all"),
        "make_nd_compute_dataproto_dispatch_fn": lambda **kwargs: None,
        "DistProfiler": SimpleNamespace(annotate=_identity_decorator),
        "OmegaConf": OmegaConf,
        "get_device_id": lambda: "cpu",
        "Timer": Timer,
        "DataLoader": torch.utils.data.DataLoader,
        "TensorDataset": torch.utils.data.TensorDataset,
        "MonotoneCalibration": MonotoneCalibration,
        "fit_pava": fit_pava,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    worker_type = namespace["PredictorAsyncActorRolloutRefWorker"]
    actor_type = namespace["PredictorDataParallelPPOActor"]
    value = worker_type.__new__(worker_type)
    value.config = OmegaConf.create(
        {
            "predictor_reorder": {"enable": True, "epochs": 1, "batch_size": 32},
            "actor": {"ulysses_sequence_parallel_size": 1},
            "rollout": {"n": 2, "response_length": 64, "temperature": 1.0},
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 128,
                "log_prob_use_dynamic_bsz": False,
            },
        }
    )
    value._is_actor = True
    value._is_offload_param = False
    value._pending_offload_param_restore = None
    value.actor_module_fsdp = torch.nn.Linear(3, 3)
    value.actor = SimpleNamespace(
        predictor_scorer=torch.nn.Linear(3, 1, bias=False),
        predictor_calibration=MonotoneCalibration((0.0, 10.0), (1.0, 100.0)),
        predictor_calibration_hidden=torch.empty((0, 3)),
        predictor_calibration_tokens=torch.empty(0),
        listmle_loss=actor_type.listmle_loss,
    )
    value.actor.predictor_scorer.weight.data.fill_(1.0)
    value.actor.extract_hidden_states = lambda batch=None, data=None: torch.tensor(
        [[1.0, 2.0, 3.0]], dtype=torch.bfloat16
    ).repeat((batch if batch is not None else data).batch["input_ids"].shape[0], 1)
    value.ulysses_sharding_manager = contextlib.nullcontext()
    value.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    return value


def _prompts(rows=2):
    return _Proto(
        {
            "input_ids": torch.ones((rows, 3), dtype=torch.long),
            "attention_mask": torch.ones((rows, 3), dtype=torch.long),
            "position_ids": torch.arange(3).repeat(rows, 1),
        }
    )


def test_bfloat16_actor_features_can_be_scored_by_float32_head(worker):
    output = worker.compute_predictor_score(_prompts())
    assert output.batch["predictor_scores"].dtype == torch.float32
    assert output.batch["predictor_scores"].tolist() == [6.0, 6.0]
    assert torch.isfinite(output.batch["epws_predicted_work"]).all()
    assert not output.batch["predictor_scores"].requires_grad


def test_single_prompt_update_keeps_calibration_and_metrics_valid(worker):
    responses = _Proto({"responses": torch.tensor([[5, 6, 0], [7, 8, 9]])})
    output = worker.update_predictor(_prompts(), responses)
    assert output.meta_info["metrics"]["predictor/observed_rollouts"] == 2
    assert output.meta_info["metrics"]["predictor/final_loss"] == pytest.approx(0.0)
    assert worker.actor.predictor_calibration_tokens.tolist() == [3.0]


def test_response_mask_counts_valid_tokens_equal_to_pad_id(worker):
    prompts = _prompts(4)
    responses = _Proto(
        {
            "responses": torch.tensor([[5, 0, 0], [6, 0, 0], [7, 0, 0], [8, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
        }
    )
    worker.update_predictor(prompts, responses)
    assert worker.actor.predictor_calibration_tokens.tolist() == [2.0, 2.0]
