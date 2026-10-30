from __future__ import annotations

# ruff: noqa: E402 -- backend imports intentionally follow pytest.importorskip("torch")
import math
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dapo_predictor.length_scheduler.calibration import MonotoneCalibration
from dapo_predictor.length_scheduler.distribution import GatedDistributionHead
from dapo_predictor.length_scheduler.predictor import (
    CensoredLogNormalPredictor,
    DistributionPrediction,
    LengthPrediction,
    LinearListMLEPredictor,
    PredictionProvenance,
)


def _d2_checkpoint() -> dict:
    head = GatedDistributionHead(projection_dim=2, tap_count=1, sigma_floor=0.05)
    return {
        "feature_keys": ["tap"],
        "projection_dim": 2,
        "projection_bank": {
            "tap": {
                "center": torch.zeros(3),
                "components": torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
                "scale": torch.ones(2),
            }
        },
        "standardized_sigma_floor": 0.05,
        "head_state_dict": head.state_dict(),
        "log_length_center": math.log1p(100.0),
        "log_length_scale": 0.5,
        "schema": {"actor_revision": "actor-r1", "tokenizer_hash": "tokenizer-h1"},
        "calibration_version": "lognormal-test-v1",
    }


def test_length_prediction_invariants() -> None:
    with pytest.raises(ValueError, match="finite"):
        LengthPrediction(math.nan, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        LengthPrediction(-1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="median"):
        LengthPrediction(2.0, 1.0, 3.0, 2.0)
    with pytest.raises(ValueError, match="risk_weight"):
        LengthPrediction(1.0, 2.0, 3.0, 2.0).risk(-1.0)


def test_linear_batch_and_input_validation() -> None:
    predictor = LinearListMLEPredictor(
        torch.tensor([1.0, -1.0]),
        MonotoneCalibration((0.0, 2.0), (100.0, 300.0)),
        feature_key="hidden",
    )
    batch = predictor.predict_batch({"hidden": torch.tensor([[2.0, 1.0], [3.0, 1.0]])})
    assert [row.median for row in batch] == [200.0, 300.0]
    with pytest.raises(ValueError, match="one row"):
        predictor.predict(torch.ones(2, 2))
    with pytest.raises(KeyError, match="hidden"):
        predictor.predict({"wrong": torch.ones(2)})
    with pytest.raises(ValueError, match="feature dim"):
        predictor.predict(torch.ones(3))
    with pytest.raises(ValueError, match="NaN or Inf"):
        predictor.predict(torch.tensor([math.inf, 1.0]))


def test_distribution_prediction_survival_extremes() -> None:
    prediction = DistributionPrediction(
        mu=math.log1p(100.0),
        sigma=0.5,
        expected=113.0,
        median=100.0,
        p90=190.0,
        p95=228.0,
    )
    assert prediction.exceedance_probability(-1.0) == 1.0
    assert prediction.exceedance_probability(math.inf) == 0.0
    assert prediction.exceedance_probability(100.0) == pytest.approx(0.5)
    assert prediction.exceedance_probability(1000.0) < prediction.exceedance_probability(100.0)
    with pytest.raises(ValueError, match="NaN"):
        prediction.exceedance_probability(math.nan)


def test_d2_distribution_api_validation_and_reload() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "d2.pt"
        torch.save(_d2_checkpoint(), path)
        predictor = CensoredLogNormalPredictor.from_checkpoint(path)
        reloaded = CensoredLogNormalPredictor.from_checkpoint(path)

        one = {"tap": torch.tensor([1.0, 2.0, 3.0])}
        prediction = predictor.predict_distribution(one)
        assert prediction == reloaded.predict_distribution(one)
        assert prediction.provenance.checkpoint_sha256 is not None
        assert prediction.provenance.actor_revision == "actor-r1"
        assert prediction.provenance.tokenizer_hash == "tokenizer-h1"
        assert prediction.provenance.tap_schema == ("tap",)
        assert prediction.provenance.calibration_version == "lognormal-test-v1"
        assert predictor.predict(one) == prediction.as_length_prediction()

        batch = {"tap": torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])}
        assert len(predictor.predict_distribution_batch(batch)) == 2
        with pytest.raises(ValueError, match="one row"):
            predictor.predict_distribution(batch)
        with pytest.raises(KeyError, match="tap"):
            predictor.predict_distribution({"wrong": torch.ones(3)})
        with pytest.raises(ValueError, match="expects dim"):
            predictor.predict_distribution({"tap": torch.ones(4)})
        with pytest.raises(ValueError, match="NaN or Inf"):
            predictor.predict_distribution({"tap": torch.tensor([math.nan, 1.0, 2.0])})


def test_provenance_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="64-character"):
        PredictionProvenance(checkpoint_sha256="bad")
    with pytest.raises(ValueError, match="duplicates"):
        PredictionProvenance(tap_schema=("tap", "tap"))
