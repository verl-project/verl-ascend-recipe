"""Predictor backends consumed by the rollout scheduler."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
import torch.nn.functional as F

from .calibration import MonotoneCalibration
from .distribution import PrefillDistributionPrior


@dataclass(frozen=True)
class PredictionProvenance:
    """Version envelope used to make prediction caches auditable."""

    checkpoint_sha256: str | None = None
    actor_revision: str | None = None
    model_revision: str | None = None
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    tap_schema: tuple[str, ...] = ()
    calibration_version: str | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256.lower()
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "checkpoint_sha256", digest)
        for name in (
            "actor_revision",
            "model_revision",
            "tokenizer_hash",
            "chat_template_hash",
            "calibration_version",
        ):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} must be non-empty when provided")
        if len(set(self.tap_schema)) != len(self.tap_schema):
            raise ValueError("tap_schema must not contain duplicates")


@dataclass(frozen=True)
class LengthPrediction:
    """Token-scale work prediction used by EPWS."""

    median: float
    p90: float
    p95: float
    expected: float
    provenance: PredictionProvenance = field(default_factory=PredictionProvenance, compare=False)

    def __post_init__(self) -> None:
        values = (self.median, self.p90, self.p95, self.expected)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("length statistics must be finite")
        if any(value < 0 for value in values):
            raise ValueError("length statistics must be non-negative")
        if not self.median <= self.p90 <= self.p95:
            raise ValueError("length quantiles must satisfy median <= p90 <= p95")

    def risk(self, risk_weight: float = 0.5) -> float:
        if not math.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError("risk_weight must be finite and non-negative")
        return self.median + float(risk_weight) * max(self.p90 - self.median, 0.0)


@dataclass(frozen=True)
class DistributionPrediction:
    """Unstandardized LogNormal parameters and token-scale summaries."""

    mu: float
    sigma: float
    expected: float
    median: float
    p90: float
    p95: float
    provenance: PredictionProvenance = field(default_factory=PredictionProvenance)

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        LengthPrediction(self.median, self.p90, self.p95, self.expected, self.provenance)

    def as_length_prediction(self) -> LengthPrediction:
        return LengthPrediction(self.median, self.p90, self.p95, self.expected, self.provenance)

    def exceedance_probability(self, token_threshold: float) -> float:
        """Return ``P(response_tokens > token_threshold)`` stably."""
        threshold = float(token_threshold)
        if math.isnan(threshold):
            raise ValueError("token_threshold must not be NaN")
        if threshold < 0:
            return 1.0
        if threshold == math.inf:
            return 0.0
        z = (math.log1p(threshold) - self.mu) / self.sigma
        return min(max(0.5 * math.erfc(z / math.sqrt(2.0)), 0.0), 1.0)


class PredictorBackend(Protocol):
    """Backend-neutral prompt predictor interface."""

    name: str

    def predict(self, features: Any) -> LengthPrediction: ...


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value(checkpoint: Mapping[str, Any], name: str) -> Any:
    for container in (checkpoint, checkpoint.get("schema"), checkpoint.get("metadata"), checkpoint.get("provenance")):
        if isinstance(container, Mapping) and container.get(name) is not None:
            return container[name]
    return None


def _checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    path: str | Path,
    *,
    tap_schema: tuple[str, ...],
    default_calibration_version: str,
    override: PredictionProvenance | None,
) -> PredictionProvenance:
    def choose(name: str, default: Any = None) -> Any:
        supplied = getattr(override, name) if override is not None else None
        return supplied if supplied is not None and supplied != () else _metadata_value(checkpoint, name) or default

    return PredictionProvenance(
        checkpoint_sha256=_checkpoint_sha256(path),
        actor_revision=choose("actor_revision"),
        model_revision=choose("model_revision"),
        tokenizer_hash=choose("tokenizer_hash"),
        chat_template_hash=choose("chat_template_hash"),
        tap_schema=tuple(choose("tap_schema", tap_schema)),
        calibration_version=str(choose("calibration_version", default_calibration_version)),
    )


class LinearListMLEPredictor:
    """Bias-free rank head with a train-only monotone token calibration."""

    name = "linear_listmle"

    def __init__(
        self,
        weight: torch.Tensor,
        calibration: MonotoneCalibration,
        *,
        normalize_features: bool = False,
        feature_key: str | None = None,
        provenance: PredictionProvenance | None = None,
    ):
        self.weight = weight.detach().cpu().float().reshape(-1)
        self.calibration = calibration
        self.normalize_features = bool(normalize_features)
        self.feature_key = feature_key
        self.provenance = provenance or PredictionProvenance(
            tap_schema=(feature_key,) if feature_key else (), calibration_version="pava-v1"
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, provenance: PredictionProvenance | None = None
    ) -> LinearListMLEPredictor:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        calibration = checkpoint["length_calibration"]
        feature_key = checkpoint.get("feature_key")
        envelope = _checkpoint_provenance(
            checkpoint,
            path,
            tap_schema=(feature_key,) if feature_key else (),
            default_calibration_version="pava-v1",
            override=provenance,
        )
        return cls(
            checkpoint["weight"],
            MonotoneCalibration(tuple(calibration["scores"]), tuple(calibration["tokens"])),
            normalize_features=bool(checkpoint.get("normalize_features", False)),
            feature_key=feature_key,
            provenance=envelope,
        )

    def _feature_matrix(self, features: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(features, Mapping):
            if self.feature_key is None:
                raise ValueError("checkpoint does not declare feature_key")
            if self.feature_key not in features:
                raise KeyError(f"missing feature key {self.feature_key!r}")
            features = features[self.feature_key]
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor or a mapping of tensors")
        if features.ndim == 1:
            rows = features.unsqueeze(0)
        elif features.ndim == 2:
            rows = features
        else:
            raise ValueError(f"features must be 1D or 2D, got shape {tuple(features.shape)}")
        if rows.shape[-1] != self.weight.numel():
            raise ValueError(f"expected feature dim {self.weight.numel()}, got {rows.shape[-1]}")
        if not torch.isfinite(rows).all():
            raise ValueError("features contain NaN or Inf")
        rows = rows.detach().cpu().float()
        return F.normalize(rows, dim=-1) if self.normalize_features else rows

    def predict_batch(self, features: torch.Tensor | Mapping[str, torch.Tensor]) -> tuple[LengthPrediction, ...]:
        rows = self._feature_matrix(features)
        scores = rows @ self.weight
        return tuple(
            LengthPrediction(tokens, tokens, tokens, tokens, self.provenance)
            for tokens in (float(self.calibration(float(score))) for score in scores)
        )

    def predict(self, features: torch.Tensor | Mapping[str, torch.Tensor]) -> LengthPrediction:
        predictions = self.predict_batch(features)
        if len(predictions) != 1:
            raise ValueError("predict() requires one row; use predict_batch() for a batch")
        return predictions[0]


class CensoredLogNormalPredictor:
    """Optional multi-tap distribution predictor."""

    name = "censored_lognormal"

    def __init__(self, prior: PrefillDistributionPrior, provenance: PredictionProvenance | None = None):
        self.prior = prior
        self.provenance = provenance or PredictionProvenance(
            tap_schema=prior.feature_keys, calibration_version="lognormal-v1"
        )

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, provenance: PredictionProvenance | None = None
    ) -> CensoredLogNormalPredictor:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        prior = PrefillDistributionPrior(checkpoint)
        envelope = _checkpoint_provenance(
            checkpoint,
            path,
            tap_schema=prior.feature_keys,
            default_calibration_version="lognormal-v1",
            override=provenance,
        )
        return cls(prior, envelope)

    def predict_distribution_batch(self, features: Mapping[str, torch.Tensor]) -> tuple[DistributionPrediction, ...]:
        summary = self.prior.predict(features)
        return tuple(
            DistributionPrediction(
                mu=float(summary["mu"].reshape(-1)[index]),
                sigma=float(summary["sigma"].reshape(-1)[index]),
                median=float(summary["median"].reshape(-1)[index]),
                p90=float(summary["p90"].reshape(-1)[index]),
                p95=float(summary["p95"].reshape(-1)[index]),
                expected=float(summary["expected"].reshape(-1)[index]),
                provenance=self.provenance,
            )
            for index in range(int(summary["mu"].numel()))
        )

    def predict_distribution(self, features: Mapping[str, torch.Tensor]) -> DistributionPrediction:
        predictions = self.predict_distribution_batch(features)
        if len(predictions) != 1:
            raise ValueError("predict_distribution() requires one row; use predict_distribution_batch() for a batch")
        return predictions[0]

    def predict_batch(self, features: Mapping[str, torch.Tensor]) -> tuple[LengthPrediction, ...]:
        return tuple(row.as_length_prediction() for row in self.predict_distribution_batch(features))

    def predict(self, features: Mapping[str, torch.Tensor]) -> LengthPrediction:
        return self.predict_distribution(features).as_length_prediction()
