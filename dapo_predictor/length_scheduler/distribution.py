"""Right-censored LogNormal inference utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

_Z90 = 1.2815515655446004
_Z95 = 1.6448536269514722


def distribution_summary(mu: torch.Tensor, sigma: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize ``log(1 + response_tokens) ~ Normal(mu, sigma)``."""
    if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
        raise ValueError("mu and sigma must be finite")
    if (sigma <= 0).any():
        raise ValueError("sigma must be positive")

    def token_scale(log_value: torch.Tensor) -> torch.Tensor:
        return (torch.exp(log_value.clamp_max(15.0)) - 1.0).clamp_min(0.0)

    return {
        "mu": mu,
        "sigma": sigma,
        "expected": token_scale(mu + 0.5 * sigma.square()),
        "median": token_scale(mu),
        "p90": token_scale(mu + _Z90 * sigma),
        "p95": token_scale(mu + _Z95 * sigma),
    }


class GatedDistributionHead(torch.nn.Module):
    """Fuse projected hidden-state taps and predict standardized mu/sigma."""

    def __init__(self, projection_dim: int, tap_count: int, sigma_floor: float):
        super().__init__()
        self.projection_dim = int(projection_dim)
        self.tap_count = int(tap_count)
        self.sigma_floor = float(sigma_floor)
        self.gate = torch.nn.Linear(self.projection_dim, 1)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.projection_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
        )
        self.out = torch.nn.Linear(128, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = features.reshape(features.shape[0], self.tap_count, self.projection_dim)
        weights = torch.softmax(self.gate(blocks).squeeze(-1), dim=1)
        fused = (blocks * weights.unsqueeze(-1)).sum(dim=1)
        output = self.out(self.net(fused))
        return output[:, 0], F.softplus(output[:, 1]) + self.sigma_floor


class PrefillDistributionPrior:
    """Frozen multi-tap prompt representation and LogNormal head."""

    def __init__(self, checkpoint: Mapping):
        self.feature_keys = tuple(checkpoint["feature_keys"])
        if not self.feature_keys or len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("feature_keys must be non-empty and unique")
        self.projection_dim = int(checkpoint["projection_dim"])
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.bank = {
            key: {name: value.detach().cpu().clone() for name, value in checkpoint["projection_bank"][key].items()}
            for key in self.feature_keys
        }
        self.head = GatedDistributionHead(
            self.projection_dim,
            len(self.feature_keys),
            float(checkpoint["standardized_sigma_floor"]),
        )
        self.head.load_state_dict(checkpoint["head_state_dict"])
        self.head.eval()
        self.center = float(checkpoint["log_length_center"])
        self.scale = float(checkpoint["log_length_scale"])
        if not math.isfinite(self.center):
            raise ValueError("log_length_center must be finite")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("log_length_scale must be finite and positive")

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> PrefillDistributionPrior:
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def project(self, hidden_by_tap: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        batch_size: int | None = None
        for key in self.feature_keys:
            if key not in hidden_by_tap:
                available = ", ".join(sorted(str(name) for name in hidden_by_tap))
                raise KeyError(f"missing hidden tap {key!r}; available taps: {available}")
            raw = hidden_by_tap[key]
            if not isinstance(raw, torch.Tensor):
                raise TypeError(f"hidden tap {key!r} must be a torch.Tensor")
            if raw.ndim == 1:
                raw = raw.unsqueeze(0)
            elif raw.ndim != 2:
                raise ValueError(f"hidden tap {key!r} must be 1D or 2D, got shape {tuple(raw.shape)}")
            if batch_size is None:
                batch_size = int(raw.shape[0])
            elif raw.shape[0] != batch_size:
                raise ValueError("all hidden taps must have the same batch dimension")
            if not torch.isfinite(raw).all():
                raise ValueError(f"hidden tap {key!r} contains NaN or Inf")
            transform = self.bank[key]
            center = transform["center"]
            components = transform["components"]
            scale = transform["scale"]
            expected_dim = int(center.numel())
            if raw.shape[-1] != expected_dim:
                raise ValueError(f"hidden tap {key!r} expects dim {expected_dim}, got {raw.shape[-1]}")
            if components.ndim != 2 or components.shape != (expected_dim, self.projection_dim):
                raise ValueError(f"invalid PCA components for hidden tap {key!r}")
            if not torch.isfinite(center).all() or not torch.isfinite(components).all():
                raise ValueError(f"projection bank for hidden tap {key!r} contains NaN or Inf")
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"projection scale for hidden tap {key!r} must be finite and positive")
            values = F.normalize(raw.detach().cpu().float(), dim=-1)
            parts.append(((values - center) @ components) / scale)
        return F.normalize(torch.cat(parts, dim=-1), dim=-1)

    def predict(self, hidden_by_tap: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.project(hidden_by_tap)
        with torch.inference_mode():
            mu_z, sigma_z = self.head(features)
        return distribution_summary(mu_z * self.scale + self.center, sigma_z * self.scale)


def censored_lognormal_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    log_lengths: torch.Tensor,
    censored: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for exact and right-censored observations."""
    sigma = sigma.clamp_min(1e-6)
    z = (log_lengths - mu) / sigma
    exact_log_prob = -torch.log(sigma) - 0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)
    survival = (0.5 * torch.erfc(z / math.sqrt(2.0))).clamp_min(1e-12)
    return torch.where(censored.bool(), -torch.log(survival), -exact_log_prob).mean()
