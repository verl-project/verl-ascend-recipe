"""Backend-aware predictor lifecycle checkpoint metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 2
LINEAR_BACKEND = "linear_listmle"
DISTRIBUTION_BACKEND = "censored_lognormal"
SUPPORTED_BACKENDS = frozenset({LINEAR_BACKEND, DISTRIBUTION_BACKEND})


def _validate_backend(backend: str) -> str:
    backend = str(backend)
    if backend not in SUPPORTED_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise ValueError(f"predictor backend must be one of: {choices}")
    return backend


def file_sha256(path: str | Path) -> str:
    """Hash a frozen predictor checkpoint without loading it into memory."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"predictor checkpoint does not exist: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lifecycle_state(
    *,
    backend: str,
    observed_rollouts: int,
    ready: bool,
    d2_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Build auditable state shared by the online D1 and frozen D2 paths."""
    backend = _validate_backend(backend)
    if isinstance(observed_rollouts, bool) or not isinstance(observed_rollouts, int) or observed_rollouts < 0:
        raise ValueError("observed_rollouts must be a non-negative integer")
    if not isinstance(ready, bool):
        raise ValueError("ready must be a boolean")
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "observed_rollouts": observed_rollouts,
        "ready": ready,
    }
    if backend == DISTRIBUTION_BACKEND:
        if d2_checkpoint is None:
            raise ValueError("d2_checkpoint is required for censored_lognormal")
        state["d2_checkpoint_sha256"] = file_sha256(d2_checkpoint)
    return state


def validate_lifecycle_state(
    state: Mapping[str, Any],
    *,
    backend: str,
    d2_checkpoint: str | Path | None = None,
) -> tuple[int, bool]:
    """Validate a saved lifecycle state against the configured backend."""
    backend = _validate_backend(backend)
    schema_version = state.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("predictor lifecycle schema_version must be an integer")
    if schema_version not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported predictor lifecycle schema_version: {schema_version}")

    saved_backend = state.get("backend")
    if schema_version == 1 and saved_backend is None:
        # Version 1 was written before D2 had a backend-specific resume contract.
        saved_backend = LINEAR_BACKEND
    if saved_backend != backend:
        raise ValueError(f"predictor backend mismatch: checkpoint={saved_backend!r}, configured={backend!r}")

    observed_rollouts = state.get("observed_rollouts", 0)
    if isinstance(observed_rollouts, bool) or not isinstance(observed_rollouts, int) or observed_rollouts < 0:
        raise ValueError("checkpoint observed_rollouts must be a non-negative integer")
    ready = state.get("ready", False)
    if not isinstance(ready, bool):
        raise ValueError("checkpoint ready must be a boolean")

    if backend == DISTRIBUTION_BACKEND:
        if schema_version < SCHEMA_VERSION:
            raise ValueError("legacy lifecycle state cannot safely resume censored_lognormal")
        expected_digest = state.get("d2_checkpoint_sha256")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("D2 lifecycle state is missing a valid checkpoint SHA-256")
        if d2_checkpoint is None:
            raise ValueError("d2_checkpoint is required for censored_lognormal")
        actual_digest = file_sha256(d2_checkpoint)
        if actual_digest != expected_digest.lower():
            raise ValueError(
                "D2 checkpoint SHA-256 does not match the checkpoint used when the trainer state was saved"
            )

    return observed_rollouts, ready
