from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dapo_predictor.length_scheduler.checkpoint import (
    DISTRIBUTION_BACKEND,
    LINEAR_BACKEND,
    build_lifecycle_state,
    file_sha256,
    validate_lifecycle_state,
)


def test_d1_lifecycle_round_trip_and_legacy_resume() -> None:
    state = build_lifecycle_state(backend=LINEAR_BACKEND, observed_rollouts=2560, ready=True)
    assert state == {
        "schema_version": 2,
        "backend": LINEAR_BACKEND,
        "observed_rollouts": 2560,
        "ready": True,
    }
    assert validate_lifecycle_state(state, backend=LINEAR_BACKEND) == (2560, True)
    assert validate_lifecycle_state(
        {"schema_version": 1, "observed_rollouts": 16, "ready": False},
        backend=LINEAR_BACKEND,
    ) == (16, False)


def test_d2_lifecycle_binds_the_frozen_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "d2.pt"
    checkpoint.write_bytes(b"frozen-d2")
    state = build_lifecycle_state(
        backend=DISTRIBUTION_BACKEND,
        observed_rollouts=2560,
        ready=True,
        d2_checkpoint=checkpoint,
    )
    assert state["d2_checkpoint_sha256"] == hashlib.sha256(b"frozen-d2").hexdigest()
    assert validate_lifecycle_state(
        state,
        backend=DISTRIBUTION_BACKEND,
        d2_checkpoint=checkpoint,
    ) == (2560, True)

    checkpoint.write_bytes(b"replaced-d2")
    with pytest.raises(ValueError, match="does not match"):
        validate_lifecycle_state(state, backend=DISTRIBUTION_BACKEND, d2_checkpoint=checkpoint)


def test_lifecycle_rejects_backend_mismatch_and_unsafe_legacy_d2(tmp_path: Path) -> None:
    checkpoint = tmp_path / "d2.pt"
    checkpoint.write_bytes(b"d2")
    d1_state = build_lifecycle_state(backend=LINEAR_BACKEND, observed_rollouts=0, ready=False)
    with pytest.raises(ValueError, match="backend mismatch"):
        validate_lifecycle_state(d1_state, backend=DISTRIBUTION_BACKEND, d2_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="legacy lifecycle"):
        validate_lifecycle_state(
            {"schema_version": 1, "backend": DISTRIBUTION_BACKEND},
            backend=DISTRIBUTION_BACKEND,
            d2_checkpoint=checkpoint,
        )


def test_lifecycle_rejects_invalid_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_lifecycle_state(backend=LINEAR_BACKEND, observed_rollouts=-1, ready=False)
    with pytest.raises(ValueError, match="non-negative"):
        build_lifecycle_state(backend=LINEAR_BACKEND, observed_rollouts="1", ready=False)
    with pytest.raises(ValueError, match="boolean"):
        build_lifecycle_state(backend=LINEAR_BACKEND, observed_rollouts=1, ready="yes")
    with pytest.raises(ValueError, match="boolean"):
        validate_lifecycle_state(
            {"schema_version": 2, "backend": LINEAR_BACKEND, "observed_rollouts": 1, "ready": "yes"},
            backend=LINEAR_BACKEND,
        )
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.pt")
