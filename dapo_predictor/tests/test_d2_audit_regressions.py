"""Adversarial cases found during the two-pass pre-review audit."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from dapo_predictor.length_scheduler.d2_quality import D2QualityLifecycle, QualityWindow


def valid_pending_state():
    return D2QualityLifecycle().state_dict()


@pytest.mark.parametrize(
    "field,value", [("last_boundary", -2), ("last_boundary", 12), ("bad_windows", -1), ("bad_windows", True)]
)
def test_corrupt_lifecycle_counters_are_rejected(field, value):
    state = valid_pending_state()
    state[field] = value
    with pytest.raises(ValueError):
        D2QualityLifecycle().load_state_dict(state)


def test_corrupt_pending_statistics_are_rejected_before_monitoring():
    state = valid_pending_state()
    state["window"] = {"invalid": {"epoch": 11, "prompt_id": "p", "n": 0}}
    with pytest.raises(ValueError):
        D2QualityLifecycle().load_state_dict(state)


@pytest.mark.parametrize("censored", ["false", 1, None])
def test_censoring_marker_must_be_a_boolean(censored):
    with pytest.raises(ValueError):
        QualityWindow().add(prompt_id="p", epoch=10, observed=10, expected=10, median=10, p90=20, censored=censored)


def test_aggregate_overflow_does_not_create_nonfinite_monitor_state():
    window = QualityWindow()
    row = dict(prompt_id="p", epoch=10, observed=1e308, expected=1e308, median=1e308, p90=1e308)
    window.add(**row)
    before = copy.deepcopy(window.groups)
    with pytest.raises(ValueError):
        window.add(**row)
    assert window.groups == before


def test_checkpoint_snapshot_does_not_alias_live_window():
    lifecycle = D2QualityLifecycle()
    snapshot = lifecycle.state_dict()
    lifecycle.window.add(prompt_id="p", epoch=11, observed=10, expected=10, median=10, p90=20)
    assert snapshot["window"] == {}


def test_quality_metrics_match_naive_pairwise_reference():
    from itertools import combinations

    from dapo_predictor.length_scheduler.d2_quality import D2QualityConfig

    observed = [100, 200, 205, 500, 1000, 1400, 2000, 3000]
    scores = [100, 300, 200, 400, 3000, 1000, 2000, 1400]
    pairs = [
        (i, j)
        for i, j in combinations(range(len(observed)), 2)
        if abs(observed[i] - observed[j]) > 0.1 * max(observed[i], observed[j], 1)
    ]
    expected = sum((observed[i] - observed[j]) * (scores[i] - scores[j]) > 0 for i, j in pairs) / len(pairs)
    window = QualityWindow()
    for i, (actual, score) in enumerate(zip(observed, scores, strict=True)):
        window.add(prompt_id=str(i), epoch=10, observed=actual, expected=score, median=score, p90=score)
    metrics = window.metrics(D2QualityConfig())
    assert metrics["comparable_pairs"] == len(pairs)
    assert metrics["rank_accuracy"] == expected


@pytest.mark.parametrize("target", ["checkpoint", "rows"])
def test_validation_publication_cannot_overwrite_sources(tmp_path, target):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"preserve-model")
    source = tmp_path / "rows.jsonl"
    row = dict(prompt_id="p", epoch=10, observed=10, expected=10, median=10, p90=20, censored=False)
    source.write_text(json.dumps(row) + "\n")
    original_model = checkpoint.read_bytes()
    original_rows = source.read_bytes()
    destination = checkpoint if target == "checkpoint" else source
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dapo_predictor.validate_d2",
            "--checkpoint",
            str(checkpoint),
            "--rows",
            str(source),
            "--trained-through-epoch",
            "9",
            "--validation-epoch",
            "10",
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert checkpoint.read_bytes() == original_model
    assert source.read_bytes() == original_rows
