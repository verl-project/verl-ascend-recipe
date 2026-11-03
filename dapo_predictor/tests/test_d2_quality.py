from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from dapo_predictor.length_scheduler.checkpoint import build_lifecycle_state, validate_lifecycle_state
from dapo_predictor.length_scheduler.d2_quality import (
    D2QualityConfig,
    D2QualityLifecycle,
    QualityWindow,
    assess,
    make_validation_report,
)


def rows(epoch=10, *, reverse=False, bias=1.0, prompts=40, siblings=2):
    for prompt in range(prompts):
        actual = (prompt + 1) * 100
        prediction = ((prompts - prompt) * 100 if reverse else actual) * bias
        for _ in range(siblings):
            yield dict(
                prompt_id=str(prompt),
                epoch=epoch,
                observed=actual,
                expected=prediction,
                median=prediction,
                p90=prediction * 1.5,
                censored=False,
            )


def report(tmp_path, *, epoch=10, **kwargs):
    checkpoint = tmp_path / f"candidate{epoch}.pt"
    checkpoint.write_bytes(f"candidate {epoch}".encode())
    result = make_validation_report(
        list(rows(epoch, **kwargs)), checkpoint=checkpoint, trained_through_epoch=epoch - 1, validation_epoch=epoch
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(result))
    return path, result


def window(**kwargs):
    result = QualityWindow()
    for row in rows(**kwargs):
        result.add(**row)
    return result


def test_rank_is_primary_not_spearman_or_point_accuracy():
    config = D2QualityConfig()
    # Scale error is tolerated while ranking is perfect and safety guards pass.
    decision = assess(window(bias=1.5).metrics(config), config)
    assert decision["passed"]
    assert decision["metrics"]["rank_accuracy"] == 1
    assert decision["metrics"]["normalized_mae"] == pytest.approx(0.5)
    reversed_decision = assess(window(reverse=True).metrics(config), config)
    assert not reversed_decision["passed"]
    assert "ranking_below_threshold" in reversed_decision["reasons"]


def test_siblings_and_repeated_epochs_do_not_inflate_identity_count():
    config = D2QualityConfig()
    w = window(prompts=4, siblings=100)
    for row in rows(epoch=11, prompts=4, siblings=100):
        w.add(**row)
    result = w.metrics(config)
    assert result["distinct_prompts"] == 4
    assert result["rollouts"] == 800
    assert result["comparable_pairs"] <= 6
    assert not assess(result, config)["passed"]


def test_constant_scores_do_not_pass_the_ranking_gate():
    w = QualityWindow()
    for row in rows():
        row.update(expected=2000, median=2000, p90=3000)
        w.add(**row)
    assert w.metrics(D2QualityConfig())["rank_accuracy"] == 0.5
    assert not assess(w.metrics(D2QualityConfig()), D2QualityConfig())["passed"]


def test_pair_sampling_is_bounded_and_does_not_touch_global_rng():
    import random

    config = D2QualityConfig(max_rank_pairs=100)
    initial_rng = random.getstate()
    first = window(prompts=500, siblings=1).metrics(config)
    second = window(prompts=500, siblings=1).metrics(config)
    assert first == second
    assert first["comparable_pairs"] <= 100
    assert first["rank_accuracy"] == 1
    assert random.getstate() == initial_rng


def test_tail_overprediction_is_not_a_free_coverage_pass():
    config = D2QualityConfig()
    w = QualityWindow()
    for row in rows():
        row["p90"] *= 100
        w.add(**row)
    decision = assess(w.metrics(config), config)
    assert decision["metrics"]["p90_coverage_lower"] == 1
    assert "inflated_p90" in decision["reasons"]


def test_right_censoring_is_not_an_exact_cap_label():
    w = window()
    for i in range(40):
        w.add(prompt_id=f"c{i}", epoch=10, observed=5000, expected=5000, median=5000, p90=6000, censored=True)
    metrics = w.metrics(D2QualityConfig())
    assert metrics["distinct_prompts"] == 80
    assert metrics["exact_prompts"] == 40
    assert metrics["p90_coverage_lower"] == 0.5
    assert metrics["p90_coverage_upper"] == 1
    assert not assess(metrics, D2QualityConfig())["passed"]


def test_default_boundaries_and_frozen_after_admission(tmp_path):
    path, _ = report(tmp_path)
    lifecycle = D2QualityLifecycle()
    for epoch in range(10):
        assert lifecycle.boundary(epoch, path) is None
    assert lifecycle.boundary(10, path)["passed"]
    digest = lifecycle.checkpoint_sha256
    assert lifecycle.boundary(10, path) is None
    assert lifecycle.boundary(11, path) is None
    lifecycle.window = window(epoch=11)
    # Active monitoring does not read or load the newly published candidate.
    report(tmp_path, epoch=15, reverse=True)
    assert lifecycle.boundary(15, path)["passed"]
    assert lifecycle.checkpoint_sha256 == digest
    assert lifecycle.active


def test_rejected_candidate_waits_five_epochs_then_can_pass(tmp_path):
    path, _ = report(tmp_path, reverse=True)
    lifecycle = D2QualityLifecycle()
    assert not lifecycle.boundary(10, path)["passed"]
    assert not lifecycle.active
    path, _ = report(tmp_path, epoch=15)
    assert lifecycle.boundary(14, path) is None
    assert lifecycle.boundary(15, path)["passed"]


def test_missing_candidate_never_activates(tmp_path):
    lifecycle = D2QualityLifecycle()
    assert not lifecycle.boundary(10, tmp_path / "missing")["passed"]
    assert not lifecycle.active


def test_prevalidated_historical_candidate_requires_explicit_zero_warmup(tmp_path):
    path, _ = report(tmp_path)
    normal = D2QualityLifecycle()
    assert normal.boundary(0, path) is None
    historical = D2QualityLifecycle(D2QualityConfig(warmup_epochs=0))
    assert historical.boundary(0, path)["passed"]
    historical.window = window(epoch=1)
    assert historical.boundary(5)["passed"]


def test_new_admission_discards_old_candidate_monitor_rows(tmp_path):
    path, _ = report(tmp_path, epoch=15)
    lifecycle = D2QualityLifecycle()
    lifecycle.status = "suspended"
    lifecycle.last_boundary = 10
    lifecycle.window = window(epoch=11, reverse=True)
    assert lifecycle.boundary(15, path)["passed"]
    assert lifecycle.window.groups == {}


def test_temporal_overlap_and_future_candidate_rejected(tmp_path):
    path, data = report(tmp_path)
    for train, end in [(10, 10), (9, 11), (1, 5)]:
        modified = copy.deepcopy(data)
        modified.update(trained_through_epoch=train, validation_epoch=end)
        path.write_text(json.dumps(modified))
        assert not D2QualityLifecycle().boundary(10, path)["passed"]


def test_validation_writer_rejects_fitting_epoch():
    with pytest.raises(ValueError, match="strictly after"):
        make_validation_report([], checkpoint="unused", trained_through_epoch=10, validation_epoch=10)
    with pytest.raises(ValueError, match="overlaps"):
        make_validation_report(list(rows(epoch=9)), checkpoint="unused", trained_through_epoch=9, validation_epoch=10)


def test_checkpoint_hash_and_risk_score_must_match(tmp_path):
    path, data = report(tmp_path)
    data["risk_weight"] = 1
    path.write_text(json.dumps(data))
    assert not D2QualityLifecycle().boundary(10, path)["passed"]
    path, data = report(tmp_path)
    (tmp_path / "candidate10.pt").write_bytes(b"replacement")
    assert not D2QualityLifecycle().boundary(10, path)["passed"]


@pytest.mark.parametrize(
    "mutation",
    [lambda d: d.update(groups=[]), lambda d: d.update(groups={"x": {}}), lambda d: d.update(validation_epoch=True)],
)
def test_malformed_candidate_fails_closed(tmp_path, mutation):
    path, data = report(tmp_path)
    mutation(data)
    path.write_text(json.dumps(data))
    assert not D2QualityLifecycle().boundary(10, path)["passed"]


def test_persistent_rank_failure_suspends_and_needs_new_candidate(tmp_path):
    path, _ = report(tmp_path)
    lifecycle = D2QualityLifecycle()
    lifecycle.boundary(10, path)

    def bad_rank(epoch):
        result = QualityWindow()
        for row in rows(epoch=epoch, reverse=True):
            # Isolate ranking degradation from the separate severe-tail veto.
            row["p90"] = 6000
            result.add(**row)
        return result

    lifecycle.window = bad_rank(11)
    lifecycle.boundary(15)
    assert lifecycle.active and lifecycle.bad_windows == 1
    lifecycle.window = bad_rank(16)
    lifecycle.boundary(20)
    assert lifecycle.status == "suspended"
    path, new = report(tmp_path, epoch=25)
    # Same bytes as the suspended checkpoint must not be automatically retried.
    (tmp_path / "candidate25.pt").write_bytes(b"candidate 10")
    new["checkpoint_sha256"] = lifecycle.checkpoint_sha256
    path.write_text(json.dumps(new))
    assert not lifecycle.boundary(25, path)["passed"]
    path, _ = report(tmp_path, epoch=30)
    assert lifecycle.boundary(30, path)["passed"]


def test_gross_error_and_missing_monitor_evidence_suspend(tmp_path):
    path, _ = report(tmp_path)
    for monitor in [QualityWindow(), window(epoch=11, bias=10)]:
        lifecycle = D2QualityLifecycle()
        lifecycle.boundary(10, path)
        lifecycle.window = monitor
        lifecycle.boundary(15)
        assert lifecycle.status == "suspended"


def test_resume_preserves_pending_window_and_checks_policy(tmp_path):
    path, _ = report(tmp_path)
    original = D2QualityLifecycle()
    original.boundary(10, path)
    original.window = window(epoch=11)
    restored = D2QualityLifecycle()
    restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
    assert restored.state_dict() == original.state_dict()
    assert restored.boundary(10, path) is None
    with pytest.raises(ValueError, match="configuration mismatch"):
        D2QualityLifecycle(D2QualityConfig(interval_epochs=3)).load_state_dict(original.state_dict())
    (tmp_path / "candidate10.pt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="SHA-256"):
        D2QualityLifecycle().load_state_dict(original.state_dict())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval_epochs": 0},
        {"min_prompts": True},
        {"min_rank_accuracy": 0.5},
        {"min_p90_coverage": 0.95},
        {"risk_weight": float("nan")},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        D2QualityConfig(**kwargs)


def test_waiting_d2_checkpoint_does_not_require_model_file():
    state = build_lifecycle_state(backend="censored_lognormal", observed_rollouts=32, ready=False)
    assert validate_lifecycle_state(state, backend="censored_lognormal") == (32, False)
    with pytest.raises(ValueError):
        build_lifecycle_state(backend="censored_lognormal", observed_rollouts=32, ready=True)


def test_validation_cli_atomically_publishes_hash_bound_report(tmp_path):
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"frozen")
    source = tmp_path / "validation.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows()))
    destination = tmp_path / "candidate.json"
    subprocess.run(
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
        check=True,
    )
    assert D2QualityLifecycle().boundary(10, destination)["passed"]
    assert not list(tmp_path.glob("*.tmp-*"))
