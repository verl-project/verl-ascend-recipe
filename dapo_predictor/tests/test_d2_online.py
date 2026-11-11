"""Online D2 fitting, publication, replay and lifecycle regression tests."""

from __future__ import annotations

# ruff: noqa: E402 -- torch is optional in the lightweight recipe CI job.
import copy
import time

import pytest

torch = pytest.importorskip("torch")

from dapo_predictor.length_scheduler.d2_online import OnlineD2, fit_candidate, spearman
from dapo_predictor.length_scheduler.distribution import PrefillDistributionPrior
from dapo_predictor.length_scheduler.predictor import CensoredLogNormalPredictor


def payload():
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(12, 2, 5, generator=generator)
    lengths = torch.exp(hidden[:, 0, 0] * 0.4 + 4.0).unsqueeze(-1).repeat(1, 2)
    return dict(
        hidden=hidden,
        lengths=lengths,
        censored=lengths >= 100,
        feature_keys=("layer_1", "layer_2"),
        config={"fit_steps": 5, "projection_dim": 3},
        previous=None,
        version=1,
        step=1,
    )


def test_real_fit_distribution_and_incremental_coordinates():
    value = payload()
    candidate = fit_candidate(value)
    predictions = CensoredLogNormalPredictor(PrefillDistributionPrior(candidate)).predict_distribution_batch(
        {key: value["hidden"][:, index] for index, key in enumerate(value["feature_keys"])}
    )
    assert len(predictions) == 12
    assert all(row.median <= row.p90 <= row.p95 and row.sigma > 0 for row in predictions)
    value.update(previous=candidate, version=2, step=2)
    second = fit_candidate(value)
    assert second["online_version"] == 2
    assert second["log_length_center"] == candidate["log_length_center"]
    assert torch.equal(
        second["projection_bank"]["layer_1"]["components"],
        candidate["projection_bank"]["layer_1"]["components"],
    )
    assert any(
        not torch.equal(second["head_state_dict"][key], candidate["head_state_dict"][key])
        for key in second["head_state_dict"]
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_lengths_rejected(bad):
    value = payload()
    value["lengths"][0, 0] = bad
    with pytest.raises(ValueError):
        fit_candidate(value)


def test_censored_extreme_tail_stays_finite():
    value = payload()
    value["lengths"].fill_(65536)
    value["censored"].fill_(True)
    assert torch.isfinite(torch.tensor(fit_candidate(value)["fit_loss"]))


def test_cpu_fitter_does_not_probe_accelerator_optimizer_paths(monkeypatch):
    import importlib

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fitting entered accelerator optimizer detection")

    monkeypatch.setattr(importlib.import_module("torch.optim.adam"), "_default_to_fused_or_foreach", forbidden)
    assert fit_candidate(payload())["online_version"] == 1


def test_spearman_ties_and_undefined_predictions():
    x = torch.tensor([2.0, 1.0, 1.0, 4.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(x, torch.ones(4)) is None


@pytest.mark.parametrize("values", [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0, 1.0, 4.0], [7.0, 7.0, 2.0, 2.0, 1.0]])
def test_spearman_stays_within_correlation_bounds(values):
    x = torch.tensor(values)
    for sign in (-1.0, 1.0):
        correlation = spearman(x, sign * x)
        assert -1.0 <= correlation <= 1.0
        assert correlation == pytest.approx(sign)


@pytest.mark.parametrize("values", [[], [1.0], [float("nan"), 1.0], [float("inf"), 1.0]])
def test_spearman_undefined_inputs(values):
    x = torch.tensor(values)
    assert spearman(x, x) is None


def test_replay_bounded_and_observation_steps_unique(tmp_path):
    value = payload()
    online = OnlineD2(tmp_path, {"max_prompt_observations": 16, "min_fit_prompts": 2})
    for step in (1, 2):
        online.observe(
            value["hidden"],
            value["lengths"],
            value["censored"],
            value["feature_keys"],
            step,
        )
    assert len(online.hidden) == 16
    assert online.observed_rollouts == 48
    with pytest.raises(ValueError, match="increase"):
        online.observe(
            value["hidden"],
            value["lengths"],
            value["censored"],
            value["feature_keys"],
            2,
        )


def test_subprocess_nonblocking_publish_and_resume(tmp_path):
    value = payload()
    config = {"fit_steps": 3, "projection_dim": 3, "min_fit_prompts": 2}
    online = OnlineD2(tmp_path / "first", config)
    online.observe(value["hidden"], value["lengths"], value["censored"], value["feature_keys"], 1)
    try:
        assert online.submit()
        assert not online.submit()
        deadline = time.monotonic() + 90
        while online.process is not None and time.monotonic() < deadline:
            online.poll()
            time.sleep(0.05)
        assert online.process is None
        assert online.version == 1, online.failure
        snapshot = copy.deepcopy(online.state_dict())
        restored = OnlineD2(tmp_path / "restored", config)
        restored.load_state_dict(snapshot)
        assert restored.version == 1
        assert restored.observed_rollouts == 24
        assert restored.path.is_file()
        assert restored.submit()
        restored.close()
        broken = copy.deepcopy(snapshot)
        broken["checkpoint"]["trained_through_step"] = 999
        with pytest.raises(ValueError, match="future"):
            OnlineD2(tmp_path / "bad", config).load_state_dict(broken)
    finally:
        online.close()


def test_checkpoint_snapshot_does_not_alias_replay_and_rejects_nan(tmp_path):
    value = payload()
    online = OnlineD2(tmp_path / "first", {"min_fit_prompts": 2})
    online.observe(value["hidden"], value["lengths"], value["censored"], value["feature_keys"], 1)
    snapshot = online.state_dict()
    snapshot["hidden"].zero_()
    assert online.hidden.abs().sum() > 0
    state = online.state_dict()
    state["hidden"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="replay"):
        OnlineD2(tmp_path / "resume", {"min_fit_prompts": 2}).load_state_dict(state)
