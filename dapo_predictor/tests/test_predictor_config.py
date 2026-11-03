"""Check opt-in wiring without creating Ray actors."""

# ruff: noqa: E402 -- optional dependencies are checked before importing the recipe.
import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from dapo_predictor.config import configure_predictor


def _config(**overrides):
    predictor = {"enable": True, "scheduler": "epws", "backend": "linear_listmle", **overrides}
    return OmegaConf.create(
        {
            "trainer": {"predictor_reorder": predictor},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp"},
                "rollout": {"agent": {"agent_loop_manager_class": "user.Manager", "num_workers": 2}},
            },
        }
    )


def test_disabled_predictor_preserves_user_worker_and_manager_configuration():
    config = _config(enable=False)
    before = OmegaConf.to_container(config)
    configure_predictor(config)
    assert OmegaConf.to_container(config) == before


@pytest.mark.parametrize("backend", ["linear_listmle", "censored_lognormal"])
def test_ews_wiring_preserves_other_agent_options(backend):
    config = _config(backend=backend)
    configure_predictor(config)
    assert config.actor_rollout_ref.predictor_reorder.backend == backend
    assert config.actor_rollout_ref.rollout.agent.num_workers == 2
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class.endswith(".EPWSAgentLoopManager")


def test_static_snake_keeps_existing_manager():
    config = _config(scheduler="static_snake")
    configure_predictor(config)
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class == "user.Manager"


@pytest.mark.parametrize("key,value", [("scheduler", "unknown"), ("backend", "unknown")])
def test_invalid_options_fail_before_worker_initialization(key, value):
    config = _config(**{key: value})
    before = OmegaConf.to_container(config)
    with pytest.raises(ValueError, match=key):
        configure_predictor(config)
    assert OmegaConf.to_container(config) == before


def test_unsupported_actor_strategy_is_rejected():
    config = _config()
    config.actor_rollout_ref.actor.strategy = "megatron"
    with pytest.raises(ValueError, match="fsdp"):
        configure_predictor(config)
