"""Check opt-in wiring without creating Ray actors."""

# ruff: noqa: E402 -- optional dependencies are checked before importing the recipe.
import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from dapo_predictor.config import configure_predictor


def _config(**overrides):
    predictor = {
        "enable": True,
        "scheduler": "ews",
        "backend": "linear_listmle",
        **overrides,
    }
    return OmegaConf.create(
        {
            "trainer": {"predictor_reorder": predictor},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp"},
                "rollout": {
                    "agent": {
                        "agent_loop_manager_class": "user.Manager",
                        "num_workers": 2,
                    }
                },
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
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class.endswith(".EWSAgentLoopManager")


def test_static_snake_keeps_existing_manager():
    config = _config(scheduler="static_snake")
    configure_predictor(config)
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class == "user.Manager"


@pytest.mark.parametrize("selection", ["auto", "disable", "enable"])
def test_custom_worker_selects_matching_dataproto_trainer_protocol(selection):
    config = _config()
    config.trainer.use_legacy_worker_impl = selection
    configure_predictor(config)
    assert config.trainer.use_legacy_worker_impl == "enable"


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


def test_default_enabled_recipe_selects_online_d2_and_ews():
    config = _config()
    del config.trainer.predictor_reorder.backend
    del config.trainer.predictor_reorder.scheduler
    configure_predictor(config)
    assert config.actor_rollout_ref.predictor_reorder.backend == "censored_lognormal"
    assert config.actor_rollout_ref.predictor_reorder.scheduler == "ews"
    assert config.actor_rollout_ref.predictor_reorder.d2_online.enable


def test_legacy_scheduler_and_capacity_are_normalized():
    config = _config(scheduler="epws", epws={"slots_per_server": 3})
    configure_predictor(config)
    assert config.trainer.predictor_reorder.scheduler == "ews"
    assert config.actor_rollout_ref.predictor_reorder.ews.slots_per_server == 3
    assert "epws" not in config.trainer.predictor_reorder


def test_conflicting_capacity_names_are_rejected():
    config = _config(ews={"slots_per_server": 3}, epws={"slots_per_server": 8})
    with pytest.raises(ValueError, match="conflicting"):
        configure_predictor(config)


def test_external_report_preserves_frozen_compatibility():
    config = _config(
        backend="censored_lognormal",
        d2_lifecycle={"candidate_report": "/shared/report.json"},
    )
    configure_predictor(config)
    assert not config.actor_rollout_ref.predictor_reorder.d2_online.enable
