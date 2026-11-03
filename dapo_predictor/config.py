"""Validate and connect predictor configuration before Ray workers start."""

from omegaconf import OmegaConf, open_dict


def configure_predictor(config) -> None:
    predictor = OmegaConf.select(config, "trainer.predictor_reorder", default=None)
    if predictor is None or not predictor.get("enable", False):
        return
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("DAPO predictor requires actor.strategy=fsdp or fsdp2")
    scheduler = str(predictor.get("scheduler", "static_snake"))
    if scheduler not in {"epws", "static_snake"}:
        raise ValueError("predictor_reorder.scheduler must be 'epws' or 'static_snake'")
    if predictor.get("backend", "linear_listmle") not in {"linear_listmle", "censored_lognormal"}:
        raise ValueError("predictor_reorder.backend must be 'linear_listmle' or 'censored_lognormal'")
    with open_dict(config.actor_rollout_ref):
        config.actor_rollout_ref.predictor_reorder = OmegaConf.create(OmegaConf.to_container(predictor, resolve=True))
    if scheduler == "epws":
        with open_dict(config.actor_rollout_ref.rollout):
            if config.actor_rollout_ref.rollout.get("agent") is None:
                config.actor_rollout_ref.rollout.agent = OmegaConf.create({})
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = (
                "recipe.dapo_predictor.epws_manager.EPWSAgentLoopManager"
            )
