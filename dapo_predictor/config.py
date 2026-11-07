"""Validate and connect predictor configuration before Ray workers start."""

from omegaconf import OmegaConf, open_dict


def configure_predictor(config) -> None:
    predictor = OmegaConf.select(config, "trainer.predictor_reorder", default=None)
    if predictor is None or not predictor.get("enable", False):
        return
    with open_dict(predictor):
        if "backend" not in predictor:
            predictor.backend = "censored_lognormal"
        if "scheduler" not in predictor:
            predictor.scheduler = "epws"
        if predictor.backend == "censored_lognormal" and "d2_online" not in predictor:
            external = predictor.get("d2_checkpoint") or predictor.get("d2_lifecycle", {}).get("candidate_report")
            predictor.d2_online = {"enable": not bool(external)}
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("DAPO predictor requires actor.strategy=fsdp or fsdp2")
    scheduler = str(predictor.get("scheduler", "epws"))
    if scheduler not in {"epws", "static_snake"}:
        raise ValueError("predictor_reorder.scheduler must be 'epws' or 'static_snake'")
    if predictor.get("backend", "linear_listmle") not in {
        "linear_listmle",
        "censored_lognormal",
    }:
        raise ValueError("predictor_reorder.backend must be 'linear_listmle' or 'censored_lognormal'")
    # The custom FSDP worker uses DataProto RPCs, not the engine-worker TensorDict protocol.
    with open_dict(config.trainer):
        config.trainer.use_legacy_worker_impl = "enable"
    with open_dict(config.actor_rollout_ref):
        config.actor_rollout_ref.predictor_reorder = OmegaConf.create(OmegaConf.to_container(predictor, resolve=True))
    if scheduler == "epws":
        with open_dict(config.actor_rollout_ref.rollout):
            if config.actor_rollout_ref.rollout.get("agent") is None:
                config.actor_rollout_ref.rollout.agent = OmegaConf.create({})
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = (
                "recipe.dapo_predictor.epws_manager.EPWSAgentLoopManager"
            )
