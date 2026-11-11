"""Close owned inference engines before Ray tears down their server actors."""


def _shutdown_vllm_server(server):
    engine = getattr(server, "engine", None)
    if engine is None:
        return False
    engine.shutdown(timeout=30)
    return True


def close_rollout_servers(trainer):
    if trainer.config.actor_rollout_ref.rollout.name != "vllm":
        return
    manager = getattr(trainer, "async_rollout_manager", None)
    handles = getattr(manager, "server_handles", ())
    if not handles:
        return
    import ray

    ray.get([handle.__ray_call__.remote(_shutdown_vllm_server) for handle in handles], timeout=90)
