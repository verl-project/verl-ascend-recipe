from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dapo_predictor.runtime import _shutdown_vllm_server, close_rollout_servers


def test_shutdown_calls_owned_engine_gracefully():
    engine = Mock()
    assert _shutdown_vllm_server(SimpleNamespace(engine=engine))
    engine.shutdown.assert_called_once_with(timeout=30)
    assert not _shutdown_vllm_server(SimpleNamespace())


def test_missing_manager_during_initialization_has_nothing_to_close():
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="vllm")))
    )
    close_rollout_servers(trainer)


def test_other_backend_retains_its_own_lifecycle():
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="sglang")))
    )
    close_rollout_servers(trainer)


@pytest.mark.parametrize("error", [None, TimeoutError("shutdown did not finish")])
def test_owned_server_rpcs_complete_or_surface_failure(monkeypatch, error):
    import sys

    remote = Mock(return_value="shutdown-ref")
    get = Mock(side_effect=error)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=get))
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="vllm"))),
        async_rollout_manager=SimpleNamespace(
            server_handles=[SimpleNamespace(__ray_call__=SimpleNamespace(remote=remote))]
        ),
    )
    if error:
        with pytest.raises(TimeoutError):
            close_rollout_servers(trainer)
    else:
        close_rollout_servers(trainer)
    remote.assert_called_once_with(_shutdown_vllm_server)
    get.assert_called_once_with(["shutdown-ref"], timeout=90)
