"""Run upstream verl with concrete deployment env values at the Ray boundary.

Some verl versions serialize OmegaConf runtime_env with resolve=False. Passing
oc.env expressions there therefore gives actors literal interpolation strings.
Keep deployment credentials out of Hydra overrides and argv altogether; inject
plain strings into the runtime_env passed to the real ray.init instead.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

FORWARDED_ENV = (
    "E2B_API_KEY",
    "E2B_API_URL",
    "E2B_SANDBOX_URL",
    "E2B_DEBUG",
    "AGENTENV_CALLBACK_ENABLED",
    "AGENTENV_CONTROL_DIR",
    "AGENTENV_RELAY_TARGET",
    "AGENTENV_RELAY_SSH_PORT",
    "AGENTENV_CALLBACK_PORT",
    "AGENTENV_HOST",
    "CLAUDE_CODE_SHA256",
    "AGENTENV_JAVA_M2_ARCHIVE",
    "SSH_KNOWN_HOSTS",
    "AGENTENV_ASSET_URL",
    "AGENTENV_RELAY_CALLBACK_PORT",
    "SSHPASS",
    "TRANSFORMERS_VERBOSITY",
)
REQUIRED_ENV = (
    "E2B_API_KEY",
    "E2B_API_URL",
    "E2B_SANDBOX_URL",
    "AGENTENV_CALLBACK_ENABLED",
    "AGENTENV_CONTROL_DIR",
    "AGENTENV_RELAY_TARGET",
    "AGENTENV_RELAY_SSH_PORT",
    "AGENTENV_CALLBACK_PORT",
    "AGENTENV_HOST",
)


def runtime_environment(existing=None):
    """Merge deployment values without mutating caller input or exposing secrets."""
    values = {name: os.environ[name] for name in FORWARDED_ENV if name in os.environ}
    values["TRANSFORMERS_VERBOSITY"] = "error"
    for name in REQUIRED_ENV:
        if not values.get(name):
            raise ValueError(f"Missing {name}; launch through run_reproduce.sh")
    for name, value in values.items():
        if "${oc.env:" in value:
            raise ValueError(f"{name} contains an unresolved environment interpolation")
    for name in ("E2B_API_URL", "E2B_SANDBOX_URL", "AGENTENV_ASSET_URL"):
        if name in values:
            url = urlsplit(values[name])
            if url.scheme not in ("http", "https") or not url.netloc:
                raise ValueError(f"{name} must be an absolute HTTP(S) URL")
    runtime = dict(existing or {})
    runtime["env_vars"] = {**runtime.get("env_vars", {}), **values}
    return runtime


def main():
    import functools
    import runpy

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    runtime_environment()  # Fail before loading the trainer if deployment values are invalid.
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    import ray

    original_init = ray.init

    @functools.wraps(original_init)
    def init_with_deployment(*args, **kwargs):
        kwargs["runtime_env"] = runtime_environment(kwargs.get("runtime_env"))
        return original_init(*args, **kwargs)

    ray.init = init_with_deployment
    try:
        runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
    finally:
        ray.init = original_init


if __name__ == "__main__":
    main()
