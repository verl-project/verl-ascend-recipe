# Copyright (c) 2026, HUAWEI TECHNOLOGIES CO., LTD. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import logging
import os
from pathlib import Path
from typing import Optional

from backend.config import (  # noqa: F401  (re-export for yaml _target_)
    MindSpeedActorConfig,
    MindSpeedCriticConfig,
    MindSpeedEngineConfig,
)
from backend.engine import MindSpeedLLMEngineWithLMHead  # noqa: F401  (trigger @register)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_CONFIG_RELPATHS = [
    "model_engine/mindspeed.yaml",
    "engine/mindspeed.yaml",
    "actor/mindspeed_actor.yaml",
    "critic/mindspeed_critic.yaml",
    "ref/mindspeed_ref.yaml",
]


def _get_recipe_configs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def _get_verl_config_dir() -> Optional[Path]:
    try:
        import verl.trainer.config as _cfg

        return Path(list(_cfg.__path__)[0])
    except Exception as e:
        logger.debug("Cannot locate verl trainer config dir: %s", e)
        return None


def _link_configs_to_verl():
    """Symlink recipe hydra configs into verl's trainer config directory."""
    recipe_cfg_dir = _get_recipe_configs_dir()
    verl_cfg_dir = _get_verl_config_dir()
    if verl_cfg_dir is None:
        logger.debug("verl trainer config dir not found, skipping config linking")
        return

    for rel in _CONFIG_RELPATHS:
        src = recipe_cfg_dir / rel
        if not src.exists():
            logger.warning("Recipe config not found: %s", src)
            continue

        dst = verl_cfg_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.is_symlink():
            try:
                if Path(os.readlink(dst)).resolve() == src.resolve():
                    continue
            except OSError:
                pass
            dst.unlink()
        elif dst.exists():
            dst.unlink()

        try:
            os.symlink(str(src), str(dst))
            logger.info("Linked config %s -> %s", dst, src)
        except OSError as e:
            logger.debug("Symlink failed for %s: %s. Falling back to copy.", rel, e)
            try:
                import shutil

                shutil.copy2(str(src), str(dst))
            except Exception as e2:
                logger.error("Failed to copy config %s: %s", rel, e2)


_link_configs_to_verl()

logger.info("backend loaded: engine registered + configs linked")
