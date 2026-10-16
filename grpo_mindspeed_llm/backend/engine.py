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

from verl.trainer.config import CheckpointConfig
from verl.utils.megatron.router_replay_patch import RouterReplay
from verl.utils.model import print_model_size
from verl.workers.config import HFModelConfig, McoreOptimizerConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from .config import MindSpeedEngineConfig
from .utils import apply_patch, gpt_model_provider

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="language_model", backend="mindspeed_llm", device="npu")
class MindSpeedLLMEngineWithLMHead(MegatronEngineWithLMHead):
    """MindSpeed-LLM Megatron engine for Ascend NPU."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: MindSpeedEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        apply_patch(self.model_config, self.engine_config, self.optimizer_config)
        super()._init_device_mesh()

    def _build_megatron_module(self):
        is_value_model = (
            "ForTokenClassification" in self.model_config.architectures[0]
            or "ForSequenceClassification" in self.model_config.architectures[0]
        )
        self.is_value_model = is_value_model

        import torch.distributed
        from megatron.core.enums import ModelType
        from megatron.training.training import get_model

        if self.engine_config.forward_only:
            module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
        else:
            module = get_model(gpt_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
        if self.vanilla_bridge:
            self.bridge.load_weights(module, self.model_config.local_path)
        else:
            raise ValueError(f"vanilla_bridge should be true now, but got {self.vanilla_bridge}")

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module
