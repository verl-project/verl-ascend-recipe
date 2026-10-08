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
import warnings
from dataclasses import dataclass, field
from typing import Any

from verl.workers.config.actor import ActorConfig
from verl.workers.config.checkpoint import McoreCheckpointConfig
from verl.workers.config.critic import CriticConfig
from verl.workers.config.engine import EngineConfig, McoreEngineConfig

__all__ = ["MindSpeedEngineConfig", "MindSpeedActorConfig", "MindSpeedCriticConfig"]


@dataclass
class MindSpeedEngineConfig(McoreEngineConfig):
    """Configuration for MindSpeed-LLM parallelism.

    Args:
        strategy (str): mindspeed_llm or mindspeed_mm. Also the backend key
            for EngineRegistry lookup.
        llm_kwargs (dict[str, Any]): mindspeed_llm engine kwargs, forwarded
            to MindSpeed-LLM's megatron_adaptor repatch and global megatron
            args.
        mm_kwargs (dict[str, Any]): mindspeed_mm engine kwargs (multimodal).
    """

    strategy: str = "mindspeed_llm"
    llm_kwargs: dict[str, Any] = field(default_factory=dict)
    mm_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        EngineConfig.__post_init__(self)
        assert self.strategy in ["mindspeed_llm", "mindspeed_mm"], f"strategy {self.strategy} not supported"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.vanilla_mbridge:
            warnings.warn(
                "The legacy mbridge backend selected by `vanilla_mbridge=True` is deprecated and will be removed "
                "in a future release. Use Megatron-Bridge by setting `vanilla_mbridge=False` or removing the option.",
                FutureWarning,
                stacklevel=2,
            )
        if self.dynamic_context_parallel and (
            not isinstance(self.max_seqlen_per_dp_cp_rank, int)
            or isinstance(self.max_seqlen_per_dp_cp_rank, bool)
            or self.max_seqlen_per_dp_cp_rank <= 0
        ):
            raise ValueError(
                "max_seqlen_per_dp_cp_rank must be a positive integer when dynamic_context_parallel is enabled"
            )
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class MindSpeedActorConfig(ActorConfig):
    """Configuration for MindSpeed-LLM actor models.

    Args:
        strategy (str): Training strategy, pinned to "mindspeed".
        load_weight (bool): Whether to load model weights from checkpoint.
        mindspeed (MindSpeedEngineConfig): MindSpeed-LLM engine settings.
        profile (dict[str, Any]): Profiling settings.
        use_rollout_log_probs (bool): Whether to use log probs from rollout engine.
        checkpoint (McoreCheckpointConfig): Checkpoint config.
    """

    strategy: str = "mindspeed"
    load_weight: bool = True
    mindspeed: MindSpeedEngineConfig = field(default_factory=MindSpeedEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False
    checkpoint: McoreCheckpointConfig = field(default_factory=McoreCheckpointConfig)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.mindspeed


@dataclass
class MindSpeedCriticConfig(CriticConfig):
    """Configuration for MindSpeed-LLM based critic model training.

    Args:
        strategy (str): Training strategy, pinned to "mindspeed".
        nccl_timeout (int): NCCL timeout in seconds for distributed ops.
        mindspeed (MindSpeedEngineConfig): MindSpeed-LLM engine settings.
        load_weight (bool): Whether to load initial weights.
        checkpoint (McoreCheckpointConfig): Checkpoint config.
    """

    strategy: str = "mindspeed"
    nccl_timeout: int = 600
    mindspeed: MindSpeedEngineConfig = field(default_factory=MindSpeedEngineConfig)
    load_weight: bool = True
    checkpoint: McoreCheckpointConfig = field(default_factory=McoreCheckpointConfig)

    def validate(self, n_gpus: int, train_batch_size: int):
        super().validate(n_gpus, train_batch_size)

    def __post_init__(self):
        super().__post_init__()
        self.engine = self.mindspeed
