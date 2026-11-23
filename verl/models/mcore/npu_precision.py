# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Mixed-precision wrapping that preserves the GDN checkpoint decay parameter."""

import torch
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.module import Float16Module


class GDNFloat16Module(Float16Module):
    """Keep A_log in FP32 before DDP buffers and checkpoint loading are built.

    Qwen3.5 checkpoints store this parameter in FP32. The default wrapper casts
    every parameter to BF16 and changes the decay exponential's evaluation dtype.
    This exponential affects every recurrent step. Parameter identity and TP
    attributes are preserved so native optimizers and weight export still own it.
    """

    def __init__(self, config, module):
        super().__init__(config, module)
        for child in self.module.modules():
            if isinstance(child, GatedDeltaNet):
                child.A_log.data = child.A_log.data.to(torch.float32)
