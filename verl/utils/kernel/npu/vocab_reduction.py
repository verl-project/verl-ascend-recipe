# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Scoped batch-invariant reduction for the native Megatron vocabulary CE."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class NpuVocabReductionMode(TorchDispatchMode):
    """Change only the FP32 vocabulary sum; retain native CE and its backward."""

    def __init__(self, vocab_width: int):
        super().__init__()
        self.vocab_width = vocab_width

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.sum.dim_IntList:
            value = args[0]
            dims = args[1] if len(args) > 1 else kwargs.get("dim")
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            dtype = kwargs.get("dtype")
            if (
                value.device.type == "npu"
                and value.dtype == torch.float32
                and value.ndim > 0
                and value.shape[-1] == self.vocab_width
                and dims is not None
                and list(dims) in ([-1], [value.ndim - 1])
                and dtype in (None, torch.float32)
            ):
                from vllm_ascend.ops.training_reduction import fixed_vocab_sum

                return fixed_vocab_sum(value, keepdim=keepdim)
        return func(*args, **kwargs)
