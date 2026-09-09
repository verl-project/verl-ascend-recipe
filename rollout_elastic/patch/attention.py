# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch ``verl.utils.attention_utils._get_attention_functions``.

Adds a pure-PyTorch fallback for the flash-attention padding helpers so the
elastic rollout recipe can run in environments where ``flash_attn`` is not
installed (e.g. Ascend NPU stacks that rely on ``verl.utils.npu_flash_attn_utils``).
"""

from __future__ import annotations

from typing import Callable

from verl.utils import attention_utils

from ._core import patch_module_function


def _torch_fallback_helpers() -> tuple[Callable, Callable, Callable, Callable]:
    """Pure-PyTorch helpers mirroring ``flash_attn.bert_padding`` signatures."""
    import torch
    from einops import rearrange  # noqa: F401  (re-exported as-is)

    def index_first_axis(input, indices):
        return input[indices]

    def unpad_input(hidden_states, attention_mask, unused_mask=None):
        # hidden_states: (B, S, ...), attention_mask: (B, S)
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
        return (flat[indices], indices, cu_seqlens, max_seqlen_in_batch, seqlens_in_batch)

    def pad_input(hidden_states, indices, batch, seqlen):
        # Inverse of unpad_input. hidden_states: (N, ...) packed; indices: positions in (B*S,)
        output = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
        output[indices] = hidden_states
        return output.reshape(batch, seqlen, *hidden_states.shape[1:])

    return index_first_axis, pad_input, rearrange, unpad_input


@patch_module_function(attention_utils, "_get_attention_functions")
def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input  # noqa: PLW0603

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            # Pure-PyTorch fallback when flash_attn is unavailable.
            index_first_axis, pad_input, rearrange, unpad_input = _torch_fallback_helpers()

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input
