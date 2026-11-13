# Copyright 2026
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

"""Adapt Ascend GDN recurrent state indexing to variable MTP depths."""

from functools import update_wrapper
from types import FunctionType


class _AttributeOverride:
    def __init__(self, original, **overrides):
        self._original = original
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._original, name)


def scoped_recurrent_adapter(method, adapter):
    """Replace the recurrent call only inside this GDN method."""
    native_torch = method.__globals__["torch"]
    native_ops = native_torch.ops
    native_ascend = native_ops._C_ascend
    namespace = dict(method.__globals__)
    namespace["torch"] = _AttributeOverride(
        native_torch,
        ops=_AttributeOverride(
            native_ops,
            _C_ascend=_AttributeOverride(native_ascend, npu_recurrent_gated_delta_rule=adapter),
        ),
    )
    replacement = FunctionType(method.__code__, namespace, method.__name__, method.__defaults__, method.__closure__)
    replacement.__kwdefaults__ = method.__kwdefaults__
    return update_wrapper(replacement, method)


def make_recurrent_adapter(native):
    def call(**kwargs):
        accepted = kwargs.get("num_accepted_tokens")
        if accepted is None or accepted.numel() == 0:
            return native(**kwargs)
        indices = kwargs["ssm_state_indices"]
        count = accepted.numel()
        if indices.numel() % count:
            raise ValueError("GDN state table is not rectangular")
        table = indices.reshape(count, -1)
        return recurrent_with_packed_states(state_table=table, native=native, **kwargs)

    return call


def recurrent_with_packed_states(*, state_table, native=None, **kwargs):
    import torch

    if native is None:
        native = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    lengths = kwargs["actual_seq_lengths"][1:].to(torch.int64)
    accepted = kwargs["num_accepted_tokens"].to(torch.int64)
    count = accepted.numel()
    capacity = state_table.shape[1]
    total = kwargs["query"].shape[0]
    if state_table.shape[0] != count or lengths.numel() != count:
        raise ValueError("GDN request/state table shape mismatch")
    if total == count * capacity:
        kwargs["ssm_state_indices"] = state_table.flatten()
        return native(**kwargs)
    rows = torch.arange(count, device=state_table.device, dtype=torch.int64)
    token_rows = torch.repeat_interleave(rows, lengths, output_size=total)
    starts = lengths.cumsum(0) - lengths
    offsets = torch.arange(total, device=state_table.device, dtype=torch.int64)
    offsets = offsets - starts.index_select(0, token_rows)
    packed = state_table[token_rows, offsets].contiguous()
    active = lengths > 0
    active_rows = rows[active]
    source = state_table[active_rows, accepted[active] - 1].long()
    destination = state_table[active_rows, 0].long()
    state = kwargs["state"]
    initial = state.index_select(0, source)
    state.index_copy_(0, destination, initial)
    kwargs["ssm_state_indices"] = packed
    kwargs["num_accepted_tokens"] = torch.ones_like(kwargs["num_accepted_tokens"])
    return native(**kwargs)


_MARKER = "_verl_adaptive_gdn_states_v023"


def bind_recurrent_adapter(source_class, aliases, adapter):
    current = source_class._forward_core
    if getattr(current, _MARKER, False):
        original, replacement = current.__wrapped__, current
    else:
        original = current
        replacement = scoped_recurrent_adapter(original, adapter)
        setattr(replacement, _MARKER, True)
    changed = False
    for cls in (source_class, *aliases):
        if cls._forward_core is original:
            cls._forward_core = replacement
            changed = True
    return changed


def patch_gdn():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

    from ..backend import ASCEND_GDN_SHA256
    from .mtp_copy_v0_23 import _guard

    original = AscendGatedDeltaNetAttention._forward_core
    if getattr(original, _MARKER, False):
        original = original.__wrapped__
    _guard(original, ASCEND_GDN_SHA256)
    native = original.__globals__["torch"].ops._C_ascend.npu_recurrent_gated_delta_rule
    # Ascend copies this method onto Qwen's class during worker initialization.
    return bind_recurrent_adapter(
        AscendGatedDeltaNetAttention, (QwenGatedDeltaNetAttention,), make_recurrent_adapter(native)
    )
