# SPDX-License-Identifier: Apache-2.0
"""Exercise chunk boundaries and packing, not just repeated identical calls."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.gdn_batch_invariant import chunk_gdn_training_math


def _inputs(length, seed):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    q, k = (
        torch.nn.functional.normalize(torch.randn(1, length, 2, 128, generator=generator), dim=-1)
        .bfloat16().npu()
        for _ in range(2)
    )
    v = torch.randn(1, length, 4, 128, generator=generator).bfloat16().npu()
    g = -torch.rand(1, length, 4, generator=generator).npu()
    beta = torch.rand(1, length, 4, generator=generator).bfloat16().npu()
    return q, k, v, g, beta


@pytest.mark.parametrize('length', [1, 63, 64, 65, 129])
@pytest.mark.parametrize('seed', [42, 123])
@torch.no_grad()
def test_chunk_entry_replay_and_packing_are_exact(length, seed):
    torch.npu.set_device(0)
    args = _inputs(length, seed)
    expected, expected_state = chunk_gdn_training_math(*args, scale=128**-0.5)
    rows = []
    state = None
    for end in range(1, length + 1):
        start = ((end - 1) // 64) * 64
        out, next_state = chunk_gdn_training_math(
            *(t[:, start:end].contiguous() for t in args),
            scale=128**-0.5, initial_state=state,
        )
        rows.append(out[:, -1:])
        if end % 64 == 0:
            state = next_state
    assert torch.isfinite(expected).all()
    assert torch.equal(torch.cat(rows, dim=1), expected)
    assert torch.equal(next_state, expected_state)
    packed = [t.repeat((1, 2) + (1,) * (t.ndim - 2)).contiguous() for t in args]
    boundaries = torch.tensor([0, length, 2 * length], dtype=torch.int64, device='npu')
    actual, actual_state = chunk_gdn_training_math(*packed, scale=128**-0.5, cu_seqlens=boundaries)
    assert torch.equal(actual[:, :length], expected)
    assert torch.equal(actual[:, length:], expected)
    assert torch.equal(actual_state[:1], expected_state)
    assert torch.equal(actual_state[1:], expected_state)


def test_grad_enabled_inputs_are_rejected_instead_of_silently_losing_gradients():
    args = list(_inputs(1, 42))
    args[0].requires_grad_(True)
    with pytest.raises(RuntimeError, match='inference-only'):
        chunk_gdn_training_math(*args, scale=128**-0.5)


@pytest.mark.parametrize('splits', [[1, 63, 1, 64], [17, 32, 80], [65, 64], [129]])
@torch.no_grad()
def test_chunk_replay_retains_only_partial_chunk_and_handles_prefill_splits(splits):
    from vllm_ascend.ops.gdn_batch_invariant import GDNChunkReplay

    args = _inputs(sum(splits), 123)
    expected, expected_state = chunk_gdn_training_math(*args, scale=128**-0.5)
    cache = GDNChunkReplay(scale=128**-0.5)
    offset = 0
    rows = []
    for length in splits:
        out, state = cache.append(*(t[:, offset:offset + length].contiguous() for t in args))
        rows.append(out)
        offset += length
        assert cache.tail is None or cache.tail[0].shape[1] == offset % 64
        if cache.tail is not None:
            for value in cache.tail:
                assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert torch.equal(torch.cat(rows, dim=1), expected)
    assert torch.equal(state, expected_state)


@torch.no_grad()
def test_gates_are_invariant_to_batch_composition_and_keep_parameter_precision():
    from vllm_ascend.ops.gdn_batch_invariant import gdn_gates_training_math

    torch.manual_seed(42)
    a_log = torch.randn(4, dtype=torch.float32, device='npu')
    bias = torch.randn(4, dtype=torch.bfloat16, device='npu')
    a, b = (torch.randn(65, 4, dtype=torch.bfloat16, device='npu') for _ in range(2))
    g, beta = gdn_gates_training_math(a_log, a, b, bias)
    short_g, short_beta = gdn_gates_training_math(a_log, a[:1], b[:1], bias)
    assert g.dtype == torch.float32
    assert beta.dtype == torch.bfloat16
    assert torch.equal(g[:, :1], short_g)
    assert torch.equal(beta[:, :1], short_beta)
    assert torch.isfinite(g).all()
    assert (g < 0).all()
    assert ((beta >= 0) & (beta <= 1)).all()


@pytest.mark.parametrize("length", [1024, 1088])
@pytest.mark.parametrize("seed", [42, 123])
@torch.no_grad()
def test_complete_chunk_prefix_matches_longer_history(length, seed):
    from vllm_ascend.ops.gdn_batch_invariant import GDNChunkReplay

    args = _inputs(length + 16, seed)
    cache = GDNChunkReplay(scale=128**-0.5)
    prefix, _ = cache.append(*(t[:, :length].contiguous() for t in args))
    expected, expected_state = chunk_gdn_training_math(*args, scale=128**-0.5)
    suffix, state = cache.append(*(t[:, length:].contiguous() for t in args))
    assert torch.isfinite(expected).all()
    assert torch.equal(prefix, expected[:, :length])
    assert torch.equal(suffix, expected[:, length:])
    assert torch.equal(state, expected_state)
