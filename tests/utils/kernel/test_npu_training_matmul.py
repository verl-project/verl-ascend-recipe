# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Exercise the shared GEMM dispatch without leaking registrations to other tests."""

import multiprocessing

import pytest


def _check_training_dispatch():
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent

    from verl.utils.kernel.npu.batch_invariant import enable_batch_invariant_matmul

    torch.npu.set_device(0)
    torch.manual_seed(42)
    a = torch.randn(65, 256, dtype=torch.bfloat16, device="npu", requires_grad=True)
    b = torch.randn(256, 128, dtype=torch.bfloat16, device="npu", requires_grad=True)
    dy = torch.randn(65, 128, dtype=torch.bfloat16, device="npu")
    dx_ref = dy.cpu().double() @ b.detach().cpu().double().t()
    dw_ref = a.detach().cpu().double().t() @ dy.cpu().double()
    enable_batch_invariant_matmul()
    actual = torch.matmul(a, b)
    actual.backward(dy)
    for gradient, reference in ((a.grad, dx_ref), (b.grad, dw_ref)):
        assert gradient is not None and torch.isfinite(gradient).all()
        difference = gradient.cpu().double() - reference
        assert torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference) < 0.01
    linear_input = a.detach().clone().requires_grad_(True)
    linear_weight = b.detach().t().contiguous().requires_grad_(True)
    bias = torch.zeros(128, dtype=torch.bfloat16, device="npu", requires_grad=True)
    linear_output = torch.nn.functional.linear(linear_input, linear_weight, bias)
    linear_output.backward(dy)
    assert torch.equal(linear_output, actual)
    assert torch.equal(linear_input.grad, a.grad)
    assert torch.equal(linear_weight.grad, b.grad.t())
    assert torch.equal(bias.grad, dy.sum(dim=0))
    with torch.no_grad():
        assert torch.equal(actual, linear_persistent(a, b.t()))
        assert torch.equal(actual, torch.cat([torch.matmul(part, b) for part in a.split([1, 63, 1])]))
        assert torch.equal(actual, torch.cat([torch.matmul(a, part) for part in b.split(32, dim=1)], dim=1))
        # MRoPE's batched outer products must retain broadcast semantics.
        freq = torch.randn(3, 1, 16, 1, device="npu")
        pos = torch.arange(144, dtype=torch.float32, device="npu").reshape(1, 1, 1, 144)
        assert torch.equal(torch.matmul(freq, pos), freq * pos)
        # Exact integer products check vector squeezing and general batch shapes.
        for ashape, bshape in [((16,), (16,)), ((16,), (2, 16, 7)), ((2, 7, 16), (16,)), ((2, 1, 7, 16), (3, 16, 5))]:
            left = (torch.arange(torch.tensor(ashape).prod()).reshape(ashape) % 7 - 3).float()
            right = (torch.arange(torch.tensor(bshape).prod()).reshape(bshape) % 5 - 2).float()
            expected = left @ right
            assert torch.equal(torch.matmul(left.npu(), right.npu()).cpu(), expected)


def test_native_autograd_and_rollout_gemm_agree():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    pytest.importorskip("vllm_ascend")
    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")
    process = multiprocessing.get_context("spawn").Process(target=_check_training_dispatch)
    process.start()
    try:
        process.join(timeout=180)
        assert process.exitcode == 0, f"NPU GEMM subprocess failed or timed out: {process.exitcode}"
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
