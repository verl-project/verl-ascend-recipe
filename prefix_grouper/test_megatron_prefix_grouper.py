# SPDX-License-Identifier: Apache-2.0
"""Run CPU tests with Python, or use --device=npu after installing the verl adapter on Ascend."""

import argparse
import copy
import os
import sys
import unittest

import torch
import torch.nn.functional as F
from megatron_prefix_grouper import pack_shared_prefix
from torch import nn


def apply_rope(tensor, positions):
    half = tensor.shape[-1] // 2
    frequencies = 10000.0 ** (-torch.arange(half, device=tensor.device, dtype=torch.float32) / half)
    phase = (positions[..., None] * frequencies).unsqueeze(1)
    cs = phase.cos().to(tensor.dtype)
    sn = phase.sin().to(tensor.dtype)
    a, b = tensor.chunk(2, dim=-1)
    return torch.cat([a * cs - b * sn, b * cs + a * sn], dim=-1)


class ToyDecoder(nn.Module):
    """Two causal GQA layers with rotary embeddings and an untied output head."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(48, 32)
        self.qkv = nn.ModuleList(nn.Linear(32, 64, bias=False) for _ in range(2))
        self.out = nn.ModuleList(nn.Linear(32, 32, bias=False) for _ in range(2))
        self.head = nn.Linear(32, 48, bias=False)

    def forward(self, embeddings, positions, blocked):
        hidden = embeddings
        for qkv, out in zip(self.qkv, self.out, strict=True):
            q, k, v = qkv(hidden).split([32, 16, 16], dim=-1)
            q = apply_rope(q.unflatten(-1, (4, 8)).transpose(1, 2), positions)
            k = apply_rope(k.unflatten(-1, (2, 8)).transpose(1, 2), positions).repeat_interleave(2, dim=1)
            v = v.unflatten(-1, (2, 8)).transpose(1, 2).repeat_interleave(2, dim=1)
            scores = (q @ k.transpose(-2, -1)) * (8**-0.5)
            weights = scores.masked_fill(blocked, float("-inf")).softmax(dim=-1)
            mixed = (weights @ v).transpose(1, 2).flatten(2)
            hidden = hidden + out(mixed)
        return self.head(hidden)


class PrefixGrouperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("PREFIX_GROUPER_TEST_DEVICE", "cpu")
        if cls.device.startswith("npu"):
            import torch_npu  # noqa: F401

            if not torch.npu.is_available():
                raise RuntimeError("PREFIX_GROUPER_TEST_DEVICE requests Ascend, but torch.npu.is_available() is false")

    def setUp(self):
        self.ids = torch.tensor(
            [[0, 3, 5, 7, 11, 13, 15], [0, 0, 2, 4, 8, 0, 0], [0, 3, 5, 7, 17, 19, 0]],
            device=self.device,
        )
        self.mask = self.ids != 0
        # Each branch repeats absolute positions even though its packed offsets differ.
        self.pos = torch.tensor(
            [[0, 3, 4, 5, 6, 7, 8], [0, 0, 11, 12, 13, 0, 0], [0, 3, 4, 5, 6, 7, 0]], device=self.device
        )
        self.uids = ["a", "b", "a"]
        self.width = 3
        self.packed = self.pack()

    def pack(self):
        return pack_shared_prefix(self.ids, self.mask, self.pos, self.width, self.uids)

    def test_topology_and_reconstruction(self):
        p = self.packed
        self.assertEqual(p.num_groups, 2)
        self.assertEqual(p.original_token_count, 14)
        self.assertEqual(p.input_ids.tolist(), [[3, 5, 7, 11, 13, 15, 17, 19, 2, 4, 8]])
        self.assertEqual(p.position_ids.tolist(), [[3, 4, 5, 6, 7, 8, 6, 7, 11, 12, 13]])
        torch.testing.assert_close(p.restore(p.input_ids), self.ids)
        torch.testing.assert_close(p.restore(p.position_ids), self.pos)
        self.assertEqual(p.response_predictors(p.input_ids).tolist(), [[7, 11, 13], [4, 0, 0], [7, 17, 0]])
        expected_paths = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7], [8, 9, 10]]
        expected_blocked = torch.ones_like(p.attention_mask)
        for path in expected_paths:
            for i, query in enumerate(path):
                expected_blocked[0, 0, query, path[: i + 1]] = False
        torch.testing.assert_close(p.attention_mask, expected_blocked)

    def check_equivalence(self, dtype, atol, rtol):
        torch.manual_seed(1024)
        reference = ToyDecoder().to(device=self.device, dtype=dtype)
        grouped = copy.deepcopy(reference)
        p = self.packed
        flat_logits = grouped(grouped.embedding(p.input_ids), p.position_ids, p.attention_mask)
        grouped_logits = p.restore(flat_logits)
        response_logits = p.response_predictors(flat_logits)
        targets = self.ids[:, -self.width :]
        # Distinct signed token weights catch incorrect branch aggregation or loss scaling.
        weights = torch.tensor([[1.0, -2.0, 3.0], [0.75, 0.0, 0.0], [-0.5, 1.25, 0.0]], device=self.device)
        packed_losses = F.log_softmax(response_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        packed_loss = -(packed_losses * weights).sum()
        reference_loss = 0
        for row in range(len(self.uids)):
            ids = self.ids[row, self.mask[row]][None, :]
            pos = self.pos[row, self.mask[row]][None, :]
            length = ids.shape[1]
            causal = torch.ones((length, length), dtype=torch.bool, device=self.device).triu(1)
            logits = reference(reference.embedding(ids), pos, causal)
            torch.testing.assert_close(grouped_logits[row, self.mask[row]], logits[0], atol=atol, rtol=rtol)
            prompt_len = int(self.mask[row, : -self.width].sum())
            pred = logits[0, prompt_len - 1 : -1]
            labels = ids[0, prompt_len:]
            torch.testing.assert_close(response_logits[row, : len(labels)], pred, atol=atol, rtol=rtol)
            logprobs = pred.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
            reference_loss = reference_loss - (logprobs * weights[row, : len(labels)]).sum()

        torch.testing.assert_close(packed_loss, reference_loss, atol=atol, rtol=rtol)
        reference_loss.backward()
        packed_loss.backward()
        for (name, param), (_, ref) in zip(grouped.named_parameters(), reference.named_parameters(), strict=True):
            with self.subTest(parameter=name):
                torch.testing.assert_close(param.grad, ref.grad, atol=atol, rtol=rtol)

    def test_logits_loss_and_parameter_gradients(self):
        dtype = torch.float32 if self.device.startswith("npu") else torch.float64
        tolerance = 3e-4 if dtype == torch.float32 else 1e-9
        self.check_equivalence(dtype, atol=tolerance, rtol=tolerance)

    def test_bfloat16_logits_loss_and_parameter_gradients(self):
        self.check_equivalence(torch.bfloat16, atol=0.08, rtol=0.04)

    def test_sibling_isolation_in_forward_and_backward(self):
        torch.manual_seed(802)
        model = ToyDecoder().to(self.device)
        p = self.packed
        embedded = model.embedding(p.input_ids).detach().requires_grad_(True)
        original = model(embedded, p.position_ids, p.attention_mask)
        # The last token in branch 0 has no path to the other branches.
        last = int(p.restore_indices[0, -1])
        untouched = [int(p.restore_indices[2, 4]), int(p.restore_indices[1, 4])]
        original[0, last].square().sum().backward()
        torch.testing.assert_close(embedded.grad[0, untouched], torch.zeros_like(embedded.grad[0, untouched]))
        self.assertGreater(embedded.grad[0, :3].abs().sum().item(), 0)
        changed = embedded.detach().clone()
        changed[0, untouched] += 0.9
        after = model(changed, p.position_ids, p.attention_mask)
        torch.testing.assert_close(original[0, last], after[0, last], rtol=0, atol=0)

    def test_empty_response_and_single_prompt_token(self):
        ids = torch.tensor([[0, 4, 6, 0], [0, 4, 0, 0]], device=self.device)
        pos = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]], device=self.device)
        p = pack_shared_prefix(ids, ids != 0, pos, 2, ["g", "g"])
        self.assertEqual(p.input_ids.tolist(), [[4, 6]])
        self.assertEqual(p.response_predictors(p.input_ids).tolist(), [[4, 0], [0, 0]])
        torch.testing.assert_close(p.restore(p.input_ids), ids)

    def test_ungrouped_prompts(self):
        self.uids = ["one", "two", "three"]
        p = self.pack()
        self.assertEqual(p.input_ids.numel(), p.original_token_count)
        torch.testing.assert_close(p.restore(p.input_ids), self.ids)

    def test_mismatched_shared_positions_and_tokens(self):
        self.ids[2, 2] += 1
        with self.assertRaisesRegex(ValueError, "identical prompt tokens"):
            self.pack()
        self.ids[2, 2] -= 1
        self.pos[2, 1] += 1
        with self.assertRaisesRegex(ValueError, "positions"):
            self.pack()

    def test_native_ascend_attention_and_gradients(self):
        if not self.device.startswith("npu"):
            self.skipTest("requires the patched verl installation and an Ascend NPU")
        from verl.models.mcore.prefix_grouper_forward import _masked_core_attention

        p = self.packed
        torch.manual_seed(4681)
        inputs = [
            torch.randn(p.input_ids.numel(), 1, heads, 128, device=self.device, dtype=torch.bfloat16)
            for heads in (32, 8, 8)
        ]
        actual_inputs = [v.detach().clone().requires_grad_(True) for v in inputs]
        reference_inputs = [v.float().requires_grad_(True) for v in inputs]
        actual = _masked_core_attention(*actual_inputs, p.attention_mask)
        # Accumulate one independent triangular computation per original trajectory.
        # Distinct cotangents force all sibling contributions into the shared gradient.
        actual_loss = 0
        expected_loss = 0
        for row in range(len(self.uids)):
            path = p.restore_indices[row, self.mask[row]]
            q, k, v = (x.index_select(0, path).permute(1, 2, 0, 3) for x in reference_inputs)
            k, v = (x.repeat_interleave(4, dim=1) for x in (k, v))
            scores = (q @ k.transpose(-1, -2)) * (128**-0.5)
            causal = torch.ones((len(path), len(path)), dtype=torch.bool, device=self.device).triu(1)
            expected = scores.masked_fill(causal, float("-inf")).softmax(-1) @ v
            expected = expected.permute(2, 0, 1, 3).flatten(2)
            observed = actual.index_select(0, path).float()
            torch.testing.assert_close(observed, expected, atol=0.03, rtol=0.03)
            cotangent = torch.randn_like(expected)
            expected_loss = expected_loss + (expected * cotangent).sum()
            actual_loss = actual_loss + (observed * cotangent).sum()
        actual_loss.backward()
        expected_loss.backward()
        for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
            torch.testing.assert_close(actual_input.grad.float(), reference_input.grad, atol=0.08, rtol=0.05)

    def test_malformed_padding(self):
        self.mask[0, 5] = False
        with self.assertRaisesRegex(ValueError, "right-padded responses"):
            self.pack()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=os.environ.get("PREFIX_GROUPER_TEST_DEVICE", "cpu"))
    args, unittest_args = parser.parse_known_args()
    os.environ["PREFIX_GROUPER_TEST_DEVICE"] = args.device
    unittest.main(argv=[sys.argv[0], *unittest_args], verbosity=2)
