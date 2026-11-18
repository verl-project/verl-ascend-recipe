# SPDX-License-Identifier: Apache-2.0
"""Pack GRPO trajectories into a causal attention tree for a Megatron microbatch."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch


@dataclass
class PackedPrefixBatch:
    input_ids: torch.Tensor  # [1, T]
    position_ids: torch.Tensor  # [1, T], original per-trajectory positions
    attention_mask: torch.Tensor  # [1, 1, T, T], True blocks a key in Megatron
    restore_indices: torch.Tensor  # [batch, original width]
    response_indices: torch.Tensor  # [batch, response width], preceding each label
    valid_mask: torch.Tensor  # [batch, original width]
    original_token_count: int
    num_groups: int

    def _gather(self, values: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if values.shape[:2] != self.input_ids.shape:
            raise ValueError("expected values with packed [1, T, ...] leading dimensions")
        gathered = values[0].index_select(0, indices.flatten()).reshape(*indices.shape, *values.shape[2:])
        return gathered.masked_fill(~valid.reshape(*valid.shape, *([1] * (values.ndim - 2))), 0)

    def restore(self, values: torch.Tensor) -> torch.Tensor:
        """Expand shared rows for per-trajectory outputs. Padding is filled with zero."""
        return self._gather(values, self.restore_indices, self.valid_mask)

    def response_predictors(self, values: torch.Tensor) -> torch.Tensor:
        """Gather the row predicting each original response token, including its first."""
        return self._gather(values, self.response_indices, self.valid_mask[:, -self.response_indices.size(1) :])


def pack_shared_prefix(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_length: int,
    uids: Sequence[Hashable],
) -> PackedPrefixBatch:
    """Pack each distinct UID's full prompt once and append each response as a private branch.

    Inputs use verl's text layout: left-padded prompts followed by right-padded
    responses. `attention_mask` marks real tokens, including any response tokens
    masked out of the PPO loss. IDs and RoPE positions must agree for a shared
    prompt. All trajectories stay in input order in the index maps.

    This implementation materializes a quadratic boolean attention mask. Call
    it on one microbatch at a time. For a vocabulary-parallel head use the
    indices directly to gather reduced outputs or bounded chunks of logits.
    Padding indices are zero and must be ignored when using the indices directly.
    """
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape != position_ids.shape:
        raise ValueError("input_ids, attention_mask and position_ids must have identical [B, L] shapes")
    if input_ids.device != attention_mask.device or input_ids.device != position_ids.device:
        raise ValueError("input_ids, attention_mask and position_ids must share a device")
    batch_size, width = input_ids.shape
    if isinstance(uids, torch.Tensor):
        if uids.ndim != 1:
            raise ValueError("expected a one-dimensional sequence of UIDs")
        uids = uids.tolist()
    if not batch_size or not 0 < response_length < width or len(uids) != batch_size:
        raise ValueError("a nonempty batch, a prompt and response region, and one uid per row are required")

    prompt_width = width - response_length
    device = input_ids.device
    valid = attention_mask.bool()
    prompt_lengths = valid[:, :prompt_width].sum(dim=1)
    response_lengths = valid[:, prompt_width:].sum(dim=1)
    columns = torch.arange(width, device=device)
    left_padding = (prompt_width - prompt_lengths).unsqueeze(1)
    expected_mask = (columns >= left_padding) & (columns < prompt_width + response_lengths.unsqueeze(1))
    if not torch.equal(expected_mask, attention_mask):
        raise ValueError("expected left-padded prompts and right-padded responses with a binary mask")

    groups = {}
    for row, uid in enumerate(uids):
        groups.setdefault(uid, []).append(row)

    lengths = torch.stack([prompt_lengths, response_lengths], dim=1).tolist()
    representatives = [0] * batch_size
    prefix_starts = [0] * batch_size
    branch_starts = [0] * batch_size
    segments = []
    cursor = 0
    for rows in groups.values():
        first = rows[0]
        prompt_size = lengths[first][0]
        if not prompt_size:
            raise ValueError("a response needs at least one prompt token to predict its first token")
        trunk = cursor
        segments.append((first, slice(prompt_width - prompt_size, prompt_width)))
        cursor += prompt_size
        for row in rows:
            prompt_len, response_len = lengths[row]
            if prompt_len != prompt_size:
                raise ValueError("trajectories sharing a uid must have identical prompts")
            representatives[row] = first
            prefix_starts[row] = trunk
            branch_starts[row] = cursor
            if response_len:
                segments.append((row, slice(prompt_width, prompt_width + response_len)))
                cursor += response_len

    if len(groups) < batch_size:
        representatives_tensor = torch.tensor(representatives, device=device)
        is_prompt = valid[:, :prompt_width]
        for tensor in (input_ids[:, :prompt_width], position_ids[:, :prompt_width]):
            equal = (tensor == tensor.index_select(0, representatives_tensor)) | ~is_prompt
            if not equal.all().item():
                raise ValueError("trajectories sharing a uid must have identical prompt tokens and positions")

    packed_ids = torch.cat([input_ids[row, span] for row, span in segments])[None, :]
    packed_positions = torch.cat([position_ids[row, span] for row, span in segments])[None, :]
    blocked = torch.ones((cursor, cursor), dtype=torch.bool, device=device)
    for rows in groups.values():
        first = rows[0]
        trunk = prefix_starts[first]
        trunk_end = trunk + lengths[first][0]
        blocked[trunk:trunk_end, trunk:trunk_end].triu_(diagonal=1)
        for row in rows:
            start = branch_starts[row]
            end = start + lengths[row][1]
            if end > start:
                blocked[start:end, trunk:trunk_end] = False
                blocked[start:end, start:end].triu_(diagonal=1)

    prefix = torch.tensor(prefix_starts, device=device).unsqueeze(1)
    branch = torch.tensor(branch_starts, device=device).unsqueeze(1)
    restore = torch.where(columns < prompt_width, prefix + columns - left_padding, branch + columns - prompt_width)
    restore.masked_fill_(~valid, 0)
    # Shifting each trajectory's reconstruction map selects its own preceding
    # token. In particular, siblings may give different labels to the same
    # final prompt logit. Autograd sums their contributions at that logit.
    predictors = restore[:, prompt_width - 1 : -1].masked_fill(~valid[:, prompt_width:], 0)
    return PackedPrefixBatch(
        input_ids=packed_ids,
        position_ids=packed_positions,
        attention_mask=blocked[None, None],
        restore_indices=restore,
        response_indices=predictors,
        valid_mask=valid,
        original_token_count=sum(p + r for p, r in lengths),
        num_groups=len(groups),
    )
