# SPDX-License-Identifier: Apache-2.0
"""Eager causal attention using the same CANN FA arithmetic as Megatron."""

import torch
import torch_npu

MIN_TRAINING_QUERY_ROWS = 16


def training_causal_attention(query, key, value, key_cache, value_cache, metadata, scale, mask, cache_layout):
    """Use actual request lengths and current cache blocks, one request at a time.

    Per-request calls keep unrelated batch members out of kernel selection.
    Cache ownership, writes, block allocation and eviction remain in vLLM.
    """
    if query.dtype != torch.bfloat16 or query.ndim != 3:
        raise ValueError("Training attention requires BF16 TND queries")
    q_ends = metadata.actual_seq_lengths_q
    kv_lengths = metadata.seq_lens_list
    if len(q_ends) != len(kv_lengths):
        raise ValueError("Training attention requires one KV length per request")
    no_cache = metadata.attn_state.name == "PrefillNoCache"
    output = torch.empty_like(query)
    start = 0
    for request, (end, kv_len) in enumerate(zip(q_ends, kv_lengths)):
        end, kv_len = int(end), int(kv_len)
        if end == start:
            continue
        if kv_len < end - start:
            raise ValueError("Causal query length cannot exceed its KV history")
        if no_cache:
            if kv_len != end - start:
                raise ValueError("Uncached attention requires a complete prefill")
            keys, values = key[start:end], value[start:end]
        else:
            if key_cache is None or value_cache is None or key_cache.ndim != 4:
                raise ValueError("Training attention requires an unquantized four-dimensional KV cache")
            hnd = cache_layout in ("LBHNC", "HND")
            block_size = key_cache.shape[2 if hnd else 1]
            count = (kv_len + block_size - 1) // block_size
            indices = metadata.block_tables[request, :count].long()
            keys, values = key_cache.index_select(0, indices), value_cache.index_select(0, indices)
            if hnd:
                keys, values = keys.transpose(1, 2), values.transpose(1, 2)
            keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])[:kv_len]
            values = values.reshape(-1, values.shape[-2], values.shape[-1])[:kv_len]
        if keys.dtype != query.dtype or values.dtype != query.dtype:
            raise ValueError("Training attention does not support quantized KV caches")
        queries = query[start:end].contiguous()
        query_padding = 0
        if (
            not no_cache
            and queries.shape[0] < MIN_TRAINING_QUERY_ROWS
            and kv_len >= MIN_TRAINING_QUERY_ROWS
        ):
            # CANN selects different accumulation for a short query tail.
            # Left padding retains bottom-right causal positions: the real
            # queries still attend to exactly the same cached key prefixes.
            query_padding = MIN_TRAINING_QUERY_ROWS - queries.shape[0]
            queries = torch.cat((queries.new_zeros((query_padding, *queries.shape[1:])), queries))
        result = torch_npu.npu_fusion_attention(
            queries, keys.contiguous(), values.contiguous(),
            query.shape[1], "TND", atten_mask=mask, scale=scale,
            pre_tockens=2147483647, next_tockens=0, keep_prob=1.0,
            inner_precise=0, sparse_mode=3,
            actual_seq_qlen=(queries.shape[0],), actual_seq_kvlen=(kv_len,),
        )[0]
        output[start:end] = result[query_padding:]
        start = end
    # Eager model runners can still supply padding beyond the actual queries.
    output[start:] = 0
    return output
