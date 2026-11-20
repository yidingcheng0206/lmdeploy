# Copyright (c) OpenMMLab. All rights reserved.
"""Small GQA reference tests for the DCP query and output communication path."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_GQA_CASES = (
    ('prefill', (2, 3), (2, 3)),
    ('cached_prefix_prefill', (2, 3), (5, 7)),
    ('single_token_decode', (1, 1), (4, 7)),
    ('multi_token_decode', (2, 3), (6, 8)),
)


def _make_gqa_inputs(device, q_lens, kv_lens, *, kv_heads, head_dim, fp8, rank):
    total_q = sum(q_lens)
    local_heads = 2
    global_heads = local_heads * 2
    torch.manual_seed(1234 + rank)
    local_query = torch.randn(total_q, local_heads, head_dim, device=device, dtype=torch.bfloat16)
    torch.manual_seed(4321)
    keys = []
    values = []
    for kv_len in kv_lens:
        keys.append(torch.randn(kv_len, kv_heads, head_dim, device=device, dtype=torch.bfloat16))
        values.append(torch.randn(kv_len, kv_heads, head_dim, device=device, dtype=torch.bfloat16))
    if fp8:
        fp8_dtype = torch.float8_e4m3fn
        keys = [key.to(fp8_dtype).to(torch.bfloat16) for key in keys]
        values = [value.to(fp8_dtype).to(torch.bfloat16) for value in values]
    assert global_heads == 4
    return local_query, keys, values


def _gqa_attention(query, keys, values, q_lens, kv_lens, *, dcp_rank=None, dcp_size=1):
    """Compute a tiny causal GQA attention reference on one DCP shard."""
    total_q, global_heads, head_dim = query.shape
    kv_heads = values[0].size(1)
    group_size = global_heads // kv_heads
    output = torch.zeros(total_q, global_heads, head_dim, device=query.device, dtype=torch.bfloat16)
    lse = torch.full((total_q, global_heads), -torch.inf, device=query.device, dtype=torch.float32)
    valid_counts = torch.zeros(total_q, device=query.device, dtype=torch.int32)
    query_offset = 0
    scale = head_dim**-0.5
    for request, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        history = kv_len - q_len
        key = keys[request].float()
        value = values[request].float()
        for query_index in range(q_len):
            end = history + query_index + 1
            positions = torch.arange(end, device=query.device)
            if dcp_rank is not None:
                positions = positions[positions.remainder(dcp_size) == dcp_rank]
            count = positions.numel()
            row = query_offset + query_index
            valid_counts[row] = count
            if count == 0:
                output[row] = torch.nan
                continue
            selected_key = key.index_select(0, positions)
            selected_value = value.index_select(0, positions)
            query_row = query[row].float()
            kv_index = torch.arange(global_heads, device=query.device) // group_size
            selected_key = selected_key.index_select(1, kv_index)
            selected_value = selected_value.index_select(1, kv_index)
            scores = torch.einsum('hd,khd->hk', query_row, selected_key) * scale
            row_lse = torch.logsumexp(scores, dim=-1)
            weights = torch.softmax(scores, dim=-1)
            row_output = torch.einsum('hk,khd->hd', weights, selected_value)
            output[row] = row_output.to(torch.bfloat16)
            lse[row] = row_lse
        query_offset += q_len
    return output, lse, valid_counts


def _gqa_worker(rank, rendezvous):
    from lmdeploy.pytorch.backends.cuda.dcp import DCPManager
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext

    world_size = 2
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=180))
    context = None
    try:
        config = DistConfig(tp=world_size, dcp=world_size)
        context = DistContext.build(rank, config, communicator_builder=lambda **kwargs: None)
        for _, q_lens, kv_lens in _GQA_CASES:
            for fp8 in (False, True):
                local_query, keys, values = _make_gqa_inputs(
                    torch.device('cuda', rank), q_lens, kv_lens, kv_heads=2, head_dim=16, fp8=fp8, rank=rank)
                for backend in ('ag_rs', 'a2a'):
                    manager = DCPManager(context.dcp_group, world_size, comm_backend=backend)
                    manager.prepare_attention(2, 16)
                    gathered_query = manager.gather_query(local_query)
                    local_output, local_lse, valid_counts = _gqa_attention(
                        gathered_query, keys, values, q_lens, kv_lens, dcp_rank=rank, dcp_size=world_size)
                    reference, _, _ = _gqa_attention(gathered_query, keys, values, q_lens, kv_lens)
                    actual = manager.combine(local_output, local_lse, valid_counts)
                    expected = reference[:, rank * 2:(rank + 1) * 2]
                    torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
                    manager.close()
                    dist.barrier()
    finally:
        if context is not None:
            context.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires two CUDA devices')
@pytest.mark.skipif(not hasattr(torch, 'float8_e4m3fn'), reason='Requires float8 support')
def test_gqa_dcp_matches_dense_reference(tmp_path):
    mp.spawn(_gqa_worker, args=('file://' + str(tmp_path / 'gqa-dcp-init'),), nprocs=2, join=True)
