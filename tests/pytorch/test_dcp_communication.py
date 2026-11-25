# Copyright (c) OpenMMLab. All rights reserved.
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _dcp_worker(rank, rendezvous, world_size):
    from lmdeploy.pytorch.backends.cuda.attention.cp import (
        gather_dcp_query,
        merge_dcp_attention,
    )
    from lmdeploy.pytorch.backends.cuda.dcp import DCPManager
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, get_dist_manager

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=180))
    try:
        dp = 2 if world_size == 8 else 1
        attn_tp = world_size // dp
        for dcp in ((1, 2, 4) if world_size == 8 else (1, 2)):
            config = DistConfig(tp=world_size, dp=dp, ep=world_size, dp_rank=rank // attn_tp, dcp=dcp)
            context = DistContext.build(rank, config, communicator_builder=lambda **kwargs: None)
            with get_dist_manager().context(context), torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
                tokens = 3 if rank < attn_tp else 7  # Different batches across DP groups.
                torch.manual_seed(717 + rank // dcp)
                heads = 5 * dcp
                states = torch.randn(dcp, tokens, heads, 512, dtype=torch.bfloat16, device='cuda')
                lses = torch.randn(dcp, tokens, heads, device='cuda') * 10
                counts = torch.ones(dcp, tokens, dtype=torch.int32, device='cuda')
                if dcp > 1:
                    counts[0, 0] = 0
                    states[0, 0] = torch.nan
                    lses[:, -1, 0] = -torch.inf
                    states[:, -1, 0] = torch.nan
                dcp_rank = rank % dcp
                local = states[dcp_rank]
                lse = lses[dcp_rank]
                count = counts[dcp_rank]
                query = torch.full((tokens, 5, 17), rank, dtype=torch.bfloat16, device='cuda')
                gathered = gather_dcp_query(query, dcp_world_size=dcp)
                expected_query = torch.cat([torch.full_like(query, r)
                                            for r in range(rank // dcp * dcp, (rank // dcp + 1) * dcp)], dim=1)
                torch.testing.assert_close(gathered, expected_query, rtol=0, atol=0)
                for backend in ('ag_rs', 'a2a'):
                    manager = DCPManager(context.dcp_group, dcp, comm_backend=backend)
                    manager.prepare_attention(5, 17)
                    torch.testing.assert_close(manager.gather_query(query), expected_query, rtol=0, atol=0)
                    workspace = manager.attention_workspace

                    def merge():
                        return manager.combine(local, lse, count)

                    def check(output):
                        if dcp == 1:
                            expected = local
                        else:
                            safe_lse = torch.where((counts[:, :, None] > 0) & torch.isfinite(lses), lses, -torch.inf)
                            maximum = safe_lse.max(dim=0).values
                            weights = (safe_lse - torch.where(torch.isfinite(maximum), maximum, 0.)).exp()
                            weights /= weights.sum(dim=0).clamp_min(1e-30)
                            expected = torch.where(weights[..., None] == 0, 0.,
                                                   states.float() * weights[..., None]).sum(dim=0)
                            expected = expected[:, dcp_rank*5:(dcp_rank+1)*5]
                        torch.testing.assert_close(output.float(), expected.float(), rtol=2e-2, atol=2e-2)

                    for _ in range(3):
                        check(merge())
                    legacy = merge_dcp_attention(local, lse, count, dcp_world_rank=(dcp, dcp_rank),
                                                  comm_backend=backend, workspace=workspace)
                    torch.testing.assert_close(merge(), legacy, rtol=0, atol=0)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                        output = merge()
                    graph.replay()
                    check(output)
                    if backend == 'a2a' and dcp > 1:
                        # Eager growth must not invalidate previously captured pointers.
                        workspace.get(local.new_empty(tokens * 4, heads, 512), dcp)
                        for history in workspace._buffers.values():
                            assert sum(pair[0].numel() for pair in history) < 2 * history[-1][0].numel()
                    # Exercise captured pack/merge again with new finite data.
                    states.add_(0.25)
                    lses.mul_(0.7)
                    graph.replay()
                    check(output)
                    torch.cuda.synchronize()
                    dist.barrier()
                    del graph
                    if dcp > 1:
                        _exercise_manager(manager, local, lse, count, rank, attn_tp)
                    manager.close()
                if dcp > 1:
                    from lmdeploy.pytorch.models.deepseek_v2 import DeepseekV2BMM
                    from lmdeploy.pytorch.nn.dcp import DCPQueryProjection
                    from lmdeploy.pytorch.nn.linear.default import BaseLinear

                    torch.manual_seed(911 + rank)
                    projection = BaseLinear(32, 5 * attn_tp * 16, bias=False, dtype=torch.bfloat16,
                                             device=torch.device('cuda', rank), is_tp=True)
                    key = DeepseekV2BMM(5 * attn_tp, 8, 16, dtype=torch.bfloat16,
                                        device=torch.device('cuda', rank))
                    projection.weight.normal_(0, .1)
                    key.weight.normal_(0, .1)
                    torch.manual_seed(812 + rank // attn_tp)
                    hidden = torch.randn(1, tokens, 32, dtype=torch.bfloat16, device='cuda')

                    def local_query():
                        q = projection(hidden).view(tokens, 5, 16)
                        result = hidden.new_empty(tokens, 5, 24)
                        key(q[..., :8], result[..., :16])
                        result[..., 16:] = q[..., 8:]
                        return result

                    before = local_query()
                    replica = DCPQueryProjection(projection)
                    replica.prepare(projection, key)
                    assert projection.weight.shape == (80, 32)
                    assert key.weight.shape == (5, 8, 16)
                    assert (projection.weight.untyped_storage().data_ptr()
                            == replica.query_weight.untyped_storage().data_ptr())
                    assert key.weight.untyped_storage().data_ptr() == replica.key_weight.untyped_storage().data_ptr()
                    torch.testing.assert_close(local_query(), before, rtol=0, atol=0)

                    def replicated_query():
                        q = replica.project(hidden).view(tokens, 5 * dcp, 16)
                        result = hidden.new_empty(tokens, 5 * dcp, 24)
                        replica.absorb(q[..., :8], result[..., :16])
                        result[..., 16:] = q[..., 8:]
                        return result

                    expected = gather_dcp_query(local_query(), dcp_world_size=dcp)
                    torch.testing.assert_close(replicated_query(), expected, rtol=2e-2, atol=2e-2)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                        query_output = replicated_query()
                    hidden.add_(.25)
                    graph.replay()
                    expected = gather_dcp_query(local_query(), dcp_world_size=dcp)
                    torch.testing.assert_close(query_output, expected, rtol=2e-2, atol=2e-2)
                    torch.cuda.synchronize()
                    del graph
                    # Refresh after a hot weight update; stale replicas must not survive.
                    projection.weight.add_(rank * .01)
                    key.weight.add_(rank * .01)
                    replica.prepare(projection, key)
                    expected = gather_dcp_query(local_query(), dcp_world_size=dcp)
                    torch.testing.assert_close(replicated_query(), expected, rtol=2e-2, atol=2e-2)
                    dist.barrier()
    finally:
        dist.destroy_process_group()


def _exercise_manager(manager, local, lse, count, rank, attn_tp):
    from lmdeploy.pytorch.backends.cp_utils import DCPPrefixChunk, get_dcp_local_seq_lens

    dcp = manager.world_size
    # An empty DP0 and a nonempty DP1 must not communicate across DCP groups.
    active = 0 if rank < attn_tp else local.size(0)
    assert manager.combine(local[:active], lse[:active], count[:active]).size(0) == active
    reference = manager.combine(local, lse, count).clone()
    torch.testing.assert_close(manager.combine(local, lse, count), reference, rtol=0, atol=0)

    packed = (torch.arange(12, dtype=torch.int32, device='cuda') + rank * 100).view(torch.float32).view(3, 2, 2)
    manager.prepare_candidate_gather(2)
    gathered = manager.gather_candidates(packed)
    for peer in range(dcp):
        expected = torch.arange(12, dtype=torch.int32, device='cuda') + (rank // dcp * dcp + peer) * 100
        assert torch.equal(gathered[peer].view(torch.int32).flatten(), expected)

    length = torch.tensor([4 * dcp - 1], device='cuda', dtype=torch.int32)
    lengths = torch.stack([get_dcp_local_seq_lens(length, (dcp, r)) for r in range(dcp)])
    chunk = DCPPrefixChunk(0, 4 * dcp, length, torch.cat((length.new_zeros(1), length)),
                           lengths, length.new_tensor([0, int(lengths[manager.rank, 0])]))
    local_kv = torch.arange(manager.rank, 4 * dcp, dcp, device='cuda', dtype=torch.bfloat16).view(4, 1, 1)
    manager.prepare_prefix_gather(1)
    prefix = manager.gather_prefix(local_kv, chunk, zero_padding=True)
    torch.testing.assert_close(prefix[:-1, 0, 0], torch.arange(4 * dcp - 1, device='cuda', dtype=torch.bfloat16),
                               rtol=0, atol=0)
    assert prefix[-1].item() == 0

    if manager.comm_backend != 'a2a':
        return
    workspace = manager.attention_workspace
    # Same-shape layers reuse scratch but every returned output owns storage.
    before_bytes = workspace.allocated_bytes
    expected = [manager.combine(local + i, lse, count).clone() for i in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        outputs = [manager.combine(local + i, lse, count) for i in range(3)]
    for _ in range(3):
        graph.replay()
        for actual, wanted in zip(outputs, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert workspace.allocated_bytes == before_bytes
    small_graph = torch.cuda.CUDAGraph()
    manager.combine(local[:1], lse[:1], count[:1])
    with torch.cuda.graph(small_graph, stream=torch.cuda.current_stream()):
        small_output = manager.combine(local[:1], lse[:1], count[:1])
    for _ in range(3):
        small_graph.replay()
        torch.testing.assert_close(small_output, reference[:1], rtol=0, atol=0)
        graph.replay()
        for actual, wanted in zip(outputs, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    original = workspace.get(local, dcp)[0]
    with torch.cuda.stream(torch.cuda.Stream()):
        alternate = workspace.get(local, dcp)[0]
    assert alternate.data_ptr() != original.data_ptr()
    torch.cuda.synchronize()
    del small_graph, graph


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason='Requires eight CUDA devices for DP2/EP8 topology')
def test_dcp_nccl_and_cuda_graph(tmp_path):
    mp.spawn(_dcp_worker, args=('file://' + str(tmp_path / 'nccl-init'), 8), nprocs=8, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Requires two CUDA devices')
def test_dcp_two_gpu_nccl_and_cuda_graph(tmp_path):
    mp.spawn(_dcp_worker, args=('file://' + str(tmp_path / 'nccl-init'), 2), nprocs=2, join=True)
