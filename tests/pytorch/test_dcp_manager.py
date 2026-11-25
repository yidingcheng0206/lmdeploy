# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.communicator import DeviceCommunicator
from lmdeploy.pytorch.backends.cp_utils import (
    DCPPrefixChunk,
    get_dcp_a2a_workspace_size,
    get_dcp_workspace_size,
)
from lmdeploy.pytorch.backends.cuda.dcp import DCPManager, get_dcp_manager


@pytest.mark.parametrize('dim', [0, -1])
@pytest.mark.parametrize('rows', [0, 3])
def test_native_gather_order_bits_and_ownership(monkeypatch, dim, rows):
    group = object()
    communicator = DeviceCommunicator(group)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda g: 2 if g is group else 99)
    # Includes arbitrary float bit patterns used by the packed top-k IDs.
    first = torch.arange(rows * 16, dtype=torch.int32).view(rows, 16)[:, ::2].view(torch.float32)
    second = (torch.arange(rows * 8, dtype=torch.int32) + 100).view(rows, 8).view(torch.float32)
    gather = Mock()

    def copy(output, input, group):
        assert group is communicator.device_group
        output.copy_(torch.cat((input, second), dim=0))

    gather.side_effect = copy
    monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', gather)
    output = communicator.all_gather(first, dim=dim, copy_output=False)
    expected = torch.cat((first, second), dim=dim)
    assert torch.equal(output.view(torch.int32), expected.view(torch.int32))
    assert gather.call_count == int(rows > 0)
    if rows:
        first.zero_()
        assert torch.equal(output.view(torch.int32), expected.view(torch.int32))


def test_manager_registration_is_shared_and_layout_specific(monkeypatch):
    from lmdeploy.pytorch.distributed import get_dist_manager

    communicator = Mock()
    communicator.create_all_gather_workspace.return_value = None
    context = SimpleNamespace(dcp_manager=None,
                              dcp_group=SimpleNamespace(gpu_group=object(), rank=1, communicator=communicator),
                              dist_config=SimpleNamespace(dcp=2, dcp_comm_backend='a2a'))
    monkeypatch.setattr(get_dist_manager(), 'current_context', lambda: context)
    manager = get_dcp_manager()
    for _ in range(48):
        assert get_dcp_manager() is manager
        manager.prepare_attention(8, 576)
        manager.prepare_prefix_gather(576)
        manager.prepare_candidate_gather(512)
    assert communicator.create_all_gather_workspace.call_count == 4
    manager.prepare_attention(16, 128)
    assert communicator.create_all_gather_workspace.call_count == 6
    assert manager.world_size == 2 and manager.rank == 1


def test_dcp_communicator_does_not_inherit_tp_provider_restrictions(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.comm import communicator as module

    provider = Mock(side_effect=AssertionError('DCP must not initialize a TP all-reduce provider'))
    monkeypatch.setattr(module, 'CudaCommunicator', provider)
    group = object()
    config = SimpleNamespace(dp=2, ep=8, enable_microbatch=False)
    result = module.build_cuda_communicator(None, group, config, group_name='dcp')
    assert result.device_group is group
    provider.assert_not_called()


def test_manager_empty_merge_and_missing_workspace():
    from lmdeploy.pytorch.backends.cuda.attention.cp import merge_dcp_attention

    group = SimpleNamespace(gpu_group=object(), communicator=Mock(), rank=0)
    output = torch.empty(0, 12, 512, dtype=torch.bfloat16)
    lse = torch.empty(0, 12)
    counts = torch.empty(0, dtype=torch.int32)
    for backend in ('ag_rs', 'a2a'):
        manager = DCPManager(group, 4, comm_backend=backend)
        assert manager.combine(output, lse, counts).shape == (0, 3, 512)
    group.communicator.all_to_all_single.assert_not_called()
    group.communicator.all_gather.assert_not_called()
    with pytest.raises(ValueError, match='requires a prepared workspace'):
        merge_dcp_attention(output, lse, counts, dcp_world_rank=(4, 0), comm_backend='a2a')


def test_prefix_chunk_geometry():
    chunk = DCPPrefixChunk(256, 512, torch.tensor([300, 0, 512]), torch.tensor([0, 300, 300, 812]),
                           torch.empty(4, 3), torch.empty(4))
    assert chunk.local_capacity == 384
    assert chunk.local_block_slice(64) == slice(1, 3)


def test_single_rank_prefix_does_not_use_default_world_group(monkeypatch):
    from lmdeploy.pytorch.kernels.cuda import dcp as kernels

    group = SimpleNamespace(gpu_group=None, communicator=Mock(), rank=0)
    manager = DCPManager(group, 1)
    local = torch.arange(8).view(4, 1, 2)
    chunk = DCPPrefixChunk(0, 4, torch.tensor([4]), torch.tensor([0, 4]),
                           torch.tensor([[4]]), torch.tensor([0, 4]))
    monkeypatch.setattr(kernels, 'reorder_dcp_prefill_kv', lambda source, dest, **kw: dest.copy_(source))
    output = manager.gather_prefix(local, chunk)
    assert torch.equal(output, local)
    assert output.data_ptr() != local.data_ptr()
    group.communicator.all_gather.assert_not_called()


@pytest.mark.parametrize('topk', [None, 512])
def test_a2a_reserve_survives_prefill_and_indexer(topk):
    kwargs = dict(num_prefill_tokens=2048, num_decode_tokens=32, batch_size=32, num_heads=8,
                  head_dim=576, dtype=torch.bfloat16, block_size=64, dcp_size=4,
                  topk=topk, score_workspace_bytes=64 << 20)
    baseline = get_dcp_workspace_size(**kwargs)
    a2a = get_dcp_workspace_size(**kwargs, comm_backend='a2a')
    retained = get_dcp_a2a_workspace_size(num_tokens=32, num_heads=8, head_dim=576,
                                         dtype=torch.bfloat16, dcp_size=4)
    assert a2a - baseline == retained > 0


def test_reset_after_all_graphs_and_before_groups(monkeypatch):
    from lmdeploy.pytorch.distributed import DistContext

    events = []
    manager = Mock()
    manager.close.side_effect = lambda: events.append('workspace')
    group = Mock()
    group.close.side_effect = lambda: events.append('group')
    context = DistContext(dcp_manager=manager, dcp_group=group)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    context.close()
    assert events == ['workspace', 'group']
    assert context.dcp_manager is None


def test_shared_workspace_graph_rejects_another_replay_stream(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.graph_runner.full_graph import CUDASingleGraphRunner

    runner = object.__new__(CUDASingleGraphRunner)
    runner._dcp_workspace_stream = SimpleNamespace(device='cuda')
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: object())
    with pytest.raises(RuntimeError, match='capture execution stream'):
        runner.forward()


@pytest.mark.parametrize('shared', [True, False])
def test_model_reset_retires_target_and_draft_before_shared_scratch(shared):
    from contextlib import nullcontext

    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    events = []
    target = Mock()
    target.reset.side_effect = lambda: events.append('target_workspace')
    draft = target if shared else Mock()
    if not shared:
        draft.reset.side_effect = lambda: events.append('draft_workspace')
    agent = object.__new__(BaseModelAgent)
    agent.all_context = nullcontext
    agent.cache_config = SimpleNamespace(compact_state_capacity=None)
    agent.dist_ctx = SimpleNamespace(dcp_manager=target)
    agent.spec_agent = SimpleNamespace(draft_dist_ctx=SimpleNamespace(dcp_manager=draft),
                                       reset_graph_runner=lambda: events.append('draft_graphs'))
    agent.patched_model = SimpleNamespace(reset=lambda: events.append('target_graphs'))
    agent.memdecode_agent = None
    agent.reset_graph_runner()
    assert events[:2] == ['target_graphs', 'draft_graphs']
    target.reset.assert_called_once()
    draft.reset.assert_called_once()
    assert events[2:] == (['target_workspace'] if shared else ['target_workspace', 'draft_workspace'])
