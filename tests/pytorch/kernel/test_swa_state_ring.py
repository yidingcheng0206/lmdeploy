from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.swa_state_ring import SWAStateRingMetadata
from lmdeploy.pytorch.kernels.cuda.swa_state_ring import flatten_swa_state_ring, scatter_swa_state_ring


def test_swa_state_ring_metadata_builds_ragged_layout():
    attn_metadata = SimpleNamespace(
        q_seqlens=torch.tensor([2, 1], dtype=torch.int64),
        kv_seqlens=torch.tensor([5, 10], dtype=torch.int64),
    )
    step_context = SimpleNamespace(max_q_seqlen=2)

    metadata = SWAStateRingMetadata.from_step_context(
        attn_metadata,
        step_context,
        state_slots=torch.tensor([1, 3]),
        num_state_slots=4,
        window_size=8,
    )

    torch.testing.assert_close(metadata.start_positions, torch.tensor([3, 9]))
    torch.testing.assert_close(metadata.history_lens, torch.tensor([3, 7], dtype=torch.int32))
    torch.testing.assert_close(metadata.cu_q_seqlens, torch.tensor([0, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(metadata.cu_kv_seqlens, torch.tensor([0, 5, 13], dtype=torch.int32))
    assert metadata.max_q_seqlen == 2
    assert metadata.max_kv_seqlen == 9


def test_swa_state_ring_metadata_rejects_invalid_window():
    with pytest.raises(ValueError, match='window_size must be greater than one'):
        SWAStateRingMetadata.from_step_context(
            SimpleNamespace(q_seqlens=torch.ones(1), kv_seqlens=torch.ones(1)),
            SimpleNamespace(max_q_seqlen=1),
            state_slots=torch.zeros(1),
            num_state_slots=1,
            window_size=1,
        )


def test_swa_state_ring_flattens_history_then_updates_current_tokens():
    ring = torch.zeros((1, 4, 1, 8), dtype=torch.bfloat16, device='cuda')
    ring[0, 0].fill_(10)
    ring[0, 1].fill_(20)
    current = torch.stack((torch.full((1, 8), 30), torch.full((1, 8), 40))).to(
        device='cuda', dtype=torch.bfloat16)
    state_slots = torch.tensor([0], dtype=torch.int64, device='cuda')
    start_positions = torch.tensor([2], dtype=torch.int64, device='cuda')
    q_seqlens = torch.tensor([2], dtype=torch.int32, device='cuda')
    cu_q_seqlens = torch.tensor([0, 2], dtype=torch.int32, device='cuda')
    history_lens = torch.tensor([2], dtype=torch.int32, device='cuda')
    cu_kv_seqlens = torch.tensor([0, 4], dtype=torch.int32, device='cuda')

    flattened = flatten_swa_state_ring(
        ring,
        current,
        state_slots,
        start_positions,
        q_seqlens,
        cu_q_seqlens,
        history_lens,
        cu_kv_seqlens,
        max_q_seqlen=2,
    )
    expected = torch.tensor([10, 20, 30, 40], dtype=torch.bfloat16, device='cuda')
    torch.testing.assert_close(flattened[:4, 0, 0], expected)

    scatter_swa_state_ring(current, ring, state_slots, start_positions, q_seqlens, cu_q_seqlens)
    torch.testing.assert_close(ring[0, :, 0, 0], expected)
