# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.dcp import combine_dcp_attention, pack_dcp_attention


def reference_merge(outputs, lses, counts):
    valid = (counts[:, :, None] > 0) & torch.isfinite(lses)
    lses = torch.where(valid, lses, -torch.inf)
    maximum = lses.max(dim=0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.)
    weights = (lses - maximum).exp()
    weights = weights / weights.sum(dim=0).clamp_min(1e-30)
    return torch.where(weights[..., None] == 0, 0., outputs.float() * weights[..., None]).sum(dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('dcp_size', [1, 2, 4])
@pytest.mark.parametrize(('tokens', 'dim'), [(1, 3), (8, 512), (17, 513)])
def test_dcp_pack_combine(dtype, dcp_size, tokens, dim):
    torch.manual_seed(217)
    heads = 3 * dcp_size
    # Both tensors have padded rows and nontrivial strides.
    outputs = torch.randn(dcp_size, tokens, heads + 2, dim * 2, device='cuda', dtype=dtype)[:, :, :heads, ::2]
    lses = (torch.randn(dcp_size, tokens, heads + 5, device='cuda') * 20)[:, :, :heads]
    counts = torch.ones(dcp_size, tokens, 2, dtype=torch.int32, device='cuda')[:, :, 0]
    counts[0, 0] = 0
    outputs[0, 0] = torch.nan
    lses[:, -1, 0] = -torch.inf
    outputs[:, -1, 0] = torch.nan
    lses[-1, -1, 1] = torch.nan
    lses[-1, -1, 2] = torch.inf
    expected = reference_merge(outputs, lses, counts)
    packets = torch.stack([pack_dcp_attention(outputs[rank], lses[rank], counts[rank],
                                              dcp_world_size=dcp_size) for rank in range(dcp_size)])
    for rank in range(dcp_size):
        wire_lse = packets[rank].view(torch.float32)[..., -1].transpose(0, 1).reshape(tokens, heads)
        valid = (counts[rank, :, None] > 0) & torch.isfinite(lses[rank])
        torch.testing.assert_close(wire_lse, torch.where(valid, lses[rank], -torch.inf), rtol=0, atol=0)
        received = packets[:, rank].contiguous()
        actual = combine_dcp_attention(received, head_dim=dim, dtype=dtype)
        tolerance = 2e-2 if dtype == torch.bfloat16 else (2e-3 if dtype == torch.float16 else 2e-5)
        torch.testing.assert_close(actual.float(), expected[:, rank*3:(rank+1)*3], rtol=tolerance, atol=tolerance)
        assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_dcp_empty_batch():
    output = torch.empty(0, 12, 512, dtype=torch.bfloat16, device='cuda')
    lse = torch.empty(0, 12, device='cuda')
    counts = torch.empty(0, dtype=torch.int32, device='cuda')
    packet = pack_dcp_attention(output, lse, counts, dcp_world_size=4)
    merged = combine_dcp_attention(packet, head_dim=512, dtype=output.dtype)
    assert merged.shape == (0, 3, 512)
