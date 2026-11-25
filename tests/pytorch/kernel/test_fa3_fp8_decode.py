# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.fa3 import FA3Impl
from lmdeploy.pytorch.backends.cuda.step_metadata import CudaStepMetaPlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


@pytest.fixture(scope='module', autouse=True)
def require_fa3():
    pytest.importorskip('flash_attn_interface')


def _make_case(query_len=3, dtype=torch.bfloat16, policy=QuantPolicy.FP8,
               head_shape=(8, 2, 64, 64), window=None, softcap=0.0):
    if policy == QuantPolicy.FP8 and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('Triton E4M3 cache loads require SM90 or newer')
    torch.manual_seed(321)
    heads, kv_heads, head_dim, value_dim = head_shape
    batch, pages, block_size = 4, 5, 16
    impl = FA3Impl(heads, head_dim, num_kv_heads=kv_heads, v_head_size=value_dim,
                   sliding_window=window, logit_softcapping=softcap)
    plan = CudaStepMetaPlan.from_implementations([impl])
    assert plan.is_supported
    blocks = torch.randperm(batch * pages, device='cuda', dtype=torch.int32).view(batch, pages)
    lengths = torch.tensor([query_len, 17, 34, 67], device='cuda', dtype=torch.int32)
    query = torch.randn(batch * query_len, heads, head_dim, device='cuda', dtype=dtype) * 0.5
    fp8_dtype = torch.float8_e4m3fn if policy == QuantPolicy.FP8 else torch.float8_e5m2
    k_scale = torch.tensor(0.37, device='cuda')
    v_scale = torch.tensor(0.83, device='cuda')
    k = (torch.randn(batch * pages, block_size, kv_heads, head_dim, device='cuda') / k_scale).to(fp8_dtype)
    v = (torch.randn(batch * pages, block_size, kv_heads, value_dim, device='cuda') / v_scale).to(fp8_dtype)
    metadata = TritonAttentionMetadata(
        is_decoding=True, block_offsets=blocks, kv_seqlens=lengths,
        q_seqlens=torch.full((batch,), query_len, device='cuda', dtype=torch.int32), quant_policy=policy)
    ctx = SimpleNamespace(is_decoding=True, input_ids=torch.zeros(1, batch * query_len, device='cuda'),
                          kv_quant_policy=policy, model_config=SimpleNamespace(block_size=block_size, dtype=dtype))
    plan.prepare(ctx, metadata, metadata)
    return SimpleNamespace(impl=impl, plan=plan, ctx=ctx, metadata=metadata, q=query, k=k, v=v,
                           k_scale=k_scale, v_scale=v_scale, query_len=query_len)


def _run(case):
    return case.impl.forward(case.q, None, None, case.k, case.v, case.metadata,
                             case.k_scale, case.v_scale).reshape(case.q.shape[0], case.q.shape[1], case.v.shape[-1])


def _reference(case):
    """Compute dense FP32 attention from dequantized cache values."""
    outputs = []
    heads = case.q.shape[1]
    repeat = heads // case.k.shape[2]
    k_cache = case.k.float() * case.k_scale
    v_cache = case.v.float() * case.v_scale
    for batch, length in enumerate(case.metadata.kv_seqlens.tolist()):
        pages = case.metadata.block_offsets[batch].long()
        k = k_cache[pages].flatten(0, 1)[:length].repeat_interleave(repeat, dim=1)
        v = v_cache[pages].flatten(0, 1)[:length].repeat_interleave(repeat, dim=1)
        for pos in range(case.query_len):
            q = case.q[batch * case.query_len + pos].float()
            logits = torch.einsum('hd,khd->hk', q, k) * case.impl.scale
            if case.impl.logit_softcapping > 0:
                logits = torch.tanh(logits / case.impl.logit_softcapping) * case.impl.logit_softcapping
            end = length - case.query_len + pos
            keys = torch.arange(length, device='cuda')
            mask = keys <= end
            window = case.impl.sliding_window
            if isinstance(window, tuple):
                window = window[0]
            if window is not None and window > 0:
                mask &= keys >= end - window
            probs = logits.masked_fill(~mask[None, :], -torch.inf).softmax(-1)
            outputs.append(torch.einsum('hk,khd->hd', probs, v))
    return torch.stack(outputs)


def _assert_reference(case, output):
    # BF16 probability/output rounding dominates; FP8 storage error is already
    # included in the reference. These bounds are fixed across all shapes.
    tol = 1e-2 if case.q.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(output.float(), _reference(case), atol=tol, rtol=tol)
    assert torch.isfinite(output).all()
    assert output.dtype == case.q.dtype


@pytest.mark.parametrize('query_len', [1, 3, 5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('policy', [QuantPolicy.FP8, QuantPolicy.FP8_E5M2])
@pytest.mark.parametrize('head_shape', [(8, 2, 64, 64), (8, 1, 128, 64), (8, 1, 256, 256)])
@pytest.mark.parametrize('window,softcap', [(None, 0.0), (17, 0.0), ((17, 0), 10.0)])
def test_fp8_decode_reference(query_len, dtype, policy, head_shape, window, softcap):
    case = _make_case(query_len, dtype, policy, head_shape, window, softcap)
    before = case.q.clone()
    output = _run(case)
    _assert_reference(case, output)
    torch.testing.assert_close(case.q, before, atol=0, rtol=0)


def test_fp8_verify_cannot_read_future_tokens():
    case = _make_case()
    before = _run(case).clone()
    # Change only the final candidate's KV of request 2. Earlier candidates
    # and all other requests must be unaffected, despite sharing cache pages.
    last = int(case.metadata.kv_seqlens[2]) - 1
    page = int(case.metadata.block_offsets[2, last // 16])
    case.k[page, last % 16].fill_(12.0)
    case.v[page, last % 16].fill_(24.0)
    after = _run(case)
    earlier = torch.arange(case.q.shape[0], device='cuda') != 8
    torch.testing.assert_close(before[earlier], after[earlier], atol=0, rtol=0)
    assert not torch.equal(before[8], after[8])
    _assert_reference(case, after)


@pytest.mark.parametrize('policy', [QuantPolicy.FP8, QuantPolicy.FP8_E5M2])
@pytest.mark.parametrize('query_len', [3, 5])
def test_fp8_verify_graph_replays_updated_metadata(policy, query_len):
    case = _make_case(query_len=query_len, policy=policy)
    inputs = dict(q_seqlens=case.metadata.q_seqlens, kv_seqlens=case.metadata.kv_seqlens,
                  block_offsets=case.metadata.block_offsets)
    graph_meta = SimpleNamespace(decode_query_len=query_len, num_blocks=5, block_size=16)
    buffers = case.plan.make_cudagraph_buffers(graph_meta, inputs, case.ctx)
    case.plan.fill_cudagraph_buffers(graph_meta, inputs, case.ctx, buffers, case.metadata)
    rows = case.metadata.kernel_metadata[0]
    pointers = (rows.decode_kv_seqlens.data_ptr(), rows.decode_block_offsets.data_ptr())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(case)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = _run(case)
    torch.cuda.current_stream().wait_stream(stream)
    for lengths in ([query_len, 31, 48, 79], [query_len, query_len, 20, 33], [16, 32, 48, 64]):
        # Short first rows exercise the same lengths used for graph padding;
        # decreasing lengths also exercises rejected speculative candidates.
        case.metadata.kv_seqlens.copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
        case.metadata.block_offsets.copy_(case.metadata.block_offsets.roll(1, dims=0))
        case.q.normal_()
        case.plan.fill_cudagraph_buffers(graph_meta, inputs, case.ctx, buffers, case.metadata)
        assert pointers == (rows.decode_kv_seqlens.data_ptr(), rows.decode_block_offsets.data_ptr())
        graph.replay()
        _assert_reference(case, output)
        torch.testing.assert_close(output, _run(case), atol=0, rtol=0)


def test_fp8_decode_metadata_is_shared_by_layers():
    case = _make_case()
    other = FA3Impl(8, 64, num_kv_heads=2)
    plan = CudaStepMetaPlan.from_implementations([case.impl, other])
    assert len(plan.attention_builders) == 1
    plan.prepare(case.ctx, case.metadata, case.metadata)
    rows = case.metadata.kernel_metadata[0]
    out = other.forward(case.q, None, None, case.k, case.v, case.metadata, case.k_scale, case.v_scale)
    assert case.metadata.kernel_metadata[0] is rows
    torch.testing.assert_close(out.reshape_as(_run(case)), _run(case), atol=0, rtol=0)


def test_fp8_decode_rejects_noncausal_metadata():
    case = _make_case()
    impl = FA3Impl(8, 64, num_kv_heads=2, causal=False)
    with pytest.raises(NotImplementedError, match='causal attention'):
        impl.get_step_metadata_provider().build(case.ctx, case.metadata)


def test_native_fa3_graph_metadata_still_replays():
    case = _make_case(head_shape=(8, 1, 128, 128))
    case.ctx.kv_quant_policy = QuantPolicy.NONE
    case.metadata.quant_policy = QuantPolicy.NONE
    case.k = (case.k.float() * case.k_scale).to(case.q.dtype)
    case.v = (case.v.float() * case.v_scale).to(case.q.dtype)
    case.k_scale.fill_(1.0)
    case.v_scale.fill_(1.0)
    inputs = dict(q_seqlens=case.metadata.q_seqlens, kv_seqlens=case.metadata.kv_seqlens,
                  block_offsets=case.metadata.block_offsets)
    graph_meta = SimpleNamespace(decode_query_len=3, num_blocks=5, block_size=16)
    buffers = case.plan.make_cudagraph_buffers(graph_meta, inputs, case.ctx)
    case.plan.fill_cudagraph_buffers(graph_meta, inputs, case.ctx, buffers, case.metadata)
    assert case.metadata.kernel_metadata[0].scheduler_metadata is not None
    assert case.metadata.kernel_metadata[0].decode_kv_seqlens is None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _run(case)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = _run(case)
    torch.cuda.current_stream().wait_stream(stream)
    case.metadata.kv_seqlens.copy_(torch.tensor([15, 32, 48, 79], device='cuda', dtype=torch.int32))
    case.plan.fill_cudagraph_buffers(graph_meta, inputs, case.ctx, buffers, case.metadata)
    graph.replay()
    _assert_reference(case, output)


def test_fp8_decode_uses_its_own_metadata_group():
    case = _make_case()
    other = FA3Impl(8, 64, num_kv_heads=2, sliding_window=17)
    plan = CudaStepMetaPlan.from_implementations([other, case.impl])
    plan.prepare(case.ctx, case.metadata, case.metadata)
    assert len(plan.attention_builders) == 2
    # Native FA3 and FP8 decode must agree on which group's metadata to use.
    case.metadata.kernel_metadata[0].decode_kv_seqlens.zero_()
    _assert_reference(case, _run(case))


def test_fp8_decode_requires_kv_scales():
    case = _make_case()
    case.k_scale = None
    with pytest.raises(AssertionError, match='requires k scale'):
        _run(case)
