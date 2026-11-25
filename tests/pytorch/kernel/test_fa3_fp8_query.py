# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.backends.cuda.attention.fa3_fp8_query import FA3FP8QueryImpl
from lmdeploy.pytorch.backends.cuda.step_metadata import CudaStepMetaPlan
from lmdeploy.pytorch.kernels.cuda.quant_fp8_query import quant_fp8_query

from .test_fa3_fp8_decode import _make_case, _reference, _run

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


@pytest.fixture(scope='module', autouse=True)
def require_fa3():
    pytest.importorskip('flash_attn_interface')
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('Native FP8 FA3 requires SM90 or newer')


def _native_case(query_len=3, shape=(8, 2, 64, 64), window=None, softcap=0.0):
    case = _make_case(query_len=query_len, head_shape=shape, window=window, softcap=softcap)
    scale = torch.linspace(.01, .02, shape[1], device='cuda')
    case.impl = FA3FP8QueryImpl(shape[0], shape[2], num_kv_heads=shape[1], v_head_size=shape[3],
                              query_scale=scale, sliding_window=window, logit_softcapping=softcap)
    case.plan = CudaStepMetaPlan.from_implementations([case.impl])
    case.plan.prepare(case.ctx, case.metadata, case.metadata)
    return case


def _dequant_q(case):
    groups = case.k.shape[2]
    scale = case.impl.query_scale.view(1, groups, 1, 1)
    q = case.q.float().view(case.q.shape[0], groups, -1, case.q.shape[-1])
    return ((q / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(case.q.shape)


def _check(case, output):
    original_q = case.q
    r0 = _reference(case)
    case.q = _dequant_q(case) if case.query_len > 1 else original_q
    r1 = _reference(case)
    probability_ref = _fp8_probability_reference(case)
    case.q = original_q
    # Follow hopper/test_flash_attn.py's kvcache check: bound max/mean error by
    # 4x/3x an independent FP8-probability reference, plus final BF16 rounding.
    # A direct native-API comparison separately protects our integration.
    diff = (output.float() - r1).abs()
    probability_error = (probability_ref - r1).abs()
    output_rounding = (r1.to(original_q.dtype).float() - r1).abs()
    assert diff.max() <= 4 * probability_error.max() + output_rounding.max() + 1e-5
    assert diff.mean() <= 3 * probability_error.mean() + output_rounding.mean() + 1e-5
    relative_rms = (output.float() - r0).square().mean().sqrt() / r0.square().mean().sqrt()
    assert relative_rms < .06
    assert torch.isfinite(output).all()
    assert output.dtype == original_q.dtype
    return r0


def _fp8_probability_reference(case):
    outputs = []
    repeat = case.q.shape[1] // case.k.shape[2]
    for batch, length in enumerate(case.metadata.kv_seqlens.tolist()):
        pages = case.metadata.block_offsets[batch].long()
        k = (case.k[pages].float() * case.k_scale).flatten(0, 1)[:length].repeat_interleave(repeat, 1)
        v = (case.v[pages].float() * case.v_scale).flatten(0, 1)[:length].repeat_interleave(repeat, 1)
        for pos in range(case.query_len):
            q = case.q[batch * case.query_len + pos].float()
            scores = torch.einsum('hd,khd->hk', q, k) * case.impl.scale
            if case.impl.logit_softcapping > 0:
                scores = torch.tanh(scores / case.impl.logit_softcapping) * case.impl.logit_softcapping
            end = length - case.query_len + pos
            keys = torch.arange(length, device='cuda')
            mask = keys <= end
            window = case.impl.sliding_window
            if isinstance(window, tuple):
                window = window[0]
            if window is not None and window > 0:
                mask &= keys >= end - window
            probs = scores.masked_fill(~mask[None, :], -torch.inf).softmax(-1)
            probs = probs.to(torch.float8_e4m3fn).float()
            outputs.append(torch.einsum('hk,khd->hd', probs, v))
    return torch.stack(outputs)


def test_native_fp8_wrapper_matches_direct_api():
    from flash_attn_interface import flash_attn_with_kvcache

    case = _native_case(shape=(8, 1, 256, 256))
    # Quantize independently of our Triton launcher and omit our scheduler metadata.
    q = (case.q.float() / case.impl.query_scale[0]).clamp(-448, 448).to(torch.float8_e4m3fn)
    direct = flash_attn_with_kvcache(
        q.unflatten(0, (4, 3)), case.k, case.v, page_table=case.metadata.block_offsets,
        cache_seqlens=case.metadata.kv_seqlens, causal=True,
        q_descale=case.impl.query_scale.expand(4, 1),
        k_descale=case.k_scale.expand(4, 1), v_descale=case.v_scale.expand(4, 1))
    torch.testing.assert_close(_run(case), direct.flatten(0, 1), atol=0, rtol=0)


@pytest.mark.parametrize('length', [1024, 8192, 32768])
def test_native_fp8_long_context_reference(length):
    case = _native_case(shape=(8, 1, 256, 256))
    pages, block = (length + 63) // 64, 64
    case.k = (torch.randn(4 * pages, block, 1, 256, device='cuda') / case.k_scale).to(torch.float8_e4m3fn)
    case.v = (torch.randn(4 * pages, block, 1, 256, device='cuda') / case.v_scale).to(torch.float8_e4m3fn)
    case.metadata.block_offsets = torch.randperm(4 * pages, device='cuda', dtype=torch.int32).view(4, pages)
    case.metadata.kv_seqlens.copy_(torch.tensor([3, length // 3, length - 11, length], device='cuda'))
    case.ctx.model_config.block_size = block
    case.plan.prepare(case.ctx, case.metadata, case.metadata)
    output = _run(case)
    reference = _check(case, output)
    # Check each request separately so the short-prefix row cannot hide long-context errors.
    error = (output.float() - reference).unflatten(0, (4, 3)).square().mean(dim=(1, 2, 3))
    energy = reference.unflatten(0, (4, 3)).square().mean(dim=(1, 2, 3))
    assert ((error / energy).sqrt() < .06).all()


@pytest.mark.parametrize('query_len', [1, 3, 5])
@pytest.mark.parametrize('shape', [(8, 2, 64, 64), (8, 1, 128, 128), (8, 1, 256, 256)])
@pytest.mark.parametrize('window,softcap', [(None, 0.0), (17, 0.0), ((17, 0), 10.0)])
def test_native_fp8_reference(query_len, shape, window, softcap):
    case = _native_case(query_len, shape, window, softcap)
    before = case.q.clone()
    _check(case, _run(case))
    torch.testing.assert_close(case.q, before, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_query_quantization_strides_saturation_and_independence(dtype):
    torch.manual_seed(65)
    q = torch.randn(9, 16, 130, device='cuda', dtype=dtype)[:, ::2, ::2]
    q[0, 0, 0] = 100
    q[0, 0, 1] = -100
    q[0, 0, 2] = 0
    q[0, 0, 3] = float('nan')
    q[0, 0, 4] = float('inf')
    q[0, 0, 5] = -float('inf')
    scale = torch.tensor([.013, .03], device='cuda')
    expanded = scale.repeat_interleave(4).view(1, 8, 1)
    expected = (q.float() / expanded).clamp(-448, 448).to(torch.float8_e4m3fn)
    before = quant_fp8_query(q, scale)
    torch.testing.assert_close(before.float(), expected.float(), atol=0, rtol=0, equal_nan=True)
    q[3:].mul_(50)
    after = quant_fp8_query(q, scale)
    torch.testing.assert_close(before[:3].float(), after[:3].float(), atol=0, rtol=0, equal_nan=True)
    empty = quant_fp8_query(q[:0], scale)
    assert empty.shape == q[:0].shape


def test_native_fp8_future_tokens_are_masked():
    case = _native_case()
    before = _run(case).clone()
    last = int(case.metadata.kv_seqlens[2]) - 1
    page = int(case.metadata.block_offsets[2, last // 16])
    case.k[page, last % 16].fill_(12)
    case.v[page, last % 16].fill_(24)
    case.q[8].fill_(1.5)
    after = _run(case)
    earlier = torch.arange(case.q.shape[0], device='cuda') != 8
    torch.testing.assert_close(before[earlier], after[earlier], atol=0, rtol=0)
    assert not torch.equal(before[8], after[8])
    _check(case, after)


@pytest.mark.parametrize('query_len', [3, 5])
def test_native_fp8_graph_metadata_replay(query_len):
    case = _native_case(query_len, (8, 1, 256, 256))
    inputs = dict(q_seqlens=case.metadata.q_seqlens, kv_seqlens=case.metadata.kv_seqlens,
                  block_offsets=case.metadata.block_offsets)
    graph_meta = SimpleNamespace(decode_query_len=query_len, num_blocks=5, block_size=16)
    buffers = case.plan.make_cudagraph_buffers(graph_meta, inputs, case.ctx)
    case.plan.fill_cudagraph_buffers(graph_meta, inputs, case.ctx, buffers, case.metadata)
    metadata = case.metadata.kernel_metadata[0]
    assert metadata.decode_kv_seqlens is None
    pointer = metadata.scheduler_metadata.data_ptr()
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
        case.metadata.kv_seqlens.copy_(torch.tensor(lengths, device='cuda', dtype=torch.int32))
        case.metadata.block_offsets.copy_(case.metadata.block_offsets.roll(1, dims=0))
        case.q.normal_(std=.5)
        case.plan.fill_cudagraph_buffers(graph_meta, inputs, case.ctx, buffers, case.metadata)
        assert case.metadata.kernel_metadata[0].scheduler_metadata.data_ptr() == pointer
        graph.replay()
        _check(case, output)
        torch.testing.assert_close(output, _run(case), atol=0, rtol=0)


def test_native_fp8_metadata_is_distinct_from_b():
    case = _native_case()
    baseline = _make_case()
    plan = CudaStepMetaPlan.from_implementations([case.impl, baseline.impl])
    assert len(plan.attention_builders) == 2


def test_native_fp8_rejects_unsupported_cache():
    case = _native_case()
    case.metadata.quant_policy = case.ctx.kv_quant_policy = QuantPolicy.FP8_E5M2
    with pytest.raises(ValueError, match='E4M3'):
        case.plan.prepare(case.ctx, case.metadata, case.metadata)
    with pytest.raises(ValueError, match='E4M3'):
        _run(case)


@pytest.mark.parametrize('value', [0.0, -1.0, float('nan'), float('inf')])
def test_native_fp8_rejects_invalid_scale(value):
    with pytest.raises(ValueError, match='finite and positive'):
        FA3FP8QueryImpl(8, 64, num_kv_heads=1, query_scale=torch.tensor([value], device='cuda'))
