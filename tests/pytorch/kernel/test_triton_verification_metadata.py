from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.cuda.attention.triton_verification import (
    TritonVarlenVerificationMetaBuilder,
)


def _make_graph_inputs():
    return {
        'block_offsets': torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        'q_start_loc': torch.tensor([0, 3], dtype=torch.int32),
        'q_seqlens': torch.tensor([3, 3], dtype=torch.int32),
        'kv_seqlens': torch.tensor([10, 12], dtype=torch.int32),
        'cu_seqlens_q': torch.tensor([0, 3, 6], dtype=torch.int32),
        'cu_seqlens_k': torch.tensor([0, 10, 22], dtype=torch.int32),
    }


def test_single_token_decode_keeps_legacy_metadata():
    builder = TritonVarlenVerificationMetaBuilder()
    graph_meta = SimpleNamespace(decode_query_len=1)

    metadata = builder.fill_cudagraph_buffer(graph_meta, {}, SimpleNamespace(), None)

    assert metadata is None


def test_multi_token_verification_builds_varlen_graph_metadata():
    builder = TritonVarlenVerificationMetaBuilder()
    graph_meta = SimpleNamespace(decode_query_len=3, num_blocks=8, block_size=64, max_batchs=2)
    step_context = SimpleNamespace(kv_quant_policy=0)
    input_buffers = _make_graph_inputs()

    metadata = builder.fill_cudagraph_buffer(graph_meta, input_buffers, step_context, None)

    assert not metadata.is_decoding
    assert metadata.max_q_seqlen == 3
    assert metadata.max_kv_seqlen == 512
    assert metadata.kv_flatten_size == 1024
    assert metadata.kv_start_loc.data_ptr() == input_buffers['cu_seqlens_k'][:-1].data_ptr()
    assert metadata.block_offsets is input_buffers['block_offsets']
