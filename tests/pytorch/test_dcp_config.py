# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy import PytorchEngineConfig
from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.pytorch.config import DistConfig, TPMode


@pytest.mark.parametrize('backend', ['ag_rs', 'a2a'])
@pytest.mark.parametrize('replicate', [False, True])
def test_dcp_options_reach_resolved_config(backend, replicate):
    parser = argparse.ArgumentParser()
    ArgumentHelper.dcp(parser)
    argv = ['--dcp', '2', '--dcp-comm-backend', backend]
    if replicate:
        argv.append('--dcp-replicate-q-proj')
    args = parser.parse_args(argv)
    config = DistConfig.from_engine_config(PytorchEngineConfig(tp=8, dp=2, ep=8, **vars(args)))
    assert config.attn_tp == 4 and config.mlp_tp == 8 and config.world_size == 8
    assert config.dcp_comm_backend == backend and config.dcp_replicate_q_proj == replicate


@pytest.mark.parametrize('kwargs', [
    {'dcp_comm_backend': 'invalid'},
    {'dcp_replicate_q_proj': True},
    {'tp': 2, 'dcp': 2, 'dcp_replicate_q_proj': True, 'adapters': {'test': '/unused'}},
])
def test_dcp_rejects_unsupported_options(kwargs):
    with pytest.raises(AssertionError):
        PytorchEngineConfig(**kwargs)


def test_dcp_divisibility_uses_resolved_attention_tp():
    # Input TP8 divides DCP4, but DP4 resolves attention TP to 2.
    engine = PytorchEngineConfig(tp=8, dp=4, ep=8, dcp=4)
    with pytest.raises(AssertionError, match='attn_tp 2 must be divisible by dcp 4'):
        DistConfig.from_engine_config(engine)
    with pytest.raises(AssertionError, match='attn_tp 2 must be divisible by dcp 4'):
        DistConfig(tp=8, dp=4, ep=8, dcp=4)


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        ({'tp': 8, 'dp': 2, 'ep': 8, 'dcp': 2},
         {'world_size': 8, 'attn_tp': 4, 'mlp_tp': 8, 'moe_tp': 1}),
        ({'tp': 8, 'dp': 2, 'ep': 8, 'dcp': 4},
         {'world_size': 8, 'attn_tp': 4, 'mlp_tp': 8, 'moe_tp': 1}),
        ({'tp': 4, 'dp': 2, 'ep': 4, 'dcp': 2},
         {'world_size': 4, 'attn_tp': 2, 'mlp_tp': 4, 'moe_tp': 1}),
    ])
def test_dcp_dp_ep_topology_resolves_expected_attention_groups(kwargs, expected):
    config = DistConfig(**kwargs)
    for field, value in expected.items():
        assert getattr(config, field) == value
    assert config.attn_tp % config.dcp == 0
    assert config.world_size % config.dp == 0
    assert config.world_size % config.ep == 0
    assert config.world_size % config.mlp_tp == 0
    assert config.world_size % config.moe_tp == 0


def test_dcp_dp_ep_layer_tp_modes_are_explicit():
    config = DistConfig(tp=8, dp=2, ep=8, dcp=4)
    assert config.get_tp_by_layer('attn') == (4, TPMode.DEFAULT)
    assert config.get_tp_by_layer('mlp') == (8, TPMode.DP_TP)
    assert config.get_tp_by_layer('moe') == (1, TPMode.DEFAULT)


def test_q_replication_rejects_quantized_or_lora_projection(monkeypatch):
    from lmdeploy.pytorch.nn import dcp as dcp_module
    from lmdeploy.pytorch.nn.linear.default import BaseLinear

    monkeypatch.setattr(dcp_module, 'get_dcp_world_rank', lambda: (2, 0))
    with pytest.raises(ValueError, match='unquantized BF16'):
        dcp_module.DCPQueryProjection(torch.nn.Linear(4, 4, bias=False))
    projection = BaseLinear(4, 4, bias=False, dtype=torch.bfloat16, device='cpu')
    projection.lora_adapters['test'] = torch.nn.Identity()
    with pytest.raises(ValueError, match='LoRA'):
        dcp_module.DCPQueryProjection(projection)


def test_q_replication_requires_explicit_model_opt_in(monkeypatch):
    from lmdeploy.pytorch.distributed import get_dist_manager
    from lmdeploy.pytorch.models import patch

    class SupportedModel:
        supports_dcp_q_replication = True

    class UnknownSubclass(SupportedModel):
        pass

    # Subclasses may replace the attention forward path. Inheriting a marker
    # must not silently enable an option that the subclass never consumes.
    monkeypatch.setattr(patch, '_get_model_class', lambda *args: UnknownSubclass)
    monkeypatch.setattr(get_dist_manager(), 'current_config',
                        lambda: SimpleNamespace(dcp_replicate_q_proj=True))
    with pytest.raises(ValueError, match='UnknownSubclass does not support'):
        patch.build_model_from_hf_config(SimpleNamespace(), ctx_mgr=object())


@pytest.mark.parametrize('sparse', [False, True])
def test_replicated_query_skips_gather_and_preserves_output_heads(monkeypatch, sparse):
    from lmdeploy.pytorch.backends.cuda.attention import mla, sparse_mla
    from lmdeploy.pytorch.kernels.cuda import dcp as kernels

    cls = sparse_mla.FlashMLASparseImpl if sparse else mla.FlashMLAImpl
    impl = object.__new__(cls)
    impl.num_heads = 5
    impl.dcp_world_size = 2
    impl.dcp_rank = 0
    impl.dcp_manager = Mock()
    query = torch.randn(3, 10, 8)
    output = torch.randn(3, 10, 8)
    lse = torch.randn(3, 10)
    metadata = SimpleNamespace(q_seqlens=torch.ones(3), dcp_local_kv_seqlens=torch.ones(3))
    gather = Mock(side_effect=AssertionError('Replicated query must not be gathered'))
    impl.dcp_manager.gather_query = gather
    decoder = Mock(return_value=(output, lse))
    indices = None
    if sparse:
        impl._decoding_sparse = decoder
        indices = torch.ones(3, 2, dtype=torch.int32)
        monkeypatch.setattr(kernels, 'filter_and_compact_dcp_indices',
                            lambda *args, **kwargs: (indices, torch.ones(3)))
    else:
        impl._decoding_paged = decoder
    merged = torch.randn(3, 5, 8)
    merge = Mock(return_value=merged)
    impl.dcp_manager.combine = merge
    actual = impl._forward_decoding(query, None, metadata, indices, query_is_dcp_replicated=True)
    assert actual is merged
    assert decoder.call_args.args[0] is query
    assert merge.call_args.args[:2] == (output, lse)
    gather.assert_not_called()
    with pytest.raises(ValueError, match='heads'):
        impl._prepare_dcp_query(query[:, :5], True)
