# Copyright (c) OpenMMLab. All rights reserved.
"""DCP-local replicas of the MLA query and absorbed key projections."""
import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.distributed import all_gather_into_tensor, get_dcp_world_rank

from .linear.default import BaseLinear


class DCPQueryProjection(nn.Module):
    """Keep BF16 decode replicas while local parameters remain prefill views.

    Prepare after linear backend weight updates and MLA KV-B absorption.
    Only DCP peers exchange weights; attention TP and output heads do not
    change. Calling prepare again refreshes replicas after weight updates;
    the engine must invalidate captured graphs, as its weight-update path
    already does.
    """

    def __init__(self, projection: nn.Module):
        super().__init__()
        self.world_size, self.rank = get_dcp_world_rank()
        if self.world_size <= 1:
            raise ValueError('DCP query projection replication requires dcp > 1')
        self._validate_projection(projection)
        self.register_buffer('query_weight', None, persistent=False)
        self.register_buffer('key_weight', None, persistent=False)

    @staticmethod
    def _validate_projection(projection):
        # Exact type is intentional: subclasses may change weight layouts or
        # forward semantics that the replicated F.linear path would bypass.
        if type(projection) is not BaseLinear or projection.weight.dtype != torch.bfloat16:
            raise ValueError('DCP query projection replication currently requires an unquantized BF16 Q projection')
        if projection.bias is not None or not projection.colwise or projection.all_reduce:
            raise ValueError('DCP query projection replication requires a bias-free column projection')
        if projection.lora_adapters:
            raise ValueError('DCP query projection replication does not support LoRA')

    def prepare(self, projection: nn.Module, key_projection: nn.Module):
        """Gather effective weights once and reuse replica slices locally."""
        self._validate_projection(projection)
        if key_projection.weight.dtype != torch.bfloat16:
            raise ValueError('DCP query projection replication requires BF16 absorbed key weights')
        query = projection.get_unquantized_weight(torch.bfloat16).contiguous()
        key = key_projection.weight.contiguous()
        query_replica = query.new_empty(self.world_size * query.size(0), *query.shape[1:])
        key_replica = key.new_empty(self.world_size * key.size(0), *key.shape[1:])
        all_gather_into_tensor(query_replica, query, group='dcp')
        all_gather_into_tensor(key_replica, key, group='dcp')
        self.query_weight = query_replica
        self.key_weight = key_replica
        # Preserve Parameter objects/loaders and their original local shapes.
        # No duplicate copy of the local shards is needed after initialization.
        projection.weight.data = query_replica.chunk(self.world_size, dim=0)[self.rank]
        key_projection.weight.data = key_replica.chunk(self.world_size, dim=0)[self.rank]

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.query_weight is None:
            raise RuntimeError('DCP query weights must be prepared before decode')
        return F.linear(hidden_states, self.query_weight)

    def absorb(self, query: torch.Tensor, output: torch.Tensor) -> None:
        torch.bmm(query.transpose(0, 1), self.key_weight, out=output.transpose(0, 1))
