# Copyright (c) OpenMMLab. All rights reserved.
"""Context-owned DCP collectives and workspaces.

The ownership and gather interfaces build on InternLM/lmdeploy PR #4995
(c0fbb225). Native collectives and the existing AG/RS and A2A arithmetic are
kept here; symmetric-memory transport is a separate provider change.
"""
import torch

from lmdeploy.pytorch.backends.communicator import DeviceCommunicator
from lmdeploy.pytorch.backends.cp_utils import DCPPrefixChunk


class DCPManager:
    """Share resources across layers of one distributed context.

    The group is bound at construction, never looked up during a collective.
    Query heads cover this DCP subgroup, not necessarily all attention-TP heads.
    A2A scratch is isolated by device/stream. Graphs sharing that scratch must
    be replayed serially on their capture execution stream: replay does not
    run the Python workspace lookup again.
    """

    def __init__(self, group, world_size: int | None = None, *, comm_backend: str = 'ag_rs'):
        from .attention.cp import DCPAttentionWorkspace

        if world_size is None:
            world_size = torch.distributed.get_world_size(group.gpu_group)
        if world_size < 1 or (world_size > 1 and group.gpu_group is None):
            raise ValueError('DCP requires an explicit process group when dcp > 1')
        if comm_backend not in ('ag_rs', 'a2a'):
            raise ValueError(f'Unsupported DCP communication backend: {comm_backend}')
        self.world_size = world_size
        self.rank = group.rank
        self.communicator = group.communicator or DeviceCommunicator(group.gpu_group)
        self.comm_backend = comm_backend
        self.attention_workspace = DCPAttentionWorkspace() if comm_backend == 'a2a' else None
        self._gather_workspaces = {}

    def _prepare_gather(self, purpose: str, width: int, dtype: torch.dtype, dim: int,
                        capacity_bytes: int = 64 * 1024 * 1024):
        key = (purpose, width, dtype, dim)
        if self.world_size > 1 and key not in self._gather_workspaces:
            self._gather_workspaces[key] = self.communicator.create_all_gather_workspace(
                width, torch.device('cuda'), dtype, dim=dim, capacity_bytes=capacity_bytes)

    def prepare_attention(self, num_heads: int, head_size: int):
        """Register each distinct layer layout; native gather needs no arena."""
        self._prepare_gather('query', num_heads * head_size * self.world_size, torch.bfloat16, -1)
        self._prepare_gather('lse', num_heads * self.world_size, torch.float32, 0, 1024 * 1024)

    def prepare_candidate_gather(self, topk: int):
        self._prepare_gather('candidate', topk * 2, torch.float32, 0)

    def prepare_prefix_gather(self, *kv_widths: int):
        for width in kv_widths:
            self._prepare_gather('prefix', width, torch.bfloat16, 0)

    def _workspace(self, purpose: str, width: int, dtype: torch.dtype, dim: int):
        return self._gather_workspaces.get((purpose, width, dtype, dim))

    def prepare(self):
        """Re-prepare registered providers after reset, before graph warmup."""
        for workspace in self._gather_workspaces.values():
            if workspace is not None:
                workspace.prepare()

    def gather_query(self, query: torch.Tensor) -> torch.Tensor:
        """Gather contiguous rank-local head slices without numeric casts."""
        if self.world_size == 1:
            return query
        tokens, heads, dim = query.shape
        width = heads * dim * self.world_size
        gathered = self.communicator.all_gather(
            query.reshape(tokens, heads * dim), dim=-1,
            workspace=self._workspace('query', width, query.dtype, -1), copy_output=False)
        return gathered.view(tokens, heads * self.world_size, dim)

    def gather_candidates(self, packed: torch.Tensor) -> torch.Tensor:
        """Transport packed score/ID words exactly; top-k stays with the caller."""
        if self.world_size == 1:
            return packed.unsqueeze(0)
        flat = packed.flatten(1)
        output = self.communicator.all_gather(
            flat, dim=0, workspace=self._workspace('candidate', flat.size(1), flat.dtype, 0),
            copy_output=False)
        return output.view(self.world_size, *packed.shape)

    def gather_prefix(self, local_kv: torch.Tensor, chunk: DCPPrefixChunk,
                      *, zero_padding: bool = False) -> torch.Tensor:
        """Gather and reorder into owned storage before the next arena reuse."""
        from lmdeploy.pytorch.kernels.cuda.dcp import reorder_dcp_prefill_kv

        flat = local_kv.flatten(1)
        gathered = flat
        if self.world_size > 1:
            gathered = self.communicator.all_gather(
                flat, dim=0, workspace=self._workspace('prefix', flat.size(1), flat.dtype, 0),
                copy_output=False)
        gathered = gathered.view(-1, *local_kv.shape[1:])
        output = torch.zeros_like(gathered) if zero_padding else torch.empty_like(gathered)
        reorder_dcp_prefill_kv(gathered, output, chunk_kv_seqlens=chunk.kv_seqlens,
                               kv_start_loc=chunk.cu_seqlens[:-1], local_lens=chunk.local_kv_seqlens)
        return output

    def combine(self, local_output: torch.Tensor, local_lse: torch.Tensor,
                valid_counts: torch.Tensor) -> torch.Tensor:
        """Run the selected, unchanged arithmetic and return owned local heads."""
        from .attention.cp import merge_dcp_attention

        return merge_dcp_attention(
            local_output, local_lse, valid_counts, dcp_world_rank=(self.world_size, self.rank),
            comm_backend=self.comm_backend, workspace=self.attention_workspace,
            communicator=self.communicator,
            lse_workspace=self._workspace('lse', local_lse.size(1), local_lse.dtype, 0))

    def reset(self):
        """Retire buffers only after ALL context-owned graphs have retired."""
        if self.attention_workspace is not None:
            self.attention_workspace.close()
        for workspace in self._gather_workspaces.values():
            if workspace is not None:
                workspace.close()

    def close(self):
        self.reset()
        self._gather_workspaces.clear()


def get_dcp_manager() -> DCPManager:
    """Get the context-owned manager during layer construction."""
    from lmdeploy.pytorch.distributed import get_dist_manager

    context = get_dist_manager().current_context()
    if context.dcp_manager is None:
        config = context.dist_config
        context.dcp_manager = DCPManager(context.dcp_group, config.dcp, comm_backend=config.dcp_comm_backend)
    return context.dcp_manager
