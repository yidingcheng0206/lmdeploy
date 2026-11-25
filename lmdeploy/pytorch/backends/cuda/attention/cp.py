# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA DCP query gathering and attention-output merging."""

import torch


class DCPAttentionWorkspace:
    """Reuse A2A buffers per CUDA stream, including graph replay.

    Grow geometrically and keep previous allocations alive for captured
    graphs. Each send/receive buffer's retained capacity stays below twice
    its largest allocation per stream, rather than growing with every batch
    size seen during decode.
    """

    def __init__(self):
        self._buffers = {}

    def get(self, local_output: torch.Tensor, dcp_world_size: int):
        from lmdeploy.pytorch.kernels.cuda.dcp import dcp_attention_packet_shape

        shape = dcp_attention_packet_shape(local_output, dcp_world_size)
        stream = torch.cuda.current_stream(local_output.device).cuda_stream
        key = (local_output.device, stream)
        numel = shape[0] * shape[1] * shape[2] * shape[3]
        history = self._buffers.setdefault(key, [])
        if not history or history[-1][0].numel() < numel:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Prepare DCP A2A workspace on the capture stream during warmup')
            capacity = 1 << max(0, numel - 1).bit_length()
            history.append(tuple(torch.empty(capacity, dtype=torch.int32, device=local_output.device)
                                  for _ in range(2)))
        return tuple(buffer[:numel].view(shape) for buffer in history[-1])

    @property
    def allocated_bytes(self) -> int:
        """Live send/receive bytes, including generations retained by graphs."""
        return sum(buffer.numel() * buffer.element_size()
                   for history in self._buffers.values() for pair in history for buffer in pair)

    def close(self):
        """Release after graph retirement and queued users finish."""
        for device in {key[0] for key in self._buffers}:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Cannot release DCP workspace during graph capture')
            torch.cuda.synchronize(device)
        self._buffers.clear()


def gather_dcp_query(query: torch.Tensor, *, dcp_world_size: int) -> torch.Tensor:
    """Gather [tokens, local_heads, dim] queries along the head axis."""
    if dcp_world_size == 1:
        return query
    from lmdeploy.pytorch.distributed import all_gather_into_tensor

    # Keep the existing packing path for strided queries. Direct gather
    # saves a copy only when the token-major input is already contiguous.
    if not query.is_contiguous():
        transposed = query.transpose(0, 1).contiguous()
        gathered = transposed.new_empty(
            dcp_world_size * transposed.size(0), *transposed.shape[1:])
        all_gather_into_tensor(gathered, transposed, group='dcp')
        return gathered.transpose(0, 1).contiguous()

    gathered = query.new_empty(dcp_world_size * query.size(0), *query.shape[1:])
    all_gather_into_tensor(gathered, query, group='dcp')
    # Gather token-major queries directly, then join rank-local heads.
    # Contiguous inputs need only this final layout conversion.
    gathered = gathered.view(dcp_world_size, *query.shape)
    return gathered.transpose(0, 1).reshape(query.size(0), -1, query.size(2)).contiguous()


def merge_dcp_attention(local_output: torch.Tensor,
                        local_lse: torch.Tensor,
                        valid_counts: torch.Tensor,
                        *,
                        dcp_world_rank: tuple[int, int],
                        comm_backend: str = 'ag_rs',
                        workspace: DCPAttentionWorkspace | None = None,
                        communicator=None,
                        lse_workspace=None) -> torch.Tensor:
    """Merge normalized CUDA shard outputs and scatter heads back to each rank.

    Inputs are [tokens, gathered_heads, value_dim] outputs and natural-log [tokens, gathered_heads] LSE. Empty rows
    contribute zero. LSE and correction arithmetic use FP32. ``ag_rs`` reduces
    contributions in the output dtype; ``a2a`` combines received states in
    FP32 before the final output cast.
    """
    dcp_world_size, dcp_rank = dcp_world_rank
    if comm_backend not in ('ag_rs', 'a2a'):
        raise ValueError(f'Unsupported DCP communication backend: {comm_backend}')
    if dcp_world_size == 1:
        return local_output
    if local_output.size(1) % dcp_world_size:
        raise ValueError('DCP output heads must be divisible by dcp')
    if comm_backend == 'a2a' and workspace is None:
        raise ValueError('DCP A2A requires a prepared workspace')
    # Token counts agree inside each DCP group. Zero *KV* counts with a
    # nonempty query batch still participate in communication and sanitization.
    if local_output.size(0) == 0:
        return local_output.new_empty(0, local_output.size(1) // dcp_world_size, local_output.size(2))
    if comm_backend == 'a2a':
        from lmdeploy.pytorch.distributed import all_to_all_single
        from lmdeploy.pytorch.kernels.cuda.dcp import combine_dcp_attention, pack_dcp_attention

        send, receive = workspace.get(local_output, dcp_world_size)
        send = pack_dcp_attention(local_output, local_lse, valid_counts,
                                  dcp_world_size=dcp_world_size, out=send)
        if communicator is None:
            all_to_all_single(receive, send, group='dcp')
        else:
            communicator.all_to_all_single(receive, send)
        return combine_dcp_attention(receive, head_dim=local_output.size(-1), dtype=local_output.dtype)
    from lmdeploy.pytorch.distributed import all_gather_into_tensor, reduce_scatter_tensor
    from lmdeploy.pytorch.kernels.cuda.dcp import correct_dcp_attention_output, sanitize_dcp_lse

    local_lse = sanitize_dcp_lse(local_lse, valid_counts)
    if communicator is None:
        gathered_lse = local_lse.new_empty(
            dcp_world_size * local_lse.size(0), local_lse.size(1))
        all_gather_into_tensor(gathered_lse, local_lse, group='dcp')
    else:
        gathered_lse = communicator.all_gather(local_lse, dim=0, workspace=lse_workspace, copy_output=False)
    gathered_lse = gathered_lse.view(dcp_world_size, *local_lse.shape)
    contribution = correct_dcp_attention_output(
        local_output, gathered_lse, dcp_rank=dcp_rank)

    num_heads = contribution.size(0)
    # The correction kernel writes [heads, tokens, dim] directly so the
    # head-sharded reduce-scatter needs no separate transpose/copy.
    assert num_heads % dcp_world_size == 0
    local_heads = num_heads // dcp_world_size
    scattered_output = contribution.new_empty(local_heads,
                                              contribution.size(1),
                                              contribution.size(2))
    if communicator is None:
        reduce_scatter_tensor(scattered_output, contribution, group='dcp')
    else:
        communicator.reduce_scatter_tensor(scattered_output, contribution)
    return scattered_output.transpose(0, 1)
