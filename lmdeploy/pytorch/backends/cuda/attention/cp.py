# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA DCP query gathering and attention-output merging."""

import torch

from lmdeploy.pytorch.backends.cp_utils import DCPPrefixChunk, get_dcp_local_causal_seq_lens

from ..dcp import DCPManager, get_dcp_manager
from .default import TritonAttentionImpl, TritonAttentionMetaBuilder, TritonAttentionMetadata


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


class DCPAttentionImpl(TritonAttentionImpl):
    """Keep TP query heads local in prefill and gather them for decode.

    Model configuration ensures that every rank in a DCP group holds the same KV head. Cached prefill gathers bounded
    K/V chunks; decode evaluates each context shard and reduce-scatters softmax-corrected outputs to the TP heads.
    """

    def __init__(self, num_heads: int, head_size: int, *, use_fa3: bool = False, **kwargs):
        super().__init__(num_heads=num_heads, head_size=head_size, **kwargs)
        self.use_fa3 = use_fa3
        assert not self.alibi, 'GQA DCP does not support ALiBi'
        if use_fa3:
            from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
            self._fa3_prefill = flash_attn_varlen_func
        self.dcp_manager = get_dcp_manager()
        self.dcp_manager.prepare_attention(self.num_heads, self.head_size)
        self.dcp_manager.prepare_prefix_gather(self.num_kv_heads * self.head_size,
                                               self.num_kv_heads * self.v_head_size)

    def get_step_metadata_provider(self):
        """Use common DCP lengths without a separate kernel scheduler."""
        return TritonAttentionMetaBuilder()

    def supports_piecewise_cuda_graph(self) -> bool:
        return True

    def _decode(self, query: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                metadata: TritonAttentionMetadata, k_scales_zeros: torch.Tensor = None,
                v_scales_zeros: torch.Tensor = None) -> torch.Tensor:
        """Evaluate interleaved shards and restore rank-local query heads."""
        query = self.dcp_manager.gather_query(query)
        query_len = query.size(0) // metadata.q_seqlens.numel()
        local_lens = metadata.dcp_local_kv_seqlens
        blocks = metadata.block_offsets
        if query_len > 1:
            # A global query step need not advance this shard's causal length.
            local_lens = get_dcp_local_causal_seq_lens(
                metadata.kv_seqlens, query_len, (self.dcp_world_size, self.dcp_rank))
            blocks = blocks.repeat_interleave(query_len, dim=0)
        output, lse = self.paged_attention_fwd(
            query, k_cache, v_cache, cache_seqlens=local_lens,
            page_table=blocks, max_seqlen_q=1, softmax_scale=self.scale,
            softcap=self.logit_softcapping, return_lse=True, quant_policy=metadata.quant_policy,
            k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
        return self.dcp_manager.combine(output, lse, valid_counts=local_lens)

    def _prefill_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *,
                           cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
                           causal: bool, return_lse: bool):
        """Evaluate one prefill partition with optional natural-log LSE."""
        kwargs = dict(cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                      max_seqlen_q=max_q, max_seqlen_k=max_k,
                      softmax_scale=self.scale, softcap=max(0.0, self.logit_softcapping), causal=causal)
        if self.use_fa3:
            return self._fa3_prefill(query, key, value, return_lse=return_lse, **kwargs)
        return self.flash_attention_fwd(query, key, value, kv_layout='shd', return_lse=return_lse, **kwargs)

    def _gather_prefix(self, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       metadata: TritonAttentionMetadata, chunk: DCPPrefixChunk,
                       dtype: torch.dtype, k_scales_zeros: torch.Tensor = None,
                       v_scales_zeros: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather and reorder separate K/V caches within the chunk budget."""
        local_k, local_v = self.flatten_kv_cache(
            k_cache, v_cache, chunk.local_kv_seqlens[self.dcp_rank],
            metadata.block_offsets[:, chunk.local_block_slice(k_cache.size(1))],
            start_loc=chunk.local_cu_seqlens[:-1], out_size=chunk.local_capacity,
            out_dtype=dtype, flatten_kv_layout='shd', quant_policy=metadata.quant_policy,
            k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
        return (self.dcp_manager.gather_prefix(local_k, chunk, zero_padding=True),
                self.dcp_manager.gather_prefix(local_v, chunk, zero_padding=True))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor, attn_metadata: TritonAttentionMetadata,
                k_scales_zeros: torch.Tensor = None, v_scales_zeros: torch.Tensor = None,
                learnable_sink: torch.Tensor = None, inplace: bool = True, **kwargs) -> torch.Tensor:
        """Write owned KV tokens, then evaluate GQA prefill or decode."""
        assert learnable_sink is None, 'GQA DCP does not support attention sinks'
        max_q = (query.size(0) // attn_metadata.q_seqlens.numel()
                 if attn_metadata.is_decoding else self._get_max_q_seqlen(query, attn_metadata))
        if key is not None and value is not None:
            self._fill_kv_cache_impl(key, value, k_cache, v_cache, attn_metadata, max_q,
                                     k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
        if attn_metadata.is_decoding:
            return self._decode(query, k_cache, v_cache, attn_metadata,
                                k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)

        assert key is not None and value is not None, 'GQA DCP prefill requires current K/V'
        chunks = attn_metadata.dcp_prefix_chunks
        result = self._prefill_attention(
            query, key, value, cu_q=attn_metadata.cu_seqlens_q, cu_k=attn_metadata.cu_seqlens_q,
            max_q=max_q, max_k=max_q, causal=True, return_lse=bool(chunks))
        if not chunks:
            return result
        from lmdeploy.pytorch.kernels.cuda.dcp import merge_attention_states
        output, lse = result
        del result
        for chunk in chunks:
            context_k, context_v = self._gather_prefix(k_cache, v_cache, attn_metadata, chunk, query.dtype,
                                                      k_scales_zeros=k_scales_zeros, v_scales_zeros=v_scales_zeros)
            chunk_output, chunk_lse = self._prefill_attention(
                query, context_k, context_v, cu_q=attn_metadata.cu_seqlens_q, cu_k=chunk.cu_seqlens,
                max_q=max_q, max_k=chunk.size, causal=False, return_lse=True)
            output, lse = merge_attention_states(output, lse, chunk_output, chunk_lse)
            del context_k, context_v, chunk_output, chunk_lse
        return output
