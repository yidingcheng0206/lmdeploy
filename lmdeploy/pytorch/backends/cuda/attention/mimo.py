# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Hashable
from dataclasses import dataclass

from ..step_metadata import CudaAttentionMetaBuilder
from .default import TritonAttentionMetadata


@dataclass(frozen=True)
class MiMoVerificationMetaBuilder(CudaAttentionMetaBuilder[None, TritonAttentionMetadata | None]):
    """Build rollback-safe MiMo target-verification graph metadata.

    The speculative scheduler classifies verification as decoding, while its q>1 attention operation is a varlen paged-
    cache extend.  This provider is injected only by MiMo's Full and paged-SWA target attention operators.
    """

    @property
    def key(self) -> Hashable:
        """Return a stable provider key shared by all MiMo attention ops."""
        return type(self)

    def build(self, step_context, sequence_metadata) -> None:
        """Use eager metadata derivation outside CUDA Graph replay."""
        # Eager verification already derives exact lengths in MiMoV2Attention.
        return None

    def apply_legacy_metadata(self, attn_metadata, metadata: TritonAttentionMetadata | None) -> None:
        """Copy replay-safe varlen fields into legacy attention metadata."""
        if metadata is None:
            return
        attn_metadata.is_decoding = metadata.is_decoding
        attn_metadata.block_offsets = metadata.block_offsets
        attn_metadata.q_start_loc = metadata.q_start_loc
        attn_metadata.q_seqlens = metadata.q_seqlens
        attn_metadata.kv_start_loc = metadata.kv_start_loc
        attn_metadata.kv_seqlens = metadata.kv_seqlens
        attn_metadata.kv_flatten_size = metadata.kv_flatten_size
        attn_metadata.cu_seqlens_q = metadata.cu_seqlens_q
        attn_metadata.cu_seqlens_k = metadata.cu_seqlens_k
        attn_metadata.max_q_seqlen = metadata.max_q_seqlen
        attn_metadata.max_kv_seqlen = metadata.max_kv_seqlen

    def make_cudagraph_buffer(self, graph_meta, input_buffers, step_context) -> None:
        """Declare that this provider needs no private CUDA Graph buffer."""
        return None

    def fill_cudagraph_buffer(self, graph_meta, input_buffers, step_context,
                              buffer: None) -> TritonAttentionMetadata | None:
        """Build varlen verification metadata from graph-owned buffers."""
        del buffer
        if graph_meta.decode_query_len <= 1:
            return None

        max_kv_seqlen = graph_meta.num_blocks * graph_meta.block_size
        return TritonAttentionMetadata(
            is_decoding=False,
            block_offsets=input_buffers['block_offsets'],
            q_start_loc=input_buffers['q_start_loc'],
            q_seqlens=input_buffers['q_seqlens'],
            kv_start_loc=input_buffers['cu_seqlens_k'][:-1],
            kv_seqlens=input_buffers['kv_seqlens'],
            quant_policy=step_context.kv_quant_policy,
            kv_flatten_size=graph_meta.max_batchs * max_kv_seqlen,
            cu_seqlens_q=input_buffers['cu_seqlens_q'],
            cu_seqlens_k=input_buffers['cu_seqlens_k'],
            max_q_seqlen=graph_meta.decode_query_len,
            max_kv_seqlen=max_kv_seqlen,
        )
