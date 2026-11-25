# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import replace

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.kernels.cuda.quant_fp8_query import quant_fp8_query

from .fa3 import FA3Impl


class FA3FP8QueryImpl(FA3Impl):
    """Experimental native FP8 multi-token decode with frozen Q descales.

    Construct before the step metadata plan, after loading independently calibrated scales. This implementation is
    deliberately not selected by the public backend dispatcher. Single-token decode and prefill retain FA3Impl behavior.
    """

    def __init__(self, *args, query_scale: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.causal or self._normalize_sliding_window(self.sliding_window)[0] == 0:
            raise ValueError('FP8 Q experiment requires causal attention and a nonzero sliding window.')
        if (not query_scale.is_cuda or query_scale.dtype != torch.float32
                or query_scale.shape != (self.num_kv_heads,)):
            raise ValueError('Q descales must be CUDA FP32 with one value per local KV head.')
        # Construction is outside CUDA graph capture. Validate once, never synchronize in forward.
        if not bool(torch.isfinite(query_scale).all() & (query_scale > 0).all()):
            raise ValueError('Q descales must be finite and positive.')
        self.query_scale = query_scale.detach().clone().contiguous()

    def get_step_metadata_provider(self):
        """Use native FP8 scheduling instead of the expanded Triton decode
        rows."""
        return replace(super().get_step_metadata_provider(), fp8_query=True)

    def _decoding_fp8_speculative(self, query, k_cache, v_cache, attn_metadata, max_q_seqlen,
                                   k_scales_zeros, v_scales_zeros):
        if (attn_metadata.quant_policy != QuantPolicy.FP8 or k_cache.dtype != torch.float8_e4m3fn
                or v_cache.dtype != torch.float8_e4m3fn):
            raise ValueError('Native FP8 FA3 requires E4M3 KV cache.')
        if k_scales_zeros is None or v_scales_zeros is None:
            raise ValueError('Native FP8 FA3 requires K and V descales.')
        # Existing regular FP8 KV stores use scalar descales, expanded without copying the cache.
        if k_scales_zeros.numel() != 1 or v_scales_zeros.numel() != 1:
            raise ValueError('The FP8 Q experiment requires scalar K and V descales.')
        batch = attn_metadata.q_seqlens.size(0)
        q_fp8 = quant_fp8_query(query, self.query_scale).unflatten(0, (batch, max_q_seqlen))
        output = self.flash_attn_with_kvcache_v3(
            q_fp8, k_cache, v_cache,
            cache_seqlens=attn_metadata.kv_seqlens.to(torch.int32),
            page_table=attn_metadata.block_offsets,
            scheduler_metadata=self._get_scheduler_metadata(attn_metadata),
            max_seqlen_q=max_q_seqlen,
            q_descale=self.query_scale[None, :].expand(batch, -1),
            k_descale=k_scales_zeros.reshape(1, 1).expand(batch, self.num_kv_heads),
            v_descale=v_scales_zeros.reshape(1, 1).expand(batch, self.num_kv_heads),
            softmax_scale=self.scale, causal=self.causal,
            window_size=self._normalize_sliding_window(self.sliding_window), softcap=self.logit_softcapping,
        )
        return output.to(query.dtype)
