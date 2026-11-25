# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _quant_fp8_query(Q, Scale, Out, stride_t: tl.constexpr, stride_h: tl.constexpr,
                    stride_d: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                    GROUP: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    group = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    head = group * GROUP + offsets // DIM
    dim = offsets % DIM
    values = tl.load(Q + token * stride_t + head * stride_h + dim * stride_d,
                     offsets < GROUP * DIM, other=0).to(tl.float32)
    scale = tl.load(Scale + group)
    values = tl.div_rn(values, scale)
    # Preserve NaNs instead of letting min/max turn invalid input into a finite endpoint.
    values = tl.where(values == values, tl.minimum(tl.maximum(values, -448.0), 448.0), values)
    tl.store(Out + token * HEADS * DIM + head * DIM + dim, values, offsets < GROUP * DIM)


def quant_fp8_query(query: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize Q with frozen descales shared by each GQA head group."""
    tokens, heads, dim = query.shape
    groups = scale.numel()
    assert query.is_cuda and scale.is_cuda and query.device == scale.device
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert scale.dtype == torch.float32 and scale.ndim == 1 and scale.is_contiguous()
    assert groups > 0 and heads % groups == 0
    output = torch.empty(query.shape, dtype=torch.float8_e4m3fn, device=query.device)
    if tokens:
        group_size = heads // groups
        _quant_fp8_query[(tokens, groups)](query, scale, output, *query.stride(), heads, dim,
                                         group_size, triton.next_power_of_2(group_size * dim))
    return output
