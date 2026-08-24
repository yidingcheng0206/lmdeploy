from types import SimpleNamespace

import torch

from lmdeploy.pytorch.nn.linear.blocked_fp8 import QKVBlockedF8Linear


def test_qkv_blocked_fp8_splits_partial_scale_blocks():
    """Each logical Q/K/V projection keeps its partial scale block."""
    linear = SimpleNamespace(
        qkv_split_section=[256, 192, 128],
        block_size=128,
        fp8_dtype=torch.float8_e4m3fn,
    )
    scales = torch.arange(5, dtype=torch.float32).unsqueeze(1)

    q_scale, k_scale, v_scale = QKVBlockedF8Linear.weight_spliter(linear, scales)

    assert [part.size(0) for part in (q_scale, k_scale, v_scale)] == [2, 2, 1]
    torch.testing.assert_close(torch.cat((q_scale, k_scale, v_scale)), scales)
