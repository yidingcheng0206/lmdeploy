import sys
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.backends.cuda.attention.fa3_capabilities import fa3_build_supports_head_dims


@pytest.fixture
def fa3_config(monkeypatch):

    def _set(flags):
        config = SimpleNamespace(CONFIG={'build_flags': flags})
        monkeypatch.setitem(sys.modules, 'flash_attn_config', config)

    return _set


def test_missing_build_metadata_accepts_only_symmetric_dims(monkeypatch):
    monkeypatch.setitem(sys.modules, 'flash_attn_config', None)

    assert fa3_build_supports_head_dims(128)
    assert not fa3_build_supports_head_dims(192, 128)


def test_asymmetric_dims_require_both_template_families(fa3_config):
    fa3_config({
        'FLASHATTENTION_DISABLE_HDIM192': False,
        'FLASH_ATTENTION_DISABLE_HDIMDIFF192': False,
    })
    assert fa3_build_supports_head_dims(192, 128)

    fa3_config({
        'FLASHATTENTION_DISABLE_HDIM192': False,
        'FLASH_ATTENTION_DISABLE_HDIMDIFF192': True,
    })
    assert not fa3_build_supports_head_dims(192, 128)


def test_head_dim_template_must_be_enabled(fa3_config):
    fa3_config({'FLASHATTENTION_DISABLE_HDIM192': True})

    assert not fa3_build_supports_head_dims(192, 192)
    assert not fa3_build_supports_head_dims(192, 128)


def test_unsupported_head_dim_is_rejected(fa3_config):
    fa3_config({})

    assert not fa3_build_supports_head_dims(320)
