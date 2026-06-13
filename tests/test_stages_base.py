"""Tests for stage base helpers that don't need torch/GPU."""

import pytest

from zymera.stages.base import build_quantization_config


def test_quantization_none_when_missing_or_disabled():
    assert build_quantization_config(None, None) is None
    assert build_quantization_config({}, None) is None
    assert build_quantization_config({"enabled": False}, None) is None


def test_quantization_unknown_backend_rejected():
    pytest.importorskip("bitsandbytes")
    with pytest.raises(ValueError, match="Unknown quantization backend"):
        build_quantization_config({"enabled": True, "backend": "gguf_q8"}, None)
