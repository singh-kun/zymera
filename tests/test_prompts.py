import json

import pytest

from zymera.prompts import PromptBuilder


def test_style_applied():
    built = PromptBuilder().build("a portrait of a synthetic persona", style="photorealistic")
    assert built.prompt.startswith("RAW photo, a portrait of a synthetic persona")
    assert "natural skin texture" in built.prompt
    assert "best quality" in built.prompt
    assert "bad anatomy" in built.negative
    assert "cartoon" in built.negative


def test_style_none_keeps_prompt_clean():
    built = PromptBuilder().build("just this", style="none")
    assert built.prompt.startswith("just this")
    assert "RAW photo" not in built.prompt


def test_no_enhance_is_raw():
    built = PromptBuilder().build("raw prompt", style="none", enhance=False)
    assert built.prompt == "raw prompt"
    assert built.negative == ""


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="Unknown style"):
        PromptBuilder().build("x", style="nope")


def test_styles_listed():
    styles = PromptBuilder().styles()
    assert "photorealistic" in styles
    assert "cinematic" in styles


def test_from_file_merges_custom_styles(tmp_path):
    spec = {"styles": {"custom": {"prefix": "custom style, ", "negative": "ugly"}}}
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(spec))
    builder = PromptBuilder.from_file(path)
    assert "custom" in builder.styles()
    assert "photorealistic" in builder.styles()  # defaults preserved
    built = builder.build("hello", style="custom")
    assert built.prompt.startswith("custom style, hello")
    assert "ugly" in built.negative


def test_from_file_missing_falls_back():
    builder = PromptBuilder.from_file("missing/prompts.json")
    assert "photorealistic" in builder.styles()
