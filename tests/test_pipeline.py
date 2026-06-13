import pytest

from zymera.config import Config
from zymera.pipeline import PHASES, Pipeline


def _pipeline(tmp_path) -> Pipeline:
    cfg = Config()
    cfg.set("paths.identities_dir", str(tmp_path / "identities"))
    cfg.set("paths.outputs_dir", str(tmp_path / "outputs"))
    return Pipeline(cfg)


def test_phase_aliases_cover_legacy_and_semantic_names():
    assert PHASES["phase1"] == "image"
    assert PHASES["phase2"] == "identity_image"
    assert PHASES["phase3"] == "video"
    assert PHASES["phase4"] == "talking_video"
    assert PHASES["talking-video"] == "talking_video"


def test_unknown_phase_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown phase"):
        _pipeline(tmp_path).run("phase99", "prompt")


def test_phase4_requires_text(tmp_path):
    with pytest.raises(ValueError, match="--text"):
        _pipeline(tmp_path).run("phase4", "prompt", text=None)


def test_phase2_requires_identity(tmp_path):
    with pytest.raises(ValueError, match="--identity"):
        _pipeline(tmp_path).run("phase2", "prompt")


def test_phase2_requires_reference_image(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.identities.create("no_refs")
    with pytest.raises(FileNotFoundError, match="no usable reference image"):
        pipeline.run("phase2", "prompt", identity="no_refs")


def test_unknown_identity_listed(tmp_path):
    with pytest.raises(KeyError, match="not found"):
        _pipeline(tmp_path).run("phase2", "prompt", identity="ghost")


def test_naming_explicit_output():
    basename, ext = Pipeline._naming("image", "my_file.png", "id", 1)
    assert (basename, ext) == ("my_file", ".png")


def test_naming_auto_contains_seed_and_kind():
    basename, ext = Pipeline._naming("video", None, "persona", 123)
    assert basename.startswith("video_persona_")
    assert basename.endswith("_123")
    assert ext == ""
