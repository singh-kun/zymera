import json

import pytest

from zymera.config import Config, load_preset, parse_override


def test_defaults_present():
    cfg = Config()
    assert cfg.get("image.model") == "stabilityai/stable-diffusion-xl-base-1.0"
    assert cfg.get("video.num_frames") == 16
    assert cfg.get("runtime.device") == "auto"


def test_dot_path_get_set():
    cfg = Config()
    cfg.set("image.steps", 50)
    assert cfg.get("image.steps") == 50
    assert cfg.get("does.not.exist", "fallback") == "fallback"


def test_merge_is_deep():
    cfg = Config({"image": {"steps": 12}})
    assert cfg.get("image.steps") == 12
    assert cfg.get("image.model")  # untouched sibling keys survive


def test_section_returns_copy():
    cfg = Config()
    section = cfg.section("image")
    section["steps"] = 999
    assert cfg.get("image.steps") != 999


def test_parse_override_types():
    assert parse_override("runtime.seed=42") == ("runtime.seed", 42)
    assert parse_override("image.guidance_scale=6.5") == ("image.guidance_scale", 6.5)
    assert parse_override("runtime.model_cpu_offload=false") == ("runtime.model_cpu_offload", False)
    assert parse_override("image.model=some/model") == ("image.model", "some/model")


def test_parse_override_invalid():
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")
    with pytest.raises(ValueError):
        parse_override("=value")


def test_builtin_presets():
    assert load_preset("fast")["image"]["steps"] == 18
    assert load_preset("balanced") == {}
    with pytest.raises(ValueError, match="Unknown preset"):
        load_preset("nope")


def test_low_vram_presets_use_sd15_img2img_for_identity():
    # VRAM-limited presets must use SD1.5 + img2img for phase2.
    # IP-Adapter on StableDiffusionPipeline has a diffusers 0.37.1 incompatibility
    # ('tuple has no .shape'); img2img is the reliable fallback on 6 GB GPUs.
    for preset in ("fast", "low_vram", "quantized"):
        cfg = Config(load_preset(preset))
        assert "stable-diffusion-v1-5" in cfg.get("identity_image.model"), preset
        assert cfg.get("identity_image.method") == "img2img", preset


def test_quantization_defaults_disabled():
    cfg = Config()
    for section in ("image", "identity_image", "video"):
        assert cfg.get(f"{section}.quantization.enabled") is False, section


def test_quantized_preset():
    cfg = Config(load_preset("quantized"))
    assert cfg.get("image.quantization.enabled") is True
    # quantized weights can't be CPU-offloaded
    assert cfg.get("runtime.model_cpu_offload") is False
    assert cfg.get("runtime.vae_tiling") is True


def test_load_layering(tmp_path):
    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps({"image": {"steps": 11, "width": 640}}))
    cfg = Config.load(config_file=config_file, preset="fast", overrides=["image.width=800"])
    # preset overrides file; explicit --set overrides preset
    assert cfg.get("image.steps") == 18
    assert cfg.get("image.width") == 800


def test_load_missing_explicit_file():
    with pytest.raises(FileNotFoundError):
        Config.load(config_file="does_not_exist.json")
