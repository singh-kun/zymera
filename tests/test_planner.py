"""Heuristic planner: deterministic mapping of requirement+GPU to a plan."""

from zymera.capabilities import CapabilityProfile
from zymera.config import Config
from zymera.planner import plan
from zymera.planner.types import GenerationPlan, section_for_phase
from zymera.registry.catalog import Catalog


def _prof(vram=6.0, cuda=True, bnb=False):
    return CapabilityProfile(has_cuda=cuda, vram_gb=vram, has_bitsandbytes=bnb, gpu_name="Test")


CAT = Catalog()


def test_phase_inference():
    assert plan("a talking avatar that speaks", _prof(), CAT).phase == "phase4"
    assert plan("a short video clip", _prof(), CAT).phase == "phase3"
    assert plan("keep the same face in a new scene", _prof(), CAT).phase == "phase2"
    assert plan("a portrait of a woman", _prof(), CAT).phase == "phase1"


def test_style_inference():
    assert plan("cinematic film still", _prof(), CAT).style == "cinematic"
    assert plan("anime girl", _prof(), CAT).style == "anime"
    assert plan("a tree", _prof(), CAT).style is None


def test_preset_by_tier():
    assert plan("x", _prof(vram=4, bnb=True), CAT).preset == "quantized"
    assert plan("x", _prof(vram=4, bnb=False), CAT).preset == "low_vram"
    assert plan("x", _prof(vram=6), CAT).preset == "low_vram"
    assert plan("x", _prof(vram=10), CAT).preset == "balanced"
    assert plan("x", _prof(vram=24), CAT).preset == "quality"
    assert plan("x", _prof(cuda=False), CAT).preset == "sd15"


def test_speed_keyword_adds_lcm_lora():
    p = plan("a quick draft portrait, few steps", _prof(vram=10), CAT)
    # phase1 on a medium GPU → sdxl family → lcm-lora-sdxl
    assert any(l["name"] == "lcm-lora-sdxl" for l in p.loras)
    assert "lcm-lora-sdxl" in p.assets


def test_no_lora_when_not_requested():
    assert plan("a calm portrait", _prof(vram=10), CAT).loras == []


def test_determinism():
    a = plan("cinematic video clip, fast", _prof(vram=8), CAT)
    b = plan("cinematic video clip, fast", _prof(vram=8), CAT)
    assert a.to_dict() == b.to_dict()


def test_plan_applies_to_config():
    p = plan("anime portrait, few steps", _prof(vram=10), CAT)
    cfg = Config()
    p.apply_to(cfg)
    assert cfg.get("prompt.style") == "anime"
    if p.loras:
        section = section_for_phase(p.phase)
        assert cfg.get(f"{section}.lora.enabled") is True


def test_roundtrip_dict():
    p = plan("cinematic portrait", _prof(), CAT)
    assert GenerationPlan.from_dict(p.to_dict()).to_dict() == p.to_dict()
