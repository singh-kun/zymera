"""Deterministic, rule-based planner: (requirement + capabilities) -> plan.

No LLM, no network. Maps keywords to phase/style and the detected GPU tier to a
preset, then picks compatible LoRAs from the catalog only when the requirement
clearly calls for one (so nothing is downloaded by surprise).
"""

from __future__ import annotations

from zymera.capabilities import CapabilityProfile
from zymera.planner.types import GenerationPlan

# Keyword → built-in prompt style (see zymera.prompts.DEFAULT_PROMPT_SPEC).
_STYLE_KEYWORDS = [
    (("anime", "manga", "illustration", "cartoon"), "anime"),
    (("cinematic", "film", "movie", "dramatic"), "cinematic"),
    (("fashion", "editorial", "magazine", "runway"), "fashion_editorial"),
    (("studio", "headshot", "portrait"), "studio_portrait"),
    (("photo", "photoreal", "realistic", "candid"), "photorealistic"),
]

# GPU tier → preset. Tiny GPUs prefer NF4 quantization when bitsandbytes exists.
_TIER_PRESET = {
    "cpu": "sd15",
    "tiny": "low_vram",
    "small": "low_vram",
    "medium": "balanced",
    "large": "quality",
}


def _infer_phase(req: str) -> str:
    if any(w in req for w in ("talking", "speak", "speech", "voice", "say ", "saying")):
        return "phase4"
    if any(w in req for w in ("video", "animate", "animation", "clip", "motion", "moving")):
        return "phase3"
    if any(w in req for w in ("identity", "same face", "same person", "consistent face",
                              "this face", "reference face")):
        return "phase2"
    return "phase1"


def _infer_style(req: str) -> str | None:
    for keywords, style in _STYLE_KEYWORDS:
        if any(k in req for k in keywords):
            return style
    return None


def _preset_for_profile(profile: CapabilityProfile) -> str:
    preset = _TIER_PRESET.get(profile.tier, "balanced")
    if profile.tier == "tiny" and profile.has_bitsandbytes:
        return "quantized"  # full SDXL quality via NF4 on <6 GB
    return preset


def _family_for(phase: str, preset: str) -> str:
    """Which base-model family the chosen phase+preset will run on."""
    if phase in ("phase3", "phase4"):
        return "sd15"  # AnimateDiff requires SD1.5
    if phase == "phase2":
        return "sd15" if preset in ("fast", "low_vram", "quantized", "sd15") else "sdxl"
    # phase1
    return "sd15" if preset == "sd15" else "sdxl"


def _select_loras(req: str, family: str, catalog) -> tuple[list[dict], list[str]]:
    """Pick compatible LoRAs only when the requirement clearly asks for one."""
    loras: list[dict] = []
    assets: list[str] = []

    def _add(entry, scale):
        loras.append({"name": entry["name"], "scale": scale})
        assets.append(entry["name"])

    wants_speed = any(w in req for w in ("fast", "quick", "few steps", "few-steps", "speed", "draft"))
    if wants_speed:
        for e in catalog.search(family=family, type="lora", tags=["lcm"]):
            _add(e, 1.0)
            break

    wants_anime = any(w in req for w in ("anime", "manga"))
    if wants_anime:
        for e in catalog.search(family=family, type="lora", tags=["anime"]):
            _add(e, 0.8)
            break

    return loras, assets


def plan(requirement: str, profile: CapabilityProfile, catalog) -> GenerationPlan:
    """Build a deterministic GenerationPlan from a requirement + capabilities."""
    req = requirement.lower()
    phase = _infer_phase(req)
    preset = _preset_for_profile(profile)
    style = _infer_style(req)
    family = _family_for(phase, preset)
    loras, assets = _select_loras(req, family, catalog)

    bits = [
        f"phase={phase} (from requirement keywords)",
        f"preset={preset} (GPU tier '{profile.tier}', {profile.summary()})",
    ]
    if style:
        bits.append(f"style={style}")
    if loras:
        bits.append("LoRAs: " + ", ".join(a for a in assets))
    rationale = "; ".join(bits)

    return GenerationPlan(
        requirement=requirement,
        phase=phase,
        preset=preset,
        style=style,
        base_family=None,  # keep the preset's base model; LoRAs constrain family
        loras=loras,
        assets=assets,
        rationale=rationale,
    )
