"""Prompt construction: style presets, quality tags, and curated negative prompts.

The library is data-driven — edit ``configs/prompts.json`` to add or tune
styles without code changes. File entries are deep-merged over the built-in
defaults below.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

from zymera.config import deep_merge

log = logging.getLogger(__name__)

DEFAULT_PROMPT_SPEC: dict[str, Any] = {
    "quality_suffix": "highly detailed, sharp focus, best quality",
    "negative_base": (
        "lowres, worst quality, low quality, jpeg artifacts, blurry, watermark, "
        "text, logo, signature, cropped, out of frame, bad anatomy, deformed, "
        "disfigured, mutated, extra limbs, extra fingers, fused fingers, "
        "bad hands, bad eyes, cross-eyed, asymmetric eyes"
    ),
    "styles": {
        "none": {},
        "photorealistic": {
            "prefix": "RAW photo, ",
            "suffix": (
                ", photorealistic, natural skin texture, detailed eyes, "
                "85mm lens, f/1.8, soft natural lighting, subtle film grain"
            ),
            "negative": "cartoon, anime, 3d render, painting, illustration, cgi, plastic skin, doll, airbrushed",
        },
        "studio_portrait": {
            "prefix": "professional studio portrait photo, ",
            "suffix": (
                ", softbox key light, rim light, seamless backdrop, "
                "shallow depth of field, crisp focus on eyes, editorial retouching"
            ),
            "negative": "cartoon, anime, 3d render, harsh shadows, flat lighting, busy background",
        },
        "cinematic": {
            "prefix": "cinematic film still, ",
            "suffix": (
                ", dramatic volumetric lighting, shallow depth of field, "
                "anamorphic bokeh, moody color grade, film grain"
            ),
            "negative": "cartoon, anime, flat lighting, oversaturated, video game screenshot",
        },
        "fashion_editorial": {
            "prefix": "high fashion editorial photo, ",
            "suffix": (
                ", designer styling, dynamic pose, magazine cover quality, "
                "studio strobes, bold composition"
            ),
            "negative": "cartoon, anime, 3d render, casual snapshot, cluttered background",
        },
        "anime": {
            "prefix": "anime artwork, ",
            "suffix": ", anime key visual, clean line art, vibrant colors, studio quality",
            "negative": "photo, photorealistic, 3d render, live action",
        },
    },
}


class BuiltPrompt(NamedTuple):
    prompt: str
    negative: str


class PromptBuilder:
    def __init__(self, spec: dict[str, Any] | None = None):
        self.spec = copy.deepcopy(DEFAULT_PROMPT_SPEC)
        if spec:
            deep_merge(self.spec, spec)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "PromptBuilder":
        if path and Path(path).is_file():
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        if path:
            log.debug("Prompts file %s not found; using built-in styles", path)
        return cls()

    def styles(self) -> list[str]:
        return sorted(self.spec["styles"])

    def build(self, prompt: str, style: str | None = None, enhance: bool = True) -> BuiltPrompt:
        style = style or "none"
        style_spec = self.spec["styles"].get(style)
        if style_spec is None:
            raise ValueError(f"Unknown style '{style}'. Available: {', '.join(self.styles())}")

        parts = [style_spec.get("prefix", ""), prompt.strip(), style_spec.get("suffix", "")]
        if enhance:
            parts.append(f", {self.spec['quality_suffix']}")
        positive = "".join(parts).strip().strip(",").strip()

        negatives = [self.spec["negative_base"]] if enhance else []
        if style_spec.get("negative"):
            negatives.append(style_spec["negative"])
        return BuiltPrompt(positive, ", ".join(negatives))
