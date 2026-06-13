"""Layered configuration: built-in defaults <- config file <- preset <- overrides.

Every generation parameter lives here so behaviour can be changed without
touching code: ``zymera generate --preset quality --set image.steps=50``.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = Path("configs") / "default.json"
PRESETS_DIR = Path("configs") / "presets"

DEFAULTS: dict[str, Any] = {
    "runtime": {
        "device": "auto",            # auto | cuda | cpu
        "dtype": "auto",             # auto | float16 | bfloat16 | float32
        "seed": None,                # null = random per run (logged for reproducibility)
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": False,         # enable for outputs >1024px on low VRAM
        "model_cpu_offload": True,   # safe default for 8GB-class GPUs; disable for speed
        "tf32": True,                # TF32 matmul on Ampere/Ada GPUs (~15% faster, lossless)
        "enable_xformers": False,    # torch>=2 SDPA already uses flash-attention kernels
    },
    "paths": {
        "identities_dir": "identities",
        "outputs_dir": "outputs",
        "prompts_file": "configs/prompts.json",
        "registry_file": "configs/registry.json",  # user asset catalog (LoRAs/models)
        "recipes_dir": "configs/presets",            # saved "skill" recipes = preset JSONs
        "assets_dir": None,                          # None -> ~/.cache/zymera/assets
    },
    # Downloadable-asset policy. content_mode is the SFW/NSFW toggle (axis 2);
    # real-person content (axis 1) is ALWAYS blocked regardless. See
    # zymera.registry.policy and CLAUDE.md "Responsible use".
    "registry": {
        "content_mode": "sfw",       # sfw | nsfw — nsfw only permits SYNTHETIC nsfw
    },
    # Optional Claude-powered planner/executor (zymera auto). Heuristic fallback
    # runs when ANTHROPIC_API_KEY is unset, so a key is never required.
    "agent": {
        "model": "claude-opus-4-8",  # planner model id (claude-sonnet-4-6 = cheaper)
        "max_tokens": 4096,
    },
    "prompt": {
        "style": "photorealistic",
        "enhance": True,             # append quality tags / curated negatives
    },
    # phase1 — pure text-to-image
    "image": {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",            # base-model family; gates LoRA compatibility
        "variant": "fp16",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "scheduler": "DPMSolverMultistepScheduler",
        "scheduler_options": {"use_karras_sigmas": True},
        "steps": 30,
        "guidance_scale": 6.0,
        "width": 1024,
        "height": 1024,
        "num_images": 1,
        # NF4 4-bit quantization (requires bitsandbytes): SDXL UNet 5.1GB -> ~1.7GB.
        "quantization": {
            "enabled": False,
            "backend": "bitsandbytes_4bit",   # bitsandbytes_4bit | bitsandbytes_8bit
            "components": ["unet", "text_encoder_2"],
        },
        # LoRA adapters (peft). Each adapter: a registry asset name or HF repo.
        "lora": {
            "enabled": False,
            "adapters": [],  # [{"name": "lcm-lora-sdxl", "scale": 0.8, "weight_name": null}]
            "fuse": False,    # fuse into weights (faster, but can't switch adapters)
        },
    },
    # phase2 — identity-conditioned image (IP-Adapter by default)
    "identity_image": {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "variant": "fp16",
        "vae": "madebyollin/sdxl-vae-fp16-fix",
        "scheduler": "DPMSolverMultistepScheduler",
        "scheduler_options": {"use_karras_sigmas": True},
        "method": "ip_adapter",      # ip_adapter | img2img
        "ip_adapter": {
            "repo": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight_name": "ip-adapter_sdxl.bin",
            "image_encoder_folder": "image_encoder",
            "scale": 0.6,            # identity strength vs prompt freedom
        },
        "strength": 0.55,            # img2img method only
        "steps": 30,
        "guidance_scale": 6.0,
        "width": 1024,
        "height": 1024,
        "num_images": 1,
        "quantization": {
            "enabled": False,
            "backend": "bitsandbytes_4bit",
            "components": ["unet", "text_encoder_2"],
        },
        "lora": {
            "enabled": False,
            "adapters": [],
            "fuse": False,
        },
    },
    # phase3 — AnimateDiff text-to-video, identity via IP-Adapter when refs exist
    "video": {
        "model": "emilianJR/epiCRealism",  # SD1.5-family checkpoint (required by the motion adapter)
        "family": "sd15",
        "variant": None,
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2",
        "scheduler": "DDIMScheduler",
        "scheduler_options": {
            "clip_sample": False,
            "timestep_spacing": "linspace",
            "beta_schedule": "linear",
            "steps_offset": 1,
        },
        "ip_adapter": {
            "enabled": True,
            "repo": "h94/IP-Adapter",
            "subfolder": "models",
            "weight_name": "ip-adapter_sd15.bin",
            "image_encoder_folder": "image_encoder",
            "scale": 0.6,
        },
        "steps": 25,
        "guidance_scale": 7.5,
        "num_frames": 16,
        "fps": 8,
        "width": 512,
        "height": 512,
        "quantization": {
            "enabled": False,            # SD1.5 UNet is small; rarely needed here
            "backend": "bitsandbytes_4bit",
            "components": ["unet"],
        },
        "lora": {
            "enabled": False,
            "adapters": [],
            "fuse": False,
        },
    },
    # phase4 — speech synthesis
    "speech": {
        "model": "tts_models/en/ljspeech/vits",
        "speaker_wav": None,         # reference voice for cloning models (e.g. XTTS v2)
        "language": "en",
    },
    # phase4 — final audio/video composition
    "compose": {
        "lipsync": {
            "method": "none",        # none | wav2lip
            "command": None,         # list of args; {video} {audio} {output} placeholders
        },
        "video_codec": "libx264",
        "audio_codec": "aac",
        "keep_intermediates": False,
    },
}

PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {},
    "quality": {
        "image": {"steps": 45, "guidance_scale": 7.0},
        "identity_image": {"steps": 45, "guidance_scale": 7.0},
        "video": {"steps": 30},
    },
    "fast": {
        "image": {"steps": 18, "width": 832, "height": 832},
        # SD1.5 + img2img for identity_image: ip_adapter on StableDiffusionPipeline
        # has a 'tuple has no .shape' bug in diffusers 0.37.1 on both SD1.5 and SDXL.
        # img2img directly conditions on the reference face and is reliable on 6 GB VRAM.
        "identity_image": {
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "family": "sd15",
            "variant": None,
            "vae": None,
            "method": "img2img",
            "scheduler": "DPMSolverMultistepScheduler",
            "scheduler_options": {"use_karras_sigmas": True},
            "strength": 0.65,
            "steps": 20,
            "width": 512,
            "height": 512,
        },
        "video": {"steps": 14, "num_frames": 12},
    },
    "low_vram": {
        "runtime": {"model_cpu_offload": True, "attention_slicing": True, "vae_slicing": True},
        "image": {"width": 768, "height": 768},
        "identity_image": {
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "family": "sd15",
            "variant": None,
            "vae": None,
            "method": "img2img",
            "scheduler": "DPMSolverMultistepScheduler",
            "scheduler_options": {"use_karras_sigmas": True},
            "strength": 0.65,
            "steps": 30,
            "width": 512,
            "height": 512,
        },
        "video": {"num_frames": 8},
    },
    # Full SDXL quality for phase1 on 6-8 GB GPUs via NF4 4-bit weights (needs
    # bitsandbytes). Quantized components can't be CPU-offloaded, so offload is
    # disabled and VAE tiling enabled instead. Phase2 keeps the SD1.5 IP-Adapter
    # stack: the SDXL IP-Adapter's 3.7 GB ViT-H image encoder is not covered by
    # pipeline quantization and would overrun 6 GB without offload.
    "quantized": {
        "runtime": {"model_cpu_offload": False, "vae_tiling": True},
        "image": {"quantization": {"enabled": True}},
        "identity_image": {
            "model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "family": "sd15",
            "variant": None,
            "vae": None,
            "method": "img2img",
            "strength": 0.65,
            "steps": 30,
            "width": 512,
            "height": 512,
        },
    },
}


def deep_merge(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_override(item: str) -> tuple[str, Any]:
    """Parse a ``section.key=value`` override; value is JSON-decoded when possible."""
    if "=" not in item:
        raise ValueError(f"Invalid override '{item}'. Expected format: section.key=value")
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override '{item}'. Empty key.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def load_preset(name: str) -> dict[str, Any]:
    """Resolve a preset: explicit file path, configs/presets/<name>.json, or built-in."""
    for candidate in (Path(name), PRESETS_DIR / f"{name}.json"):
        if candidate.suffix == ".json" and candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    if name in PRESETS:
        return copy.deepcopy(PRESETS[name])
    file_presets = sorted(p.stem for p in PRESETS_DIR.glob("*.json")) if PRESETS_DIR.is_dir() else []
    available = sorted(set(PRESETS) | set(file_presets))
    raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(available)}")


class Config:
    """Nested configuration with dot-path access."""

    def __init__(self, data: dict[str, Any] | None = None):
        self._data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        if data:
            deep_merge(self._data, data)

    @classmethod
    def load(
        cls,
        config_file: str | Path | None = None,
        preset: str | None = None,
        overrides: list[str] | None = None,
    ) -> "Config":
        cfg = cls()
        path = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
        if path.is_file():
            cfg.merge(json.loads(path.read_text(encoding="utf-8")))
            log.debug("Loaded config file %s", path)
        elif config_file:
            raise FileNotFoundError(f"Config file not found: {path}")
        if preset:
            cfg.merge(load_preset(preset))
            log.info("Applied preset '%s'", preset)
        for item in overrides or []:
            key, value = parse_override(item)
            cfg.set(key, value)
        return cfg

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set '{path}': '{part}' is not a section")
        node[parts[-1]] = value

    def merge(self, data: dict[str, Any]) -> None:
        deep_merge(self._data, data)

    def section(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self._data.get(name, {}))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)
