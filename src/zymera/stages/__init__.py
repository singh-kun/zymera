"""Stage registry with lazy imports (torch is only loaded when generating)."""

from __future__ import annotations

import importlib

_STAGES: dict[str, tuple[str, str]] = {
    "text2image": ("zymera.stages.text2image", "Text2ImageStage"),
    "identity_image": ("zymera.stages.identity_image", "IdentityImageStage"),
    "video": ("zymera.stages.video", "VideoStage"),
    "tts": ("zymera.stages.tts", "SpeechStage"),
}

_cache: dict[str, object] = {}


def get_stage(name: str, cfg):
    """Get a (cached) stage instance. Models stay loaded for the process lifetime."""
    if name not in _STAGES:
        raise KeyError(f"Unknown stage '{name}'. Available: {', '.join(sorted(_STAGES))}")
    if name not in _cache:
        module_name, class_name = _STAGES[name]
        stage_cls = getattr(importlib.import_module(module_name), class_name)
        _cache[name] = stage_cls(cfg)
    return _cache[name]


def clear_stages() -> None:
    for stage in _cache.values():
        stage.unload()
    _cache.clear()
