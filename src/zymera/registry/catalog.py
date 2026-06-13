"""Asset catalog: named model / LoRA / adapter entries.

A built-in curated set (safe, SFW, known-good) is deep-merged with the user's
``configs/registry.json`` (its ``assets`` object), mirroring how prompt styles
merge over built-in defaults in :mod:`zymera.prompts`.

Entry schema (all optional unless noted)::

    "<name>": {
      "type":        "checkpoint" | "lora" | "vae" | "ip_adapter" | "motion_adapter",
      "source":      "hf" | "civitai",          # required
      "repo":        "org/model",               # hf
      "weight_name": "file.safetensors",        # hf single-file (e.g. a LoRA)
      "model_id":    12345,                      # civitai
      "version_id":  67890,                      # civitai (optional; default latest)
      "family":      "sdxl" | "sd15" | null,    # base-model family for compatibility
      "vram_gb":     0.2,                         # rough load cost, for the planner
      "tags":        ["detail", "quality"],
      "nsfw":        false,                       # axis-2 flag (default false)
      "real_person": false,                       # axis-1 flag (default false)
      "sha256":      null                         # optional integrity check
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from zymera.config import deep_merge

log = logging.getLogger(__name__)

DEFAULT_REGISTRY_FILE = Path("configs") / "registry.json"

# Curated, SFW, real-person-free starter catalog. Base checkpoints mirror the
# defaults in config.py so the planner can refer to them by name; LoRAs are
# well-known speed/quality adapters published on the HF Hub.
BUILTIN_ASSETS: dict[str, dict[str, Any]] = {
    "sdxl-base": {
        "type": "checkpoint", "source": "hf",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "family": "sdxl", "vram_gb": 7.0, "tags": ["base", "photoreal"],
    },
    "sd15-base": {
        "type": "checkpoint", "source": "hf",
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "family": "sd15", "vram_gb": 4.0, "tags": ["base"],
    },
    "epicrealism": {
        "type": "checkpoint", "source": "hf", "repo": "emilianJR/epiCRealism",
        "family": "sd15", "vram_gb": 4.0, "tags": ["base", "photoreal", "video"],
    },
    "sdxl-vae-fp16": {
        "type": "vae", "source": "hf", "repo": "madebyollin/sdxl-vae-fp16-fix",
        "family": "sdxl", "vram_gb": 0.3, "tags": ["vae"],
    },
    # Latent Consistency LoRAs — drop step counts dramatically; SFW, no real people.
    "lcm-lora-sdxl": {
        "type": "lora", "source": "hf", "repo": "latent-consistency/lcm-lora-sdxl",
        "family": "sdxl", "vram_gb": 0.2, "tags": ["speed", "lcm", "few-steps"],
    },
    "lcm-lora-sd15": {
        "type": "lora", "source": "hf", "repo": "latent-consistency/lcm-lora-sdv1-5",
        "family": "sd15", "vram_gb": 0.1, "tags": ["speed", "lcm", "few-steps"],
    },
}


class Catalog:
    """A merged, queryable set of asset entries keyed by name."""

    def __init__(self, assets: dict[str, dict[str, Any]] | None = None):
        # deepcopy via deep_merge into a fresh dict keeps BUILTIN_ASSETS pristine.
        self.assets: dict[str, dict[str, Any]] = {}
        deep_merge(self.assets, {k: dict(v) for k, v in BUILTIN_ASSETS.items()})
        if assets:
            deep_merge(self.assets, assets)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        """Load built-in catalog merged with a user registry file's ``assets``."""
        path = Path(path) if path else DEFAULT_REGISTRY_FILE
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            user_assets = data.get("assets", data) if isinstance(data, dict) else {}
            log.debug("Loaded %d catalog entries from %s", len(user_assets), path)
            return cls(user_assets)
        return cls()

    def names(self) -> list[str]:
        return sorted(self.assets)

    def resolve(self, name: str) -> dict[str, Any]:
        """Return a copy of the entry with its ``name`` injected, or raise."""
        if name not in self.assets:
            raise KeyError(
                f"Unknown asset '{name}'. Known: {', '.join(self.names()) or '(none)'}"
            )
        entry = dict(self.assets[name])
        entry["name"] = name
        return entry

    def search(
        self,
        query: str | None = None,
        *,
        family: str | None = None,
        type: str | None = None,  # noqa: A002 - mirrors the entry field name
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter catalog entries by free-text query, family, type, and/or tags."""
        results: list[dict[str, Any]] = []
        q = query.lower() if query else None
        want_tags = {t.lower() for t in tags} if tags else None
        for name in self.names():
            entry = self.resolve(name)
            if family and entry.get("family") != family:
                continue
            if type and entry.get("type") != type:
                continue
            entry_tags = {str(t).lower() for t in entry.get("tags", [])}
            if want_tags and not (want_tags & entry_tags):
                continue
            if q:
                blob = f"{name} {entry.get('repo', '')} {' '.join(entry_tags)}".lower()
                if q not in blob:
                    continue
            results.append(entry)
        return results

    def by_family(self, family: str) -> list[dict[str, Any]]:
        return self.search(family=family)
