"""Recipes = saved "skills": a named, re-runnable generation setup.

A recipe is just a preset JSON in the recipes dir (``configs/presets`` by
default) so it works directly with ``--preset <name>``. The base preset chosen by
the planner is baked in, so the file is self-contained and reproducible. Recipe
metadata (requirement, phase, assets to fetch) lives under a ``_recipe`` key.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from zymera.config import deep_merge, load_preset

log = logging.getLogger(__name__)


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    """Expand dot-path keys into a nested dict."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


class RecipeStore:
    def __init__(self, recipes_dir: str | Path):
        self.dir = Path(recipes_dir)

    def save(self, name: str, plan) -> Path:
        """Bake the plan (base preset + overrides) into a self-contained recipe."""
        data: dict[str, Any] = {}
        if plan.preset:
            try:
                data = copy.deepcopy(load_preset(plan.preset))
            except ValueError:
                log.debug("Base preset '%s' not resolvable; saving overrides only", plan.preset)
        deep_merge(data, _nest(plan.to_config_overrides()))
        data["_recipe"] = {
            "requirement": plan.requirement,
            "phase": plan.phase,
            "style": plan.style,
            "base_preset": plan.preset,
            "assets": plan.assets,
            "loras": plan.loras,
            "rationale": plan.rationale,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Saved recipe '%s' -> %s", name, path)
        return path

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def show(self, name: str) -> dict[str, Any]:
        path = self.path(name)
        if not path.is_file():
            raise KeyError(f"Recipe '{name}' not found in {self.dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def meta(self, name: str) -> dict[str, Any]:
        return self.show(name).get("_recipe", {})

    def list(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        names = []
        for p in sorted(self.dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            if "_recipe" in data:  # only files we created, not hand-written presets
                names.append(p.stem)
        return names

    def materialize(self, name: str, cfg) -> str:
        """Ensure all of a recipe's assets are downloaded; return the preset name
        to pass to ``Config.load(preset=...)``."""
        meta = self.meta(name)
        assets = meta.get("assets", [])
        if assets:
            from zymera.registry import AssetManager

            manager = AssetManager.from_config(cfg)
            for asset in assets:
                log.info("Ensuring recipe asset '%s'", asset)
                manager.ensure(asset)
        return name
