"""Executor: turn a GenerationPlan into downloaded assets, a saved recipe, and
(optionally) a generation run. Fully deterministic — no LLM calls here."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def screen_and_download(plan, cfg) -> list[str]:
    """Policy-screen and download every asset the plan needs. Returns the list of
    asset names successfully ensured. Raises PolicyError if any is blocked."""
    from zymera.registry import AssetManager

    if not plan.assets:
        return []
    manager = AssetManager.from_config(cfg)
    ensured = []
    for name in plan.assets:
        log.info("Ensuring asset '%s'", name)
        manager.ensure(name)  # PolicyGate.check runs inside; raises if blocked
        ensured.append(name)
    return ensured


def save_recipe(plan, cfg, name: str):
    """Persist the plan as a reusable recipe preset. Returns the recipe name."""
    from zymera.recipes import RecipeStore

    store = RecipeStore(cfg.get("paths.recipes_dir"))
    store.save(name, plan)
    return name


def run_generation(plan, cfg, *, prompt, identity=None, text=None, output=None, seed=None):
    """Apply the plan to config and run the pipeline. Returns produced paths."""
    from zymera.config import Config
    from zymera.pipeline import Pipeline

    # Rebuild config layering the plan's preset, then the plan's overrides.
    run_cfg = Config.load(preset=plan.preset) if plan.preset else Config()
    # Preserve the content_mode the user selected for this run.
    run_cfg.set("registry.content_mode", cfg.get("registry.content_mode", "sfw"))
    plan.apply_to(run_cfg)

    return Pipeline(run_cfg).run(
        phase=plan.phase,
        prompt=prompt,
        identity=identity,
        text=text,
        output=output,
        style=plan.style,
        seed=seed,
    )
