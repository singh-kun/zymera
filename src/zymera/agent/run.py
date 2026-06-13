"""Orchestrate the agentic ``zymera auto`` flow: plan -> confirm -> execute.

Planner selection:
- Claude (Anthropic SDK) when ANTHROPIC_API_KEY is set and the SDK is installed.
- The deterministic heuristic planner otherwise (always available, no network).
"""

from __future__ import annotations

import logging

from zymera.config import Config

log = logging.getLogger(__name__)


def _make_plan(requirement, profile, catalog, cfg):
    """Pick the planner: Claude if available, else heuristic. Falls back to the
    heuristic planner if the Claude call errors."""
    from zymera.agent import planner_agent
    from zymera.planner import plan as heuristic_plan

    if planner_agent.is_available():
        try:
            print("Planning with Claude...")
            return planner_agent.plan_with_claude(requirement, profile, catalog, cfg)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to heuristics
            log.warning("Claude planner failed (%s); using heuristic planner", exc)
    else:
        print("ANTHROPIC_API_KEY not set — using the deterministic heuristic planner.")
    return heuristic_plan(requirement, profile, catalog)


def _print_plan(plan, profile) -> None:
    print("\n=== Generation plan ===")
    print(f"  requirement : {plan.requirement}")
    print(f"  GPU         : {profile.summary()}")
    print(f"  phase       : {plan.phase}")
    print(f"  preset      : {plan.preset}")
    print(f"  style       : {plan.style or '(none)'}")
    if plan.loras:
        for l in plan.loras:
            print(f"  lora        : {l['name']} (scale {l.get('scale', 1.0)})")
    if plan.assets:
        print(f"  downloads   : {', '.join(plan.assets)}")
    print(f"  rationale   : {plan.rationale}")
    print()


def run_auto(args) -> int:
    from zymera.capabilities import probe
    from zymera.registry import Catalog

    cfg = Config.load(config_file=getattr(args, "config", None))
    if getattr(args, "nsfw", False):
        cfg.set("registry.content_mode", "nsfw")

    profile = probe()
    catalog = Catalog.load(cfg.get("paths.registry_file"))
    plan = _make_plan(args.requirement, profile, catalog, cfg)

    # An explicit --preset forces the base preset regardless of the planner.
    if getattr(args, "preset", None):
        plan.preset = args.preset

    _print_plan(plan, profile)

    needs_action = bool(args.save or args.run or plan.assets)
    if needs_action and not args.yes:
        try:
            answer = input("Proceed (download assets / save recipe / generate)? [y/N] ")
        except EOFError:
            answer = "n"
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted. (Re-run with --yes to skip this prompt.)")
            return 0

    # 1. Screen + download assets (policy-gated).
    from zymera.agent.executor_agent import run_generation, save_recipe, screen_and_download

    try:
        screen_and_download(plan, cfg)
    except Exception as exc:  # PolicyError or download failure
        log.error("Asset preparation failed: %s", exc)
        return 1

    # 2. Save recipe if requested.
    if args.save:
        save_recipe(plan, cfg, args.save)
        print(f"Saved recipe '{args.save}'. Re-run with: zymera recipe run {args.save} --prompt \"...\"")

    # 3. Generate if requested.
    if args.run:
        prompt = args.prompt or args.requirement
        paths = run_generation(
            plan, cfg, prompt=prompt, identity=args.identity,
            text=args.text, output=args.output, seed=args.seed,
        )
        for path in paths:
            print(f"Generated: {path}")

    if not args.run and not args.save:
        print("Plan only (no --run / --save). Add --save <name> to keep it, or --run to generate.")
    return 0
