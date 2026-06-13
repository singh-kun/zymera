"""Zymera command line interface."""

from __future__ import annotations

import argparse
import logging
import sys

from zymera.config import Config
from zymera.pipeline import PHASES

log = logging.getLogger("zymera")

# Which config section the convenience flags (--steps etc.) should target per phase.
_FLAG_SECTION = {
    "image": "image",
    "identity_image": "identity_image",
    "video": "video",
    "talking_video": "video",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zymera",
        description="Multi-stage synthetic identity media generation (images, video, talking video).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate an image, video, or talking video")
    gen.add_argument("--phase", default="phase1", choices=sorted(PHASES),
                     help="phase1=image, phase2=identity image, phase3=video, phase4=talking video")
    gen.add_argument("--prompt", required=True, help="What to generate")
    gen.add_argument("--identity", help="Identity ID (required for phase2; used by phase3/4 when set)")
    gen.add_argument("--text", help="Speech text (phase4)")
    gen.add_argument("--output", help="Output filename (saved under the outputs dir)")
    gen.add_argument("--config", help="Path to a config JSON file (default: configs/default.json)")
    gen.add_argument("--preset", help="Preset name (quality, balanced, fast, low_vram, quantized) or JSON path")
    gen.add_argument("--style", help="Prompt style (see 'zymera styles')")
    gen.add_argument("--no-enhance", action="store_true", help="Disable quality tags / curated negatives")
    gen.add_argument("--seed", type=int, help="Seed for reproducible output")
    gen.add_argument("--steps", type=int, help="Inference steps")
    gen.add_argument("--guidance-scale", type=float, dest="guidance_scale")
    gen.add_argument("--width", type=int)
    gen.add_argument("--height", type=int)
    gen.add_argument("--frames", type=int, help="Video frame count (phase3/4)")
    gen.add_argument("--num-images", type=int, dest="num_images", help="Images per run (phase1/2)")
    gen.add_argument("--offload", choices=["on", "off"], help="Model CPU offload (VRAM vs speed)")
    gen.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                     help="Override any config value, e.g. --set image.model=...")

    ident = sub.add_parser("identity", help="Manage identities")
    ident_sub = ident.add_subparsers(dest="identity_command", required=True)
    create = ident_sub.add_parser("create", help="Create an identity")
    create.add_argument("id")
    create.add_argument("--images", nargs="*", default=[], help="Reference image files")
    create.add_argument("--note", help="Free-form note (e.g. provenance/consent)")
    add = ident_sub.add_parser("add", help="Add a reference image to an identity")
    add.add_argument("id")
    add.add_argument("--image", required=True)
    ident_sub.add_parser("list", help="List identities")

    styles = sub.add_parser("styles", help="List available prompt styles")
    styles.add_argument("--config", help="Path to a config JSON file")

    doctor = sub.add_parser("doctor", help="Check the environment (GPU, deps, ffmpeg)")
    doctor.add_argument("--config", help="Path to a config JSON file")

    # Agentic planner/executor: requirement -> plan -> assets -> recipe -> (run).
    auto = sub.add_parser("auto", help="Plan models/LoRAs/config for a requirement (agentic)")
    auto.add_argument("requirement", help="Natural-language description of what you want")
    auto.add_argument("--run", action="store_true", help="Generate after planning")
    auto.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    auto.add_argument("--save", metavar="NAME", help="Save the plan as a reusable recipe")
    auto.add_argument("--prompt", help="Generation prompt (defaults to the requirement)")
    auto.add_argument("--identity", help="Identity ID (phase2+)")
    auto.add_argument("--text", help="Speech text (phase4)")
    auto.add_argument("--output", help="Output filename")
    auto.add_argument("--seed", type=int)
    auto.add_argument("--nsfw", action="store_true",
                      help="Allow SYNTHETIC NSFW assets (content_mode=nsfw); real people stay blocked")
    auto.add_argument("--preset", help="Force a base preset instead of auto-detecting")
    auto.add_argument("--config", help="Path to a config JSON file")

    # Recipes = saved "skills".
    recipe = sub.add_parser("recipe", help="Manage saved recipes (skills)")
    recipe_sub = recipe.add_subparsers(dest="recipe_command", required=True)
    recipe_sub.add_parser("list", help="List saved recipes").add_argument("--config")
    show = recipe_sub.add_parser("show", help="Print a recipe's JSON")
    show.add_argument("name")
    show.add_argument("--config")
    run = recipe_sub.add_parser("run", help="Download a recipe's assets and generate")
    run.add_argument("name")
    run.add_argument("--prompt", required=True)
    run.add_argument("--identity")
    run.add_argument("--text")
    run.add_argument("--output")
    run.add_argument("--seed", type=int)
    run.add_argument("--config")

    return parser


def normalize_argv(argv: list[str]) -> list[str]:
    """Allow flag-first invocation ('zymera --phase ...') by implying 'generate'."""
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        return ["generate", *argv]
    return argv


def _load_config(args) -> Config:
    return Config.load(
        config_file=getattr(args, "config", None),
        preset=getattr(args, "preset", None),
        overrides=getattr(args, "overrides", []),
    )


def _cmd_generate(args) -> int:
    cfg = _load_config(args)
    section = _FLAG_SECTION[PHASES[args.phase]]
    flag_map = {
        "steps": f"{section}.steps",
        "guidance_scale": f"{section}.guidance_scale",
        "width": f"{section}.width",
        "height": f"{section}.height",
        "frames": "video.num_frames",
        "num_images": f"{section}.num_images",
    }
    for flag, key in flag_map.items():
        value = getattr(args, flag)
        if value is not None:
            cfg.set(key, value)
    if args.offload:
        cfg.set("runtime.model_cpu_offload", args.offload == "on")
    if args.no_enhance:
        cfg.set("prompt.enhance", False)

    from zymera.pipeline import Pipeline

    paths = Pipeline(cfg).run(
        phase=args.phase,
        prompt=args.prompt,
        identity=args.identity,
        text=args.text,
        output=args.output,
        style=args.style,
        seed=args.seed,
    )
    for path in paths:
        print(f"Generated: {path}")
    return 0


def _cmd_identity(args) -> int:
    from zymera.identity import IdentityStore

    cfg = Config.load()
    store = IdentityStore(cfg.get("paths.identities_dir"))
    if args.identity_command == "create":
        attributes = {"note": args.note} if args.note else {}
        identity = store.create(args.id, args.images, attributes)
        print(f"Created identity '{identity.identity_id}' with {len(identity.reference_images)} reference image(s)")
    elif args.identity_command == "add":
        identity = store.add_reference(args.id, args.image)
        print(f"Identity '{identity.identity_id}' now has {len(identity.reference_images)} reference image(s)")
    else:
        ids = store.list_ids()
        if not ids:
            print("No identities yet. Create one: zymera identity create <id> --images ref.jpg")
        for identity_id in ids:
            refs = len(store.load(identity_id).reference_images)
            print(f"  {identity_id}  ({refs} reference image(s))")
    return 0


def _cmd_styles(args) -> int:
    from zymera.prompts import PromptBuilder

    cfg = Config.load(config_file=getattr(args, "config", None))
    builder = PromptBuilder.from_file(cfg.get("paths.prompts_file"))
    for style in builder.styles():
        print(f"  {style}")
    return 0


def _cmd_doctor(args) -> int:
    from zymera.doctor import run_doctor

    return run_doctor(Config.load(config_file=getattr(args, "config", None)))


def _cmd_auto(args) -> int:
    from zymera.agent.run import run_auto

    return run_auto(args)


def _cmd_recipe(args) -> int:
    import json

    from zymera.recipes import RecipeStore

    cfg = Config.load(config_file=getattr(args, "config", None))
    store = RecipeStore(cfg.get("paths.recipes_dir"))

    if args.recipe_command == "list":
        names = store.list()
        if not names:
            print('No recipes yet. Create one: zymera auto "<requirement>" --save <name>')
        for name in names:
            meta = store.meta(name)
            print(f"  {name}  [{meta.get('phase', '?')}]  {meta.get('requirement', '')}")
        return 0

    if args.recipe_command == "show":
        print(json.dumps(store.show(args.name), indent=2))
        return 0

    # run
    preset = store.materialize(args.name, cfg)
    meta = store.meta(args.name)
    run_cfg = Config.load(config_file=getattr(args, "config", None), preset=preset)
    from zymera.pipeline import Pipeline

    paths = Pipeline(run_cfg).run(
        phase=meta.get("phase", "phase1"),
        prompt=args.prompt,
        identity=args.identity,
        text=args.text,
        output=args.output,
        style=meta.get("style"),
        seed=args.seed,
    )
    for path in paths:
        print(f"Generated: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = normalize_argv(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 1

    handler = {
        "generate": _cmd_generate,
        "identity": _cmd_identity,
        "styles": _cmd_styles,
        "doctor": _cmd_doctor,
        "auto": _cmd_auto,
        "recipe": _cmd_recipe,
    }[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exc:
        log.error("%s", exc.args[0] if exc.args else exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
