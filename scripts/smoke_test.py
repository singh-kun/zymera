"""End-to-end GPU smoke test: runs all four phases with a self-bootstrapped
synthetic identity (phase1 output becomes the identity reference image).

Slow on first run (model downloads). Usage:
    python scripts/smoke_test.py [--preset low_vram] [--phases phase1 phase2]
    python scripts/smoke_test.py --preset fast --traceback
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Use hf_transfer for faster, more reliable model downloads when available.
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zymera.config import Config
from zymera.pipeline import Pipeline

IDENTITY = "smoke_synthetic"
PROMPT = (
    "a friendly synthetic persona in their 30s with short dark hair, "
    "soft smile, wearing a navy sweater, plain grey backdrop, medium close-up"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--phases", nargs="*", default=["phase1", "phase2", "phase3", "phase4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traceback", action="store_true", help="print full tracebacks on failure")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = Config.load(preset=args.preset)
    pipeline = Pipeline(cfg)
    results: dict[str, str] = {}
    tracebacks: dict[str, str] = {}

    for phase in args.phases:
        start = time.time()
        try:
            kwargs = {"identity": IDENTITY} if phase != "phase1" else {}
            if phase == "phase4":
                kwargs["text"] = "Hello, this is a Zymera smoke test."
            paths = pipeline.run(
                phase, PROMPT, seed=args.seed, output=f"smoke_{phase}", **kwargs
            )
            results[phase] = f"PASS  {time.time() - start:.0f}s  {paths[0]}"
            if phase == "phase1":
                # Bootstrap: the generated face becomes the identity reference.
                pipeline.identities.create(IDENTITY, [paths[0]], {"note": "smoke test, fully synthetic"})
        except Exception as exc:
            results[phase] = f"FAIL  {exc}"
            tracebacks[phase] = traceback.format_exc()

    print("\n=== Smoke test results ===")
    for phase, outcome in results.items():
        print(f"  {phase}: {outcome}")

    if args.traceback:
        for phase, tb in tracebacks.items():
            print(f"\n--- {phase} traceback ---\n{tb}")

    failed = [p for p, r in results.items() if not r.startswith("PASS")]
    if failed and not args.traceback:
        print(f"\nRe-run with --traceback to see full stack traces for: {', '.join(failed)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
