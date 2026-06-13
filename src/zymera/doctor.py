"""Environment health checks: ``zymera doctor``.

Probing lives in :mod:`zymera.capabilities` (shared with the planner); this
module just formats the results for the terminal.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from zymera.capabilities import probe, probe_ffmpeg


def _check_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "installed")
    except ImportError:
        return False, "not installed"


def run_doctor(cfg) -> int:
    prof = probe()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 10), sys.version.split()[0]))

    torch_ok = bool(prof.torch_version)
    if not torch_ok:
        checks.append(("torch / GPU", False, "not installed — pip install -r requirements.txt"))
    elif prof.has_cuda:
        detail = (
            f"{prof.torch_version} | CUDA {prof.cuda_version} | "
            f"{prof.gpu_name} ({prof.vram_gb:.1f} GB)"
        )
        checks.append(("torch / GPU", True, detail))
    else:
        checks.append(
            ("torch / GPU", True,
             f"{prof.torch_version} | CUDA unavailable — generation will be very slow on CPU")
        )

    if torch_ok:
        if not prof.has_cuda:
            checks.append(("attention", True, "CPU SDPA (no GPU acceleration)"))
        elif prof.flash_sdpa:
            checks.append(("attention", True,
                           "SDPA with flash-attention kernels (no flash-attn package needed)"))
        else:
            checks.append(("attention", True,
                           "SDPA mem-efficient kernels (flash unavailable on this GPU)"))
        if prof.has_bitsandbytes:
            import bitsandbytes

            checks.append(("quantization", True,
                           f"bitsandbytes {bitsandbytes.__version__} — 4/8-bit model loading available"))
        else:
            checks.append(("quantization", True,
                           "bitsandbytes not installed — 'pip install bitsandbytes' enables --preset quantized"))

    for module in ("diffusers", "transformers", "accelerate", "peft"):
        ok, detail = _check_module(module)
        checks.append((module, ok, detail))
    ok, detail = _check_module("TTS")
    checks.append(("TTS (phase4 speech)", ok, detail))
    ffmpeg_ok, ffmpeg_detail = probe_ffmpeg()
    checks.append(("ffmpeg (phase3/4)", ffmpeg_ok, ffmpeg_detail))

    identities_dir = Path(cfg.get("paths.identities_dir", "identities"))
    count = len(list(identities_dir.glob("*/metadata.json"))) if identities_dir.is_dir() else 0
    checks.append(("identities", True, f"{count} found in {identities_dir}"))
    checks.append(("outputs dir", True, str(Path(cfg.get("paths.outputs_dir", "outputs")).resolve())))
    checks.append(("content policy", True, f"mode={cfg.get('registry.content_mode', 'sfw')} (real-person always blocked)"))

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'OK' if ok else '!!'}] {name.ljust(width)}  {detail}")

    if not torch_ok:
        print("\nCritical: torch is required for all generation phases.")
        return 1
    print("\nEnvironment looks usable. Run 'zymera generate --help' to get started.")
    return 0
