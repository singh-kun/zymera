"""Environment health checks: ``zymera doctor``."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path


def _check_torch() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "not installed — pip install -r requirements.txt"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        return True, f"{torch.__version__} | CUDA {torch.version.cuda} | {props.name} ({vram:.1f} GB)"
    return True, f"{torch.__version__} | CUDA unavailable — generation will be very slow on CPU"


def _check_module(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "installed")
    except ImportError:
        return False, "not installed"


def _check_attention() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch missing"
    if not torch.cuda.is_available():
        return True, "CPU SDPA (no GPU acceleration)"
    flash = torch.backends.cuda.flash_sdp_enabled()
    if flash:
        return True, "SDPA with flash-attention kernels (no flash-attn package needed)"
    return True, "SDPA mem-efficient kernels (flash unavailable on this GPU)"


def _check_quantization() -> tuple[bool, str]:
    try:
        import bitsandbytes

        return True, f"bitsandbytes {bitsandbytes.__version__} — 4/8-bit model loading available"
    except ImportError:
        return True, "bitsandbytes not installed — 'pip install bitsandbytes' enables --preset quantized"


def _check_ffmpeg() -> tuple[bool, str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return True, exe
    try:
        import imageio_ffmpeg

        return True, f"bundled ({imageio_ffmpeg.get_ffmpeg_exe()})"
    except Exception:
        return False, "not found — install ffmpeg or pip install imageio-ffmpeg"


def run_doctor(cfg) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 10), sys.version.split()[0]))
    ok, detail = _check_torch()
    checks.append(("torch / GPU", ok, detail))
    torch_ok = ok
    if torch_ok:
        ok, detail = _check_attention()
        checks.append(("attention", ok, detail))
        ok, detail = _check_quantization()
        checks.append(("quantization", ok, detail))
    for module in ("diffusers", "transformers", "accelerate"):
        ok, detail = _check_module(module)
        checks.append((module, ok, detail))
    ok, detail = _check_module("TTS")
    checks.append(("TTS (phase4 speech)", ok, detail))
    ok, detail = _check_ffmpeg()
    checks.append(("ffmpeg (phase3/4)", ok, detail))

    identities_dir = Path(cfg.get("paths.identities_dir", "identities"))
    count = len(list(identities_dir.glob("*/metadata.json"))) if identities_dir.is_dir() else 0
    checks.append(("identities", True, f"{count} found in {identities_dir}"))
    checks.append(("outputs dir", True, str(Path(cfg.get("paths.outputs_dir", "outputs")).resolve())))

    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"  [{'OK' if ok else '!!'}] {name.ljust(width)}  {detail}")

    if not torch_ok:
        print("\nCritical: torch is required for all generation phases.")
        return 1
    print("\nEnvironment looks usable. Run 'zymera generate --help' to get started.")
    return 0
