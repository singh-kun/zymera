"""Hardware/software capability probing.

A small, dependency-light layer that answers "what can this machine run?" so the
planner (and ``zymera doctor``) can choose presets and models that actually fit.
All probes degrade gracefully when a library is missing — nothing here imports
torch at module load time.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass, field


@dataclass
class CapabilityProfile:
    """A snapshot of the current machine's generation-relevant capabilities."""

    has_cuda: bool = False
    gpu_name: str = ""
    vram_gb: float = 0.0
    torch_version: str = ""
    cuda_version: str = ""
    flash_sdpa: bool = False
    has_bitsandbytes: bool = False
    has_ffmpeg: bool = False
    has_coqui_tts: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def tier(self) -> str:
        """Coarse capability tier used by the heuristic planner."""
        if not self.has_cuda:
            return "cpu"
        if self.vram_gb < 6:
            return "tiny"      # <6 GB — quantized / sd15
        if self.vram_gb < 8:
            return "small"     # 6-8 GB — low_vram / fast
        if self.vram_gb < 12:
            return "medium"    # 8-12 GB — balanced
        return "large"         # 12 GB+ — quality

    def summary(self) -> str:
        if not self.has_cuda:
            return "CPU only (no CUDA GPU) — generation will be very slow"
        return f"{self.gpu_name} ({self.vram_gb:.1f} GB VRAM), tier={self.tier}"


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except ImportError:
        return None


def probe_ffmpeg() -> tuple[bool, str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return True, exe
    try:
        import imageio_ffmpeg

        return True, f"bundled ({imageio_ffmpeg.get_ffmpeg_exe()})"
    except Exception:
        return False, "not found — install ffmpeg or pip install imageio-ffmpeg"


def probe() -> CapabilityProfile:
    """Build a CapabilityProfile by probing torch, bitsandbytes, ffmpeg, TTS."""
    prof = CapabilityProfile()

    try:
        import torch

        prof.torch_version = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            prof.has_cuda = True
            prof.gpu_name = props.name
            prof.vram_gb = props.total_memory / 1024**3
            prof.cuda_version = torch.version.cuda or ""
            prof.flash_sdpa = bool(torch.backends.cuda.flash_sdp_enabled())
        else:
            prof.notes.append("CUDA unavailable — CPU fallback")
    except ImportError:
        prof.notes.append("torch not installed")

    prof.has_bitsandbytes = _module_version("bitsandbytes") is not None
    prof.has_coqui_tts = _module_version("TTS") is not None
    prof.has_ffmpeg, _ = probe_ffmpeg()
    return prof
