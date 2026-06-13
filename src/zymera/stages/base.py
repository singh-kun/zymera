"""Base class and shared helpers for model-backed pipeline stages."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

_backends_configured = False


def configure_backends(runtime: dict) -> None:
    """One-time global torch backend setup (TF32, SDPA, log filtering).

    PyTorch's scaled_dot_product_attention automatically dispatches to
    flash-attention kernels on Ampere/Ada GPUs — no flash-attn package needed.
    TF32 speeds up fp32 matmuls (scheduler math, VAE) with negligible quality
    impact and is the recommended setting for RTX 30/40-series cards.
    """
    global _backends_configured
    if _backends_configured:
        return
    _backends_configured = True
    import warnings
    import torch

    if runtime.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        flash = torch.backends.cuda.flash_sdp_enabled()
        log.info(
            "Attention backend: SDPA (flash kernels %s)",
            "active" if flash else "unavailable — falling back to mem-efficient",
        )

    # Silence chatty third-party deprecation warnings that add noise without
    # any action a user can take.
    _noisy_patterns = [
        ".*torch_dtype.*deprecated.*",
        ".*CLIPFeatureExtractor.*deprecated.*",
        ".*config attributes.*will be ignored.*",
        ".*huggingface_hub.*cache-system.*symlinks.*",
    ]
    for pat in _noisy_patterns:
        warnings.filterwarnings("ignore", message=pat)

    # Suppress INFO/DEBUG logs from model libraries — tqdm progress bars
    # come from stderr directly and are unaffected.
    for lib in ("diffusers", "transformers", "accelerate", "comtypes", "pyttsx3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def build_quantization_config(qcfg: dict | None, dtype):
    """Build a diffusers PipelineQuantizationConfig from a stage's
    ``quantization`` config block, or None when disabled/unavailable.

    NF4 4-bit quantization shrinks the SDXL UNet from ~5.1 GB to ~1.7 GB,
    letting SDXL-class models fit comfortably in 6 GB VRAM.
    """
    if not qcfg or not qcfg.get("enabled"):
        return None
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        log.warning(
            "quantization.enabled is set but bitsandbytes is not installed; "
            "loading unquantized. Fix with: pip install bitsandbytes"
        )
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    backend = qcfg.get("backend", "bitsandbytes_4bit")
    if backend == "bitsandbytes_4bit":
        quant_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": qcfg.get("quant_type", "nf4"),
            "bnb_4bit_compute_dtype": dtype,
        }
    elif backend == "bitsandbytes_8bit":
        quant_kwargs = {"load_in_8bit": True}
    else:
        raise ValueError(
            f"Unknown quantization backend '{backend}' "
            "(expected: bitsandbytes_4bit, bitsandbytes_8bit)"
        )
    components = qcfg.get("components", ["unet"])
    log.info("Quantizing %s with %s", ", ".join(components), backend)
    return PipelineQuantizationConfig(
        quant_backend=backend,
        quant_kwargs=quant_kwargs,
        components_to_quantize=components,
    )


class Stage(ABC):
    """Lazy-loading stage. Subclasses set ``section`` and implement ``_load``."""

    section: str = ""

    def __init__(self, cfg):
        self.cfg = cfg
        self.runtime = cfg.section("runtime")
        self.params = cfg.section(self.section) if self.section else {}
        self.pipe = None

    @property
    def device(self) -> str:
        device = self.runtime.get("device", "auto")
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def dtype(self):
        import torch

        name = self.runtime.get("dtype", "auto")
        if name == "auto":
            return torch.float16 if self.device == "cuda" else torch.float32
        return getattr(torch, name)

    def ensure_loaded(self) -> None:
        if self.pipe is None:
            configure_backends(self.runtime)
            log.info("Loading %s [%s]", type(self).__name__, self.params.get("model", "-"))
            self.pipe = self._load()
            self._place_and_optimize()

    @abstractmethod
    def _load(self):
        """Build and return the underlying pipeline/model."""

    def _place_and_optimize(self) -> None:
        pipe = self.pipe
        if self.runtime.get("model_cpu_offload") and hasattr(pipe, "enable_model_cpu_offload"):
            try:
                pipe.enable_model_cpu_offload()
            except Exception as exc:
                # Quantized components may refuse offload hooks; stay on-device.
                log.warning("model_cpu_offload unavailable (%s); using direct placement", exc)
                self._to_device(pipe)
        elif hasattr(pipe, "to"):
            self._to_device(pipe)
        if self.runtime.get("attention_slicing") and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "vae"):
            if self.runtime.get("vae_slicing") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if self.runtime.get("vae_tiling") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
        if self.runtime.get("enable_xformers") and hasattr(
            pipe, "enable_xformers_memory_efficient_attention"
        ):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # optional accelerator, never fatal
                log.warning("xFormers unavailable: %s", exc)

    def _to_device(self, pipe) -> None:
        try:
            pipe.to(self.device)
        except Exception as exc:
            # bitsandbytes-quantized modules are already placed by the loader.
            log.debug("pipe.to(%s) skipped: %s", self.device, exc)

    def generator(self, seed: int):
        import torch

        return torch.Generator(device=self.device).manual_seed(seed)

    def unload(self) -> None:
        if self.pipe is not None:
            self.pipe = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_diffusers_pipeline(
    pipeline_cls,
    model: str,
    dtype,
    variant: str | None = None,
    quantization: dict | None = None,
    **kwargs,
):
    """from_pretrained with graceful fallback when the fp16 variant doesn't exist.

    ``quantization`` is the stage's ``quantization`` config block; when enabled
    the selected components are loaded in 4/8-bit via bitsandbytes.
    """
    quant_config = build_quantization_config(quantization, dtype)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    if variant:
        try:
            return pipeline_cls.from_pretrained(model, torch_dtype=dtype, variant=variant, **kwargs)
        except Exception:
            log.debug("Variant '%s' unavailable for %s, retrying without it", variant, model)
    return pipeline_cls.from_pretrained(model, torch_dtype=dtype, **kwargs)


def set_scheduler(pipe, name: str | None, options: dict | None = None) -> None:
    if not name:
        return
    import diffusers

    scheduler_cls = getattr(diffusers, name, None)
    if scheduler_cls is None:
        raise ValueError(f"Unknown scheduler '{name}' (must be a diffusers scheduler class name)")
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **(options or {}))


def load_ip_adapter(pipe, ip_cfg: dict) -> None:
    pipe.load_ip_adapter(
        ip_cfg["repo"],
        subfolder=ip_cfg.get("subfolder"),
        weight_name=ip_cfg["weight_name"],
        image_encoder_folder=ip_cfg.get("image_encoder_folder", "image_encoder"),
    )
    pipe.set_ip_adapter_scale(ip_cfg.get("scale", 0.6))
