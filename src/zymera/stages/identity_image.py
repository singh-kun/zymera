"""Phase 2: identity-conditioned image generation.

Default method in VRAM-limited presets (fast/low_vram/quantized) is img2img —
the reference image is noise-blended into the generation directly and is
reliable on all GPU sizes. The ip_adapter method is available for balanced/
quality presets on 8 GB+ GPUs: it gives stronger style independence at the cost
of needing the ViT-H image encoder (~300 MB for SD1.5, 3.7 GB for SDXL).

If ip_adapter inference fails at runtime the stage automatically retries with
img2img. The fallback correctly strips IP-Adapter state from the shared UNet so
the img2img pipeline does not require ip_adapter_image embeddings.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from zymera.stages.base import Stage, load_diffusers_pipeline, load_ip_adapter, set_scheduler

log = logging.getLogger(__name__)


class IdentityImageStage(Stage):
    section = "identity_image"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._ip_loaded = False

    def _load(self):
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

        method = self.params.get("method", "ip_adapter")
        pipeline_cls = (
            AutoPipelineForImage2Image if method == "img2img" else AutoPipelineForText2Image
        )

        kwargs = {}
        if self.params.get("vae"):
            from diffusers import AutoencoderKL

            kwargs["vae"] = AutoencoderKL.from_pretrained(
                self.params["vae"], torch_dtype=self.dtype
            )
        pipe = load_diffusers_pipeline(
            pipeline_cls,
            self.params["model"],
            self.dtype,
            self.params.get("variant"),
            quantization=self.params.get("quantization"),
            **kwargs,
        )
        set_scheduler(pipe, self.params.get("scheduler"), self.params.get("scheduler_options"))
        self._load_loras(pipe)
        if method != "img2img" and self.params.get("ip_adapter"):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch_dtype.*")
                load_ip_adapter(pipe, self.params["ip_adapter"])
            self._ip_loaded = True
        return pipe

    def run(self, prompt: str, negative: str, reference: Path, seed: int) -> list:
        self.ensure_loaded()
        from PIL import Image

        p = self.params
        ref_image = Image.open(reference).convert("RGB")
        common = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance_scale"],
            num_images_per_prompt=p.get("num_images", 1),
            generator=self.generator(seed),
        )
        if p.get("method", "ip_adapter") == "img2img":
            return self._run_img2img(ref_image, p, common)
        try:
            result = self.pipe(
                ip_adapter_image=ref_image,
                width=p["width"],
                height=p["height"],
                **common,
            )
            return result.images
        except Exception as exc:
            log.warning(
                "IP-Adapter inference failed (%s: %s); retrying with img2img fallback.",
                type(exc).__name__,
                exc,
            )
            return self._run_img2img(ref_image, p, common)

    def _run_img2img(self, ref_image, p: dict, common: dict) -> list:
        # If IP-Adapter was loaded (and failed), its attention processors are still
        # installed on the shared UNet. img2img calls the same UNet without providing
        # ip_adapter_image, so encoder_hid_proj receives None → crash. Strip it first.
        self._strip_ip_adapter()

        from diffusers import AutoPipelineForImage2Image

        img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
        ref_resized = ref_image.resize((p["width"], p["height"]))
        result = img2img_pipe(
            image=ref_resized,
            strength=p.get("strength", 0.65),
            **common,
        )
        return result.images

    def _strip_ip_adapter(self) -> None:
        if not self._ip_loaded:
            return
        unet = getattr(self.pipe, "unet", None)
        if unet is None:
            return
        if getattr(unet, "encoder_hid_proj", None) is not None:
            unet.set_default_attn_processor()
            unet.encoder_hid_proj = None
            log.debug("Stripped IP-Adapter from UNet for img2img")
        self._ip_loaded = False
