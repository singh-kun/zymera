"""Phase 1: prompt-driven text-to-image generation."""

from __future__ import annotations

from zymera.stages.base import Stage, load_diffusers_pipeline, set_scheduler


class Text2ImageStage(Stage):
    section = "image"

    def _load(self):
        from diffusers import AutoPipelineForText2Image

        kwargs = {}
        if self.params.get("vae"):
            from diffusers import AutoencoderKL

            kwargs["vae"] = AutoencoderKL.from_pretrained(self.params["vae"], torch_dtype=self.dtype)
        pipe = load_diffusers_pipeline(
            AutoPipelineForText2Image,
            self.params["model"],
            self.dtype,
            self.params.get("variant"),
            quantization=self.params.get("quantization"),
            **kwargs,
        )
        set_scheduler(pipe, self.params.get("scheduler"), self.params.get("scheduler_options"))
        self._load_loras(pipe)
        return pipe

    def run(self, prompt: str, negative: str, seed: int) -> list:
        self.ensure_loaded()
        p = self.params
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance_scale"],
            width=p["width"],
            height=p["height"],
            num_images_per_prompt=p.get("num_images", 1),
            generator=self.generator(seed),
        )
        return result.images
