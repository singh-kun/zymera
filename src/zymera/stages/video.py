"""Phase 3: AnimateDiff short-video generation.

Identity consistency comes from IP-Adapter conditioning on a reference image
(when the identity has one), which keeps the face stable across frames. The
base model must be an SD1.5-family checkpoint to match the motion adapter.
"""

from __future__ import annotations

import logging
from pathlib import Path

from zymera.stages.base import Stage, load_diffusers_pipeline, load_ip_adapter, set_scheduler

log = logging.getLogger(__name__)


class VideoStage(Stage):
    section = "video"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._ip_loaded = False

    def _load(self):
        from diffusers import AnimateDiffPipeline, MotionAdapter

        adapter = MotionAdapter.from_pretrained(
            self.params["motion_adapter"], torch_dtype=self.dtype
        )
        pipe = load_diffusers_pipeline(
            AnimateDiffPipeline,
            self.params["model"],
            self.dtype,
            self.params.get("variant"),
            quantization=self.params.get("quantization"),
            motion_adapter=adapter,
        )
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        # Load IP-Adapter before _place_and_optimize() so the image_encoder is
        # included when enable_model_cpu_offload() installs its CPU-offload hooks.
        ip_cfg = self.params.get("ip_adapter") or {}
        if ip_cfg.get("enabled"):
            load_ip_adapter(pipe, ip_cfg)
            self._ip_loaded = True
        set_scheduler(pipe, self.params.get("scheduler"), self.params.get("scheduler_options"))
        return pipe

    def run(self, prompt: str, negative: str, seed: int, reference: Path | None = None) -> list:
        self.ensure_loaded()
        p = self.params
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=p["num_frames"],
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance_scale"],
            width=p["width"],
            height=p["height"],
            generator=self.generator(seed),
        )
        if reference is not None:
            if self._ip_loaded:
                from PIL import Image

                kwargs["ip_adapter_image"] = Image.open(reference).convert("RGB")
            else:
                log.info("video.ip_adapter disabled; generating without identity conditioning")
        return self.pipe(**kwargs).frames[0]
