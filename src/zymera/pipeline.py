"""Pipeline orchestrator: maps phases to stage sequences and owns output IO.

Phases (legacy names kept for compatibility):
- phase1 / image          — text-to-image
- phase2 / identity-image — identity-conditioned image (requires reference)
- phase3 / video          — short video, identity-conditioned when refs exist
- phase4 / talking-video  — video + speech + lip-synced composition
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import random
from pathlib import Path

from zymera.config import Config
from zymera.identity import Identity, IdentityStore
from zymera.prompts import BuiltPrompt, PromptBuilder
from zymera.stages import get_stage
from zymera.stages.compose import ComposeStage

log = logging.getLogger(__name__)

PHASES: dict[str, str] = {
    "phase1": "image",
    "phase2": "identity_image",
    "phase3": "video",
    "phase4": "talking_video",
    "image": "image",
    "identity-image": "identity_image",
    "video": "video",
    "talking-video": "talking_video",
}

MAX_SEED = 2**32 - 1


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.prompts = PromptBuilder.from_file(self.cfg.get("paths.prompts_file"))
        self.identities = IdentityStore(self.cfg.get("paths.identities_dir"))

    def run(
        self,
        phase: str,
        prompt: str,
        identity: str | None = None,
        text: str | None = None,
        output: str | None = None,
        style: str | None = None,
        seed: int | None = None,
    ) -> list[Path]:
        """Run one phase end to end. Returns the list of produced files."""
        kind = PHASES.get(phase)
        if kind is None:
            raise ValueError(f"Unknown phase '{phase}'. Available: {', '.join(sorted(PHASES))}")

        ident = self.identities.load(identity) if identity else None
        built = self.prompts.build(
            prompt,
            style=style or self.cfg.get("prompt.style"),
            enhance=bool(self.cfg.get("prompt.enhance", True)),
        )
        if seed is None:
            seed = self.cfg.get("runtime.seed")
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        log.info("Phase=%s style=%s seed=%d", kind, style or self.cfg.get("prompt.style"), seed)
        log.debug("Prompt: %s", built.prompt)

        out_dir = Path(self.cfg.get("paths.outputs_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        basename, ext = self._naming(kind, output, identity, seed)

        runner = {
            "image": self._run_image,
            "identity_image": self._run_identity_image,
            "video": self._run_video,
            "talking_video": self._run_talking_video,
        }[kind]
        return runner(built, ident, text, seed, out_dir, basename, ext)

    # ----------------------------------------------------------- phase runners

    def _run_image(self, built, ident, text, seed, out_dir, basename, ext) -> list[Path]:
        images = get_stage("text2image", self.cfg).run(built.prompt, built.negative, seed)
        return self._save_images(images, out_dir, basename, ext or ".png", "image", built, seed)

    def _run_identity_image(self, built, ident, text, seed, out_dir, basename, ext) -> list[Path]:
        reference = self._require_reference(ident, "phase2 (identity-image)")
        images = get_stage("identity_image", self.cfg).run(
            built.prompt, built.negative, reference, seed
        )
        return self._save_images(
            images, out_dir, basename, ext or ".png", "identity_image", built, seed,
            reference=str(reference),
        )

    def _run_video(self, built, ident, text, seed, out_dir, basename, ext) -> list[Path]:
        path = out_dir / f"{basename}{ext or '.mp4'}"
        self._render_video(built, ident, seed, path)
        self._write_metadata(out_dir, basename, "video", built, seed, [path])
        return [path]

    def _run_talking_video(self, built, ident, text, seed, out_dir, basename, ext) -> list[Path]:
        if not text:
            raise ValueError("phase4 (talking-video) requires --text for speech generation")

        video_path = out_dir / f"{basename}.video.mp4"
        audio_path = out_dir / f"{basename}.wav"
        final_path = out_dir / f"{basename}{ext or '.mp4'}"

        self._render_video(built, ident, seed, video_path)
        get_stage("tts", self.cfg).run(text, audio_path)
        ComposeStage(self.cfg).run(video_path, audio_path, final_path)

        if not self.cfg.get("compose.keep_intermediates", False):
            for intermediate in (video_path, audio_path):
                intermediate.unlink(missing_ok=True)
        self._write_metadata(out_dir, basename, "talking_video", built, seed, [final_path], text=text)
        return [final_path]

    # ------------------------------------------------------------------ helpers

    def _render_video(self, built: BuiltPrompt, ident: Identity | None, seed: int, path: Path) -> None:
        from diffusers.utils import export_to_video

        reference = ident.primary_reference() if ident else None
        frames = get_stage("video", self.cfg).run(built.prompt, built.negative, seed, reference)
        export_to_video(frames, str(path), fps=self.cfg.get("video.fps", 8))

    @staticmethod
    def _require_reference(ident: Identity | None, label: str) -> Path:
        if ident is None:
            raise ValueError(f"{label} requires --identity")
        reference = ident.primary_reference()
        if reference is None:
            raise FileNotFoundError(
                f"Identity '{ident.identity_id}' has no usable reference image; "
                f"add one with: zymera identity add {ident.identity_id} --image <path>"
            )
        return reference

    @staticmethod
    def _naming(kind: str, output: str | None, identity: str | None, seed: int) -> tuple[str, str]:
        if output:
            path = Path(output)
            return path.stem, path.suffix
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{kind}_{identity or 'anon'}_{timestamp}_{seed}", ""

    def _save_images(
        self, images, out_dir: Path, basename: str, ext: str, kind: str,
        built: BuiltPrompt, seed: int, **extra,
    ) -> list[Path]:
        paths = []
        for i, image in enumerate(images):
            name = basename if len(images) == 1 else f"{basename}_{i}"
            path = out_dir / f"{name}{ext}"
            image.save(path)
            paths.append(path)
        self._write_metadata(out_dir, basename, kind, built, seed, paths, **extra)
        return paths

    def _write_metadata(
        self, out_dir: Path, basename: str, kind: str,
        built: BuiltPrompt, seed: int, paths: list[Path], **extra,
    ) -> None:
        payload = {
            "phase": kind,
            "prompt": built.prompt,
            "negative_prompt": built.negative,
            "seed": seed,
            "files": [p.name for p in paths],
            "params": self.cfg.section(kind if kind != "talking_video" else "video"),
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        (out_dir / f"{basename}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
