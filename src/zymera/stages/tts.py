"""Phase 4: text-to-speech.

Primary backend: Coqui TTS (downloaded from HuggingFace/GitHub on first use).
Fallback: pyttsx3 (system TTS, no download, works offline) — used automatically
when the Coqui model cannot be loaded (e.g. network failure, disk quota).

For voice cloning set ``speech.model`` to an XTTS v2 model and provide
``speech.speaker_wav`` pointing at a consented reference voice file.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from zymera.stages.base import Stage

log = logging.getLogger(__name__)


class SpeechStage(Stage):
    section = "speech"

    def _load(self):
        model = self.params["model"]
        try:
            import contextlib
            import io
            from TTS.api import TTS

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # Coqui TTS prints verbose audio-processor config to stdout.
                with contextlib.redirect_stdout(io.StringIO()):
                    tts = TTS(model)
            try:
                tts.to(self.device)
            except Exception as exc:
                log.debug("TTS device placement: %s", exc)
            return ("coqui", tts)
        except Exception as exc:
            log.warning(
                "Coqui TTS '%s' unavailable (%s) — using pyttsx3 system TTS. "
                "For production quality re-run after confirming 'zymera doctor' shows TTS OK.",
                model, type(exc).__name__,
            )
            return self._load_pyttsx3()

    def _load_pyttsx3(self):
        import logging as _logging
        # pyttsx3 initialises COM/comtypes on Windows and emits INFO messages
        # through the comtypes logger; keep them out of the user's terminal.
        _logging.getLogger("comtypes").setLevel(_logging.WARNING)
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            return ("pyttsx3", engine)
        except Exception as exc:
            raise RuntimeError(
                "Both Coqui TTS and pyttsx3 are unavailable.\n"
                "  Coqui: check 'zymera doctor' or pick a cached model.\n"
                "  pyttsx3: pip install pyttsx3"
            ) from exc

    def _place_and_optimize(self) -> None:
        pass  # device placement is handled in _load

    def run(self, text: str, output: Path) -> Path:
        self.ensure_loaded()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        backend, engine = self.pipe
        if backend == "coqui":
            kwargs = {}
            if self.params.get("speaker_wav"):
                kwargs["speaker_wav"] = self.params["speaker_wav"]
                kwargs["language"] = self.params.get("language", "en")
            engine.tts_to_file(text=text, file_path=str(output), **kwargs)
        else:
            log.info("Using pyttsx3 (system TTS) — output quality will be basic")
            engine.save_to_file(text, str(output))
            engine.runAndWait()
        return output
