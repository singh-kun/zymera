"""Phase 4: final composition — mux speech onto video, optional lip sync.

Lip sync strategies:
- ``none``: loop the video to the audio duration and mux (ffmpeg).
- ``wav2lip``: delegate to an external Wav2Lip install via a configurable
  command template (``compose.lipsync.command``), e.g.::

      ["python", "C:/tools/Wav2Lip/inference.py",
       "--checkpoint_path", "C:/tools/Wav2Lip/checkpoints/wav2lip_gan.pth",
       "--face", "{video}", "--audio", "{audio}", "--outfile", "{output}"]
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class ComposeStage:
    def __init__(self, cfg):
        self.params = cfg.section("compose")

    def run(self, video: Path, audio: Path, output: Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        method = (self.params.get("lipsync") or {}).get("method", "none")
        if method == "wav2lip":
            return self._wav2lip(video, audio, output)
        if method != "none":
            raise ValueError(f"Unknown lipsync method '{method}' (expected: none, wav2lip)")
        return self._mux(video, audio, output)

    def _ffmpeg_exe(self) -> str:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()

    def _mux(self, video: Path, audio: Path, output: Path) -> Path:
        cmd = [
            self._ffmpeg_exe(), "-y",
            "-stream_loop", "-1", "-i", str(video),
            "-i", str(audio),
            "-map", "0:v", "-map", "1:a",
            "-c:v", self.params.get("video_codec", "libx264"),
            "-pix_fmt", "yuv420p",
            "-c:a", self.params.get("audio_codec", "aac"),
            "-shortest",
            str(output),
        ]
        self._run(cmd)
        return output

    def _wav2lip(self, video: Path, audio: Path, output: Path) -> Path:
        template = (self.params.get("lipsync") or {}).get("command")
        if not template:
            raise ValueError(
                "compose.lipsync.command must be a list of arguments with "
                "{video}, {audio}, {output} placeholders when method is 'wav2lip'"
            )
        cmd = [str(part).format(video=video, audio=audio, output=output) for part in template]
        self._run(cmd)
        if not output.is_file():
            raise RuntimeError(f"Lip sync command finished but produced no file at {output}")
        return output

    @staticmethod
    def _run(cmd: list[str]) -> None:
        log.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-8:]
            raise RuntimeError(f"Command failed ({cmd[0]}):\n" + "\n".join(tail))
