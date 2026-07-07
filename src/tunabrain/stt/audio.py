"""Extract audio from a video for STT, using ffmpeg.

Both STT backends accept video directly, but extracting a clean 16 kHz mono PCM
WAV first removes a variable from STT performance and is cheap (~0.5s for a 5min
video). ffmpeg is CPU-bound, so we run it out of the event loop via
``asyncio.create_subprocess_exec`` (spec §4.4).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract_audio(video_path: Path, *, out_path: Path | None = None) -> Path:
    """Extract a 16 kHz mono PCM WAV from ``video_path`` and return its path.

    Raises RuntimeError if ffmpeg exits non-zero.
    """
    video_path = Path(video_path)
    if out_path is None:
        out_path = video_path.with_suffix(".wav")
    out_path = Path(out_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    logger.info("Extracting audio: %s -> %s", video_path, out_path)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg audio extraction failed (exit {proc.returncode}): {detail}")
    return out_path
