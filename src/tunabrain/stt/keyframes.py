"""Extract evenly-spaced keyframes from a video, using ffmpeg.

For long-form media we grab a small set of frames spread across the runtime and
caption them to add visual grounding. When the media duration is known we space
the frames evenly (interval = duration / count); otherwise we fall back to one
frame per 60s (spec §4.4). ffmpeg runs off the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract_keyframes(
    video_path: Path,
    count: int = 5,
    *,
    duration_seconds: float | None = None,
    out_dir: Path | None = None,
    scale_width: int = 320,
) -> list[Path]:
    """Extract up to ``count`` evenly-spaced keyframes; return their paths.

    Raises RuntimeError if ffmpeg exits non-zero.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir) if out_dir is not None else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{video_path.stem}-frame-%02d.jpg"

    # Choose a sampling rate that yields ~count frames across the runtime.
    if duration_seconds and duration_seconds > 0:
        interval = max(duration_seconds / count, 1.0)
    else:
        interval = 60.0
    fps_filter = f"fps=1/{interval:g},scale={scale_width}:-1"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        fps_filter,
        "-frames:v",
        str(count),
        "-f",
        "image2",
        str(pattern),
    ]
    logger.info("Extracting %s keyframes: %s (interval=%.1fs)", count, video_path, interval)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg keyframe extraction failed (exit {proc.returncode}): {detail}")

    frames = sorted(out_dir.glob(f"{video_path.stem}-frame-*.jpg"))
    return frames[:count]
