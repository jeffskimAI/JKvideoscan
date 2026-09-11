"""Video inspection and metadata probing using ffprobe."""
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.logger import logger


@dataclass
class VideoStreamInfo:
    """Probed video attributes."""
    duration: float
    total_chunks: int
    width: Optional[int] = None
    height: Optional[int] = None
    codec_name: Optional[str] = None
    has_audio: bool = False
    size_bytes: int = 0


class VideoProbeError(RuntimeError):
    """Raised when probing video metadata fails."""
    pass


def probe_video(video_path: Path, chunk_duration_sec: float = 10.0) -> VideoStreamInfo:
    """Probes a video file using ffprobe.

    Args:
        video_path: Path to the video file.
        chunk_duration_sec: Duration of each chunk in seconds (default 10.0).

    Returns:
        VideoStreamInfo containing duration, chunk count, resolution, and audio presence.

    Raises:
        FileNotFoundError: If video_path does not exist.
        VideoProbeError: If ffprobe fails or duration cannot be determined.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,size:stream=width,height,codec_name,codec_type",
        "-of", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ffprobe failed for {video_path}: {e.stderr}")
        raise VideoProbeError(f"ffprobe failed: {e.stderr.strip()}") from e

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise VideoProbeError(f"Failed to parse ffprobe JSON output: {e}") from e

    # Extract duration
    format_info = data.get("format", {})
    raw_duration = format_info.get("duration")
    if not raw_duration:
        # Fallback to streams
        streams = data.get("streams", [])
        for s in streams:
            if "duration" in s:
                raw_duration = s["duration"]
                break

    if not raw_duration:
        raise VideoProbeError(f"Could not determine duration for video: {video_path}")

    try:
        duration = float(raw_duration)
    except ValueError as e:
        raise VideoProbeError(f"Invalid duration value '{raw_duration}': {e}") from e

    if duration <= 0:
        raise VideoProbeError(f"Invalid non-positive duration: {duration}")

    # Inspect streams
    streams = data.get("streams", [])
    width = None
    height = None
    codec_name = None
    has_audio = False

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and width is None:
            width = s.get("width")
            height = s.get("height")
            codec_name = s.get("codec_name")
        elif codec_type == "audio":
            has_audio = True

    size_bytes = int(format_info.get("size", 0))
    total_chunks = max(1, math.ceil(duration / chunk_duration_sec))

    logger.info(
        f"Probed video {video_path.name}: duration={duration:.2f}s, "
        f"total_chunks={total_chunks}, res={width}x{height}, "
        f"codec={codec_name}, has_audio={has_audio}"
    )

    return VideoStreamInfo(
        duration=duration,
        total_chunks=total_chunks,
        width=width,
        height=height,
        codec_name=codec_name,
        has_audio=has_audio,
        size_bytes=size_bytes,
    )
