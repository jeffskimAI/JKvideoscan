"""FFmpeg video segmentation into strict 10-second chunks."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import List

from src.logger import logger
from src.video_probe import VideoStreamInfo, probe_video


@dataclass
class VideoChunk:
    """Represents a segmented video chunk."""
    index: int
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    file_path: Path


class VideoSegmentationError(RuntimeError):
    """Raised when video segmentation fails."""
    pass


def segment_video(
    video_path: Path,
    output_dir: Path,
    chunk_duration_sec: float = 10.0,
) -> List[VideoChunk]:
    """Segments a video file into strict 10-second sequential chunks using FFmpeg.

    Args:
        video_path: Absolute or relative path to the input video.
        output_dir: Directory to write chunk files.
        chunk_duration_sec: Segment length in seconds (default 10.0).

    Returns:
        List of VideoChunk objects sorted by chunk index.

    Raises:
        FileNotFoundError: If input video does not exist.
        VideoSegmentationError: If ffmpeg fails or produces zero chunks.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Probe video to determine duration, stream properties
    stream_info: VideoStreamInfo = probe_video(video_path, chunk_duration_sec=chunk_duration_sec)
    total_duration = stream_info.duration

    logger.info(
        f"Segmenting '{video_path.name}' ({total_duration:.2f}s) "
        f"into {stream_info.total_chunks} chunks of {chunk_duration_sec}s each."
    )

    # 2. Attempt fast stream copy (-c copy) first without re-encoding
    output_pattern = str(output_dir / "chunk_%04d.mp4")
    copy_cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-i", str(video_path),
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(chunk_duration_sec),
        "-reset_timestamps", "1",
        output_pattern,
    ]

    use_reencode = False
    try:
        subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        chunk_files = sorted(output_dir.glob("chunk_*.mp4"))
        if not chunk_files or (stream_info.total_chunks > 1 and len(chunk_files) < max(2, stream_info.total_chunks // 2)):
            logger.info("Stream copy produced insufficient keyframe splits; falling back to ultrafast re-encode...")
            use_reencode = True
            for f in chunk_files:
                f.unlink(missing_ok=True)
    except subprocess.CalledProcessError as e:
        logger.info(f"Stream copy failed ({e.stderr.strip()}); falling back to ultrafast re-encode...")
        use_reencode = True

    if use_reencode:
        reencode_cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
        ]
        if stream_info.has_audio:
            reencode_cmd.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            reencode_cmd.extend(["-an"])

        reencode_cmd.extend([
            "-f", "segment",
            "-segment_time", str(chunk_duration_sec),
            "-reset_timestamps", "1",
            "-force_key_frames", f"expr:gte(t,n_forced*{chunk_duration_sec})",
            output_pattern,
        ])
        try:
            subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg segmentation failed: {e.stderr}")
            raise VideoSegmentationError(f"FFmpeg failed: {e.stderr.strip()}") from e

    # 3. Collect and validate generated chunks
    chunk_files = sorted(output_dir.glob("chunk_*.mp4"))
    if not chunk_files:
        raise VideoSegmentationError(f"FFmpeg completed without generating chunk files in {output_dir}")

    chunks: List[VideoChunk] = []
    for idx, chunk_file in enumerate(chunk_files):
        if chunk_file.stat().st_size == 0:
            raise VideoSegmentationError(f"Generated chunk file is empty: {chunk_file}")

        start_time = round(idx * chunk_duration_sec, 2)
        end_time = round(min((idx + 1) * chunk_duration_sec, total_duration), 2)
        duration = round(end_time - start_time, 2)

        chunks.append(
            VideoChunk(
                index=idx,
                start_time_seconds=start_time,
                end_time_seconds=end_time,
                duration_seconds=duration,
                file_path=chunk_file,
            )
        )

    logger.info(f"Successfully generated {len(chunks)} chunks for {video_path.name}")
    return chunks
