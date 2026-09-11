"""Tests for video probing and FFmpeg video segmentation."""
import subprocess
from pathlib import Path
import pytest

from src.video_probe import VideoProbeError, probe_video
from src.video_segmenter import VideoSegmentationError, segment_video


def create_synthetic_video(
    output_path: Path,
    duration_sec: float = 12.0,
    with_audio: bool = True,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Generates a synthetic mp4 test video with visual counter using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use testsrc filter to generate video frames
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size={width}x{height}:rate=24",
    ]

    if with_audio:
        # Add a 440Hz sine wave audio track
        cmd.extend([
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_sec}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
        ])

    cmd.append(str(output_path))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_path


def test_probe_video_with_audio(tmp_path: Path):
    """Test probing a video file that includes an audio track."""
    test_video = tmp_path / "test_with_audio.mp4"
    create_synthetic_video(test_video, duration_sec=15.0, with_audio=True)

    info = probe_video(test_video, chunk_duration_sec=10.0)
    assert abs(info.duration - 15.0) < 0.5
    assert info.total_chunks == 2
    assert info.has_audio is True
    assert info.width == 320
    assert info.height == 240
    assert info.codec_name == "h264"


def test_probe_video_without_audio(tmp_path: Path):
    """Test probing a video file without audio."""
    test_video = tmp_path / "test_no_audio.mp4"
    create_synthetic_video(test_video, duration_sec=8.0, with_audio=False)

    info = probe_video(test_video, chunk_duration_sec=10.0)
    assert abs(info.duration - 8.0) < 0.5
    assert info.total_chunks == 1
    assert info.has_audio is False


def test_probe_nonexistent_file(tmp_path: Path):
    """Test probing nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "does_not_exist.mp4")


def test_segment_video_single_chunk(tmp_path: Path):
    """Test segmenting video shorter than chunk duration."""
    test_video = tmp_path / "short.mp4"
    create_synthetic_video(test_video, duration_sec=6.0, with_audio=True)

    out_dir = tmp_path / "chunks_short"
    chunks = segment_video(test_video, out_dir, chunk_duration_sec=10.0)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.index == 0
    assert chunk.start_time_seconds == 0.0
    assert abs(chunk.end_time_seconds - 6.0) < 0.5
    assert chunk.file_path.exists()
    assert chunk.file_path.stat().st_size > 0


def test_segment_video_multi_chunk(tmp_path: Path):
    """Test segmenting a 25-second video into 3 chunks."""
    test_video = tmp_path / "multi.mp4"
    create_synthetic_video(test_video, duration_sec=25.0, with_audio=True)

    out_dir = tmp_path / "chunks_multi"
    chunks = segment_video(test_video, out_dir, chunk_duration_sec=10.0)

    assert len(chunks) == 3

    # Chunk 0: 0s to 10s
    assert chunks[0].index == 0
    assert chunks[0].start_time_seconds == 0.0
    assert chunks[0].end_time_seconds == 10.0
    assert chunks[0].duration_seconds == 10.0
    assert chunks[0].file_path.name == "chunk_0000.mp4"

    # Chunk 1: 10s to 20s
    assert chunks[1].index == 1
    assert chunks[1].start_time_seconds == 10.0
    assert chunks[1].end_time_seconds == 20.0
    assert chunks[1].duration_seconds == 10.0
    assert chunks[1].file_path.name == "chunk_0001.mp4"

    # Chunk 2: 20s to 25s (remainder)
    assert chunks[2].index == 2
    assert chunks[2].start_time_seconds == 20.0
    assert abs(chunks[2].end_time_seconds - 25.0) < 0.5
    assert chunks[2].file_path.name == "chunk_0002.mp4"

    for chunk in chunks:
        assert chunk.file_path.exists()
        assert chunk.file_path.stat().st_size > 0


def test_segment_video_exact_boundary(tmp_path: Path):
    """Test segmenting a video that is an exact multiple of 10s (20.0s)."""
    test_video = tmp_path / "exact.mp4"
    create_synthetic_video(test_video, duration_sec=20.0, with_audio=False)

    out_dir = tmp_path / "chunks_exact"
    chunks = segment_video(test_video, out_dir, chunk_duration_sec=10.0)

    assert len(chunks) == 2
    assert chunks[0].start_time_seconds == 0.0
    assert chunks[0].end_time_seconds == 10.0
    assert chunks[1].start_time_seconds == 10.0
    assert abs(chunks[1].end_time_seconds - 20.0) < 0.5
