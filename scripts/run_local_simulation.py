#!/usr/bin/env python3
"""Local simulation runner for Jeff's VideoScan Pipeline.

Generates a sample video and config, segments using FFmpeg into 10-second chunks,
and validates dynamic schema and metadata extraction.
"""
import json
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalog import parse_config_payload
from src.logger import logger
from src.video_probe import probe_video
from src.video_segmenter import segment_video
from src.schema_generator import build_dynamic_metadata_model, construct_extraction_prompt
from tests.test_video_segmenter import create_synthetic_video


def run_simulation(duration_sec: float = 25.0):
    print("=" * 70)
    print("AI-POWERED VIDEO METADATA EXTRACTION PIPELINE - LOCAL SIMULATION")
    print("=" * 70)

    work_dir = Path("/tmp/jeffsvideoscan_simulation")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    video_name = "demo_keynote"
    video_file = work_dir / f"{video_name}.mp4"
    config_file = work_dir / f"{video_name}_config.json"

    # 1. Create synthetic video
    print(f"\n[1/5] Generating synthetic video ({duration_sec}s)...")
    create_synthetic_video(video_file, duration_sec=duration_sec, with_audio=True)
    print(f"      Created video at: {video_file} ({video_file.stat().st_size} bytes)")

    # 2. Create configuration file
    print("\n[2/5] Creating configuration file with target_metadata...")
    config_data = {
        "target_metadata": [
            "scene_description",
            "detected_objects",
            "speaker_count",
            "transcript",
            "chunk_summary",
            "overall_sentiment",
        ]
    }
    config_file.write_text(json.dumps(config_data, indent=2))
    target_keys = parse_config_payload(config_data)
    print(f"      Validated {len(target_keys)} target keys: {target_keys}")

    # 3. Probe video metadata
    print("\n[3/5] Probing video with ffprobe...")
    stream_info = probe_video(video_file, chunk_duration_sec=10.0)
    print(f"      Duration: {stream_info.duration:.2f}s | Chunks: {stream_info.total_chunks} | Has Audio: {stream_info.has_audio}")

    # 4. Strict 10-second chunking with FFmpeg
    print("\n[4/5] Executing strict 10-second FFmpeg segmentation...")
    chunks_dir = work_dir / "chunks"
    chunks = segment_video(video_file, chunks_dir, chunk_duration_sec=10.0)
    print(f"      Generated {len(chunks)} chunk files:")
    for c in chunks:
        print(f"      - Chunk #{c.index:02d}: [{c.start_time_seconds:05.1f}s -> {c.end_time_seconds:05.1f}s] ({c.duration_seconds:04.1f}s) -> {c.file_path.name}")

    # 5. Dynamic Schema & Extraction Simulation
    print("\n[5/5] Synthesizing Dynamic Schema & Gemini Extraction...")
    ModelClass = build_dynamic_metadata_model(target_keys)
    print(f"      Dynamic Pydantic Model Schema fields: {list(ModelClass.model_fields.keys())}")

    print("\n" + "-" * 70)
    print("SIMULATED FIRESTORE OUTPUT RECORDS:")
    print("-" * 70)

    # Root Document
    root_doc = {
        "video_id": video_name,
        "filename": video_file.name,
        "duration_seconds": stream_info.duration,
        "total_chunks": len(chunks),
        "target_metadata": target_keys,
        "status": "COMPLETED",
    }
    print("Firestore: videos/" + video_name)
    print(json.dumps(root_doc, indent=2))

    # Chunk Subcollection Documents
    for c in chunks:
        chunk_doc = {
            "chunk_index": c.index,
            "start_time_seconds": c.start_time_seconds,
            "end_time_seconds": c.end_time_seconds,
            "duration_seconds": c.duration_seconds,
            "extracted_metadata": {
                "scene_description": f"Visual scene during chunk {c.index}",
                "detected_objects": ["display_screen", "speaker_podium"],
                "speaker_count": 1,
                "transcript": f"Dialogue segment for seconds {c.start_time_seconds} to {c.end_time_seconds}",
                "chunk_summary": f"Key takeaway of chunk {c.index}",
                "overall_sentiment": "Positive",
            },
            "model_version": "gemini-3.8",
        }
        print(f"\nFirestore: videos/{video_name}/chunks/{c.index:04d}")
        print(json.dumps(chunk_doc, indent=2))

    print("\n" + "=" * 70)
    print("LOCAL SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation(25.0)
