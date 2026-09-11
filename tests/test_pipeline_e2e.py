"""End-to-end integration tests for the video metadata extraction pipeline."""
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from src.main import app, orchestrator, extract_storage_event_data
from src.video_segmenter import VideoChunk
from tests.test_firestore_repo import MockFirestoreClient
from tests.test_video_segmenter import create_synthetic_video


@pytest.fixture
def test_client():
    return TestClient(app)


def test_cloudevent_extraction():
    """Test extracting bucket and object from structured and binary CloudEvents."""
    # Structured mode
    structured = {
        "specversion": "1.0",
        "type": "google.cloud.storage.object.v1.finalized",
        "data": {"bucket": "video-bucket", "name": "clip.mp4"},
    }
    b, n = extract_storage_event_data(structured, {})
    assert b == "video-bucket"
    assert n == "clip.mp4"

    # Direct/Binary mode
    binary = {"bucket": "video-bucket", "name": "clip_config.json"}
    b, n = extract_storage_event_data(binary, {})
    assert b == "video-bucket"
    assert n == "clip_config.json"


def test_health_endpoints(test_client):
    """Test health and liveness endpoints."""
    res = test_client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}

    res = test_client.get("/livez")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}

    res = test_client.get("/")
    assert res.status_code == 200
    assert res.json()["service"] == "jeffsvideoscan"


def test_pipeline_e2e_successful_processing(tmp_path: Path, monkeypatch, test_client):
    """Simulates an entire end-to-end CloudEvent trigger with synthetic video chunking and metadata extraction."""
    # 1. Create a 22-second synthetic video (will chunk into 3 chunks: 0-10, 10-20, 20-22)
    sample_video = tmp_path / "sample_keynote.mp4"
    create_synthetic_video(sample_video, duration_sec=22.0, with_audio=True)

    config_content = {
        "target_metadata": [
            "scene_description",
            "detected_objects",
            "speaker_count",
            "chunk_summary",
        ]
    }

    # 2. Mock StorageManager
    mock_storage = MagicMock()
    mock_storage.derive_pair_paths.return_value = (
        "sample_keynote",
        "sample_keynote.mp4",
        "sample_keynote_config.json",
    )
    mock_storage.check_blob_exists.return_value = True

    def fake_download(bucket, obj, dest):
        dest.write_bytes(sample_video.read_bytes())
        return dest

    mock_storage.download_blob_to_file.side_effect = fake_download
    mock_storage.download_json_as_dict.return_value = config_content

    # 3. Mock FirestoreRepository using our MockFirestoreClient
    mock_db = MockFirestoreClient()
    from src.firestore_repo import FirestoreRepository
    firestore_repo = FirestoreRepository(client=mock_db, collection_name="videos")

    # 4. Mock GeminiExtractor
    from src.gemini_extractor import GeminiExtractor
    mock_gemini = MagicMock(spec=GeminiExtractor)
    mock_gemini.model_name = "gemini-3.8"

    def fake_extract(chunk: VideoChunk, target_keys: list):
        return {
            "scene_description": f"Auditorium stage during chunk {chunk.index}",
            "detected_objects": ["stage", "screen", "microphone"],
            "speaker_count": 1,
            "chunk_summary": f"Summary for chunk interval {chunk.start_time_seconds}-{chunk.end_time_seconds}s",
        }

    mock_gemini.extract_chunk_metadata.side_effect = fake_extract

    # Wire up mock orchestrator dependencies
    orchestrator.storage_manager = mock_storage
    orchestrator.firestore_repo = firestore_repo
    orchestrator.gemini_extractor = mock_gemini
    orchestrator.temp_base_dir = tmp_path / "scratch"

    # 5. Send CloudEvent HTTP POST request
    cloudevent_payload = {
        "specversion": "1.0",
        "type": "google.cloud.storage.object.v1.finalized",
        "source": "//storage.googleapis.com/buckets/test-bucket",
        "id": "event-12345",
        "data": {
            "bucket": "test-bucket",
            "name": "sample_keynote.mp4",
        },
    }

    response = test_client.post("/events", json=cloudevent_payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "COMPLETED"
    assert res_data["video_id"] == "sample_keynote"
    assert res_data["total_chunks"] == 3

    # 6. Verify Firestore persistence
    video_doc = firestore_repo.get_video_record("sample_keynote")
    assert video_doc is not None
    assert video_doc["status"] == "COMPLETED"
    assert video_doc["total_chunks"] == 3
    assert abs(video_doc["duration_seconds"] - 22.0) < 0.5
    assert video_doc["target_metadata"] == config_content["target_metadata"]

    # Verify chunk subcollection
    chunks = firestore_repo.get_video_chunks("sample_keynote")
    assert len(chunks) == 3

    # Check Chunk 0
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["start_time_seconds"] == 0.0
    assert chunks[0]["end_time_seconds"] == 10.0
    assert chunks[0]["extracted_metadata"]["speaker_count"] == 1
    assert chunks[0]["model_version"] == "gemini-3.8"

    # Check Chunk 1
    assert chunks[1]["chunk_index"] == 1
    assert chunks[1]["start_time_seconds"] == 10.0
    assert chunks[1]["end_time_seconds"] == 20.0

    # Check Chunk 2 (remainder)
    assert chunks[2]["chunk_index"] == 2
    assert chunks[2]["start_time_seconds"] == 20.0
    assert abs(chunks[2]["end_time_seconds"] - 22.0) < 0.5


def test_pipeline_awaiting_pair(test_client):
    """Test that pipeline gracefully waits when only the video or config is present."""
    mock_storage = MagicMock()
    mock_storage.derive_pair_paths.return_value = (
        "unpaired_vid",
        "unpaired_vid.mp4",
        "unpaired_vid_config.json",
    )
    # Video exists, but config does not exist yet
    mock_storage.check_blob_exists.side_effect = lambda bucket, obj: obj.endswith(".mp4")

    orchestrator.storage_manager = mock_storage

    payload = {
        "bucket": "test-bucket",
        "name": "unpaired_vid.mp4",
    }
    response = test_client.post("/events", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "AWAITING_PAIR"
    assert res_data["video_id"] == "unpaired_vid"
    assert res_data["missing_file"] == "unpaired_vid_config.json"
