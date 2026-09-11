"""Tests for Web Interface, Upload, and Firestore Catalog API endpoints."""
import io
import json
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.main import app, orchestrator
from tests.test_firestore_repo import MockFirestoreClient


@pytest.fixture
def client():
    return TestClient(app)


def test_ui_and_root_html(client):
    """Test that GET / with text/html and GET /ui deliver the HTML interface."""
    # Requesting HTML via Accept header
    res_root = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert res_root.status_code == 200
    assert "Jeff's VideoScan" in res_root.text
    assert "<!DOCTYPE html>" in res_root.text

    # Direct /ui endpoint
    res_ui = client.get("/ui")
    assert res_ui.status_code == 200
    assert "Jeff's VideoScan" in res_ui.text


def test_api_info(client):
    """Test /api/info returns service configuration."""
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "jeffsvideoscan"
    assert data["status"] == "ready"
    assert "chunk_duration_seconds" in data
    assert "gemini_model" in data


def test_api_catalog(client):
    """Test /api/catalog returns all categories and 17 attributes."""
    res = client.get("/api/catalog")
    assert res.status_code == 200
    data = res.json()
    assert "categories" in data
    assert "all_keys" in data
    assert len(data["all_keys"]) >= 16
    assert "Visual & Spatial Metadata" in data["categories"]
    assert "scene_description" in data["all_keys"]
    assert "detected_objects" in data["all_keys"]


def test_api_videos_list_and_detail(client):
    """Test listing videos and retrieving video detail with chunks."""
    # Mock firestore repository
    mock_db = MockFirestoreClient()
    from src.firestore_repo import FirestoreRepository
    repo = FirestoreRepository(client=mock_db, collection_name="videos")

    # Add a mock video record
    repo.init_video_record(
        video_id="test_vid_1",
        filename="test_vid_1.mp4",
        gcs_bucket="test-bucket",
        gcs_path="test_vid_1.mp4",
        config_path="test_vid_1_config.json",
        target_metadata=["scene_description", "overall_sentiment"],
        total_chunks=2,
        duration_seconds=18.5,
    )
    from src.video_segmenter import VideoChunk
    from pathlib import Path
    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=Path("/tmp/chunk_0000.mp4"),
    )
    repo.save_chunk_metadata(
        video_id="test_vid_1",
        chunk=chunk,
        extracted_metadata={"scene_description": "Office scene", "overall_sentiment": "positive"},
        model_version="gemini-2.5-flash",
    )

    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.firestore_repo = repo

        # Test list endpoint
        res_list = client.get("/api/videos")
        assert res_list.status_code == 200
        videos = res_list.json()["videos"]
        assert len(videos) == 1
        assert videos[0]["video_id"] == "test_vid_1"
        assert videos[0]["gcs_uri"] == "gs://test-bucket/test_vid_1.mp4"

        # Test detail endpoint
        res_detail = client.get("/api/videos/test_vid_1")
        assert res_detail.status_code == 200
        detail = res_detail.json()
        assert detail["video"]["video_id"] == "test_vid_1"
        assert len(detail["chunks"]) == 1
        assert detail["chunks"][0]["extracted_metadata"]["overall_sentiment"] == "positive"
        assert detail["gcs_uri"] == "gs://test-bucket/test_vid_1.mp4"

        # Test 404 for non-existent video
        res_404 = client.get("/api/videos/non_existent_video")
        assert res_404.status_code == 404
    finally:
        orchestrator.firestore_repo = orig_repo


def test_api_video_streaming_and_range(client):
    """Test streaming video with full content and HTTP 206 Range headers."""
    mock_storage = MagicMock()
    mock_blob = MagicMock()
    fake_video_bytes = b"0123456789" * 100  # 1000 bytes
    mock_blob.size = len(fake_video_bytes)
    mock_blob.download_as_bytes.side_effect = lambda start=0, end=None: (
        fake_video_bytes[start : (end + 1 if end is not None else None)]
        if end is not None
        else fake_video_bytes[start:]
    )
    mock_storage.get_blob.return_value = mock_blob

    orig_storage = orchestrator.storage_manager
    try:
        orchestrator.storage_manager = mock_storage

        # 1. Full content request
        res_full = client.get("/api/videos/stream_test/video")
        assert res_full.status_code == 200
        assert res_full.headers["Content-Type"] == "video/mp4"
        assert res_full.content == fake_video_bytes

        # 2. Byte Range request (bytes=0-99)
        res_range = client.get(
            "/api/videos/stream_test/video",
            headers={"Range": "bytes=0-99"},
        )
        assert res_range.status_code == 206
        assert res_range.headers["Content-Range"] == "bytes 0-99/1000"
        assert len(res_range.content) == 100
        assert res_range.content == fake_video_bytes[:100]
    finally:
        orchestrator.storage_manager = orig_storage


def test_api_upload_with_builder_json(client):
    """Test uploading a video file with an interactive JSON configuration payload."""
    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "gs://jeffsvideoscan-ingest/uploaded_demo.mp4"
    mock_storage.upload_json.return_value = "gs://jeffsvideoscan-ingest/uploaded_demo_config.json"

    orig_storage = orchestrator.storage_manager
    try:
        orchestrator.storage_manager = mock_storage

        video_content = b"fake video bytes"
        config_json = json.dumps({
            "target_metadata": ["scene_description", "detected_objects", "chunk_summary"]
        })

        files = {
            "video": ("my_cool_video.mp4", io.BytesIO(video_content), "video/mp4"),
        }
        data = {
            "video_id": "custom_demo_id",
            "config_json": config_json,
            "run_pipeline": "false",  # Don't trigger background pipeline in test
        }

        res = client.post("/api/upload", files=files, data=data)
        assert res.status_code == 200
        resp_data = res.json()
        assert resp_data["status"] == "PROCESSING"
        assert resp_data["video_id"] == "custom_demo_id"
        assert resp_data["filename"] == "custom_demo_id.mp4"
        assert resp_data["gcs_video_path"] == "custom_demo_id.mp4"
        assert resp_data["gcs_config_path"] == "custom_demo_id_config.json"
        assert "scene_description" in resp_data["target_metadata"]

        # Verify GCS upload was called for both video and config
        mock_storage.upload_json.assert_called_once()
        mock_storage.upload_file.assert_called_once()
    finally:
        orchestrator.storage_manager = orig_storage


def test_api_upload_with_config_file(client):
    """Test uploading a video file paired with a _config.json file."""
    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "gs://jeffsvideoscan-ingest/clip1.mp4"
    mock_storage.upload_json.return_value = "gs://jeffsvideoscan-ingest/clip1_config.json"

    orig_storage = orchestrator.storage_manager
    try:
        orchestrator.storage_manager = mock_storage

        video_content = b"fake video bytes"
        config_bytes = json.dumps({
            "target_metadata": ["overall_sentiment", "transcript"]
        }).encode("utf-8")

        files = {
            "video": ("clip1.mp4", io.BytesIO(video_content), "video/mp4"),
            "config_file": ("clip1_config.json", io.BytesIO(config_bytes), "application/json"),
        }
        data = {
            "run_pipeline": "false",
        }

        res = client.post("/api/upload", files=files, data=data)
        assert res.status_code == 200
        resp_data = res.json()
        assert resp_data["video_id"] == "clip1"
        assert resp_data["target_metadata"] == ["overall_sentiment", "transcript"]
    finally:
        orchestrator.storage_manager = orig_storage


def test_api_upload_validation_error(client):
    """Test that invalid metadata in config is rejected with HTTP 400."""
    video_content = b"fake video bytes"
    invalid_config = json.dumps({
        "target_metadata": ["non_existent_metadata_attribute"]
    })

    files = {
        "video": ("invalid.mp4", io.BytesIO(video_content), "video/mp4"),
    }
    data = {
        "config_json": invalid_config,
    }

    res = client.post("/api/upload", files=files, data=data)
    assert res.status_code == 400
    assert "Unsupported metadata keys requested" in res.json()["detail"]


def test_api_delete_video(client):
    """Test deleting a video from Firestore via API."""
    mock_db = MockFirestoreClient()
    from src.firestore_repo import FirestoreRepository
    repo = FirestoreRepository(client=mock_db, collection_name="videos")
    repo.init_video_record(
        video_id="to_delete",
        filename="to_delete.mp4",
        gcs_bucket="test-bucket",
        gcs_path="to_delete.mp4",
        config_path="to_delete_config.json",
        target_metadata=["scene_description"],
        total_chunks=1,
        duration_seconds=10.0,
    )

    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.firestore_repo = repo
        assert repo.get_video_record("to_delete") is not None

        res = client.delete("/api/videos/to_delete")
        assert res.status_code == 200
        assert res.json()["status"] == "DELETED"
        assert repo.get_video_record("to_delete") is None
    finally:
        orchestrator.firestore_repo = orig_repo
