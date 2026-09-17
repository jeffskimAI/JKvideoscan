"""Tests for Web Interface, Upload, Authentication, and Firestore Catalog API endpoints."""
import io
import json
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.main import app, orchestrator, get_auth_token
from tests.test_firestore_repo import MockFirestoreClient


@pytest.fixture
def auth_headers():
    return {"X-App-Password": "joanisawful"}


@pytest.fixture
def client(auth_headers):
    return TestClient(app, headers=auth_headers)


@pytest.fixture
def unauth_client():
    return TestClient(app)


def test_ui_and_root_html(unauth_client):
    """Test that GET / with text/html and GET /ui deliver the HTML interface without requiring prior auth."""
    # Requesting HTML via Accept header
    res_root = unauth_client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert res_root.status_code == 200
    assert "Jeff's VideoScan" in res_root.text
    assert "<!DOCTYPE html>" in res_root.text
    assert "authGate" in res_root.text

    # Direct /ui endpoint
    res_ui = unauth_client.get("/ui")
    assert res_ui.status_code == 200
    assert "Jeff's VideoScan" in res_ui.text
    assert "authGate" in res_ui.text


def test_auth_login_and_verification(unauth_client):
    """Test authentication login, rejection of invalid password, and token verification."""
    # 1. Unauthenticated access to protected route is rejected with 401
    res_unauth = unauth_client.get("/api/info")
    assert res_unauth.status_code == 401
    assert "Authentication required" in res_unauth.json()["detail"]
    assert "joanisawful" in res_unauth.json()["detail"]

    # 2. Login with incorrect password returns 401
    res_bad = unauth_client.post("/api/auth/login", json={"password": "wrong_password!"})
    assert res_bad.status_code == 401
    assert "Incorrect password" in res_bad.json()["detail"]

    # 3. Login with correct password 'joanisawful' returns 200 and session token
    res_login = unauth_client.post("/api/auth/login", json={"password": "joanisawful"})
    assert res_login.status_code == 200
    login_data = res_login.json()
    assert login_data["success"] is True

    # 3b. Login with whitespace and mixed casing also succeeds
    res_case = unauth_client.post("/api/auth/login", json={"password": " JoanIsAwful "})
    assert res_case.status_code == 200
    assert res_case.json()["success"] is True
    assert "token" in login_data
    assert "videoscan_auth_token" in res_login.cookies

    token = login_data["token"]
    assert token == get_auth_token()

    # 4. Verify endpoint with token header
    res_verify = unauth_client.get("/api/auth/verify", headers={"Authorization": f"Bearer {token}"})
    assert res_verify.status_code == 200
    assert res_verify.json()["authenticated"] is True

    # 5. Access protected route with Bearer token
    res_authed = unauth_client.get("/api/info", headers={"Authorization": f"Bearer {token}"})
    assert res_authed.status_code == 200
    assert res_authed.json()["service"] == "jeffsvideoscan"

    # 6. Logout clears cookie
    res_logout = unauth_client.post("/api/auth/logout")
    assert res_logout.status_code == 200
    assert res_logout.json()["success"] is True


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

        # 3. Query param token access
        token = get_auth_token()
        unauth = TestClient(app)
        res_token = unauth.get(f"/api/videos/stream_test/video?token={token}")
        assert res_token.status_code == 200
        assert res_token.content == fake_video_bytes
    finally:
        orchestrator.storage_manager = orig_storage


def test_api_upload_with_builder_json(client):
    """Test uploading a video file with an interactive JSON configuration payload."""
    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "gs://jeffsvideoscan-ingest/uploaded_demo.mp4"
    mock_storage.upload_json.return_value = "gs://jeffsvideoscan-ingest/uploaded_demo_config.json"
    mock_repo = MagicMock()

    orig_storage = orchestrator.storage_manager
    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.storage_manager = mock_storage
        orchestrator.firestore_repo = mock_repo

        video_content = b"fake video bytes for testing"
        config_json = json.dumps({
            "target_metadata": ["scene_description", "chunk_summary", "detected_objects"]
        })

        files = {
            "video": ("my_test_video.mp4", io.BytesIO(video_content), "video/mp4"),
        }
        data = {
            "config_json": config_json,
            "video_id": "custom_test_id",
            "run_pipeline": "false",
        }

        res = client.post("/api/upload", files=files, data=data)
        assert res.status_code == 200
        resp_data = res.json()
        assert resp_data["status"] == "PROCESSING"
        assert resp_data["video_id"] == "custom_test_id"
        assert resp_data["target_metadata"] == ["scene_description", "chunk_summary", "detected_objects"]
        assert resp_data["gcs_uri"] == "gs://jeffsvideoscan-ingest/custom_test_id.mp4"

        # Verify storage manager was called with video and config
        assert mock_storage.upload_json.called
        assert mock_storage.upload_file.called

        # Verify firestore record was initialized immediately with status PROCESSING
        assert mock_repo.init_video_record.called
        call_kwargs = mock_repo.init_video_record.call_args.kwargs
        assert call_kwargs["video_id"] == "custom_test_id"
        assert call_kwargs["target_metadata"] == ["scene_description", "chunk_summary", "detected_objects"]
    finally:
        orchestrator.storage_manager = orig_storage
        orchestrator.firestore_repo = orig_repo


def test_api_upload_with_config_file(client):
    """Test uploading a video file with a separate .json config file."""
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


def test_api_upload_oversized_file(client):
    """Test that videos exceeding 32MB are rejected with HTTP 413."""
    oversized_bytes = b"x" * (33 * 1024 * 1024)  # 33 MB
    config_json = json.dumps({
        "target_metadata": ["scene_description"]
    })
    files = {
        "video": ("large.mp4", io.BytesIO(oversized_bytes), "video/mp4"),
    }
    data = {
        "config_json": config_json,
    }

    res = client.post("/api/upload", files=files, data=data)
    assert res.status_code == 413
    assert "exceeds Cloud Run's 32 MB HTTP request limit" in res.json()["detail"]


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


def test_api_system_logs(client):
    """Test retrieving system and execution logs via GET /api/logs."""
    from src.logger import logger

    logger.info("Test log message for system log query")
    logger.error("Test error message for system log query")

    res = client.get("/api/logs?limit=50")
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data
    assert isinstance(data["logs"], list)
    assert data["count"] > 0

    # Filter by level
    res_err = client.get("/api/logs?level=ERROR")
    assert res_err.status_code == 200
    err_data = res_err.json()
    for item in err_data["logs"]:
        assert item["level"] == "ERROR"


def test_api_video_logs_and_failure_diagnostics(client):
    """Test retrieving failure diagnostics and logs for a failed video."""
    mock_db = MockFirestoreClient()
    from src.firestore_repo import FirestoreRepository
    repo = FirestoreRepository(client=mock_db, collection_name="videos")

    sample_logs = [
        {"timestamp": "2026-09-17T01:00:00Z", "level": "INFO", "stage": "video_download", "message": "Downloading video..."},
        {"timestamp": "2026-09-17T01:00:05Z", "level": "ERROR", "stage": "video_probing", "message": "ffprobe: Invalid data found when processing input"},
    ]

    repo.init_video_record(
        video_id="failed_vid_01",
        filename="corrupted.mp4",
        gcs_bucket="test-bucket",
        gcs_path="corrupted.mp4",
        config_path="corrupted_config.json",
        target_metadata=["scene_description"],
        total_chunks=0,
        duration_seconds=0.0,
    )
    repo.fail_video_processing(
        video_id="failed_vid_01",
        error_message="Invalid data found when processing input",
        error_description="FFprobe failed to inspect the video stream metadata. Verify that the video is a valid MP4/H.264 file.",
        logs=sample_logs,
    )

    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.firestore_repo = repo

        # 1. Detail endpoint returns error_description and logs
        res_detail = client.get("/api/videos/failed_vid_01")
        assert res_detail.status_code == 200
        vid_data = res_detail.json()["video"]
        assert vid_data["status"] == "FAILED"
        assert "Invalid data found" in vid_data["error_message"]
        assert "FFprobe failed to inspect" in vid_data["error_description"]
        assert len(vid_data["logs"]) == 2

        # 2. Logs-specific endpoint returns logs
        res_logs = client.get("/api/videos/failed_vid_01/logs")
        assert res_logs.status_code == 200
        logs_data = res_logs.json()
        assert logs_data["video_id"] == "failed_vid_01"
        assert logs_data["status"] == "FAILED"
        assert logs_data["error_description"] is not None
        assert len(logs_data["logs"]) == 2
        assert logs_data["logs"][1]["level"] == "ERROR"
    finally:
        orchestrator.firestore_repo = orig_repo


def test_api_upload_signed_url_success(client):
    """Test generating a GCS V4 Signed URL for direct client upload."""
    mock_storage = MagicMock()
    mock_storage.upload_json.return_value = "gs://jeffsvideoscan-ingest/signed_vid_config.json"
    mock_storage.generate_signed_upload_url.return_value = "https://storage.googleapis.com/jeffsvideoscan-ingest/signed_vid.mp4?X-Goog-Signature=test_sig"
    mock_repo = MagicMock()

    orig_storage = orchestrator.storage_manager
    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.storage_manager = mock_storage
        orchestrator.firestore_repo = mock_repo

        payload = {
            "video_id": "signed_vid",
            "filename": "my_camera_roll.mp4",
            "content_type": "video/mp4",
            "target_metadata": ["scene_description", "chunk_summary"],
        }
        res = client.post("/api/upload/signed-url", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["video_id"] == "signed_vid"
        assert "storage.googleapis.com" in data["signed_url"]
        assert data["gcs_uri"] == "gs://jeffsvideoscan-ingest/signed_vid.mp4"
        assert data["video_object"] == "signed_vid.mp4"
        assert data["config_object"] == "signed_vid_config.json"
        assert data["target_metadata"] == ["scene_description", "chunk_summary"]

        mock_storage.upload_json.assert_called_once()
        mock_storage.generate_signed_upload_url.assert_called_once()
        mock_repo.init_video_record.assert_called_once()
    finally:
        orchestrator.storage_manager = orig_storage
        orchestrator.firestore_repo = orig_repo


def test_api_upload_signed_url_invalid_catalog(client):
    """Test generating signed URL fails gracefully when invalid metadata keys are submitted."""
    payload = {
        "video_id": "bad_catalog_vid",
        "target_metadata": ["unsupported_attribute_xyz"],
    }
    res = client.post("/api/upload/signed-url", json=payload)
    assert res.status_code == 400
    assert "Configuration validation error" in res.json()["detail"]


def test_api_upload_complete_success(client):
    """Test completing a signed URL upload kicks off the pipeline."""
    mock_storage = MagicMock()
    mock_storage.check_blob_exists.return_value = True

    orig_storage = orchestrator.storage_manager
    try:
        orchestrator.storage_manager = mock_storage

        payload = {
            "video_id": "signed_vid",
            "run_pipeline": False,
        }
        res = client.post("/api/upload/complete", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PROCESSING"
        assert data["video_id"] == "signed_vid"
        assert "Upload verified in GCS" in data["message"]
        mock_storage.check_blob_exists.assert_called_with("jeffsvideoscan-ingest", "signed_vid.mp4")
    finally:
        orchestrator.storage_manager = orig_storage


def test_api_upload_complete_missing_blob(client):
    """Test complete endpoint returns 404 when file never reached GCS."""
    mock_storage = MagicMock()
    mock_storage.check_blob_exists.return_value = False

    orig_storage = orchestrator.storage_manager
    try:
        orchestrator.storage_manager = mock_storage

        payload = {
            "video_id": "missing_vid",
            "run_pipeline": True,
        }
        res = client.post("/api/upload/complete", json=payload)
        assert res.status_code == 404
        assert "not found in bucket" in res.json()["detail"]
    finally:
        orchestrator.storage_manager = orig_storage


def test_api_stop_video(client):
    """Test stopping processing for a single video."""
    mock_repo = MagicMock()
    mock_repo.get_video_record.return_value = {"video_id": "vid_to_stop", "status": "PROCESSING"}
    mock_repo.stop_video_processing.return_value = True

    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.firestore_repo = mock_repo
        res = client.post("/api/videos/vid_to_stop/stop")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "STOPPED"
        assert data["video_id"] == "vid_to_stop"
        mock_repo.stop_video_processing.assert_called_once_with("vid_to_stop", "Processing stopped by user request.")
    finally:
        orchestrator.firestore_repo = orig_repo


def test_api_stop_all_videos(client):
    """Test stopping processing for all active videos."""
    mock_repo = MagicMock()
    mock_repo.list_videos.return_value = [
        {"video_id": "vid1", "status": "PROCESSING"},
        {"video_id": "vid2", "status": "COMPLETED"},
        {"video_id": "vid3", "status": "PROCESSING"},
    ]
    mock_repo.stop_video_processing.return_value = True

    orig_repo = orchestrator.firestore_repo
    try:
        orchestrator.firestore_repo = mock_repo
        res = client.post("/api/videos/stop-all")
        assert res.status_code == 200
        data = res.json()
        assert data["stopped_count"] == 2
        assert set(data["stopped_video_ids"]) == {"vid1", "vid3"}
    finally:
        orchestrator.firestore_repo = orig_repo



