"""Unit tests for FirestoreRepository."""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.firestore_repo import FirestoreRepository
from src.video_segmenter import VideoChunk


class MockFirestoreClient:
    """In-memory mock for Firestore client to facilitate deterministic unit tests."""

    def __init__(self):
        self.store = {}

    def collection(self, col_name: str):
        return MockCollectionRef(self.store, [col_name])


class MockCollectionRef:
    def __init__(self, store: dict, path: list):
        self.store = store
        self.path = path

    def document(self, doc_id: str):
        return MockDocumentRef(self.store, self.path + [doc_id])

    def order_by(self, field_name: str, direction=None):
        return self

    def limit(self, count: int):
        return self

    def stream(self):
        # Retrieve all docs under current path
        target = self.store
        for seg in self.path:
            target = target.setdefault(seg, {})
        snapshots = []
        for doc_id, doc_val in target.items():
            if isinstance(doc_val, dict) and "_data" in doc_val:
                mock_snap = MagicMock()
                mock_snap.to_dict.return_value = doc_val["_data"]
                mock_snap.reference = MockDocumentRef(self.store, self.path + [doc_id])
                snapshots.append(mock_snap)
        return snapshots


class MockDocumentRef:
    def __init__(self, store: dict, path: list):
        self.store = store
        self.path = path

    def _get_target(self):
        target = self.store
        for seg in self.path[:-1]:
            target = target.setdefault(seg, {})
        return target

    def set(self, data: dict, merge: bool = False):
        target = self._get_target()
        doc_entry = target.setdefault(self.path[-1], {})
        if merge and "_data" in doc_entry:
            doc_entry["_data"].update(data)
        else:
            doc_entry["_data"] = data.copy()

    def delete(self):
        target = self._get_target()
        doc_id = self.path[-1]
        if doc_id in target:
            del target[doc_id]


    def update(self, data: dict):
        target = self._get_target()
        doc_entry = target.setdefault(self.path[-1], {})
        if "_data" not in doc_entry:
            doc_entry["_data"] = {}
        doc_entry["_data"].update(data)

    def get(self):
        target = self._get_target()
        doc_entry = target.get(self.path[-1])
        snapshot = MagicMock()
        if doc_entry and "_data" in doc_entry:
            snapshot.exists = True
            snapshot.to_dict.return_value = doc_entry["_data"]
        else:
            snapshot.exists = False
            snapshot.to_dict.return_value = None
        return snapshot

    def collection(self, subcol_name: str):
        return MockCollectionRef(self.store, self.path + [subcol_name])


def test_firestore_repo_init_video():
    """Test initializing a video record in Firestore."""
    mock_client = MockFirestoreClient()
    repo = FirestoreRepository(client=mock_client, collection_name="videos")

    doc = repo.init_video_record(
        video_id="video123",
        filename="test.mp4",
        gcs_bucket="test-bucket",
        gcs_path="raw/test.mp4",
        config_path="raw/test_config.json",
        target_metadata=["scene_description"],
        total_chunks=3,
        duration_seconds=25.0,
    )

    assert doc["video_id"] == "video123"
    assert doc["status"] == "PROCESSING"
    assert doc["total_chunks"] == 3

    stored = repo.get_video_record("video123")
    assert stored is not None
    assert stored["video_id"] == "video123"
    assert stored["status"] == "PROCESSING"


def test_firestore_repo_save_chunk_and_complete():
    """Test saving chunk metadata and marking processing as completed."""
    mock_client = MockFirestoreClient()
    repo = FirestoreRepository(client=mock_client, collection_name="videos")

    repo.init_video_record(
        video_id="vid456",
        filename="demo.mp4",
        gcs_bucket="bucket",
        gcs_path="demo.mp4",
        config_path="demo_config.json",
        target_metadata=["transcript", "speaker_count"],
        total_chunks=1,
        duration_seconds=9.5,
    )

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=9.5,
        duration_seconds=9.5,
        file_path=Path("/tmp/chunk_0000.mp4"),
    )

    extracted = {"transcript": "Hello world", "speaker_count": 1}
    repo.save_chunk_metadata("vid456", chunk, extracted, model_version="gemini-3.8")

    chunks = repo.get_video_chunks("vid456")
    assert len(chunks) == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["extracted_metadata"]["transcript"] == "Hello world"
    assert chunks[0]["model_version"] == "gemini-3.8"

    repo.complete_video_processing("vid456")
    updated = repo.get_video_record("vid456")
    assert updated["status"] == "COMPLETED"


def test_firestore_repo_failure():
    """Test marking video processing as failed."""
    mock_client = MockFirestoreClient()
    repo = FirestoreRepository(client=mock_client, collection_name="videos")

    repo.init_video_record(
        video_id="err_vid",
        filename="corrupted.mp4",
        gcs_bucket="b",
        gcs_path="corrupted.mp4",
        config_path="corrupted_config.json",
        target_metadata=["scene_description"],
        total_chunks=0,
        duration_seconds=0,
    )

    repo.fail_video_processing("err_vid", "FFmpeg failed to demux video")
    rec = repo.get_video_record("err_vid")
    assert rec["status"] == "FAILED"
    assert "FFmpeg failed" in rec["error_message"]
