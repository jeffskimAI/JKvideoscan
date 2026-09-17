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

    def collection_group(self, group_name: str):
        # find all documents under any subcollection named group_name
        snapshots = []
        for col_k, col_v in self.store.items():
            if isinstance(col_v, dict):
                for doc_id, doc_dict in col_v.items():
                    if isinstance(doc_dict, dict) and group_name in doc_dict:
                        subcol = doc_dict[group_name]
                        if isinstance(subcol, dict):
                            for sub_id, sub_data in subcol.items():
                                if isinstance(sub_data, dict) and "_data" in sub_data:
                                    mock_snap = MagicMock()
                                    mock_snap.to_dict.return_value = sub_data["_data"]
                                    mock_ref = MagicMock()
                                    mock_ref.parent.parent.id = doc_id
                                    mock_snap.reference = mock_ref
                                    snapshots.append(mock_snap)
        mock_cg = MagicMock()
        mock_cg.stream.return_value = snapshots
        return mock_cg


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


def test_firestore_repo_search_chunks():
    """Test searching chunks by key, query, and sports_key_plays."""
    mock_client = MockFirestoreClient()
    repo = FirestoreRepository(client=mock_client, collection_name="videos")

    # Seed video 1: soccer
    repo.init_video_record(
        video_id="soccer_vid",
        filename="soccer.mp4",
        gcs_bucket="b",
        gcs_path="soccer.mp4",
        config_path="soccer_config.json",
        target_metadata=["sports_key_plays", "sports_play_type", "detected_objects", "transcript"],
        total_chunks=3,
        duration_seconds=30.0,
    )
    chunk0 = VideoChunk(index=0, start_time_seconds=0.0, end_time_seconds=10.0, duration_seconds=10.0, file_path=Path("/tmp/c0.mp4"))
    repo.save_chunk_metadata("soccer_vid", chunk0, {
        "sports_play_type": "soccer_kickoff",
        "significance_score": 0.5,
        "significance_factors": ["Match start"],
        "detected_objects": ["soccer ball", "referee", "players"],
        "transcript": "The whistle blows to start the match",
    }, model_version="gemini-3.8")
    chunk1 = VideoChunk(index=1, start_time_seconds=10.0, end_time_seconds=20.0, duration_seconds=10.0, file_path=Path("/tmp/c1.mp4"))
    repo.save_chunk_metadata("soccer_vid", chunk1, {
        "sports_play_type": "soccer_goal",
        "significance_score": 0.99,
        "significance_factors": ["Upper corner strike", "Crowd eruption"],
        "sports_key_plays": [{"play_type": "soccer_goal", "score": "1-0"}],
        "detected_objects": ["soccer ball", "net", "goalkeeper"],
        "transcript": "What a magnificent goal by Ronaldo!",
    }, model_version="gemini-3.8")

    # Seed video 2: nature
    repo.init_video_record(
        video_id="nature_vid",
        filename="nature.mp4",
        gcs_bucket="b",
        gcs_path="nature.mp4",
        config_path="nature_config.json",
        target_metadata=["scene_description", "detected_objects"],
        total_chunks=1,
        duration_seconds=10.0,
    )
    chunk_n = VideoChunk(index=0, start_time_seconds=0.0, end_time_seconds=10.0, duration_seconds=10.0, file_path=Path("/tmp/cn.mp4"))
    repo.save_chunk_metadata("nature_vid", chunk_n, {
        "scene_description": "A sunny forest with tall pine trees and deer grazing.",
        "detected_objects": ["pine tree", "deer", "sunlight"],
    }, model_version="gemini-3.8")

    # 1. Search by key="sports_key_plays"
    res_kp = repo.search_chunks(key="sports_key_plays")
    assert len(res_kp) >= 1
    goal_res = [r for r in res_kp if r["chunk_index"] == 1][0]
    assert goal_res["video_id"] == "soccer_vid"
    assert goal_res["sports_play_type"] == "soccer_goal"
    assert goal_res["significance_score"] == 0.99

    # 2. Search by key="sports_key_plays" with query="Ronaldo"
    res_ronaldo = repo.search_chunks(key="sports_key_plays", query="Ronaldo")
    assert len(res_ronaldo) == 1
    assert res_ronaldo[0]["chunk_index"] == 1

    # 3. Search by key="detected_objects" with query="deer"
    res_deer = repo.search_chunks(key="detected_objects", query="deer")
    assert len(res_deer) == 1
    assert res_deer[0]["video_id"] == "nature_vid"

    # 4. Search query only across all fields (query="forest")
    res_forest = repo.search_chunks(query="forest")
    assert len(res_forest) == 1
    assert res_forest[0]["video_id"] == "nature_vid"

    # 5. Search with video_id scoping
    res_scoped = repo.search_chunks(video_id="nature_vid")
    assert len(res_scoped) == 1
    assert res_scoped[0]["video_id"] == "nature_vid"

