"""Cloud Firestore Persistence Layer for Video Metadata."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import firestore

from src.auth_utils import get_gcp_credentials
from src.config import settings
from src.logger import logger
from src.video_segmenter import VideoChunk


class FirestoreRepository:
    """Repository managing video metadata documents in Cloud Firestore."""

    def __init__(
        self,
        client: Optional[firestore.Client] = None,
        collection_name: Optional[str] = None,
    ):
        """Initializes the FirestoreRepository.

        Args:
            client: Optional firestore.Client instance (useful for testing/mocking).
            collection_name: Firestore collection name (default from settings).
        """
        self._client = client
        self.collection_name = collection_name or settings.firestore_collection

    @property
    def client(self) -> firestore.Client:
        """Lazily initializes and returns the Firestore client."""
        if self._client is None:
            logger.info(
                f"Initializing Firestore Client (project={settings.gcp_project}, "
                f"database={settings.firestore_database})"
            )
            creds = get_gcp_credentials()
            self._client = firestore.Client(
                project=settings.gcp_project,
                database=settings.firestore_database,
                credentials=creds,
            )
        return self._client

    def _now_iso(self) -> str:
        """Returns current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def init_video_record(
        self,
        video_id: str,
        filename: str,
        gcs_bucket: str,
        gcs_path: str,
        config_path: str,
        target_metadata: List[str],
        total_chunks: int,
        duration_seconds: float,
    ) -> Dict[str, Any]:
        """Initializes a video record with status 'PROCESSING'.

        Args:
            video_id: Unique video identifier.
            filename: Video file name.
            gcs_bucket: Source GCS bucket name.
            gcs_path: GCS object path of the video.
            config_path: GCS object path of the paired config.
            target_metadata: List of target metadata keys.
            total_chunks: Total segmented chunks count.
            duration_seconds: Total video duration in seconds.

        Returns:
            The created document data dictionary.
        """
        now = self._now_iso()
        doc_data = {
            "video_id": video_id,
            "filename": filename,
            "gcs_bucket": gcs_bucket,
            "gcs_path": gcs_path,
            "config_path": config_path,
            "target_metadata": target_metadata,
            "total_chunks": total_chunks,
            "duration_seconds": duration_seconds,
            "status": "PROCESSING",
            "created_at": now,
            "updated_at": now,
            "error_message": None,
            "error_description": None,
            "logs": [],
        }

        doc_ref = self.client.collection(self.collection_name).document(video_id)
        doc_ref.set(doc_data, merge=True)
        logger.info(f"Initialized Firestore video record: {self.collection_name}/{video_id}")
        return doc_data

    def save_chunk_metadata(
        self,
        video_id: str,
        chunk: VideoChunk,
        extracted_metadata: Dict[str, Any],
        model_version: str,
    ) -> Dict[str, Any]:
        """Saves extracted metadata for a single 10-second chunk in a subcollection.

        Args:
            video_id: Video identifier.
            chunk: The VideoChunk object.
            extracted_metadata: Extracted metadata dictionary.
            model_version: Name/version of the Gemini model used.

        Returns:
            The chunk document data dictionary.
        """
        chunk_id = f"{chunk.index:04d}"
        now = self._now_iso()
        chunk_data = {
            "chunk_index": chunk.index,
            "start_time_seconds": chunk.start_time_seconds,
            "end_time_seconds": chunk.end_time_seconds,
            "duration_seconds": chunk.duration_seconds,
            "extracted_metadata": extracted_metadata,
            "model_version": model_version,
            "processed_at": now,
        }

        chunk_ref = (
            self.client.collection(self.collection_name)
            .document(video_id)
            .collection("chunks")
            .document(chunk_id)
        )
        chunk_ref.set(chunk_data)
        logger.info(f"Saved chunk metadata for video={video_id}, chunk={chunk_id}")
        return chunk_data

    def complete_video_processing(self, video_id: str) -> None:
        """Marks video processing status as 'COMPLETED'."""
        now = self._now_iso()
        doc_ref = self.client.collection(self.collection_name).document(video_id)
        doc_ref.set({
            "status": "COMPLETED",
            "updated_at": now,
            "error_message": None,
            "error_description": None,
        }, merge=True)
        logger.info(f"Marked video={video_id} as COMPLETED")

    def fail_video_processing(
        self,
        video_id: str,
        error_message: str,
        error_description: Optional[str] = None,
        logs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Marks video processing status as 'FAILED' with error descriptions and execution logs."""
        now = self._now_iso()
        doc_ref = self.client.collection(self.collection_name).document(video_id)
        update_data: Dict[str, Any] = {
            "status": "FAILED",
            "error_message": error_message,
            "error_description": error_description or error_message,
            "updated_at": now,
        }
        if logs is not None:
            update_data["logs"] = logs
        doc_ref.set(update_data, merge=True)
        logger.warning(f"Marked video={video_id} as FAILED: {error_message}")

    def stop_video_processing(
        self,
        video_id: str,
        reason: str = "Processing stopped by user request.",
    ) -> None:
        """Marks video processing status as 'STOPPED'."""
        now = self._now_iso()
        doc_ref = self.client.collection(self.collection_name).document(video_id)
        doc_ref.set({
            "status": "STOPPED",
            "error_message": reason,
            "error_description": reason,
            "updated_at": now,
        }, merge=True)
        logger.info(f"Marked video={video_id} as STOPPED: {reason}")

    def get_video_logs(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves execution logs recorded for a video."""
        rec = self.get_video_record(video_id)
        if rec and "logs" in rec and isinstance(rec["logs"], list):
            return rec["logs"]
        return []

    def get_video_record(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves video document data if present."""
        doc_ref = self.client.collection(self.collection_name).document(video_id)
        snapshot = doc_ref.get()
        if snapshot.exists:
            return snapshot.to_dict()
        return None

    def get_video_chunks(self, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chunk documents for a video sorted by index."""
        chunks_ref = (
            self.client.collection(self.collection_name)
            .document(video_id)
            .collection("chunks")
        )
        docs = chunks_ref.order_by("chunk_index").stream()
        return [doc.to_dict() for doc in docs]

    def list_videos(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves all video documents from Firestore, sorted by created_at descending."""
        try:
            col_ref = self.client.collection(self.collection_name)
            try:
                docs = col_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
                results = [doc.to_dict() for doc in docs if doc.to_dict()]
            except Exception:
                docs = col_ref.limit(limit).stream()
                results = [doc.to_dict() for doc in docs if doc.to_dict()]
                results.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
            return results
        except Exception as e:
            logger.error(f"Failed to list videos from Firestore: {e}")
            return []

    def delete_video_record(self, video_id: str) -> bool:
        """Deletes a video record and its chunks subcollection from Firestore."""
        try:
            doc_ref = self.client.collection(self.collection_name).document(video_id)
            # Delete chunks subcollection
            chunks_ref = doc_ref.collection("chunks")
            for chunk_doc in chunks_ref.stream():
                chunk_doc.reference.delete()
            doc_ref.delete()
            logger.info(f"Deleted Firestore record for video_id={video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete video_id={video_id} from Firestore: {e}")
            return False

