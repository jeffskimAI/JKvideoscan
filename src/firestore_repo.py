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
            "processed_chunks": 0,
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

    def update_video_progress(
        self,
        video_id: str,
        processed_chunks: int,
        total_chunks: Optional[int] = None,
    ) -> None:
        """Updates real-time chunk progress counter on the video document."""
        try:
            now = self._now_iso()
            doc_ref = self.client.collection(self.collection_name).document(video_id)
            update_data: Dict[str, Any] = {
                "processed_chunks": processed_chunks,
                "updated_at": now,
            }
            if total_chunks is not None:
                update_data["total_chunks"] = total_chunks
            doc_ref.set(update_data, merge=True)
        except Exception as e:
            logger.warning(f"Could not update video progress in Firestore for {video_id}: {e}")

    def complete_video_processing(self, video_id: str, total_chunks: Optional[int] = None) -> None:
        """Marks video processing status as 'COMPLETED'."""
        now = self._now_iso()
        doc_ref = self.client.collection(self.collection_name).document(video_id)
        update_data: Dict[str, Any] = {
            "status": "COMPLETED",
            "updated_at": now,
            "error_message": None,
            "error_description": None,
        }
        if total_chunks is not None:
            update_data["processed_chunks"] = total_chunks
            update_data["total_chunks"] = total_chunks
        doc_ref.set(update_data, merge=True)
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
        chunks = [doc.to_dict() for doc in docs]
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        return chunks

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

    def search_chunks(
        self,
        query: Optional[str] = None,
        key: Optional[str] = None,
        video_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Searches across video segment metadata in Firestore.

        Supports filtering by:
        - `key`: Metadata attribute name (e.g. 'sports_key_plays', 'detected_objects', 'transcript').
          If key is specified, matches chunks where this attribute is meaningfully present.
        - `query`: Keyword text to match within values (or across all metadata if key is not specified).
        - `video_id`: Restrict to chunks of a specific video (or None for global library search).
        - `limit`: Maximum number of search results to return.

        Args:
            query: Optional search keyword or phrase.
            key: Optional target metadata attribute key.
            video_id: Optional video ID to filter within.
            limit: Maximum results (default 100).

        Returns:
            List of matching segment dictionaries sorted by relevance/significance.
        """
        import json
        from src.catalog import METADATA_CATALOG

        q_clean = query.strip() if query else ""
        q_lower = q_clean.lower()
        key_clean = key.strip() if key else ""

        # If user typed an exact catalog key into query and didn't select key, treat as key filter
        if q_lower in METADATA_CATALOG and not key_clean:
            key_clean = q_lower
            q_clean = ""
            q_lower = ""

        # Retrieve chunks
        chunk_items: List[tuple[str, Dict[str, Any]]] = []
        if video_id and video_id.strip() and video_id.strip() != "all":
            v_id = video_id.strip()
            for c in self.get_video_chunks(v_id):
                chunk_items.append((v_id, c))
        else:
            try:
                cg_docs = self.client.collection_group("chunks").stream()
                for doc in cg_docs:
                    d = doc.to_dict()
                    if d:
                        parent_vid = ""
                        try:
                            parent_vid = doc.reference.parent.parent.id
                        except Exception:
                            parent_vid = d.get("video_id", "")
                        chunk_items.append((parent_vid, d))
            except Exception as e:
                logger.warning(f"Collection group stream failed or unavailable, falling back to listing videos: {e}")
                for v in self.list_videos(limit=50):
                    v_id = v.get("video_id")
                    if v_id:
                        for c in self.get_video_chunks(v_id):
                            chunk_items.append((v_id, c))

        results: List[Dict[str, Any]] = []

        for vid, chunk_data in chunk_items:
            meta = chunk_data.get("extracted_metadata")
            if not meta or not isinstance(meta, dict):
                continue

            matched_key = ""
            matched_value = ""
            is_match = False

            # Special case 1: sports_key_plays
            if key_clean == "sports_key_plays":
                kp = meta.get("sports_key_plays")
                spt = meta.get("sports_play_type")
                sig = meta.get("significance_score")
                factors = meta.get("significance_factors")

                is_sports_play = False
                preview_parts = []

                if spt and str(spt).lower() not in ("routine_play", "routine_possession", "none", ""):
                    is_sports_play = True
                    play_name = str(spt).replace("_", " ").title()
                    preview_parts.append(play_name)

                if sig is not None and (float(sig) >= 0.70 or is_sports_play):
                    is_sports_play = True
                    preview_parts.append(f"Score: {float(sig):.2f}")

                if factors and isinstance(factors, list) and len(factors) > 0:
                    factors_clean = [str(f) for f in factors if f]
                    if factors_clean:
                        is_sports_play = True
                        preview_parts.append("; ".join(factors_clean))

                if kp and isinstance(kp, list) and len(kp) > 0:
                    valid_items = [item for item in kp if item and item != {} and str(item).strip() not in ("{}", "[]")]
                    if valid_items:
                        is_sports_play = True
                        preview_parts.append(json.dumps(valid_items))
                    elif not preview_parts:
                        is_sports_play = True
                        preview_parts.append(meta.get("chunk_summary") or "Key play segment")

                if is_sports_play:
                    matched_key = "sports_key_plays"
                    matched_value = " · ".join(preview_parts) if preview_parts else (meta.get("chunk_summary") or "Sports Key Play")
                    if q_lower:
                        search_corpus = f"{matched_value} {str(meta)}".lower()
                        if q_lower in search_corpus:
                            is_match = True
                    else:
                        is_match = True

            # General case 2: Specific attribute key requested
            elif key_clean:
                if key_clean in meta:
                    val = meta[key_clean]
                    is_pop = False
                    if val is not None:
                        if isinstance(val, (list, dict)):
                            if len(val) > 0:
                                if isinstance(val, list):
                                    non_empty = [x for x in val if x not in (None, {}, "", "None detected")]
                                    is_pop = len(non_empty) > 0 or (q_lower == "none")
                                else:
                                    is_pop = True
                        elif isinstance(val, str):
                            s_val = val.strip().lower()
                            if s_val not in ("none", "n/a", "none detected", "null", "") or q_lower == "none":
                                is_pop = True
                        elif isinstance(val, (int, float, bool)):
                            is_pop = True

                    if is_pop:
                        val_str = val if isinstance(val, (str, int, float)) else json.dumps(val)
                        if q_lower:
                            if q_lower in str(val).lower():
                                is_match = True
                                matched_key = key_clean
                                matched_value = str(val_str)
                        else:
                            is_match = True
                            matched_key = key_clean
                            matched_value = str(val_str)

            # General case 3: No specific key; search across all attributes or return non-empty
            else:
                if q_lower:
                    for k, v in meta.items():
                        if v is not None and q_lower in str(v).lower():
                            is_match = True
                            matched_key = k
                            matched_value = v if isinstance(v, (str, int, float)) else json.dumps(v)
                            break
                else:
                    is_match = True
                    matched_key = "chunk_summary" if "chunk_summary" in meta else (list(meta.keys())[0] if meta else "metadata")
                    matched_value = meta.get(matched_key) or "Extracted metadata segment"
                    if isinstance(matched_value, (list, dict)):
                        matched_value = json.dumps(matched_value)

            if is_match:
                results.append({
                    "video_id": vid,
                    "chunk_index": chunk_data.get("chunk_index", 0),
                    "start_time_seconds": float(chunk_data.get("start_time_seconds", 0.0)),
                    "end_time_seconds": float(chunk_data.get("end_time_seconds", 10.0)),
                    "duration_seconds": float(chunk_data.get("duration_seconds", 10.0)),
                    "matched_key": matched_key or (key_clean or "metadata"),
                    "matched_value": str(matched_value)[:400],
                    "significance_score": meta.get("significance_score"),
                    "sports_play_type": meta.get("sports_play_type"),
                    "scene_description": meta.get("scene_description"),
                    "chunk_summary": meta.get("chunk_summary"),
                    "extracted_metadata": meta,
                })

        def sort_key(item):
            score = item.get("significance_score")
            try:
                s_val = float(score) if score is not None else 0.0
            except (ValueError, TypeError):
                s_val = 0.0
            return (-s_val, item.get("video_id", ""), item.get("chunk_index", 0))

        results.sort(key=sort_key)
        return results[:limit]


