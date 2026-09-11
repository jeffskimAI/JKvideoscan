"""Pipeline Orchestrator for end-to-end video processing."""
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from src.catalog import parse_config_payload
from src.config import settings
from src.firestore_repo import FirestoreRepository
from src.gemini_extractor import GeminiExtractor
from src.logger import logger
from src.storage_manager import StorageManager
from src.video_probe import probe_video
from src.video_segmenter import segment_video


class PipelineOrchestrator:
    """Coordinates GCS ingestion, FFmpeg chunking, Gemini extraction, and Firestore storage."""

    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        firestore_repo: Optional[FirestoreRepository] = None,
        gemini_extractor: Optional[GeminiExtractor] = None,
        temp_dir: Optional[str] = None,
    ):
        self.storage_manager = storage_manager or StorageManager()
        self.firestore_repo = firestore_repo or FirestoreRepository()
        self.gemini_extractor = gemini_extractor or GeminiExtractor()
        self.temp_base_dir = Path(temp_dir or settings.temp_dir)

    def process_event(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """Processes an incoming GCS object finalization event.

        Args:
            bucket_name: GCS bucket containing the files.
            object_name: The finalized object name (.mp4 or _config.json).

        Returns:
            Dictionary with processing status and summary.
        """
        logger.info(f"Received event for gs://{bucket_name}/{object_name}")

        try:
            video_id, video_object, config_object = self.storage_manager.derive_pair_paths(object_name)
        except ValueError as e:
            logger.warning(f"Ignoring non-target object: {e}")
            return {"status": "IGNORED", "message": str(e)}

        # Check for pair readiness
        video_exists = self.storage_manager.check_blob_exists(bucket_name, video_object)
        config_exists = self.storage_manager.check_blob_exists(bucket_name, config_object)

        if not video_exists or not config_exists:
            missing = video_object if not video_exists else config_object
            logger.info(
                f"Video pair not ready for video_id='{video_id}'. "
                f"Missing '{missing}'. Awaiting paired upload."
            )
            return {
                "status": "AWAITING_PAIR",
                "video_id": video_id,
                "missing_file": missing,
            }

        # Both video and config exist in GCS -> execute pipeline
        return self.execute_video_pipeline(bucket_name, video_id, video_object, config_object)

    def execute_video_pipeline(
        self,
        bucket_name: str,
        video_id: str,
        video_object: str,
        config_object: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Executes full extraction pipeline for a paired video and config."""
        existing = self.firestore_repo.get_video_record(video_id)
        if existing and not force:
            status = existing.get("status")
            if status == "PROCESSING":
                logger.info(f"Video {video_id} is already PROCESSING. Skipping duplicate run.")
                return {"status": "PROCESSING", "video_id": video_id, "message": "Already processing"}

        work_dir = self.temp_base_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)


        try:
            # 1. Download and parse configuration
            config_dict = self.storage_manager.download_json_as_dict(bucket_name, config_object)
            target_metadata = parse_config_payload(config_dict)
            logger.info(f"Target metadata keys for {video_id}: {target_metadata}")

            # 2. Download video file
            local_video_path = work_dir / Path(video_object).name
            self.storage_manager.download_blob_to_file(bucket_name, video_object, local_video_path)

            # 3. Probe video metadata
            probe_info = probe_video(local_video_path, chunk_duration_sec=settings.chunk_duration_seconds)

            # 4. Initialize Firestore document
            self.firestore_repo.init_video_record(
                video_id=video_id,
                filename=Path(video_object).name,
                gcs_bucket=bucket_name,
                gcs_path=video_object,
                config_path=config_object,
                target_metadata=target_metadata,
                total_chunks=probe_info.total_chunks,
                duration_seconds=probe_info.duration,
            )

            # 5. Segment into strict 10-second chunks using FFmpeg
            chunks_dir = work_dir / "chunks"
            chunks = segment_video(
                local_video_path,
                chunks_dir,
                chunk_duration_sec=settings.chunk_duration_seconds,
            )

            # 6. Extract metadata for each chunk via Vertex AI Gemini
            logger.info(f"Extracting metadata across {len(chunks)} chunks for {video_id}...")
            for chunk in chunks:
                chunk_metadata = self.gemini_extractor.extract_chunk_metadata(
                    chunk=chunk,
                    target_keys=target_metadata,
                )
                self.firestore_repo.save_chunk_metadata(
                    video_id=video_id,
                    chunk=chunk,
                    extracted_metadata=chunk_metadata,
                    model_version=self.gemini_extractor.model_name,
                )

            # 7. Complete video processing in Firestore
            self.firestore_repo.complete_video_processing(video_id)

            logger.info(f"Successfully processed video pipeline for {video_id} ({len(chunks)} chunks)")
            return {
                "status": "COMPLETED",
                "video_id": video_id,
                "total_chunks": len(chunks),
                "duration_seconds": probe_info.duration,
            }

        except Exception as e:
            logger.exception(f"Pipeline failed for video_id={video_id}: {e}")
            try:
                self.firestore_repo.fail_video_processing(video_id, str(e))
            except Exception as fe:
                logger.error(f"Failed to record failure state in Firestore: {fe}")
            raise e

        finally:
            # Clean up temporary work directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary workspace: {work_dir}")
