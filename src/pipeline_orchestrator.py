"""Pipeline Orchestrator for end-to-end video processing."""
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.catalog import parse_config_payload
from src.config import settings
from src.error_handler import describe_error
from src.firestore_repo import FirestoreRepository
from src.gemini_extractor import GeminiExtractor
from src.logger import logger
from src.storage_manager import StorageManager
from src.video_probe import probe_video
from src.video_segmenter import VideoChunk, segment_video


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
        pipeline_logs: List[Dict[str, Any]] = []
        log_lock = threading.Lock()

        def log_step(level: str, msg: str, stage: str = "general") -> None:
            now_iso = datetime.now(timezone.utc).isoformat()
            with log_lock:
                pipeline_logs.append({
                    "timestamp": now_iso,
                    "level": level,
                    "stage": stage,
                    "message": msg,
                })
            log_fn = getattr(logger, level.lower(), logger.info)
            log_fn(f"[{video_id}][{stage}] {msg}")

        current_stage = "pre_check"
        log_step("INFO", f"Starting video pipeline execution for video_id='{video_id}'", current_stage)

        existing = self.firestore_repo.get_video_record(video_id)
        if existing and not force:
            status = existing.get("status")
            if status == "PROCESSING":
                log_step("INFO", f"Video {video_id} is already PROCESSING. Skipping duplicate run.", "pre_check")
                return {"status": "PROCESSING", "video_id": video_id, "message": "Already processing"}

        work_dir = self.temp_base_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Download and parse configuration
            current_stage = "config_retrieval"
            log_step("INFO", f"Downloading configuration gs://{bucket_name}/{config_object}", current_stage)
            config_dict = self.storage_manager.download_json_as_dict(bucket_name, config_object)

            current_stage = "config_validation"
            target_metadata = parse_config_payload(config_dict)
            log_step("INFO", f"Validated {len(target_metadata)} target metadata keys: {target_metadata}", current_stage)

            # 2. Download video file
            current_stage = "video_download"
            local_video_path = work_dir / Path(video_object).name
            log_step("INFO", f"Downloading video gs://{bucket_name}/{video_object} to {local_video_path.name}", current_stage)
            self.storage_manager.download_blob_to_file(bucket_name, video_object, local_video_path)
            log_step("INFO", f"Video downloaded successfully ({local_video_path.stat().st_size} bytes)", current_stage)

            # 3. Probe video metadata
            current_stage = "video_probing"
            log_step("INFO", f"Probing video stream properties using ffprobe...", current_stage)
            probe_info = probe_video(local_video_path, chunk_duration_sec=settings.chunk_duration_seconds)
            log_step(
                "INFO",
                f"Probed video: duration={probe_info.duration:.2f}s, total_chunks={probe_info.total_chunks}, codec={probe_info.codec_name}",
                current_stage,
            )

            # 4. Initialize Firestore document
            current_stage = "firestore_init"
            log_step("INFO", f"Initializing Firestore record for {video_id} with status PROCESSING", current_stage)
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
            current_stage = "video_segmentation"
            chunks_dir = work_dir / "chunks"
            log_step("INFO", f"Segmenting video into strict {settings.chunk_duration_seconds}s chunks with FFmpeg...", current_stage)
            chunks = segment_video(
                local_video_path,
                chunks_dir,
                chunk_duration_sec=settings.chunk_duration_seconds,
            )
            log_step("INFO", f"Successfully generated {len(chunks)} video segments", current_stage)

            # 6. Extract metadata for each chunk via Vertex AI Gemini
            current_stage = "gemini_extraction"
            total_chunks = len(chunks)
            log_step("INFO", f"Beginning Gemini metadata extraction across {total_chunks} chunks with model={self.gemini_extractor.model_name}...", current_stage)

            # Checkpoint / Resumability check: find already processed chunks in Firestore
            existing_chunks = self.firestore_repo.get_video_chunks(video_id)
            completed_indices = {
                c.get("chunk_index")
                for c in existing_chunks
                if "chunk_index" in c and c.get("extracted_metadata")
            }
            if completed_indices:
                log_step("INFO", f"Resuming video {video_id}: Found {len(completed_indices)} already completed chunks in Firestore. Skipping them.", current_stage)

            chunks_to_process = [c for c in chunks if c.index not in completed_indices]
            total_to_process = len(chunks_to_process)

            if not chunks_to_process:
                log_step("INFO", f"All {total_chunks} chunks are already completed in Firestore.", current_stage)
            else:
                concurrency = max(1, min(settings.max_concurrent_chunks, total_to_process))
                log_step("INFO", f"Processing {total_to_process} chunks with concurrency={concurrency} workers (max_workers={settings.max_concurrent_chunks})...", current_stage)

                stop_requested = threading.Event()
                completed_counter = len(completed_indices)
                failed_chunk_indices: List[int] = []
                counter_lock = threading.Lock()
                extraction_error: Optional[Exception] = None

                def process_chunk_task(chunk: VideoChunk):
                    nonlocal completed_counter, extraction_error
                    if stop_requested.is_set():
                        return

                    # Periodically check if video processing was stopped by user
                    if chunk.index % 5 == 0:
                        rec = self.firestore_repo.get_video_record(video_id)
                        if rec and rec.get("status") in ("STOPPED", "CANCELLED"):
                            stop_requested.set()
                            logger.warning(f"Aborting chunk worker for {video_id}: status={rec.get('status')}")
                            return

                    log_step("INFO", f"Extracting chunk {chunk.index + 1}/{total_chunks} ({chunk.start_time_seconds:.1f}s - {chunk.end_time_seconds:.1f}s)...", current_stage)
                    last_exc = None
                    for attempt in range(1, 4):
                        try:
                            chunk_metadata = self.gemini_extractor.extract_chunk_metadata(
                                chunk=chunk,
                                target_keys=target_metadata,
                            )
                            self.firestore_repo.save_chunk_metadata(
                                video_id=video_id,
                                chunk=chunk,
                                extracted_metadata=chunk_metadata,
                                model_version=getattr(self.gemini_extractor, "last_used_model", None) or self.gemini_extractor.model_name,
                            )
                            with counter_lock:
                                completed_counter += 1
                                log_step("INFO", f"Saved metadata for chunk {chunk.index} to Firestore ({completed_counter}/{total_chunks} completed)", current_stage)
                                try:
                                    self.firestore_repo.update_video_progress(video_id, completed_counter, total_chunks)
                                except Exception:
                                    pass
                            last_exc = None
                            break
                        except Exception as e:
                            last_exc = e
                            if attempt < 3:
                                logger.warning(f"Chunk {chunk.index} attempt {attempt} failed ({e}), retrying in {attempt}s...")
                                time.sleep(attempt * 1.0)
                            else:
                                logger.warning(
                                    f"Chunk {chunk.index} failed extraction after 3 attempts ({e}). "
                                    "Recording placeholder metadata so pipeline can proceed."
                                )
                                fallback_meta = {k: None for k in target_metadata}
                                fallback_meta.update({
                                    "scene_description": f"Metadata extraction failed: {e}",
                                    "chunk_summary": f"Extraction error: {e}",
                                    "overall_sentiment": "Neutral",
                                    "sports_key_plays": [],
                                    "safety_flags": [],
                                    "extraction_error": str(e),
                                })
                                try:
                                    self.firestore_repo.save_chunk_metadata(
                                        video_id=video_id,
                                        chunk=chunk,
                                        extracted_metadata=fallback_meta,
                                        model_version="error_placeholder",
                                    )
                                    with counter_lock:
                                        completed_counter += 1
                                        failed_chunk_indices.append(chunk.index)
                                        log_step("WARNING", f"Chunk {chunk.index} saved with fallback placeholder metadata after 3 attempts ({completed_counter}/{total_chunks} completed)", current_stage)
                                        try:
                                            self.firestore_repo.update_video_progress(video_id, completed_counter, total_chunks)
                                        except Exception:
                                            pass
                                    last_exc = None
                                    break
                                except Exception as save_err:
                                    logger.error(f"Failed to save fallback metadata for chunk {chunk.index}: {save_err}")
                                    with counter_lock:
                                        if extraction_error is None:
                                            extraction_error = e
                                    raise e

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    future_map = {executor.submit(process_chunk_task, chunk): chunk for chunk in chunks_to_process}
                    for future in as_completed(future_map):
                        try:
                            future.result()
                        except Exception as exc:
                            if not extraction_error:
                                extraction_error = exc

                # Check if pipeline was stopped by user
                current_rec = self.firestore_repo.get_video_record(video_id)
                if (current_rec and current_rec.get("status") in ("STOPPED", "CANCELLED")) or stop_requested.is_set():
                    status_str = current_rec.get("status") if current_rec else "STOPPED"
                    log_step("WARNING", f"Pipeline halted by user request ({status_str})", current_stage)
                    logger.warning(f"Aborting pipeline for {video_id}: status={status_str}")
                    return {
                        "status": status_str,
                        "video_id": video_id,
                        "processed_chunks": completed_counter,
                        "total_chunks": total_chunks,
                        "logs": pipeline_logs,
                    }

                if len(failed_chunk_indices) > max(10, int(total_chunks * 0.25)):
                    raise GeminiExtractionError(
                        f"{len(failed_chunk_indices)}/{total_chunks} chunks failed extraction: {failed_chunk_indices[:10]}"
                    )

                if extraction_error:
                    raise extraction_error

            # 7. Complete video processing in Firestore
            current_stage = "completion"
            log_step("INFO", f"Marking video={video_id} as COMPLETED in Firestore", current_stage)
            self.firestore_repo.complete_video_processing(video_id, total_chunks=len(chunks))

            log_step("INFO", f"Pipeline completed successfully for {video_id} ({len(chunks)} chunks, {probe_info.duration:.1f}s)", current_stage)
            return {
                "status": "COMPLETED",
                "video_id": video_id,
                "total_chunks": len(chunks),
                "duration_seconds": probe_info.duration,
                "logs": pipeline_logs,
            }

        except Exception as e:
            err_diag = describe_error(
                exception=e,
                stage=current_stage,
                context={
                    "video_id": video_id,
                    "bucket_name": bucket_name,
                    "video_object": video_object,
                    "config_object": config_object,
                },
            )
            log_step(
                "ERROR",
                f"Pipeline failed at stage '{current_stage}': {e}\n{err_diag['description']}\n{err_diag.get('traceback', '')}",
                current_stage,
            )
            logger.exception(f"Pipeline failed for video_id={video_id}: {e}")
            try:
                self.firestore_repo.fail_video_processing(
                    video_id=video_id,
                    error_message=str(e),
                    error_description=err_diag["description"],
                    logs=pipeline_logs,
                )
            except Exception as fe:
                logger.error(f"Failed to record failure state in Firestore: {fe}")
            raise e

        finally:
            # Clean up temporary work directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
                logger.debug(f"Cleaned up temporary workspace: {work_dir}")
