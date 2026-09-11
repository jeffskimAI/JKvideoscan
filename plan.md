# Implementation Plan: AI-Powered Video Metadata Extraction Pipeline

## Overview
This plan outlines the systematic implementation of the video metadata extraction pipeline specified in `spec.md` and `architecture.md`. Each task is accompanied by verification criteria to ensure high test coverage, robust error handling, and production readiness.

---

## Task Progress Tracker

### Phase 1: Project Environment & Core Infrastructure
- [x] **Task 1.1: Dependencies & Configuration**
  - Created `requirements.txt` with required dependencies: `fastapi`, `uvicorn`, `pydantic`, `google-cloud-storage`, `google-cloud-firestore`, `google-genai`, `pytest`, `pytest-asyncio`.
  - Implemented `src/config.py` using `pydantic-settings` to manage environment variables (GCS buckets, Firestore database, Gemini model name `gemini-3.8`, chunk duration = 10, log level).
- [x] **Task 1.2: Project Layout & Logging Framework**
  - Setup structured logging in `src/logger.py` with timestamps, levels, and module tagging.

### Phase 2: Metadata Catalog & Dynamic Schema Generation
- [x] **Task 2.1: Catalog Definition (`src/catalog.py`)**
  - Encoded all supported metadata keys across the 4 categories from `metadata_catalog.md` with explicit typing and descriptions.
  - Implemented configuration file validator (`validate_target_metadata`, `parse_config_payload`) to ensure input JSON matches catalog keys.
- [x] **Task 2.2: Dynamic Structured Schema Generator (`src/schema_generator.py`)**
  - Built dynamic Pydantic model generator that generates schema containing only the user-requested `target_metadata` fields.
  - Generates OpenAPI-compliant JSON schema for Vertex AI Gemini 3.8 `response_schema`.
- [x] **Task 2.3: Unit Tests for Catalog & Schema Generation (`tests/test_catalog_and_schema.py`)**
  - Tested valid configs, deduplication, invalid keys, empty list, and schema generation output (11/11 tests passing).

### Phase 3: FFmpeg Video Segmentation Engine
- [x] **Task 3.1: Video Probe Utility (`src/video_probe.py`)**
  - Implemented `ffprobe` wrapper to probe total duration, resolution, codecs, and calculate chunk bounds.
- [x] **Task 3.2: Strict 10-Second Chunking (`src/video_segmenter.py`)**
  - Implemented FFmpeg segmenter with forced keyframes at exact 10s intervals (`gte(t, n_forced*10)`) and `-reset_timestamps 1`.
  - Guarantees deterministic output chunk filenames and metadata (chunk index, start timestamp, end timestamp, duration).
- [x] **Task 3.3: Chunker Unit & Integration Tests (`tests/test_video_segmenter.py`)**
  - Built synthetic video generator helper producing test videos of varying lengths (e.g. 6s, 15s, 20s, 25s) with and without audio.
  - Tested chunk boundary precision, single chunk, multi-chunk, and remainder handling (6/6 tests passing).

### Phase 4: Vertex AI Gemini 3.8 Inference Engine
- [x] **Task 4.1: Prompt Synthesis & Model Client (`src/gemini_extractor.py`)**
  - Implemented Gemini 3.8 multimodal prompt constructor tailored to the requested metadata keys.
  - Integrated with `google-genai` / Vertex AI SDK with structured JSON output configuration.
  - Implemented exponential backoff retry policy for rate limiting and transient failures.
- [x] **Task 4.2: Gemini Extractor Unit & Mock Tests (`tests/test_gemini_extractor.py`)**
  - Mocked Gemini API responses for CI testing.
  - Verified schema compliance, missing keys handling, and fallback resilience (3/3 tests passing).

### Phase 5: Cloud Firestore Persistence Layer
- [x] **Task 5.1: Firestore Repository (`src/firestore_repo.py`)**
  - Implemented document creation for `videos/{video_id}`.
  - Implemented subcollection creation for `videos/{video_id}/chunks/{chunk_index}`.
  - Supports status transitions: `PENDING` -> `PROCESSING` -> `COMPLETED` / `FAILED`.
- [x] **Task 5.2: Firestore Repo Unit Tests (`tests/test_firestore_repo.py`)**
  - Verified document structure, timestamps, chunk storage, and idempotency (3/3 tests passing).

### Phase 6: CloudEvent Router, Ingestion Handler, and Pipeline Orchestration
- [x] **Task 6.1: GCS Storage Manager & Pairing (`src/storage_manager.py`)**
  - Handles GCS download of video and config files.
  - Implemented pairing logic for `.mp4` and `_config.json`.
- [x] **Task 6.2: Pipeline Orchestrator (`src/pipeline_orchestrator.py`)**
  - Coordinates: Ingestion -> Download -> Probe -> Chunk -> Gemini Extraction -> Firestore Logging -> Cleanup.
- [x] **Task 6.3: Cloud Run Service (`src/main.py`)**
  - CloudEvent HTTP POST endpoint (`/` and `/events`) supporting binary and structured CloudEvent formats.
  - Health check endpoints (`/healthz`, `/livez`).
- [x] **Task 6.4: End-to-End Pipeline Integration Test (`tests/test_pipeline_e2e.py`)**
  - Simulated full CloudEvent lifecycle with synthetic video and sample config (4/4 tests passing).

### Phase 7: Containerization & Deployment
- [x] **Task 7.1: Production Dockerfile**
  - Debian slim image with FFmpeg installed, non-root user (`appuser`), and port 8080 exposure.
- [x] **Task 7.2: Deployment and Trigger Scripts**
  - Created Cloud Run deployment script (`scripts/deploy.sh`) and Eventarc trigger configuration (`scripts/setup_eventarc.sh`).
- [x] **Task 7.3: Complete System Audit & Validation**
  - Built standalone simulation runner (`scripts/run_local_simulation.py`) and executed simulation verifying all constraints.
  - Executed entire test suite (27/27 tests passing).
