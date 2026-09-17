# Architecture Design: AI-Powered Video Metadata Extraction Pipeline

## 1. Executive Summary & Objective

This document specifies the technical architecture for an event-driven, serverless pipeline deployed on Google Cloud. The system automatically ingests video files (`.mp4`) and user configuration files (`<video_filename>_config.json`) uploaded to Google Cloud Storage (GCS), executes precise 10-second video chunking via FFmpeg, extracts custom user-selected metadata for each chunk using **Vertex AI Gemini 3.8** with Structured JSON Outputs, and persists the extracted insights into **Cloud Firestore** for downstream search and retrieval.

---

## 2. High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Storage [Google Cloud Storage]
        GCS_Bucket[("GCS Ingestion Bucket\ngs://<bucket-name>/")]
        MP4_File["video_sample.mp4"] --> GCS_Bucket
        JSON_File["video_sample_config.json"] --> GCS_Bucket
    end

    subgraph EventBus [Google Cloud Eventarc]
        GCS_Bucket -->|"google.cloud.storage.object.v1.finalized"| EventarcTrigger["Eventarc Trigger\n(Filtering on .mp4 / .json)"]
    end

    subgraph Compute [Cloud Run Service (FFmpeg + Python 3.13)]
        EventarcTrigger -->|"HTTP POST /events (CloudEvent)"| IngestionHandler["CloudEvent Router & Ingestion Handler"]
        
        IngestionHandler -->|"Check & Pair"| PairingManager["File Pairing Manager\n(.mp4 <-> _config.json)"]
        
        PairingManager -->|"Load Config"| ConfigParser["Config Parser & Validator\n(metadata_catalog.md)"]
        PairingManager -->|"Download/Stream"| FFmpegEngine["FFmpeg Segmentation Engine\n(Strict 10s Chunks)"]
        
        FFmpegEngine -->|"10-second chunk files"| ChunkQueue["Sequential/Parallel Chunk Processor"]
        
        ConfigParser -->|"Dynamic Schema & Prompts"| GeminiClient["Vertex AI Gemini 3.8 Client\n(Structured JSON Outputs)"]
        ChunkQueue -->|"Multimodal Video Chunk"| GeminiClient
    end

    subgraph ExternalAI [Google Cloud Vertex AI]
        GeminiClient -->|"generate_content(video_chunk, schema)"| GeminiModel["Gemini 3.8 Model\n(gemini-3.8)"]
        GeminiModel -->|"Structured JSON Response"| GeminiClient
    end

    subgraph Persistence [Cloud Firestore (Firebase)]
        GeminiClient -->|"Persist Chunk Document"| FirestoreDB[("Cloud Firestore\nCollection: videos/{videoId}/chunks/{chunkIndex}")]
    end
```

---

## 3. Core Component Specifications

### 3.1 Ingestion & Event Handling (GCS & Eventarc)

#### Trigger Mechanism
- **Eventarc Source:** Google Cloud Storage bucket notifications.
- **Event Type:** `google.cloud.storage.object.v1.finalized`.
- **Destination:** Cloud Run Service endpoint (`POST /events` or `POST /`).
- **CloudEvent Specification:** Standard CloudEvent v1.0 payload format received over HTTP:
  ```json
  {
    "specversion": "1.0",
    "type": "google.cloud.storage.object.v1.finalized",
    "source": "//storage.googleapis.com/projects/_/buckets/my-video-bucket",
    "id": "1234567890",
    "time": "2026-09-11T17:30:00Z",
    "data": {
      "bucket": "my-video-bucket",
      "name": "conference_talk.mp4",
      "generation": "1726075800000",
      "contentType": "video/mp4",
      "size": "52428800"
    }
  }
  ```

#### File Pairing and Trigger Synchronization
Because GCS upload of the video (`<name>.mp4`) and configuration (`<name>_config.json`) are distinct operations:
1. When `<name>.mp4` arrives:
   - Check if `<name>_config.json` already exists in the bucket.
   - If present, proceed immediately with processing.
   - If missing, check Firestore state: register video in `PENDING_CONFIG` state and wait or exit gracefully (idempotently).
2. When `<name>_config.json` arrives:
   - Check if `<name>.mp4` already exists in the bucket.
   - If present and status is `PENDING_CONFIG` or unstarted, trigger the processing pipeline.
   - If missing, register configuration in `PENDING_VIDEO` state.
3. This eliminates race conditions regardless of upload order.

---

### 3.2 Configuration Parser & Metadata Catalog Mapping

The configuration file format follows `spec.md`:
```json
{
  "target_metadata": [
    "chunk_summary",
    "transcript",
    "detected_objects",
    "ambient_sounds"
  ]
}
```

The system includes a strictly typed **Catalog Dictionary** mirroring `metadata_catalog.md`:

| Key | Section | Type | Schema Definition |
| :--- | :--- | :--- | :--- |
| `scene_description` | Visual & Spatial | string | Narrative of visual setting and environment |
| `detected_objects` | Visual & Spatial | list[string] | Distinct physical items present in the frame |
| `on_screen_text` | Visual & Spatial | list[string] | Text recognized within video frames (OCR) |
| `camera_movement` | Visual & Spatial | string | Camera behavior (Static shot, Pan, Handheld, etc.) |
| `brand_presence` | Visual & Spatial | list[string] | Recognizable logos or brand imagery visible |
| `lighting_and_color` | Visual & Spatial | string | Mood, color palette, lighting setup |
| `transcript` | Audio & Speech | string | Word-for-word spoken dialogue |
| `speaker_count` | Audio & Speech | integer | Number of distinct voices heard in the chunk |
| `ambient_sounds` | Audio & Speech | list[string] | Background noises and sound effects |
| `vocal_emotion` | Audio & Speech | string | Tone or emotion of the speaker(s) |
| `key_actions` | Action & Event | list[string] | Primary activities performed by subjects |
| `interactions` | Action & Event | string | Subject-object or subject-subject interactions |
| `event_anomalies` | Action & Event | list[string] | Sudden or unexpected changes in the scene |
| `chunk_summary` | Semantic | string | 1-2 sentence concise summary of the 10 seconds |
| `overall_sentiment` | Semantic | string | Emotional sentiment (Positive, Neutral, Tense) |
| `content_categories` | Semantic | list[string] | Classification tags (Business, Technology, etc.) |
| `safety_flags` | Semantic | list[string] | Unsafe/explicit content detected ("None detected", etc.) |

#### Dynamic Schema Generation
From the requested `target_metadata`, the pipeline dynamically constructs:
1. A **Pydantic Model** / **OpenAPI JSON Schema** containing *only* the requested fields.
2. A customized prompt instructing Gemini 3.8 to populate exactly those fields.
3. This guarantees response predictability, eliminates hallucinations, and minimizes output token costs.

---

### 3.3 FFmpeg Video Segmentation Engine

#### Strict 10-Second Chunking
The specification mandates strict 10-second intervals:
- Chunk 0: `[00:00 - 00:10]`
- Chunk 1: `[00:10 - 00:20]`
- Chunk $N$: `[10*N - 10*(N+1)]`
- Final Chunk: Remainder duration up to 10 seconds (e.g. 00:40 to 00:47 for a 47-second clip).

#### FFmpeg Command
To ensure exact time-slice segmenting without drifting or audio desync:
```bash
ffmpeg -i input.mp4 \
  -f segment \
  -segment_time 10 \
  -reset_timestamps 1 \
  -c:v libx264 -preset ultrafast -crf 22 \
  -c:a aac -b:a 128k \
  -force_key_frames "expr:gte(t,n_forced*10)" \
  chunk_%04d.mp4
```
- `-segment_time 10`: Target segment length of 10s.
- `-force_key_frames "expr:gte(t,n_forced*10)"`: Forces keyframes exactly at 10-second boundaries, ensuring clean cuts with no dropped leading frames.
- `-reset_timestamps 1`: Resets each chunk's internal timestamps to start from 00:00, required for standalone video chunk consumption by Gemini.

#### Video Probing
`ffprobe` is used prior to segmentation to extract:
- Total duration (seconds)
- Video codec, resolution, and frame rate
- Audio streams present/absent
- Expected chunk count: $\lceil \text{duration} / 10 \rceil$

---

### 3.4 Vertex AI Gemini 3.8 Inference Engine

#### Multimodal Payload Construction
For each chunk:
1. Video bytes or Cloud Storage URI are packaged into `Part.from_data()` or `Part.from_uri()`.
2. The dynamic JSON schema generated in Section 3.2 is passed to `generation_config.response_schema` with `response_mime_type="application/json"`.
3. The prompt specifies:
   - Analysis bounds: the 10-second segment.
   - Instruction to extract ONLY observed visual/audio data strictly matching the requested metadata fields.

#### Resilience & Rate Limiting
- Concurrent chunk analysis (up to configurable concurrency limit, e.g., 4 parallel chunks per instance) using `asyncio` or thread pools.
- Exponential backoff with jitter on `429 (ResourceExhausted)` or `503 (Unavailable)`.

---

### 3.5 Cloud Firestore Persistence Model

Firestore collections are organized hierarchically:

#### 1. Root Video Document: `videos/{video_id}`
```json
{
  "video_id": "conference_talk_2026_09",
  "filename": "conference_talk.mp4",
  "gcs_bucket": "my-video-bucket",
  "gcs_path": "conference_talk.mp4",
  "config_path": "conference_talk_config.json",
  "duration_seconds": 125.4,
  "total_chunks": 13,
  "target_metadata": ["scene_description", "detected_objects", "transcript", "chunk_summary"],
  "status": "COMPLETED", // PENDING, PROCESSING, COMPLETED, FAILED
  "created_at": "2026-09-11T17:35:00.000Z",
  "updated_at": "2026-09-11T17:36:15.000Z",
  "error_message": null
}
```

#### 2. Subcollection: `videos/{video_id}/chunks/{chunk_index}`
Document ID: e.g. `0000`, `0001`, ...
```json
{
  "chunk_index": 0,
  "start_time_seconds": 0.0,
  "end_time_seconds": 10.0,
  "duration_seconds": 10.0,
  "extracted_metadata": {
    "scene_description": "A presenter stands on a modern stage in front of a blue keynote screen.",
    "detected_objects": ["microphone", "lectern", "screen", "laptop"],
    "transcript": "Good morning everyone, welcome to our 2026 annual technology showcase.",
    "chunk_summary": "Presenter opens the keynote and welcomes the audience to the showcase."
  },
  "model_version": "gemini-3.8",
  "processed_at": "2026-09-11T17:35:45.000Z"
}
```

---

## 4. Containerization & Deployment

### 4.1 Cloud Run Architecture
- Base image: `python:3.13-slim`
- System packages: `ffmpeg`
- Application server: `uvicorn` running `FastAPI`
- Default Port: `8080`
- Resource Recommendations:
  - Memory: 2GiB - 4GiB (needed for in-memory video segment buffering or local scratch)
  - CPU: 2 vCPU
  - Concurrency: 1 - 4 requests per container instance (video processing is CPU-bound)
  - Timeout: 900s (Cloud Run standard max for long video chunk processing)

---

## 5. Security & Identity
- **IAM Roles:**
  - `roles/storage.objectViewer` on GCS bucket
  - `roles/aiplatform.user` for Vertex AI Gemini API
  - `roles/datastore.user` for Cloud Firestore access
  - `roles/run.invoker` for Eventarc trigger service account
- **Principle of Least Privilege:** Cloud Run service account has granular permissions restricted to only the designated bucket and Firestore database.

---

## 6. Testing & Validation Strategy

1. **Unit Tests:**
   - Catalog schema validator (tests valid/invalid `target_metadata`).
   - Dynamic Pydantic / JSON schema generator.
   - CloudEvent parser & pairing logic.
2. **FFmpeg Engine Integration Tests:**
   - Synthetic video generator (creates test `.mp4` with audio and visual text using ffmpeg `testsrc`).
   - Validation that generated chunks are strictly 10 seconds.
3. **Gemini Extraction Tests:**
   - Mocked Gemini responses for deterministic CI/CD testing.
   - End-to-end integration test runner against live Vertex AI Gemini 3.8 when credentials are provided.
4. **End-to-End Local Pipeline Test:**
   - Simulated CloudEvent delivery -> Config lookup -> Chunking -> Extraction -> Mock Firestore verification.
