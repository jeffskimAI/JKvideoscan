# Feature Specification: AI-Powered Video Metadata Extraction Pipeline

## 1. Goal and Problem Statement
**Goal:** Build a serverless, event-driven pipeline that ingests video files from Cloud Storage, splits them into 10-second chunks using FFmpeg, extracts dynamically defined metadata via Vertex AI Gemini 3.8, and stores the results in Firebase for downstream retrieval.
**Problem it Solves:** Automates the granular indexing and understanding of video content at scale, allowing users to define exactly what visual or audio metadata they want to extract on a per-video basis (refer to `metadata_catalog.md` for the full dictionary of supported extraction points).

## 2. Execution Flow
1. **Ingestion:** A video file (`.mp4`) and a corresponding configuration file (`.json`) are uploaded to a designated Google Cloud Storage (GCS) bucket.
2. **Trigger:** The file upload event triggers a Cloud Run service.
3. **Configuration Parsing:** The Cloud Run service reads the input configuration file to determine the specific metadata points to extract.
4. **Processing (FFmpeg):** The service streams or downloads the video and utilizes FFmpeg to segment the video into sequential 10-second chunks.
5. **AI Extraction:** For each 10-second chunk, the service sends the video segment and a dynamically constructed prompt (based on the config file) to the Vertex AI Gemini 3.8 API.
6. **Persistence:** The structured metadata response returned by Gemini is logged into a Firebase Data store (Cloud Firestore), tagged with the video ID and the chunk's start/end timestamps.

## 3. Scope and Boundaries
* **In Scope:** 
  * Cloud Run service containerization with FFmpeg installed.
  * Integration with GCS (read), Vertex AI Gemini 3.8 API (inference via Structured JSON Outputs), and Firestore (write).
  * Strict 10-second chunking logic.
* **Out of Scope (Do NOT Implement):** 
  * User Interface (UI) or frontend dashboards for upload or playback.
  * Variable chunking intervals (locked to 10 seconds).
  * Audio extraction/transcription independent of Gemini's native multimodal capabilities.

## 4. API and Contract Definitions
* **Trigger Mechanism:** Google Cloud Eventarc (Storage Object Finalize) routing to Cloud Run.
* **Input Configuration Schema (`<video_filename>_config.json`):**
  * *Note: The values inside `target_metadata` must map to the supported keys defined in `metadata_catalog.md`.*
  ```json
  {
    "target_metadata": ["scene_description", "detected_objects", "overall_sentiment", "key_actions"]
  }
