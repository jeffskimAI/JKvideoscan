# Jeff's VideoScan: AI-Powered Video Metadata Extraction Pipeline

Serverless, event-driven pipeline that ingests video files from Google Cloud Storage, segments them into strict 10-second chunks using FFmpeg, extracts dynamically defined metadata via Vertex AI Gemini 3.8 using Structured JSON Outputs, and persists the extracted insights into Cloud Firestore.

---

## Architecture Overview

```
Google Cloud Storage (GCS)
       │
       ▼
Cloud Eventarc (Storage Object Finalize)
       │
       ▼ HTTP POST (CloudEvent)
Cloud Run Service (FastAPI + FFmpeg + Python 3.13)
  ├── 1. Pairing Manager (.mp4 <-> _config.json)
  ├── 2. Config Parser & Validator (metadata_catalog.md)
  ├── 3. Video Prober (ffprobe)
  ├── 4. Segmentation Engine (strict 10s chunks with forced keyframes)
  ├── 5. Dynamic Schema Generator & Prompt Synthesizer
  ├── 6. Vertex AI Gemini 3.8 Inference Client (Structured JSON)
  └── 7. Firestore Repository (videos/{videoId}/chunks/{chunkId})
```

---

## Metadata Catalog Support

The pipeline maps directly to `metadata_catalog.md`, supporting extraction of up to 17 attributes across 4 categories:
1. **Visual & Spatial Metadata:** `scene_description`, `detected_objects`, `on_screen_text`, `camera_movement`, `brand_presence`, `lighting_and_color`.
2. **Audio & Speech Metadata:** `transcript`, `speaker_count`, `ambient_sounds`, `vocal_emotion`.
3. **Action & Event Metadata:** `key_actions`, `interactions`, `event_anomalies`.
4. **Semantic & Contextual Metadata:** `chunk_summary`, `overall_sentiment`, `content_categories`, `safety_flags`.

---

## Directory Structure

```
jeffsvideoscan/
├── spec.md                       # Core specification & constraints
├── metadata_catalog.md           # Supported metadata fields catalog
├── architecture.md               # End-to-end architecture & design
├── plan.md                       # Implementation task tracker
├── Dockerfile                    # Production container with FFmpeg
├── requirements.txt              # Production & test dependencies
├── src/
│   ├── config.py                 # Application settings (Pydantic BaseSettings)
│   ├── logger.py                 # Structured logger
│   ├── catalog.py                # Catalog definitions & config validation
│   ├── schema_generator.py       # Dynamic Pydantic & OpenAPI schema generator
│   ├── video_probe.py            # ffprobe duration & stream inspection
│   ├── video_segmenter.py        # Strict 10s FFmpeg chunker
│   ├── gemini_extractor.py       # Vertex AI Gemini 3.8 client
│   ├── firestore_repo.py         # Cloud Firestore data access layer
│   ├── storage_manager.py        # GCS download & file pairing
│   ├── pipeline_orchestrator.py  # End-to-end workflow coordinator
│   └── main.py                   # FastAPI CloudEvent listener & health checks
├── scripts/
│   ├── deploy.sh                 # Cloud Run deployment script
│   ├── setup_eventarc.sh         # Eventarc trigger configuration
│   └── run_local_simulation.py   # Standalone local simulator
└── tests/
    ├── test_catalog_and_schema.py
    ├── test_video_segmenter.py
    ├── test_gemini_extractor.py
    ├── test_firestore_repo.py
    └── test_pipeline_e2e.py
```

---

## Quickstart & Local Verification

### 1. Run All Tests
```bash
.venv/bin/pytest tests/ -v
```

### 2. Run Local Simulation
Executes an end-to-end simulation with synthetic video generation, FFmpeg 10s chunking, dynamic schema generation, and simulated Firestore output:
```bash
.venv/bin/python3 scripts/run_local_simulation.py
```

### 3. Start Local Web Server
```bash
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080
```

---

## Deployment to Google Cloud Run

```bash
# 1. Deploy Cloud Run service
./scripts/deploy.sh

# 2. Setup Eventarc trigger
./scripts/setup_eventarc.sh
```
