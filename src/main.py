"""FastAPI Cloud Run Service, Web Interface, and CloudEvent Ingestion."""
import hashlib
import hmac
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.catalog import get_catalog_summary, parse_config_payload, validate_target_metadata
from src.config import settings
from src.error_handler import describe_error
from src.logger import get_system_logs, logger
from src.pipeline_orchestrator import PipelineOrchestrator

app = FastAPI(
    title="Jeff's VideoScan - AI Video Metadata Extraction Pipeline",
    version="1.0.0",
    description="Production service for 10s video chunking, Gemini metadata extraction, and Web UI",
)

orchestrator = PipelineOrchestrator()

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def load_index_html() -> str:
    """Loads web interface HTML file."""
    if INDEX_HTML_PATH.exists():
        return INDEX_HTML_PATH.read_text(encoding="utf-8")
    return "<h1>Jeff's VideoScan</h1><p>UI loading error: index.html not found.</p>"


# ---------------------------------------------------------------------------
# Authentication Utilities & Dependencies
# ---------------------------------------------------------------------------
class LoginPayload(BaseModel):
    password: str


def get_auth_token() -> str:
    """Computes deterministic session token based on current app_password and secret key."""
    return hashlib.sha256(f"{settings.app_password}:{settings.auth_secret_key}".encode("utf-8")).hexdigest()


def is_authenticated(request: Request) -> bool:
    """Checks whether the request presents a valid authorization credential."""
    expected_token = get_auth_token()

    # 1. Bearer Token in Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if hmac.compare_digest(token, expected_token):
            return True

    # 2. Cookie 'videoscan_auth_token'
    cookie_token = request.cookies.get("videoscan_auth_token")
    if cookie_token and hmac.compare_digest(cookie_token, expected_token):
        return True

    # 3. Direct header 'X-App-Password'
    pwd_header = request.headers.get("x-app-password")
    if pwd_header and hmac.compare_digest(pwd_header, settings.app_password):
        return True

    # 4. Query parameter 'token' (vital for HTML5 <video> elements)
    query_token = request.query_params.get("token")
    if query_token and hmac.compare_digest(query_token, expected_token):
        return True

    # 5. Query parameter 'password'
    query_pwd = request.query_params.get("password")
    if query_pwd and hmac.compare_digest(query_pwd, settings.app_password):
        return True

    return False


def require_auth(request: Request):
    """Enforces authentication dependency for sensitive API routes."""
    if not is_authenticated(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please enter the app password (joanisawful).",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
async def auth_login(payload: LoginPayload, response: Response):
    """Validates the application password and creates an authenticated session."""
    if not hmac.compare_digest(payload.password, settings.app_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password. Access denied.",
        )

    token = get_auth_token()
    # Set HTTP-only session cookie valid for 30 days
    response.set_cookie(
        key="videoscan_auth_token",
        value=token,
        max_age=86400 * 30,
        httponly=True,
        samesite="lax",
    )
    return {
        "success": True,
        "token": token,
        "message": "Authenticated successfully",
    }


@app.get("/api/auth/verify")
async def auth_verify(request: Request):
    """Checks whether the client session is currently authenticated."""
    if is_authenticated(request):
        return {"authenticated": True, "token": get_auth_token()}
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"authenticated": False, "detail": "Not authenticated"},
    )


@app.post("/api/auth/logout")
async def auth_logout(response: Response):
    """Clears the authentication session cookie."""
    response.delete_cookie(key="videoscan_auth_token")
    return {"success": True, "message": "Logged out successfully"}


# ---------------------------------------------------------------------------
# Public Web & Health Check Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root(request: Request):
    """Root endpoint: serves Web UI if HTML is requested, or JSON status."""
    accept = request.headers.get("accept", "").lower()
    if "text/html" in accept:
        return HTMLResponse(content=load_index_html())
    return {
        "service": "jeffsvideoscan",
        "status": "ready",
        "model": settings.gemini_model,
        "chunk_duration": settings.chunk_duration_seconds,
        "gcp_project": settings.gcp_project,
        "gcs_bucket": settings.gcs_bucket,
    }


@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """Direct route for Web Interface."""
    return HTMLResponse(content=load_index_html())


@app.get("/healthz")
@app.get("/livez")
async def health_check():
    """Container health and liveness check."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Protected API Routes (Protected with require_auth)
# ---------------------------------------------------------------------------
@app.get("/api/info", dependencies=[Depends(require_auth)])
async def get_service_info():
    """Returns deployment configuration and runtime parameters."""
    return {
        "service": "jeffsvideoscan",
        "status": "ready",
        "gcp_project": settings.gcp_project,
        "gcp_region": settings.gcp_region,
        "gcs_bucket": settings.gcs_bucket,
        "gemini_model": settings.gemini_model,
        "chunk_duration_seconds": settings.chunk_duration_seconds,
        "max_concurrent_chunks": settings.max_concurrent_chunks,
    }


@app.get("/api/catalog", dependencies=[Depends(require_auth)])
async def get_catalog():
    """Returns metadata extraction catalog definitions and categories."""
    return get_catalog_summary()


@app.get("/api/videos", dependencies=[Depends(require_auth)])
async def list_videos(limit: int = 100):
    """Lists indexed videos from Cloud Firestore."""
    videos = orchestrator.firestore_repo.list_videos(limit=limit)
    for v in videos:
        bucket = v.get("gcs_bucket") or settings.gcs_bucket
        path = v.get("gcs_path") or f"{v.get('video_id')}.mp4"
        v["gcs_uri"] = f"gs://{bucket}/{path}"
    return {"videos": videos}


@app.get("/api/videos/{video_id}", dependencies=[Depends(require_auth)])
async def get_video_detail(video_id: str):
    """Retrieves full Firestore video document and its extracted chunk documents."""
    record = orchestrator.firestore_repo.get_video_record(video_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found in Firestore collection 'videos'.",
        )

    chunks = orchestrator.firestore_repo.get_video_chunks(video_id)
    bucket = record.get("gcs_bucket", settings.gcs_bucket)
    path = record.get("gcs_path", f"{video_id}.mp4")
    config_path = record.get("config_path", f"{video_id}_config.json")

    return {
        "video": record,
        "chunks": chunks,
        "gcs_uri": f"gs://{bucket}/{path}",
        "config_gcs_uri": f"gs://{bucket}/{config_path}",
    }


@app.get("/api/logs", dependencies=[Depends(require_auth)])
async def get_system_log_entries(
    limit: int = 100,
    level: Optional[str] = None,
    search: Optional[str] = None,
):
    """Retrieves system and pipeline logs from the in-memory circular buffer."""
    safe_limit = max(1, min(limit, 500))
    logs = get_system_logs(limit=safe_limit, level=level, search=search)
    return {
        "logs": logs,
        "count": len(logs),
        "limit": safe_limit,
        "level": level,
        "search": search,
    }


@app.get("/api/videos/{video_id}/logs", dependencies=[Depends(require_auth)])
async def get_video_pipeline_logs(video_id: str):
    """Retrieves failure diagnostics, error descriptions, and execution logs for a specific video."""
    record = orchestrator.firestore_repo.get_video_record(video_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video '{video_id}' not found.",
        )
    logs = record.get("logs") or []
    if not logs:
        # Fallback to in-memory system logs matching video_id
        logs = get_system_logs(limit=200, search=video_id)

    return {
        "video_id": video_id,
        "status": record.get("status"),
        "error_message": record.get("error_message"),
        "error_description": record.get("error_description"),
        "logs": logs,
    }


@app.get("/api/videos/{video_id}/video", dependencies=[Depends(require_auth)])
async def stream_video(video_id: str, request: Request):
    """Streams video file from Google Cloud Storage with HTTP 206 Partial Content support."""
    record = orchestrator.firestore_repo.get_video_record(video_id)
    bucket_name = record.get("gcs_bucket", settings.gcs_bucket) if record else settings.gcs_bucket
    object_name = record.get("gcs_path", f"{video_id}.mp4") if record else f"{video_id}.mp4"

    blob = orchestrator.storage_manager.get_blob(bucket_name, object_name)
    if not blob:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file not found at gs://{bucket_name}/{object_name}",
        )

    file_size = blob.size
    range_header = request.headers.get("range")

    if range_header and file_size:
        # Parse standard Range: bytes=start-end
        try:
            byte_range = range_header.replace("bytes=", "").split("-")
            start = int(byte_range[0]) if byte_range[0] else 0
            end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
            end = min(end, file_size - 1)
        except Exception:
            start = 0
            end = file_size - 1

        data = blob.download_as_bytes(start=start, end=end)
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(len(data)),
            "Content-Type": "video/mp4",
        }
        return Response(content=data, status_code=status.HTTP_206_PARTIAL_CONTENT, headers=headers)
    else:
        data = blob.download_as_bytes()
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size or len(data)),
            "Content-Type": "video/mp4",
        }
        return Response(content=data, status_code=status.HTTP_200_OK, headers=headers)


@app.post("/api/upload", dependencies=[Depends(require_auth)])
async def upload_video_and_config(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config_file: Optional[UploadFile] = File(None),
    config_json: Optional[str] = Form(None),
    video_id: Optional[str] = Form(None),
    run_pipeline: bool = Form(True),
):
    """Uploads a video and configuration file to GCS and initiates processing."""
    # 1. Determine sanitized video_id
    if video_id and video_id.strip():
        raw_id = video_id.strip()
    else:
        raw_id = Path(video.filename or "video").stem

    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_id).strip("_").lower()
    if not clean_id:
        clean_id = f"video_{int(time.time())}"

    # 2. Determine target configuration payload
    config_dict: Dict[str, Any] = {}
    if config_file and config_file.filename:
        try:
            config_bytes = await config_file.read()
            config_dict = json.loads(config_bytes.decode("utf-8"))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse uploaded config JSON file: {e}",
            )
    elif config_json:
        try:
            config_dict = json.loads(config_json)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse config_json string: {e}",
            )
    else:
        # Default target metadata
        config_dict = {
            "target_metadata": [
                "chunk_summary",
                "scene_description",
                "detected_objects",
                "overall_sentiment",
            ]
        }

    # Validate configuration against metadata catalog
    try:
        validated_metadata = parse_config_payload(config_dict)
    except Exception as e:
        err_info = describe_error(e, stage="config_validation")
        logger.error(f"Configuration validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration validation error: {e}. {err_info['description']}",
        )

    # 3. Read video contents
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty.",
        )

    # 4. Upload config and video to Google Cloud Storage
    bucket_name = settings.gcs_bucket
    video_object = f"{clean_id}.mp4"
    config_object = f"{clean_id}_config.json"

    # Upload config first
    logger.info(f"Uploading config to gs://{bucket_name}/{config_object}")
    orchestrator.storage_manager.upload_json(
        bucket_name=bucket_name,
        object_name=config_object,
        data={"target_metadata": validated_metadata},
    )

    # Upload video
    logger.info(f"Uploading video ({len(video_bytes)} bytes) to gs://{bucket_name}/{video_object}")
    orchestrator.storage_manager.upload_file(
        bucket_name=bucket_name,
        object_name=video_object,
        data=video_bytes,
        content_type=video.content_type or "video/mp4",
    )

    # Immediately initialize the Firestore document with status "PROCESSING"
    # so that My Library and Metadata Results immediately reflect
    # the new video in PROCESSING status without 404s
    try:
        orchestrator.firestore_repo.init_video_record(
            video_id=clean_id,
            filename=Path(video.filename or video_object).name,
            gcs_bucket=bucket_name,
            gcs_path=video_object,
            config_path=config_object,
            target_metadata=validated_metadata,
            total_chunks=0,
            duration_seconds=0.0,
        )
        logger.info(f"Initialized Firestore record for {clean_id} with status PROCESSING")
    except Exception as e:
        logger.warning(f"Could not immediately initialize Firestore record for {clean_id}: {e}")

    # 5. Initiate pipeline processing if requested
    if run_pipeline:
        logger.info(f"Adding background pipeline execution for video {clean_id}")
        background_tasks.add_task(
            orchestrator.execute_video_pipeline,
            bucket_name=bucket_name,
            video_id=clean_id,
            video_object=video_object,
            config_object=config_object,
            force=True,
        )

    return {
        "status": "PROCESSING",
        "video_id": clean_id,
        "filename": video_object,
        "gcs_bucket": bucket_name,
        "gcs_video_path": video_object,
        "gcs_config_path": config_object,
        "gcs_uri": f"gs://{bucket_name}/{video_object}",
        "target_metadata": validated_metadata,
        "message": f"Successfully uploaded {video_object} and {config_object}. Pipeline execution started.",
    }


@app.post("/api/videos/{video_id}/process", dependencies=[Depends(require_auth)])
async def trigger_reprocessing(video_id: str, background_tasks: BackgroundTasks):
    """Triggers or forces re-processing of a video currently in GCS."""
    record = orchestrator.firestore_repo.get_video_record(video_id)
    bucket = record.get("gcs_bucket", settings.gcs_bucket) if record else settings.gcs_bucket
    video_object = record.get("gcs_path", f"{video_id}.mp4") if record else f"{video_id}.mp4"
    config_object = record.get("config_path", f"{video_id}_config.json") if record else f"{video_id}_config.json"

    if not orchestrator.storage_manager.check_blob_exists(bucket, video_object):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file gs://{bucket}/{video_object} does not exist in GCS.",
        )

    background_tasks.add_task(
        orchestrator.execute_video_pipeline,
        bucket_name=bucket,
        video_id=video_id,
        video_object=video_object,
        config_object=config_object,
        force=True,
    )
    return {"status": "TRIGGERED", "video_id": video_id}


@app.delete("/api/videos/{video_id}", dependencies=[Depends(require_auth)])
async def delete_video(video_id: str):
    """Deletes video document and chunks from Firestore."""
    success = orchestrator.firestore_repo.delete_video_record(video_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete video from Firestore.")
    return {"status": "DELETED", "video_id": video_id}


# ---------------------------------------------------------------------------
# CloudEvent Ingestion (Eventarc Storage Webhooks)
# ---------------------------------------------------------------------------
def extract_storage_event_data(request_body: Dict[str, Any], headers: Dict[str, str]) -> tuple[str, str]:
    """Extracts bucket and object name from binary or structured CloudEvent payloads."""
    # Structured mode: {"data": {"bucket": "...", "name": "..."}}
    if "data" in request_body and isinstance(request_body["data"], dict):
        data = request_body["data"]
        bucket = data.get("bucket")
        name = data.get("name")
        if bucket and name:
            return bucket, name

    # Direct payload or binary mode body: {"bucket": "...", "name": "..."}
    bucket = request_body.get("bucket") or request_body.get("bucket_name")
    name = request_body.get("name") or request_body.get("object_name")
    if bucket and name:
        return bucket, name

    # Check CloudEvent headers if present
    ce_subject = headers.get("ce-subject")
    if ce_subject and "objects/" in ce_subject:
        name = ce_subject.split("objects/", 1)[-1]
        bucket = request_body.get("bucket") or settings.gcs_bucket
        if bucket and name:
            return bucket, name

    raise ValueError(f"Unable to parse GCS bucket and object name from payload: {request_body}")


@app.post("/")
@app.post("/events")
async def handle_cloudevent(request: Request):
    """Handles incoming CloudEvent HTTP POST requests from Eventarc."""
    headers = {k.lower(): v for k, v in request.headers.items()}

    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request JSON: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")

    logger.info(f"Received CloudEvent with headers: {dict(filter(lambda x: x[0].startswith('ce-'), headers.items()))}")

    try:
        bucket, object_name = extract_storage_event_data(body, headers)
    except ValueError as e:
        logger.warning(f"Unprocessable event payload: {e}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "IGNORED", "reason": str(e)},
        )

    try:
        result = orchestrator.process_event(bucket_name=bucket, object_name=object_name)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        err_info = describe_error(e, stage="cloudevent_ingestion", context={"bucket": bucket, "object_name": object_name})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "ERROR",
                "error": str(e),
                "error_description": err_info["description"],
                "logs": err_info.get("traceback", "").splitlines(),
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.port, reload=False)
