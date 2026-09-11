"""FastAPI Cloud Run Service and CloudEvent Router."""
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.config import settings
from src.logger import logger
from src.pipeline_orchestrator import PipelineOrchestrator

app = FastAPI(
    title="Jeff's VideoScan - AI Video Metadata Extraction Pipeline",
    version="1.0.0",
    description="Event-driven Cloud Run service for 10s video chunking and Gemini 3.8 metadata extraction",
)

orchestrator = PipelineOrchestrator()


@app.get("/")
async def root():
    """Root info endpoint."""
    return {
        "service": "jeffsvideoscan",
        "status": "ready",
        "model": settings.gemini_model,
        "chunk_duration": settings.chunk_duration_seconds,
    }


@app.get("/healthz")
@app.get("/livez")
async def health_check():
    """Container health and liveness check."""
    return {"status": "healthy"}


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
        # Format: objects/path/to/file.mp4
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
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "ERROR", "error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.port, reload=False)
