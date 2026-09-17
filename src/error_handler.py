"""Error categorization, descriptive failure analysis, and diagnostic reporting."""
import json
import traceback
from typing import Any, Dict, Optional

from src.catalog import CatalogValidationError
from src.video_probe import VideoProbeError


def describe_error(
    exception: Any,
    stage: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates a structured, human-readable error description with diagnostics.

    Args:
        exception: The caught Exception or error message.
        stage: Pipeline stage where the failure occurred.
        context: Optional dictionary containing contextual info (video_id, bucket, etc.).

    Returns:
        Dictionary containing category, stage, summary, description, and traceback.
    """
    if isinstance(exception, Exception):
        err_type = type(exception).__name__
        err_msg = str(exception)
        tb = traceback.format_exc()
        if tb.strip() == "NoneType: None":
            tb = f"{err_type}: {err_msg}"
    else:
        err_type = "ProcessingError"
        err_msg = str(exception)
        tb = f"{err_type}: {err_msg}"

    stage_str = stage or "pipeline_execution"
    ctx = context or {}

    category = "General System Failure"
    description = (
        f"An error occurred during stage '{stage_str}': {err_msg}. "
        "Review the execution logs below for detailed stack traces."
    )

    err_msg_lower = err_msg.lower()
    stage_lower = stage_str.lower()

    # 1. Catalog / Configuration Validation errors
    if isinstance(exception, (CatalogValidationError, json.JSONDecodeError)) or "config" in stage_lower:
        category = "Configuration Validation Failure"
        description = (
            "The configuration file is missing, improperly formatted, or contains unsupported metadata keys. "
            "Ensure that 'target_metadata' is a valid JSON array of supported attributes from the "
            "Metadata Extractor Dictionary (e.g. 'scene_description', 'detected_objects', 'overall_sentiment')."
        )
        if isinstance(exception, CatalogValidationError) or "unsupported" in err_msg_lower or "catalog" in err_msg_lower:
            description += f" Specific catalog issue: {err_msg}"

    # 2. Video Probing (ffprobe) errors
    elif isinstance(exception, VideoProbeError) or "probe" in stage_lower:
        category = "Video Probing Failure"
        description = (
            "FFprobe failed to inspect the video stream metadata. This usually indicates an invalid, corrupted, "
            "or empty video file, or an unsupported codec/container. Verify that the video is a valid MP4/H.264 file."
        )
        if err_msg:
            description += f" FFprobe output: {err_msg}"

    # 3. Video Segmentation (ffmpeg) errors
    elif "segment" in stage_lower or "ffmpeg" in err_msg_lower or "chunk" in stage_lower:
        category = "Video Segmentation Failure"
        description = (
            "FFmpeg failed while segmenting the video into strict 10-second chunks. "
            "Possible causes include corrupted video frames, missing audio/video streams, or disk space limits. "
            "Ensure the source MP4 is readable and re-encode if necessary."
        )

    # 4. Gemini Extraction / Vertex AI errors
    elif "gemini" in stage_lower or "extract" in stage_lower or "gemini" in err_type.lower():
        category = "Vertex AI Gemini Inference Failure"
        if "429" in err_msg or "resource_exhausted" in err_msg_lower:
            description = (
                "Vertex AI Gemini API quota exceeded (HTTP 429 / RESOURCE_EXHAUSTED). "
                "The pipeline retried with exponential backoff, but requests were throttled. "
                "Consider increasing Vertex AI Gemini API quota or retrying after a short wait."
            )
        elif "403" in err_msg or "permission_denied" in err_msg_lower:
            description = (
                "Vertex AI Gemini API access denied (HTTP 403). "
                "Ensure that the Cloud Run service account has the 'Vertex AI User' (roles/aiplatform.user) role."
            )
        else:
            description = (
                "Vertex AI Gemini failed to extract structured metadata for one or more video chunks. "
                "Verify model availability, prompt schema compliance, and Google GenAI API credentials."
            )

    # 5. Cloud Storage errors
    elif "storage" in stage_lower or "gcs" in stage_lower or "notfound" in err_type.lower() or "blob" in stage_lower:
        category = "Cloud Storage Access Failure"
        description = (
            "Failed to access or transfer files with Google Cloud Storage. "
            "Verify that the GCS bucket exists, objects are accessible, and appropriate read/write IAM permissions are granted."
        )

    # 6. Firestore errors
    elif "firestore" in stage_lower:
        category = "Firestore Persistence Failure"
        description = (
            "Cloud Firestore encountered an error persisting video document or chunk subcollection records. "
            "Verify Firestore database connectivity and IAM permissions."
        )

    # 7. Network / Connection errors
    elif any(term in err_msg_lower for term in ["connection", "timeout", "timed out", "econnreset"]):
        category = "Network Connection Timeout"
        description = (
            "A network timeout or connection reset occurred while communicating with external GCP services. "
            "Check network stability and retry the operation."
        )

    return {
        "category": category,
        "stage": stage_str,
        "summary": err_msg or err_type,
        "description": description,
        "error_type": err_type,
        "traceback": tb,
        "context": ctx,
    }
