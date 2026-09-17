"""Application settings and configuration."""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_gcp_project() -> str:
    proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if proj:
        return proj
    try:
        import subprocess
        out = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        if out and out != "(unset)":
            return out
    except Exception:
        pass
    return "jeffskim999"


class Settings(BaseSettings):
    """Pipeline runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Cloud Project Configuration
    gcp_project: str = Field(default_factory=_detect_gcp_project)
    gcp_region: str = Field(default="us-central1")

    # Cloud Storage & Firestore
    gcs_bucket: str = Field(default="jeffsvideoscan-ingest")
    service_account_email: str = Field(
        default="jeffsvideoscan-sa@jeffskim999.iam.gserviceaccount.com",
        description="Service account email used for signing GCS V4 URLs",
    )
    firestore_database: str = Field(default="(default)")
    firestore_collection: str = Field(default="videos")

    # Vertex AI Gemini Configuration
    gemini_model: str = Field(default="gemini-3.8-flash", description="Primary Gemini model")
    gemini_location: str = Field(default="global", description="Vertex AI location for Gemini models")
    gemini_fallback_models: list[str] = Field(
        default=[
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.25-flash",
            "gemini-2.5-flash",
        ],
        description="Cascading fallback model chain in descending priority order",
    )

    # Video Processing Parameters
    chunk_duration_seconds: int = Field(default=10, description="Strict 10-second chunking")
    max_concurrent_chunks: int = Field(default=35, description="Maximum parallel Gemini inferences")
    temp_dir: str = Field(default="/tmp/videoprocessing")

    # Authentication
    app_password: str = Field(default="joanisawful", description="Password required to access Web App")
    auth_secret_key: str = Field(default="jeffsvideoscan-secret-salt-2026")

    # Server Configuration
    port: int = Field(default=8080)
    log_level: str = Field(default="INFO")


settings = Settings()
