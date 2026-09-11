"""Application settings and configuration."""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Cloud Project Configuration
    gcp_project: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("GCP_PROJECT", "jeffsvideoscan-prod"))
    )
    gcp_region: str = Field(default="us-central1")

    # Cloud Storage & Firestore
    gcs_bucket: str = Field(default="jeffsvideoscan-ingest")
    firestore_database: str = Field(default="(default)")
    firestore_collection: str = Field(default="videos")

    # Vertex AI Gemini Configuration
    # spec.md mentions Vertex AI Gemini 3.8
    gemini_model: str = Field(default="gemini-2.5-flash")

    # Video Processing Parameters
    chunk_duration_seconds: int = Field(default=10, description="Strict 10-second chunking")
    max_concurrent_chunks: int = Field(default=4, description="Maximum parallel Gemini inferences")
    temp_dir: str = Field(default="/tmp/videoprocessing")

    # Server Configuration
    port: int = Field(default=8080)
    log_level: str = Field(default="INFO")


settings = Settings()
