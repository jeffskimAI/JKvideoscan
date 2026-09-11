"""Google Cloud Platform authentication helpers."""
import subprocess
from typing import Optional
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as OAuth2Credentials

from src.logger import logger


def get_gcp_credentials() -> Optional[Credentials]:
    """Resolves GCP credentials.
    
    Tries Application Default Credentials first (works on Cloud Run and standard GCP environments).
    If ADC refresh fails (e.g. local dev with expired ADC), falls back to gcloud auth token.
    """
    try:
        creds, _ = default()
        # Verify credentials can refresh or are valid
        try:
            creds.refresh(Request())
            return creds
        except Exception as refresh_err:
            logger.debug(f"Default credentials refresh check: {refresh_err}")
    except Exception as default_err:
        logger.debug(f"Application Default Credentials error: {default_err}")

    # Fallback for local development using gcloud CLI
    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        if token:
            logger.info("Using local gcloud access token for GCP authentication.")
            return OAuth2Credentials(token)
    except Exception as gcloud_err:
        logger.debug(f"gcloud auth print-access-token fallback failed: {gcloud_err}")

    return None
