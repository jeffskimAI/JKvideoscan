"""Google Cloud Storage manager and file pairing logic."""
import json
from pathlib import Path
from typing import Optional, Tuple
from google.cloud import storage

from src.auth_utils import get_gcp_credentials
from src.config import settings
from src.logger import logger


class StorageManager:
    """Manages GCS file downloads, uploads, and pairing between .mp4 and _config.json."""

    def __init__(self, client: Optional[storage.Client] = None):
        """Initializes the StorageManager.

        Args:
            client: Optional preconfigured storage.Client (useful for mocking).
        """
        self._client = client

    @property
    def client(self) -> storage.Client:
        """Lazily initializes and returns the storage client."""
        if self._client is None:
            logger.info(f"Initializing Storage Client (project={settings.gcp_project})")
            creds = get_gcp_credentials()
            self._client = storage.Client(project=settings.gcp_project, credentials=creds)
        return self._client

    @staticmethod
    def derive_pair_paths(object_name: str) -> Tuple[str, str, str]:
        """Derives video_id, video_path, and config_path from an incoming object name.

        Args:
            object_name: GCS object name (e.g. 'folder/my_video.mp4' or 'folder/my_video_config.json').

        Returns:
            Tuple of (video_id, video_object_path, config_object_path).
        """
        if object_name.endswith("_config.json"):
            base = object_name[:-len("_config.json")]
            video_path = f"{base}.mp4"
            config_path = object_name
        elif object_name.endswith(".mp4"):
            base = object_name[:-len(".mp4")]
            video_path = object_name
            config_path = f"{base}_config.json"
        else:
            raise ValueError(f"Object {object_name} is neither a .mp4 video nor a _config.json file.")

        # Extract video_id as normalized basename
        video_id = Path(base).name
        return video_id, video_path, config_path

    def check_blob_exists(self, bucket_name: str, object_name: str) -> bool:
        """Checks if a blob exists in the given GCS bucket."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        return blob.exists()

    def list_blobs(self, bucket_name: str, prefix: Optional[str] = None):
        """Lists blobs in the given GCS bucket."""
        bucket = self.client.bucket(bucket_name)
        return list(bucket.list_blobs(prefix=prefix))

    def download_blob_to_file(self, bucket_name: str, object_name: str, dest_path: Path) -> Path:
        """Downloads a GCS blob to a local file."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        if not blob.exists():
            raise FileNotFoundError(f"Blob gs://{bucket_name}/{object_name} does not exist.")

        logger.info(f"Downloading gs://{bucket_name}/{object_name} to {dest_path}")
        blob.download_to_filename(str(dest_path))
        return dest_path

    def download_json_as_dict(self, bucket_name: str, object_name: str) -> dict:
        """Downloads and parses a JSON file from GCS."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        if not blob.exists():
            raise FileNotFoundError(f"Config blob gs://{bucket_name}/{object_name} does not exist.")

        raw_bytes = blob.download_as_bytes()
        return json.loads(raw_bytes.decode("utf-8"))

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """Uploads bytes directly to a GCS object."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
        logger.info(f"Uploaded {len(data)} bytes to gs://{bucket_name}/{object_name}")
        return f"gs://{bucket_name}/{object_name}"

    def upload_json(self, bucket_name: str, object_name: str, data: dict) -> str:
        """Uploads a dictionary as a JSON file to GCS."""
        raw_json = json.dumps(data, indent=2)
        return self.upload_file(
            bucket_name=bucket_name,
            object_name=object_name,
            data=raw_json.encode("utf-8"),
            content_type="application/json",
        )

    def get_blob(self, bucket_name: str, object_name: str) -> Optional[storage.Blob]:
        """Returns the GCS Blob object if it exists."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        if blob.exists():
            blob.reload()
            return blob
        return None

    def read_blob_range(
        self,
        bucket_name: str,
        object_name: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> bytes:
        """Reads a byte range from a GCS blob."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        return blob.download_as_bytes(start=start, end=end)

    def generate_signed_upload_url(
        self,
        bucket_name: str,
        object_name: str,
        content_type: str = "video/mp4",
        expiration_minutes: int = 60,
    ) -> str:
        """Generates a V4 Signed URL for uploading directly to GCS via HTTP PUT.

        Uses IAM Credentials API (SignBlob) under the hood when running in environments
        without a local private key file (e.g. Google Cloud Run).
        """
        import datetime
        import subprocess
        import google.auth
        from google.auth.transport.requests import Request

        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        # 1. First try credentials attached to client
        creds = getattr(self.client, "_credentials", None)
        token = None
        sa_email = None

        if creds:
            try:
                if not getattr(creds, "valid", False):
                    creds.refresh(Request())
                token = getattr(creds, "token", None)
                sa_email = getattr(creds, "service_account_email", None)
            except Exception as e:
                logger.debug(f"Client credentials refresh for signed URL: {e}")

        # 2. If not found or incomplete, check google.auth.default()
        if not token or not sa_email:
            try:
                default_creds, _ = google.auth.default()
                default_creds.refresh(Request())
                token = token or getattr(default_creds, "token", None)
                sa_email = sa_email or getattr(default_creds, "service_account_email", None)
            except Exception as e:
                logger.debug(f"Default credentials check for signed URL: {e}")

        # 3. Fallback to settings and local gcloud access token
        if not sa_email:
            sa_email = settings.service_account_email
        if not token:
            try:
                token = subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"],
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).decode().strip()
            except Exception as e:
                logger.debug(f"gcloud token fallback for signed URL: {e}")

        logger.info(f"Generating V4 Signed Upload URL for gs://{bucket_name}/{object_name} (sa={sa_email})")

        return blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="PUT",
            content_type=content_type,
            service_account_email=sa_email,
            access_token=token,
        )


