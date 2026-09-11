"""Google Cloud Storage manager and file pairing logic."""
import json
from pathlib import Path
from typing import Optional, Tuple
from google.cloud import storage

from src.config import settings
from src.logger import logger


class StorageManager:
    """Manages GCS file downloads and pairing between .mp4 and _config.json."""

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
            self._client = storage.Client(project=settings.gcp_project)
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
