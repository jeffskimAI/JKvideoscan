"""Vertex AI Gemini 3.8 Inference Client with Dynamic Structured Output."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from google import genai
from google.genai import types

from src.auth_utils import get_gcp_credentials
from src.catalog import validate_target_metadata
from src.config import settings
from src.logger import logger
from src.schema_generator import (
    build_dynamic_metadata_model,
    construct_extraction_prompt,
)
from src.video_segmenter import VideoChunk


class GeminiExtractionError(RuntimeError):
    """Raised when metadata extraction via Gemini fails."""
    pass


class GeminiExtractor:
    """Client for extracting multimodal metadata from video chunks using Vertex AI Gemini."""

    def __init__(self, client: Optional[genai.Client] = None, model_name: Optional[str] = None):
        """Initializes the GeminiExtractor.

        Args:
            client: Optional preconfigured genai.Client (useful for mocking).
            model_name: Optional model name override.
        """
        self.model_name = model_name or settings.gemini_model
        self._client = client

    @property
    def client(self) -> genai.Client:
        """Lazily initializes and returns the GenAI client configured for Vertex AI."""
        if self._client is None:
            logger.info(
                f"Initializing Vertex AI GenAI Client (project={settings.gcp_project}, "
                f"region={settings.gcp_region})"
            )
            creds = get_gcp_credentials()
            self._client = genai.Client(
                vertexai=True,
                project=settings.gcp_project,
                location=settings.gcp_region,
                credentials=creds,
            )
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def _call_model(self, contents: list, config: types.GenerateContentConfig) -> str:
        """Invokes the model with retry logic and graceful fallback if the model is not found in Vertex AI."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            if not response.text:
                raise GeminiExtractionError("Gemini returned empty response text.")
            return response.text
        except Exception as e:
            if ("not found" in str(e).lower() or "404" in str(e)) and self.model_name != "gemini-2.5-flash":
                logger.warning(
                    f"Configured model '{self.model_name}' not found on Vertex AI. "
                    f"Falling back to 'gemini-2.5-flash' to complete extraction: {e}"
                )
                fallback_response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config,
                )
                if not fallback_response.text:
                    raise GeminiExtractionError("Gemini fallback returned empty response text.")
                return fallback_response.text
            raise e

    def extract_chunk_metadata(
        self,
        chunk: VideoChunk,
        target_keys: List[str],
        video_bytes: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Extracts requested metadata from a 10-second video chunk.

        Args:
            chunk: The VideoChunk object.
            target_keys: List of metadata keys to extract.
            video_bytes: Optional raw bytes of chunk; if None, reads from chunk.file_path.

        Returns:
            Dictionary mapping requested metadata keys to extracted values.

        Raises:
            GeminiExtractionError: If inference or JSON parsing fails.
        """
        validated_keys = validate_target_metadata(target_keys)
        dynamic_model = build_dynamic_metadata_model(validated_keys)
        prompt = construct_extraction_prompt(
            target_keys=validated_keys,
            chunk_start_sec=chunk.start_time_seconds,
            chunk_end_sec=chunk.end_time_seconds,
        )

        if video_bytes is None:
            chunk_path = Path(chunk.file_path)
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
            video_bytes = chunk_path.read_bytes()

        # Build multimodal payload
        video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        contents = [video_part, prompt]

        # Configure structured JSON output schema
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=dynamic_model,
        )

        logger.info(
            f"Sending chunk {chunk.index} [{chunk.start_time_seconds}s - {chunk.end_time_seconds}s] "
            f"to {self.model_name} for metadata extraction ({len(validated_keys)} fields)..."
        )

        try:
            raw_text = self._call_model(contents, config)
        except Exception as e:
            logger.error(f"Error during Gemini inference for chunk {chunk.index}: {e}")
            raise GeminiExtractionError(f"Inference failed for chunk {chunk.index}: {e}") from e

        # Validate structured JSON output against dynamic Pydantic model
        try:
            parsed_data = json.loads(raw_text)
            validated_model = dynamic_model.model_validate(parsed_data)
            output_dict = validated_model.model_dump()
        except Exception as e:
            logger.error(f"Failed to validate JSON response for chunk {chunk.index}: {raw_text}")
            raise GeminiExtractionError(
                f"Invalid structured JSON returned for chunk {chunk.index}: {e}"
            ) from e

        return output_dict
