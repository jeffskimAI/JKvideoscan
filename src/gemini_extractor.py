"""Vertex AI Gemini 3.8 Inference Client with Dynamic Structured Output."""
import json
import re
import threading
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
        self.last_used_model: Optional[str] = None
        self._custom_client = client
        self._thread_local = threading.local()

    @property
    def client(self) -> genai.Client:
        """Lazily initializes and returns the GenAI client configured for Vertex AI per thread."""
        if self._custom_client is not None:
            return self._custom_client
        current_client = getattr(self._thread_local, "client", None)
        if current_client is None:
            logger.debug(
                f"Initializing Vertex AI GenAI Client for thread {threading.get_ident()} "
                f"(project={settings.gcp_project}, location={settings.gemini_location})"
            )
            creds = get_gcp_credentials()
            current_client = genai.Client(
                vertexai=True,
                project=settings.gcp_project,
                location=settings.gemini_location,
                credentials=creds,
            )
            self._thread_local.client = current_client
        return current_client

    def reset_client(self):
        """Resets the thread-local client if a transport error or closed connection occurs."""
        if hasattr(self._thread_local, "client"):
            self._thread_local.client = None

    def _get_candidate_models(self) -> List[str]:
        """Returns the ordered list of models to try in sequence."""
        fallback_list = list(settings.gemini_fallback_models)
        if self.model_name in fallback_list:
            start_idx = fallback_list.index(self.model_name)
            return fallback_list[start_idx:]
        elif self.model_name:
            return [self.model_name] + [m for m in fallback_list if m != self.model_name]
        return fallback_list

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, RuntimeError)),
    )
    def _call_model(self, contents: list, config: types.GenerateContentConfig) -> str:
        """Invokes Gemini models with cascading fallback from latest to earlier models in step."""
        candidate_models = self._get_candidate_models()
        last_error = None

        for idx, model in enumerate(candidate_models):
            try:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                except Exception as inner_e:
                    inner_str = str(inner_e).lower()
                    if "client has been closed" in inner_str or "connection closed" in inner_str or ("closed" in inner_str and "request" in inner_str):
                        logger.warning(
                            f"GenAI client transport closed ({inner_e}), recreating thread-local client and retrying model '{model}'..."
                        )
                        self.reset_client()
                        response = self.client.models.generate_content(
                            model=model,
                            contents=contents,
                            config=config,
                        )
                    else:
                        raise inner_e

                if not response.text:
                    if idx < len(candidate_models) - 1:
                        next_model = candidate_models[idx + 1]
                        logger.warning(
                            f"Model '{model}' returned empty response text. Stepping down to fallback model '{next_model}'..."
                        )
                        continue
                    else:
                        raise GeminiExtractionError(f"All candidate models returned empty response text (last model '{model}').")

                self.last_used_model = model
                if idx > 0:
                    logger.info(
                        f"Successfully extracted metadata using fallback model '{model}' "
                        f"(attempted {idx + 1}/{len(candidate_models)} in chain)."
                    )
                return response.text
            except GeminiExtractionError:
                raise
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "client has been closed" in err_str or "connection closed" in err_str:
                    logger.warning(f"Thread-local client closed, resetting client: {e}")
                    self.reset_client()
                is_fallback_candidate = (
                    "not found" in err_str
                    or "404" in err_str
                    or "quota" in err_str
                    or "429" in err_str
                    or "503" in err_str
                    or "unavailable" in err_str
                    or "resource_exhausted" in err_str
                    or "empty response text" in err_str
                )
                if is_fallback_candidate:
                    if idx < len(candidate_models) - 1:
                        next_model = candidate_models[idx + 1]
                        logger.warning(
                            f"Model '{model}' failed ({e}). Stepping down to fallback model '{next_model}'..."
                        )
                        continue
                    else:
                        logger.error(f"All candidate models exhausted. Final model '{model}' failed: {e}")
                        raise GeminiExtractionError(f"All candidate models exhausted ({len(candidate_models)} models tried). Last error: {e}") from e
                else:
                    logger.error(f"Error during Gemini inference with model '{model}': {e}")
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
        except GeminiExtractionError:
            raise
        except Exception as e:
            logger.error(f"Error during Gemini inference for chunk {chunk.index}: {e}")
            raise GeminiExtractionError(f"Inference failed for chunk {chunk.index}: {e}") from e

        # Clean markdown fences if any
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.IGNORECASE)
            clean_text = re.sub(r"\s*```$", "", clean_text).strip()

        # Validate structured JSON output against dynamic Pydantic model
        try:
            parsed_data = json.loads(clean_text)
            validated_model = dynamic_model.model_validate(parsed_data)
            output_dict = validated_model.model_dump()
        except Exception as e:
            # If standard JSON parsing failed due to minor surrounding text, attempt regex block extraction
            json_match = re.search(r"(\{.*\})", clean_text, flags=re.DOTALL)
            if json_match:
                try:
                    parsed_data = json.loads(json_match.group(1))
                    validated_model = dynamic_model.model_validate(parsed_data)
                    return validated_model.model_dump()
                except Exception:
                    pass
            logger.error(f"Failed to validate JSON response for chunk {chunk.index}: {raw_text[:300]}")
            raise GeminiExtractionError(
                f"Invalid structured JSON returned for chunk {chunk.index}: {e}"
            ) from e

        return output_dict
