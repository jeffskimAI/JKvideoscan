"""Unit tests for Vertex AI Gemini extractor."""
import json
from unittest.mock import MagicMock
from pathlib import Path
import pytest

from src.gemini_extractor import GeminiExtractionError, GeminiExtractor
from src.video_segmenter import VideoChunk


def test_gemini_extractor_success(tmp_path: Path):
    """Test successful metadata extraction from video chunk."""
    chunk_file = tmp_path / "chunk_0000.mp4"
    chunk_file.write_bytes(b"\x00\x00\x00\x20ftypmp42")

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=chunk_file,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "scene_description": "A sunny garden with flowering trees",
        "detected_objects": ["flower", "tree", "bench"],
        "overall_sentiment": "Positive",
    })
    mock_client.models.generate_content.return_value = mock_response

    extractor = GeminiExtractor(client=mock_client, model_name="gemini-3.8")
    target_keys = ["scene_description", "detected_objects", "overall_sentiment"]

    result = extractor.extract_chunk_metadata(chunk, target_keys)

    assert result["scene_description"] == "A sunny garden with flowering trees"
    assert result["detected_objects"] == ["flower", "tree", "bench"]
    assert result["overall_sentiment"] == "Positive"
    assert mock_client.models.generate_content.called


def test_gemini_extractor_invalid_json(tmp_path: Path):
    """Test handling of invalid JSON response from model."""
    chunk_file = tmp_path / "chunk_0000.mp4"
    chunk_file.write_bytes(b"\x00\x00\x00\x20ftypmp42")

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=chunk_file,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Not a json response!"
    mock_client.models.generate_content.return_value = mock_response

    extractor = GeminiExtractor(client=mock_client)
    with pytest.raises(GeminiExtractionError) as exc_info:
        extractor.extract_chunk_metadata(chunk, ["scene_description"])

    assert "Invalid structured JSON" in str(exc_info.value)


def test_gemini_extractor_empty_response(tmp_path: Path):
    """Test handling of empty text response."""
    chunk_file = tmp_path / "chunk_0000.mp4"
    chunk_file.write_bytes(b"\x00\x00\x00\x20ftypmp42")

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=chunk_file,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""
    mock_client.models.generate_content.return_value = mock_response

    extractor = GeminiExtractor(client=mock_client)
    with pytest.raises(GeminiExtractionError):
        extractor.extract_chunk_metadata(chunk, ["scene_description"])


def test_gemini_extractor_cascading_fallback(tmp_path: Path):
    """Test stepped cascading fallback: 3.8 fails -> 3.7 fails -> 3.6 succeeds."""
    chunk_file = tmp_path / "chunk_0000.mp4"
    chunk_file.write_bytes(b"\x00\x00\x00\x20ftypmp42")

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=chunk_file,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "scene_description": "Extracted via fallback model 3.6",
    })

    def mock_generate_content(model, contents, config):
        if model in ("gemini-3.8-flash", "gemini-3.7-flash"):
            raise RuntimeError(f"Publisher model `{model}` was not found (404 NOT_FOUND)")
        if model == "gemini-3.6-flash":
            return mock_response
        raise RuntimeError(f"Unexpected model `{model}`")

    mock_client.models.generate_content.side_effect = mock_generate_content

    extractor = GeminiExtractor(client=mock_client, model_name="gemini-3.8-flash")
    result = extractor.extract_chunk_metadata(chunk, ["scene_description"])

    assert result["scene_description"] == "Extracted via fallback model 3.6"
    assert extractor.last_used_model == "gemini-3.6-flash"
    assert mock_client.models.generate_content.call_count == 3
    called_models = [call.kwargs["model"] for call in mock_client.models.generate_content.call_args_list]
    assert called_models == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]


def test_gemini_extractor_all_fallbacks_fail(tmp_path: Path):
    """Test that exhausting all candidate fallback models raises GeminiExtractionError."""
    chunk_file = tmp_path / "chunk_0000.mp4"
    chunk_file.write_bytes(b"\x00\x00\x00\x20ftypmp42")

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=chunk_file,
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("404 NOT_FOUND")

    extractor = GeminiExtractor(client=mock_client, model_name="gemini-3.8-flash")
    with pytest.raises(GeminiExtractionError) as exc_info:
        extractor.extract_chunk_metadata(chunk, ["scene_description"])

    assert "All candidate models exhausted" in str(exc_info.value)


def test_gemini_extractor_recovers_from_client_closed(tmp_path: Path):
    """Test that client automatically recovers and retries when transport is closed."""
    chunk_file = tmp_path / "chunk_0000.mp4"
    chunk_file.write_bytes(b"\x00\x00\x00\x20ftypmp42")

    chunk = VideoChunk(
        index=0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        duration_seconds=10.0,
        file_path=chunk_file,
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({"scene_description": "Recovered scene"})

    attempts = 0

    def mock_generate_content(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return mock_response

    mock_client.models.generate_content.side_effect = mock_generate_content

    extractor = GeminiExtractor(client=mock_client, model_name="gemini-3.8-flash")
    result = extractor.extract_chunk_metadata(chunk, ["scene_description"])

    assert result["scene_description"] == "Recovered scene"
    assert attempts == 2


