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

    extractor = GeminiExtractor(client=mock_client, model_name="gemini-2.5-flash")
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
