"""Unit tests for metadata catalog validation and dynamic schema generation."""
import pytest
from pydantic import ValidationError

from src.catalog import (
    METADATA_CATALOG,
    SUPPORTED_METADATA_KEYS,
    CatalogValidationError,
    parse_config_payload,
    validate_target_metadata,
)
from src.schema_generator import (
    build_dynamic_metadata_model,
    construct_extraction_prompt,
    generate_gemini_response_schema,
)


def test_catalog_contains_all_16_keys():
    """Verify that all 16 specified keys from metadata_catalog.md are present."""
    expected_keys = {
        # Visual & Spatial
        "scene_description",
        "detected_objects",
        "on_screen_text",
        "camera_movement",
        "brand_presence",
        "lighting_and_color",
        # Audio & Speech
        "transcript",
        "speaker_count",
        "ambient_sounds",
        "vocal_emotion",
        # Action & Event
        "key_actions",
        "interactions",
        "event_anomalies",
        # Semantic & Contextual
        "chunk_summary",
        "overall_sentiment",
        "content_categories",
        "safety_flags",
    }
    assert set(METADATA_CATALOG.keys()) == expected_keys
    assert len(expected_keys) == 17  # Wait, let's verify count: 6 visual + 4 audio + 3 action + 4 semantic = 17!
    assert len(SUPPORTED_METADATA_KEYS) == 17


def test_validate_target_metadata_valid():
    """Test valid metadata keys validation."""
    keys = ["scene_description", "detected_objects", "overall_sentiment"]
    validated = validate_target_metadata(keys)
    assert validated == keys


def test_validate_target_metadata_deduplication():
    """Test deduplication preserving order."""
    keys = ["scene_description", "detected_objects", "scene_description"]
    validated = validate_target_metadata(keys)
    assert validated == ["scene_description", "detected_objects"]


def test_validate_target_metadata_invalid_key():
    """Test that invalid keys raise CatalogValidationError."""
    keys = ["scene_description", "invalid_nonexistent_key"]
    with pytest.raises(CatalogValidationError) as exc_info:
        validate_target_metadata(keys)
    assert "Unsupported metadata keys" in str(exc_info.value)


def test_validate_target_metadata_empty():
    """Test that empty list raises CatalogValidationError."""
    with pytest.raises(CatalogValidationError) as exc_info:
        validate_target_metadata([])
    assert "must not be empty" in str(exc_info.value)


def test_parse_config_payload_success():
    """Test parsing a valid config dictionary."""
    payload = {
        "target_metadata": ["chunk_summary", "transcript", "detected_objects", "ambient_sounds"]
    }
    result = parse_config_payload(payload)
    assert result == ["chunk_summary", "transcript", "detected_objects", "ambient_sounds"]


def test_parse_config_payload_missing_key():
    """Test parsing config missing target_metadata."""
    payload = {"some_other_key": 123}
    with pytest.raises(CatalogValidationError):
        parse_config_payload(payload)


def test_parse_config_payload_not_a_dict():
    """Test non-dict payload."""
    with pytest.raises(CatalogValidationError):
        parse_config_payload(["not", "a", "dict"])


def test_build_dynamic_metadata_model():
    """Test dynamic Pydantic model generation."""
    keys = ["scene_description", "speaker_count", "detected_objects"]
    ModelClass = build_dynamic_metadata_model(keys)

    # Valid instantiation
    instance = ModelClass(
        scene_description="A bustling tech conference",
        speaker_count=2,
        detected_objects=["podium", "laptop", "screen"],
    )
    assert instance.scene_description == "A bustling tech conference"
    assert instance.speaker_count == 2
    assert instance.detected_objects == ["podium", "laptop", "screen"]

    # Missing required field should raise ValidationError
    with pytest.raises(ValidationError):
        ModelClass(scene_description="Only description provided")

    # Wrong type should raise ValidationError
    with pytest.raises(ValidationError):
        ModelClass(
            scene_description="Valid",
            speaker_count="not-a-number",
            detected_objects=["item"],
        )


def test_generate_gemini_response_schema():
    """Test generating JSON schema for Gemini response_schema."""
    keys = ["chunk_summary", "overall_sentiment"]
    schema = generate_gemini_response_schema(keys)

    assert "properties" in schema
    assert "chunk_summary" in schema["properties"]
    assert "overall_sentiment" in schema["properties"]
    assert "required" in schema
    assert set(schema["required"]) == {"chunk_summary", "overall_sentiment"}


def test_construct_extraction_prompt():
    """Test prompt constructor includes field instructions and time boundaries."""
    keys = ["transcript", "key_actions"]
    prompt = construct_extraction_prompt(keys, chunk_start_sec=10.0, chunk_end_sec=20.0)

    assert "10.0s to 20.0s" in prompt
    assert "'transcript'" in prompt
    assert "'key_actions'" in prompt
    assert "valid JSON object" in prompt
