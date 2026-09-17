"""Supported Metadata Extraction Catalog and Validation.

Implements the specification defined in metadata_catalog.md.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Type


@dataclass(frozen=True)
class MetadataFieldDefinition:
    """Definition of a metadata extraction field."""
    key: str
    category: str
    python_type: Type
    description: str
    example: Any


# Catalog of 16 supported extraction fields across 4 categories
METADATA_CATALOG: Dict[str, MetadataFieldDefinition] = {
    # 1. Visual & Spatial Metadata
    "scene_description": MetadataFieldDefinition(
        key="scene_description",
        category="Visual & Spatial Metadata",
        python_type=str,
        description="A detailed narrative of the visual setting and environment.",
        example="A crowded outdoor cafe during daytime with overcast lighting.",
    ),
    "detected_objects": MetadataFieldDefinition(
        key="detected_objects",
        category="Visual & Spatial Metadata",
        python_type=List[str],
        description="A list of distinct physical items present in the frame.",
        example=["coffee cup", "laptop", "yellow umbrella", "bicycle"],
    ),
    "on_screen_text": MetadataFieldDefinition(
        key="on_screen_text",
        category="Visual & Spatial Metadata",
        python_type=List[str],
        description="Text recognized within the video frames (OCR).",
        example=["CAFE OPEN 24/7", "SALE"],
    ),
    "camera_movement": MetadataFieldDefinition(
        key="camera_movement",
        category="Visual & Spatial Metadata",
        python_type=str,
        description="Description of how the camera is behaving (e.g. Static shot, Slow pan to the right, Handheld shaky).",
        example="Slow pan to the right",
    ),
    "brand_presence": MetadataFieldDefinition(
        key="brand_presence",
        category="Visual & Spatial Metadata",
        python_type=List[str],
        description="Any recognizable logos or brand imagery visible.",
        example=["Starbucks logo on cup", "Apple logo on laptop"],
    ),
    "lighting_and_color": MetadataFieldDefinition(
        key="lighting_and_color",
        category="Visual & Spatial Metadata",
        python_type=str,
        description="The mood, color palette, and lighting setup.",
        example="High contrast, neon cyberpunk colors, low light",
    ),

    # 2. Audio & Speech Metadata
    "transcript": MetadataFieldDefinition(
        key="transcript",
        category="Audio & Speech Metadata",
        python_type=str,
        description="Word-for-word transcription of spoken dialogue.",
        example="So if we look at the Q3 numbers...",
    ),
    "speaker_count": MetadataFieldDefinition(
        key="speaker_count",
        category="Audio & Speech Metadata",
        python_type=int,
        description="The number of distinct voices heard in the chunk.",
        example=2,
    ),
    "ambient_sounds": MetadataFieldDefinition(
        key="ambient_sounds",
        category="Audio & Speech Metadata",
        python_type=List[str],
        description="Background noises and sound effects.",
        example=["traffic noise", "typing on keyboard", "dog barking"],
    ),
    "vocal_emotion": MetadataFieldDefinition(
        key="vocal_emotion",
        category="Audio & Speech Metadata",
        python_type=str,
        description="The detected tone or emotion of the speaker(s).",
        example="Calm and professional",
    ),

    # 3. Action & Event Metadata
    "key_actions": MetadataFieldDefinition(
        key="key_actions",
        category="Action & Event Metadata",
        python_type=List[str],
        description="The primary activities being performed by subjects.",
        example=["person typing on laptop", "car drives past in background"],
    ),
    "interactions": MetadataFieldDefinition(
        key="interactions",
        category="Action & Event Metadata",
        python_type=str,
        description="How subjects are interacting with objects or each other.",
        example="Person A hands a coffee cup to Person B",
    ),
    "event_anomalies": MetadataFieldDefinition(
        key="event_anomalies",
        category="Action & Event Metadata",
        python_type=List[str],
        description="Any sudden or unexpected changes in the scene.",
        example=["Screen flickers", "Loud crash off-camera"],
    ),

    # 4. Semantic & Contextual Metadata
    "chunk_summary": MetadataFieldDefinition(
        key="chunk_summary",
        category="Semantic & Contextual Metadata",
        python_type=str,
        description="A concise 1-2 sentence summary of the 10 seconds.",
        example="Two colleagues discuss quarterly metrics while sitting at a cafe.",
    ),
    "overall_sentiment": MetadataFieldDefinition(
        key="overall_sentiment",
        category="Semantic & Contextual Metadata",
        python_type=str,
        description="The general vibe or emotional weight of the clip.",
        example="Positive",
    ),
    "content_categories": MetadataFieldDefinition(
        key="content_categories",
        category="Semantic & Contextual Metadata",
        python_type=List[str],
        description="High-level tags classifying the content.",
        example=["Business", "Technology", "Outdoor"],
    ),
    "safety_flags": MetadataFieldDefinition(
        key="safety_flags",
        category="Semantic & Contextual Metadata",
        python_type=List[str],
        description="Detection of any explicit, violent, or unsafe content.",
        example=["None detected"],
    ),

    # 5. Sports & Key Play Metadata
    "sport_type": MetadataFieldDefinition(
        key="sport_type",
        category="Sports & Key Play Metadata",
        python_type=str,
        description="Classification of the active sport being broadcast (e.g. Soccer, Tennis, American Football, Baseball, Boxing).",
        example="Soccer",
    ),
    "sports_play_type": MetadataFieldDefinition(
        key="sports_play_type",
        category="Sports & Key Play Metadata",
        python_type=str,
        description="Canonical play/action taxonomy classification distinguishing routine play from key plays (e.g. soccer_goal, tennis_break_point, football_interception, baseball_home_run, boxing_knockdown).",
        example="soccer_goal",
    ),
    "significance_score": MetadataFieldDefinition(
        key="significance_score",
        category="Sports & Key Play Metadata",
        python_type=float,
        description="Composite game-impact and editorial leverage score (0.00-1.00) distinguishing routine actions (~0.10-0.40) from significant moments (~0.80-1.00).",
        example=0.92,
    ),
    "detection_confidence": MetadataFieldDefinition(
        key="detection_confidence",
        category="Sports & Key Play Metadata",
        python_type=float,
        description="Model confidence probability (0.00-1.00) that the sports event occurred and matches canonical taxonomy.",
        example=0.98,
    ),
    "game_state": MetadataFieldDefinition(
        key="game_state",
        category="Sports & Key Play Metadata",
        python_type=Dict[str, Any],
        description="Structured game state context (score delta, period/round, game clock, scorebug OCR).",
        example={"score": "2-2", "period": "Stoppage Time (90+3')", "time_remaining": "00:45", "scorebug_ocr": "HOME 2 - AWAY 2 (90:15)"},
    ),
    "significance_factors": MetadataFieldDefinition(
        key="significance_factors",
        category="Sports & Key Play Metadata",
        python_type=List[str],
        description="Multimodal explainability breakdown driving the significance score (game leverage, audio spikes/crowd noise, visual/referee signals, event rarity).",
        example=["Late stoppage-time equalizer", "Crowd acoustic peak", "Team dogpile celebration", "OCR scorebug update"],
    ),
    "recommended_clip_window": MetadataFieldDefinition(
        key="recommended_clip_window",
        category="Sports & Key Play Metadata",
        python_type=Dict[str, Any],
        description="Suggested clip boundaries (seconds) including buildup, peak event, celebration, and suggested duration.",
        example={"buildup_start_sec": 2.0, "peak_event_sec": 5.5, "celebration_end_sec": 10.0, "recommended_duration_sec": 30.0},
    ),
    "sports_key_plays": MetadataFieldDefinition(
        key="sports_key_plays",
        category="Sports & Key Play Metadata",
        python_type=List[Dict[str, Any]],
        description="Complete structured array of detected key plays with taxonomy, confidence, significance, game state, and clip windows for in-stream carousels and recap reels.",
        example=[{
            "play_type": "soccer_goal",
            "detection_confidence": 0.98,
            "significance_score": 0.92,
            "game_state": {"score": "2-2", "period": "90+3'"},
            "recommended_clip_window": {"buildup_start_sec": 2.0, "peak_event_sec": 5.5, "celebration_end_sec": 10.0},
        }],
    ),
}

SUPPORTED_METADATA_KEYS = set(METADATA_CATALOG.keys())


class CatalogValidationError(ValueError):
    """Raised when target_metadata contains unsupported or invalid keys."""
    pass


def validate_target_metadata(target_keys: List[str]) -> List[str]:
    """Validates that all requested keys are supported in the metadata catalog.

    Args:
        target_keys: List of target metadata key names.

    Returns:
        List of validated target keys (deduplicated preserving order).

    Raises:
        CatalogValidationError: If any key is not in the catalog or list is empty.
    """
    if not target_keys:
        raise CatalogValidationError("target_metadata must not be empty.")

    # Deduplicate while preserving order
    deduped_keys: List[str] = []
    seen = set()
    for key in target_keys:
        if not isinstance(key, str):
            raise CatalogValidationError(f"Invalid key type: {key!r}. Must be string.")
        stripped_key = key.strip()
        if not stripped_key:
            raise CatalogValidationError("Metadata key cannot be blank.")
        if stripped_key not in seen:
            seen.add(stripped_key)
            deduped_keys.append(stripped_key)

    invalid_keys = [k for k in deduped_keys if k not in SUPPORTED_METADATA_KEYS]
    if invalid_keys:
        raise CatalogValidationError(
            f"Unsupported metadata keys requested: {invalid_keys}. "
            f"Supported keys are: {sorted(list(SUPPORTED_METADATA_KEYS))}"
        )

    return deduped_keys


def parse_config_payload(payload: dict) -> List[str]:
    """Parses and validates a `<video_filename>_config.json` payload.

    Args:
        payload: Dict loaded from JSON config.

    Returns:
        List of validated metadata keys.

    Raises:
        CatalogValidationError: If the payload structure is invalid.
    """
    if not isinstance(payload, dict):
        raise CatalogValidationError("Configuration root must be a JSON object.")

    if "target_metadata" not in payload:
        raise CatalogValidationError("Configuration missing required field: 'target_metadata'.")

    target_metadata = payload["target_metadata"]
    if not isinstance(target_metadata, list):
        raise CatalogValidationError("'target_metadata' must be an array of strings.")

    return validate_target_metadata(target_metadata)


def get_catalog_summary() -> Dict[str, Any]:
    """Returns the catalog grouped by category with JSON-serializable types."""
    categories: Dict[str, List[Dict[str, Any]]] = {}
    for field in METADATA_CATALOG.values():
        if field.category not in categories:
            categories[field.category] = []
        type_str = getattr(field.python_type, "__name__", str(field.python_type))
        categories[field.category].append({
            "key": field.key,
            "category": field.category,
            "type": type_str,
            "description": field.description,
            "example": field.example,
        })
    return {
        "categories": categories,
        "all_keys": sorted(list(SUPPORTED_METADATA_KEYS)),
        "total_fields": len(METADATA_CATALOG),
    }

