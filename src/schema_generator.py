"""Dynamic Schema Generator for Vertex AI Gemini 3.8.

Generates dynamic Pydantic models and OpenAPI JSON schemas based on
the user-requested target metadata fields.
"""
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field, create_model

from src.catalog import METADATA_CATALOG, validate_target_metadata


def build_dynamic_metadata_model(target_keys: List[str]) -> Type[BaseModel]:
    """Dynamically creates a Pydantic BaseModel class containing only the requested fields.

    Args:
        target_keys: List of metadata keys from the catalog.

    Returns:
        A Pydantic BaseModel subclass.
    """
    validated_keys = validate_target_metadata(target_keys)

    field_definitions: Dict[str, Any] = {}
    for key in validated_keys:
        meta_def = METADATA_CATALOG[key]
        # Required field (...) with description
        field_definitions[key] = (
            meta_def.python_type,
            Field(
                ...,
                description=f"{meta_def.description} (Category: {meta_def.category})",
            ),
        )

    model_name = "ChunkMetadataOutput"
    dynamic_model = create_model(model_name, **field_definitions)
    return dynamic_model


def generate_gemini_response_schema(target_keys: List[str]) -> Dict[str, Any]:
    """Generates an OpenAPI-compliant JSON schema suitable for Gemini response_schema.

    Args:
        target_keys: List of metadata keys.

    Returns:
        JSON Schema dictionary.
    """
    model = build_dynamic_metadata_model(target_keys)
    schema = model.model_json_schema()
    # Clean up schema for Vertex AI / Gemini API
    # Gemini supports standard JSON Schema subsets (type, properties, required, items, description)
    return schema


def construct_extraction_prompt(target_keys: List[str], chunk_start_sec: float, chunk_end_sec: float) -> str:
    """Constructs a detailed system/user instruction prompt for Gemini 3.8.

    Args:
        target_keys: List of metadata keys.
        chunk_start_sec: Segment start timestamp.
        chunk_end_sec: Segment end timestamp.

    Returns:
        Instruction string for the multimodal prompt.
    """
    validated_keys = validate_target_metadata(target_keys)

    field_instructions = []
    for key in validated_keys:
        item = METADATA_CATALOG[key]
        field_instructions.append(f"- '{key}' ({item.category}): {item.description} Example: {item.example}")

    prompt = f"""You are an expert video analysis AI. Analyze this exact 10-second video chunk (timestamps {chunk_start_sec:.1f}s to {chunk_end_sec:.1f}s).
Extract only the following requested metadata fields strictly according to their definitions:

{chr(10).join(field_instructions)}

Guidelines:
1. Base your extraction exclusively on visual and auditory evidence present in this specific video chunk.
2. Return a valid JSON object matching the requested schema.
3. For array fields, provide an empty list if no items are detected.
4. Do not include markdown formatting or commentary outside the JSON object.
"""
    return prompt.strip()
