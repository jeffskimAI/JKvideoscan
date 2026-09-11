# Supported Metadata Extraction Catalog

This document outlines the types of metadata that can be extracted from 10-second video chunks using Vertex AI Gemini. 

When uploading a video to the GCS bucket, you can include any of these keys in the `target_metadata` array of your `<video_filename>_config.json` file.

## 1. Visual & Spatial Metadata
Information extracted purely from the visual frames of the video chunk.

| Metadata Key | Description | Example Output from Gemini |
| :--- | :--- | :--- |
| `scene_description` | A detailed narrative of the visual setting and environment. | "A crowded outdoor cafe during daytime with overcast lighting." |
| `detected_objects` | A list of distinct physical items present in the frame. | `["coffee cup", "laptop", "yellow umbrella", "bicycle"]` |
| `on_screen_text` | Text recognized within the video frames (OCR). | "CAFE OPEN 24/7", "SALE" |
| `camera_movement` | Description of how the camera is behaving. | "Static shot", "Slow pan to the right", "Handheld shaky" |
| `brand_presence` | Any recognizable logos or brand imagery visible. | `["Starbucks logo on cup", "Apple logo on laptop"]` |
| `lighting_and_color`| The mood, color palette, and lighting setup. | "High contrast, neon cyberpunk colors, low light" |

## 2. Audio & Speech Metadata
Information extracted from the audio track of the 10-second chunk.

| Metadata Key | Description | Example Output from Gemini |
| :--- | :--- | :--- |
| `transcript` | Word-for-word transcription of spoken dialogue. | "So if we look at the Q3 numbers..." |
| `speaker_count` | The number of distinct voices heard in the chunk. | `2` |
| `ambient_sounds` | Background noises and sound effects. | `["traffic noise", "typing on keyboard", "dog barking"]` |
| `vocal_emotion` | The detected tone or emotion of the speaker(s). | "Urgent", "Calm and professional", "Frustrated" |

## 3. Action & Event Metadata
Dynamic events occurring within the 10-second timeframe.

| Metadata Key | Description | Example Output from Gemini |
| :--- | :--- | :--- |
| `key_actions` | The primary activities being performed by subjects. | `["person typing on laptop", "car drives past in background"]` |
| `interactions` | How subjects are interacting with objects or each other. | "Person A hands a coffee cup to Person B" |
| `event_anomalies` | Any sudden or unexpected changes in the scene. | "Screen flickers", "Loud crash off-camera" |

## 4. Semantic & Contextual Metadata
High-level understanding and synthesis of the chunk.

| Metadata Key | Description | Example Output from Gemini |
| :--- | :--- | :--- |
| `chunk_summary` | A concise 1-2 sentence summary of the 10 seconds. | "Two colleagues discuss quarterly metrics while sitting at a cafe." |
| `overall_sentiment` | The general vibe or emotional weight of the clip. | "Positive", "Neutral", "Tense" |
| `content_categories`| High-level tags classifying the content. | `["Business", "Technology", "Outdoor"]` |
| `safety_flags` | Detection of any explicit, violent, or unsafe content. | `["None detected"]` or `["Mild profanity"]` |

---

## Example Usage

**Input Configuration (`meeting_clip_config.json`):**
```json
{
  "target_metadata": [
    "chunk_summary",
    "transcript",
    "detected_objects",
    "ambient_sounds"
  ]
}
