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

## 5. Sports & Key Play Metadata
Specialized sports and athletic event intelligence designed to detect granular athletic actions from broadcast video, distinguish routine broadcast events from game-altering key plays, and power member-facing features like **in-stream Key Plays carousels** and **automated recap reels**.

Gemini separates two distinct scoring dimensions for sports events:
1. **Detection Confidence (0.00–1.00)**: The probability that the event occurred and was classified correctly under the canonical sports taxonomy.
2. **Significance / Game-Impact Score (0.00–1.00)**: A composite editorial and match-leverage score evaluating:
   - **Game State & Leverage**: Score delta, game clock, inning/quarter/set/round pressure (e.g., a 90th-minute equalizer ranks much higher than a goal scored while leading 4–0).
   - **Audio & Atmosphere**: Multimodal spikes in crowd cheer, stadium decibels, and commentator excitement.
   - **Visual & OCR Validation**: Scorebug changes, referee whistles/signals, player celebrations, and on-screen graphics.
   - **Event Rarity**: Low-frequency, game-altering actions (e.g., knockdowns, red cards, walk-offs, pick-sixes).

| Metadata Key | Description | Example Output from Gemini |
| :--- | :--- | :--- |
| `sport_type` | Classification of the active sport being broadcast. | `"Soccer"`, `"Tennis"`, `"American Football"`, `"Baseball"`, `"Boxing"` |
| `sports_play_type` | Canonical play/action taxonomy classification distinguishing routine play from key plays. | `"soccer_goal"`, `"tennis_break_point"`, `"football_interception"`, `"baseball_home_run"`, `"boxing_knockdown"` |
| `significance_score` | Composite game-impact and editorial leverage score (0.00–1.00). Routine plays: ~0.10–0.40; significant key plays: ~0.80–1.00. | `0.92` |
| `detection_confidence` | Model confidence probability (0.00–1.00) that the event occurred and was classified correctly. | `0.98` |
| `game_state` | Structured game state context (score delta, period/round, game clock, scorebug OCR). | `{"score": "2-2", "period": "Stoppage Time (90+3')", "time_remaining": "00:45", "scorebug_ocr": "HOME 2 - AWAY 2 (90:15)"}` |
| `significance_factors` | Multimodal explainability breakdown driving the significance score (leverage, audio, visual/OCR, rarity). | `["Late stoppage-time equalizer", "Crowd acoustic peak", "Team dogpile celebration", "OCR scorebug update"]` |
| `recommended_clip_window` | Suggested clip boundaries (seconds) including buildup, peak event, celebration, and suggested duration. | `{"buildup_start_sec": 2.0, "peak_event_sec": 5.5, "celebration_end_sec": 10.0, "recommended_duration_sec": 30.0}` |
| `sports_key_plays` | Complete structured array of detected key plays with taxonomy, confidence, significance, game state, and clip windows. | `[{"play_type": "soccer_goal", "detection_confidence": 0.98, "significance_score": 0.92, "game_state": {"score": "2-2", "period": "90+3'"}, "recommended_clip_window": {"buildup_start_sec": 2.0, "peak_event_sec": 5.5, "celebration_end_sec": 10.0}}]` |

### Key Play Taxonomy & Significance Reference (Top 5 Sports)

| Sport | Routine Detections (Low Significance ~0.10–0.40) | Significant Key Plays (High Significance ~0.80–1.00) | Canonical Play Taxonomy Examples | Recommended Clip Window |
| :--- | :--- | :--- | :--- | :--- |
| **Tennis** | Standard rallies, faults, unforced errors, routine holds | Break point winners, tiebreak mini-breaks, match points, clutch aces | `tennis_winner`, `tennis_break_point`, `tennis_ace`, `tennis_match_point` | ~15–20s (ball toss through baseline rally, shot winner, and celebration) |
| **Soccer** | Back passes, clearances, throw-ins, routine tackles | Stoppage-time goals, penalty saves, red cards, goal-line clearances | `soccer_goal`, `soccer_penalty_kick`, `soccer_shot_on_target`, `soccer_red_card` | ~25–35s (set piece/kick delivery, goal execution, and celebration) |
| **American Football** | 2-yard runs, incomplete screens, punts, routine tackles | Pick-sixes, 4th-quarter go-ahead TDs, red-zone strip sacks, walk-off FGs | `football_interception`, `football_touchdown`, `football_sack`, `football_fumble_recovery` | ~30–40s (pre-snap alignment, pass/run, score/turnover, and referee signal) |
| **Baseball** | Routine balls/strikes, foul tips, routine pop-outs | Walk-off home runs, bases-loaded strikeouts, robbed home runs at the wall | `baseball_home_run`, `baseball_run_scored`, `baseball_strikeout` | ~35–45s (pitch delivery, crack of bat, home run trot, home-plate celebration) |
| **Boxing (Combat Sports)** | Single probing jabs, clinches, blocked punches | Knockdowns, referee 10-counts, fight-ending KOs/TKOs, staggering power flurries | `boxing_knockdown`, `boxing_power_punch`, `boxing_knockout`, `boxing_technical_knockout` | ~25–30s (combination setup, clean connection, canvas fall, and referee count) |

---

## Example Usage

### 1. General Meeting / Dialogue Video (`meeting_clip_config.json`)
```json
{
  "target_metadata": [
    "chunk_summary",
    "transcript",
    "detected_objects",
    "ambient_sounds"
  ]
}
```

### 2. Sports Broadcast Video (`sports_broadcast_config.json`)
```json
{
  "target_metadata": [
    "sport_type",
    "sports_play_type",
    "significance_score",
    "detection_confidence",
    "game_state",
    "significance_factors",
    "recommended_clip_window",
    "sports_key_plays"
  ]
}
```
