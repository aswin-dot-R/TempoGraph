import json
import re
import logging
from typing import Optional
from src.models import (
    AnalysisResult,
    Entity,
    VisualEvent,
    AudioEvent,
    MultimodalCorrelation,
    BehaviorType,
    SoundType,
)


class JSONParser:
    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)

    def parse(self, raw_text: str) -> AnalysisResult:
        """
        Parse VLM output text into AnalysisResult.
        Tries multiple strategies in order.
        """
        # Strategy 1: Extract JSON from markdown code blocks
        json_str = self._extract_json_string(raw_text)
        if json_str is None:
            self.logger.warning("No JSON code block found, trying raw extraction")
            json_str = self._find_outermost_json(raw_text)

        if json_str is None:
            self.logger.error("Could not extract JSON from text")
            return AnalysisResult(summary="Unable to extract JSON")

        # Fix common issues
        fixed_json_str = self._fix_common_issues(json_str)
        self.logger.debug(f"Fixed JSON string: {fixed_json_str}")

        # Parse and validate
        try:
            return self._validate_and_build(fixed_json_str)
        except Exception as e:
            self.logger.warning(f"Failed to validate JSON after fixing: {e}")
            return AnalysisResult(summary="JSON parsing failed")

    def _extract_json_string(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown blocks."""
        # Try ```json ... ``` first
        json_block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if json_block:
            return json_block.group(1)

        # Try triple backticks without language specifier
        json_block = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if json_block:
            return json_block.group(1)

        # Try finding the outermost { ... } that looks like JSON
        # This is a fallback that may capture surrounding text
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            # Extract until matching closing brace
            brace_count = 1
            start = json_match.start()
            current = start + 1
            while current < len(text) and brace_count > 0:
                if text[current] == "{":
                    brace_count += 1
                elif text[current] == "}":
                    brace_count -= 1
                current += 1
            if brace_count == 0:
                return text[start:current]

        return None

    def _find_outermost_json(self, text: str) -> Optional[str]:
        """Find JSON by looking for outermost { ... } structure."""
        # Find first {
        start = text.find("{")
        if start == -1:
            return None

        # Find matching }
        brace_count = 1
        current = start + 1
        while current < len(text) and brace_count > 0:
            if text[current] == "{":
                brace_count += 1
            elif text[current] == "}":
                brace_count -= 1
            current += 1

        if brace_count == 0:
            return text[start:current]

        return None

    def _fix_common_issues(self, json_str: str) -> str:
        """Fix common VLM JSON mistakes."""
        # Remove trailing commas before } or ]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        # Replace single quotes with double quotes (but not apostrophes in text)
        # This is risky but VLM outputs often use single quotes for strings
        json_str = json_str.replace("'", '"')

        # Remove control characters (ASCII < 32 except \\n, \\r, \\t)
        json_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", json_str)

        return json_str

    def _validate_and_build(self, json_str: str) -> AnalysisResult:
        """Build AnalysisResult from parsed dict with validation."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            return AnalysisResult(summary=f"JSON parse error: {e}")

        # Create entities
        entities = []
        for entity_data in data.get("entities", []):
            if (
                "id" in entity_data
                and "type" in entity_data
                and "description" in entity_data
            ):
                entity = Entity(
                    id=entity_data["id"],
                    type=entity_data["type"],
                    description=entity_data["description"],
                    first_seen=entity_data.get("first_seen", "00:00"),
                    last_seen=entity_data.get("last_seen", "00:00"),
                )
                entities.append(entity)

        # Create visual events
        visual_events = []
        for visual_data in data.get("visual_events", []):
            if (
                "type" in visual_data
                and "entities" in visual_data
                and "start_time" in visual_data
                and "end_time" in visual_data
            ):
                try:
                    behavior_type = BehaviorType(visual_data["type"])
                except ValueError:
                    behavior_type = BehaviorType.OTHER
                visual_event = VisualEvent(
                    type=behavior_type,
                    entities=visual_data["entities"],
                    start_time=self._normalize_timestamp(visual_data["start_time"]),
                    end_time=self._normalize_timestamp(visual_data["end_time"]),
                    description=visual_data.get("description", ""),
                    confidence=visual_data.get("confidence", 0.5),
                )
                visual_events.append(visual_event)

        # Create audio events
        audio_events = []
        for audio_data in data.get("audio_events", []):
            if (
                "type" in audio_data
                and "start_time" in audio_data
                and "end_time" in audio_data
            ):
                try:
                    sound_type = SoundType(audio_data["type"])
                except ValueError:
                    sound_type = SoundType.SILENCE
                audio_event = AudioEvent(
                    type=sound_type,
                    start_time=self._normalize_timestamp(audio_data["start_time"]),
                    end_time=self._normalize_timestamp(audio_data["end_time"]),
                    speaker=audio_data.get("speaker"),
                    text=audio_data.get("text"),
                    label=audio_data.get("label"),
                    emotion=audio_data.get("emotion"),
                    confidence=audio_data.get("confidence", 0.5),
                )
                audio_events.append(audio_event)

        # Create multimodal correlations
        correlations = []
        for corr_data in data.get("multimodal_correlations", []):
            if "visual_event" in corr_data and "audio_event" in corr_data:
                correlation = MultimodalCorrelation(
                    visual_event=corr_data["visual_event"],
                    audio_event=corr_data["audio_event"],
                    description=corr_data.get("description", ""),
                )
                correlations.append(correlation)

        # Build AnalysisResult
        analysis_result = AnalysisResult(
            entities=entities,
            visual_events=visual_events,
            audio_events=audio_events,
            multimodal_correlations=correlations,
            summary=data.get("summary", ""),
        )

        return analysis_result

    def _normalize_timestamp(self, ts: str) -> str:
        """Normalize timestamp to MM:SS format."""
        # Handle various timestamp formats
        ts = ts.strip()

        # Remove trailing non-digit characters (like 's')
        ts = re.sub(r"[^0-9:]", "", ts)

        # Handle cases like "0:03", "00:03", "3", "0:3", "00:03.5", "3.5s"
        if ":" in ts:
            # Already in MM:SS format
            parts = ts.split(":")
            minutes = parts[0].zfill(2)
            seconds = parts[1]
        else:
            # Single number, assume seconds
            minutes = "00"
            seconds = ts

        # Handle decimal seconds
        if "." in seconds:
            sec_parts = seconds.split(".")
            seconds = sec_parts[0]
            # We'll keep the decimal part for processing but return MM:SS format
            # For now just return the integer part

        return f"{minutes.zfill(2)}:{seconds.zfill(2)}"


# Convenience function for direct use
def parse_vlm_output(raw_text: str) -> AnalysisResult:
    parser = JSONParser()
    return parser.parse(raw_text)
