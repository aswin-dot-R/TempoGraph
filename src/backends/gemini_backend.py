"""
Gemini Flash backend for TempoGraph.

Uses google-genai SDK. Key API flow:
1. client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
2. Upload video:
   video_file = client.files.upload(file=video_path)
   # Poll until file.state == "ACTIVE"
3. Generate:
   response = client.models.generate_content(
       model="gemini-2.0-flash",
       contents=[
           types.Content(parts=[
               types.Part.from_uri(
                   file_uri=video_file.uri,
                   mime_type=video_file.mime_type
               ),
               types.Part.from_text(ANALYSIS_PROMPT)
           ])
       ],
       config=types.GenerateContentConfig(
           temperature=0.1,
           max_output_tokens=8192,
           response_mime_type="application/json"  # forces JSON output
       )
   )
4. Parse: response.text → json_parser.parse()

IMPORTANT DETAILS:
- Gemini processes audio FROM the video natively — no need for separate
  audio extraction in cloud mode
- response_mime_type="application/json" forces structured output
- Video files take time to process — poll file.state until "ACTIVE"
- Free tier: 15 RPM, 1000 RPD for Flash
- Handle google.api_core.exceptions for rate limits

Audio token cost: ~32 tokens/second of audio (very cheap)
Video token cost: ~263 tokens/second at 1 FPS
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional
from src.backends.base import BaseVLMBackend
from src.models import AnalysisResult
from src.json_parser import JSONParser

# Full VLM prompt for Gemini
ANALYSIS_PROMPT = """You are a video analysis AI. Analyze the provided video and identify:

1. All distinct entities (people, animals, objects) with unique IDs
2. All temporal behaviors and interactions between entities
3. Estimated timestamps based on frame position

Output ONLY valid JSON:
{
  "entities": [
    {"id": "E1", "type": "dog", "description": "brown labrador",
     "first_seen": "00:02", "last_seen": "00:45"}
  ],
  "visual_events": [
    {"type": "approach", "entities": ["E1", "E2"],
     "start_time": "00:03", "end_time": "00:05",
     "description": "Brown labrador walks toward white poodle",
     "confidence": 0.9}
  ],
  "audio_events": [
    {"type": "speech", "start_time": "00:04", "end_time": "00:06",
     "speaker": "Speaker 1", "text": "Come here boy!",
     "label": "command", "emotion": "excited", "confidence": 0.8}
  ],
  "multimodal_correlations": [
    {"visual_event": 0, "audio_event": 0,
     "description": "Person approaches dog at 00:04"}
  ],
  "summary": "Brief description of what happens in the video."
}

Behavior types: approach, depart, interact, follow, idle, group, avoid, chase, observe.
Sound types: speech, music, animal_sound, vehicle, impact, environmental, silence.
Be precise. Include ALL entities and interactions you observe across the video."""


class GeminiBackend(BaseVLMBackend):
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.logger = logging.getLogger(__name__)
        self.parser = JSONParser()
        self._client = None

    def _get_client(self):
        """Lazy init Gemini client."""
        if self._client is None:
            try:
                from google import genai
                import google.genai.types as types

                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")
                self._client = genai.Client(api_key=api_key)
                self._types = types
                self.logger.info("Gemini client initialized")
            except ImportError as e:
                raise ImportError(
                    "google-genai SDK not installed. "
                    "Install with: pip install google-genai"
                ) from e
        return self._client

    def analyze_video(
        self,
        video_path: str,
        frames: Optional[List[Path]] = None,
        audio_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Upload video to Gemini and get comprehensive analysis.
        Handles: upload → wait for processing → generate → parse
        """
        self.logger.info(f"Analyzing video with Gemini: {video_path}")

        # Use provided prompt or default
        analysis_prompt = prompt or ANALYSIS_PROMPT

        # Upload video and wait for processing
        video_file = self._upload_and_wait(video_path)

        try:
            # Generate analysis
            response = self._generate_analysis(video_file, analysis_prompt)

            # Parse JSON response
            result = self.parser.parse(response.text)

            self.logger.info(
                f"Analysis complete: {len(result.entities)} entities, "
                f"{len(result.visual_events)} visual events, "
                f"{len(result.audio_events)} audio events"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise

        finally:
            # Clean up uploaded file
            try:
                self._get_client().files.delete(video_file.name)
                self.logger.info("Uploaded file deleted")
            except Exception as e:
                self.logger.warning(f"Failed to delete uploaded file: {e}")

    def _upload_and_wait(self, video_path: str, timeout: int = 120):
        """Upload video and wait for it to become ACTIVE."""
        client = self._get_client()

        self.logger.info(f"Uploading video: {video_path}")
        video_file = client.files.upload(file=video_path)

        self.logger.info(f"Waiting for video to process (state: {video_file.state})")

        # Poll until ACTIVE or timeout
        start_time = time.time()
        while video_file.state != "ACTIVE" and time.time() - start_time < timeout:
            time.sleep(5)
            video_file = client.files.get(name=video_file.name)
            self.logger.debug(f"Video state: {video_file.state}")

        if video_file.state != "ACTIVE":
            raise TimeoutError(
                f"Video processing timed out after {timeout}s. "
                f"Current state: {video_file.state}"
            )

        self.logger.info("Video processing complete")
        return video_file

    def _generate_analysis(self, video_file, prompt: str):
        """Generate analysis from uploaded video file."""
        client = self._get_client()

        contents = [
            self._types.Content(
                parts=[
                    self._types.Part.from_uri(
                        file_uri=video_file.uri, mime_type=video_file.mime_type
                    ),
                    self._types.Part.from_text(prompt),
                ]
            )
        ]

        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )

        self.logger.info("Generating analysis...")
        response = client.models.generate_content(
            model=self.model, contents=contents, config=config
        )

        self.logger.info("Analysis generated successfully")
        return response

    def is_available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def cleanup(self):
        pass  # No GPU resources to free

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def requires_gpu(self) -> bool:
        return False
