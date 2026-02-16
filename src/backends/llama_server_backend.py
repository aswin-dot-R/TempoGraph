"""
LLaMA Server backend for TempoGraph.

Uses Ollama native API to offload model inference to an external server.

Usage:
1. Start Ollama: ollama serve
2. Pull model: ollama pull qwen3-vl:4b

The server handles VRAM management and model loading.
"""

import base64
import logging
import time
from pathlib import Path
from typing import List, Optional

import requests

from src.backends.base import BaseVLMBackend
from src.json_parser import JSONParser
from src.models import AnalysisResult

ANALYSIS_PROMPT = """Analyze these video frames and identify entities and events.

Output ONLY valid JSON (no markdown, no explanation). Use this exact format:
{"entities": [{"id": "E1", "type": "person/animal/object", "description": "description", "first_seen": "00:00", "last_seen": "00:01"}], "visual_events": [{"type": "idle", "entities": ["E1"], "start_time": "00:00", "end_time": "00:01", "description": "what happened", "confidence": 0.9}], "audio_events": [], "multimodal_correlations": [], "summary": "Brief summary of video content."}"""


class LlamaServerBackend(BaseVLMBackend):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:4b",
        max_frames: int = 16,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_frames = max_frames
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        self.parser = JSONParser()

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _subsample_frames(self, frames: List[Path]) -> List[Path]:
        """Uniformly subsample frames to max_frames."""
        if len(frames) <= self.max_frames:
            return frames

        indices = [
            int(i * len(frames) / self.max_frames) for i in range(self.max_frames)
        ]
        return [frames[i] for i in indices]

    def analyze_video(
        self,
        video_path: str,
        frames: Optional[List[Path]] = None,
        audio_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> AnalysisResult:
        """Send frames to Ollama for analysis using native API."""
        self.logger.info(f"Analyzing video with Ollama: {self.base_url}")

        if frames is None:
            raise ValueError("frames must be provided for llama server backend")

        frame_paths = self._subsample_frames(frames)
        self.logger.info(f"Processing {len(frame_paths)} frames")

        analysis_prompt = prompt or ANALYSIS_PROMPT

        # Build content with images using Ollama native format
        images = []
        for frame_path in frame_paths:
            try:
                image_base64 = self._encode_image(Path(frame_path))
                images.append(image_base64)
            except Exception as e:
                self.logger.warning(f"Failed to encode frame {frame_path}: {e}")

        # Use Ollama native API
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": analysis_prompt,
                    "images": images,
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            inference_time = time.time() - start_time

            result = response.json()
            response_text = result.get("message", {}).get("content", "")

            self.logger.info(
                f"Got response ({len(response_text)} chars) in {inference_time:.2f}s"
            )
            self.logger.info("Analysis complete, parsing response...")

            analysis = self.parser.parse(response_text)

            self.logger.info(
                f"Parsed: {len(analysis.entities)} entities, "
                f"{len(analysis.visual_events)} visual events"
            )

            return analysis

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            ) from e
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise

    def is_available(self) -> bool:
        """Check if server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def cleanup(self):
        """No local resources to clean up."""
        pass

    @property
    def name(self) -> str:
        return "llama-server"

    @property
    def requires_gpu(self) -> bool:
        return False
