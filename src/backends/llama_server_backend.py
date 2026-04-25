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
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from src.backends.base import BaseVLMBackend
from src.json_parser import JSONParser
from src.models import AnalysisResult, ChunkCaption

ANALYSIS_PROMPT = """/no_think
Analyze these video frames and identify all entities and their behaviors over time.

You MUST output ONLY valid JSON. No thinking, no markdown, no explanation before or after.

The JSON must follow this exact schema:
{"entities": [{"id": "E1", "type": "person", "description": "description here", "first_seen": "00:00", "last_seen": "00:10"}], "visual_events": [{"type": "idle", "entities": ["E1"], "start_time": "00:00", "end_time": "00:05", "description": "what happened", "confidence": 0.8}], "audio_events": [], "multimodal_correlations": [], "summary": "Brief summary."}

Valid behavior types: approach, depart, interact, follow, idle, group, avoid, chase, observe, moving, walking, running, standing, sitting, playing, jumping, other

Instructions:
- Each entity needs a unique ID like E1, E2, etc.
- Timestamps are in MM:SS format based on frame position in the video
- Include ALL entities you can see (people, animals, vehicles, objects)
- Include ALL behaviors/interactions between entities
- Be specific in descriptions

Output the JSON now:"""


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
            self.logger.debug(f"Raw VLM response: {response_text[:2000]}")
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

    CHUNK_PROMPT_TEMPLATE = """You are watching a short segment of a video. Describe what is happening across these frames in chronological order.

Previous segment summary: {seed}

For each frame below, output ONE LINE describing the action. If consecutive frames show no significant change, write "(no change)". End with ONE LINE summarizing this segment in <= 20 words for use as context in the next segment.

Frame data:
{frame_block}

Output format:
FRAME_<idx>: <description>
...
SUMMARY: <one-line segment summary>
"""

    def caption_chunks(
        self,
        chunks: List[Tuple[int, List[int]]],
        db,
    ) -> List[ChunkCaption]:
        """Caption each chunk; previous chunk's summary becomes the seed for the next."""
        seed = "this is the start"
        results: List[ChunkCaption] = []

        for chunk_id, frame_indices in chunks:
            try:
                images_b64 = []
                frame_lines = []
                for fidx in frame_indices:
                    frow = db.get_frame(fidx)
                    if not frow:
                        continue
                    images_b64.append(self._encode_image(Path(frow["image_path"])))
                    dets = db.get_detections_for_frame(fidx)
                    det_text = self._format_detections(dets)
                    ts = self._format_timestamp_ms(frow["timestamp_ms"])
                    frame_lines.append(f"[frame {fidx} — t={ts}] YOLO: {det_text}")

                prompt = self.CHUNK_PROMPT_TEMPLATE.format(
                    seed=seed, frame_block="\n".join(frame_lines)
                )
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt, "images": images_b64}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                }
                response = requests.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=600
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
                per_frame, summary = self._parse_chunk_response(content, frame_indices)

                results.append(
                    ChunkCaption(
                        chunk_id=chunk_id,
                        frame_indices=list(frame_indices),
                        per_frame_lines=per_frame,
                        summary=summary,
                        raw_response=content,
                    )
                )
                if summary:
                    seed = summary
            except Exception as e:
                self.logger.warning(f"Chunk {chunk_id} failed: {e}")
                results.append(
                    ChunkCaption(
                        chunk_id=chunk_id,
                        frame_indices=list(frame_indices),
                        per_frame_lines={},
                        summary="",
                        raw_response="",
                    )
                )

        return results

    @staticmethod
    def _format_detections(dets) -> str:
        if not dets:
            return "(none)"
        parts = []
        for d in dets:
            base = (
                f"{d['class_name']} at [{d['x1']:.2f},{d['y1']:.2f},"
                f"{d['x2']:.2f},{d['y2']:.2f}] conf={d['confidence']:.2f}"
            )
            if d.get("mean_depth") is not None:
                base += f" depth={d['mean_depth']:.2f}"
            parts.append(base)
        return "; ".join(parts)

    @staticmethod
    def _format_timestamp_ms(ms: int) -> str:
        s = ms / 1000.0
        m = int(s // 60)
        sec = s - m * 60
        return f"{m:02d}:{sec:05.2f}"

    @staticmethod
    def _parse_chunk_response(text: str, frame_indices) -> Tuple[Dict[int, str], str]:
        per_frame: Dict[int, str] = {}
        summary = ""
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"FRAME[_ ]?(\d+)\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
                per_frame[idx] = m.group(2).strip()
                continue
            sm = re.match(r"SUMMARY\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if sm:
                summary = sm.group(1).strip()
        return per_frame, summary

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
