"""Aggregates per-chunk captions into an AnalysisResult."""

import logging
from typing import List

import requests

from src.json_parser import JSONParser
from src.models import AnalysisResult, ChunkCaption


SINGLE_PASS_PROMPT = """You are given a chronological log of per-frame and per-chunk descriptions of a video. Identify entities (people, animals, vehicles, objects), their behaviors and interactions over time, and produce structured JSON.

Schema:
{{"entities":[{{"id":"E1","type":"person","description":"...","first_seen":"MM:SS","last_seen":"MM:SS"}}],
"visual_events":[{{"type":"walking","entities":["E1"],"start_time":"MM:SS","end_time":"MM:SS","description":"...","confidence":0.8}}],
"audio_events":[],"multimodal_correlations":[],"summary":"..."}}

Valid behavior types: approach, depart, interact, follow, idle, group, avoid, chase, observe, moving, walking, running, standing, sitting, playing, jumping, other.

Caption log:
{captions}

Output ONLY the JSON.
"""

META_PROMPT = """Compress the following sequence of segment summaries into ONE paragraph that preserves who/what/when. Keep timestamps if present.

{block}

Output the compressed paragraph only.
"""


class CaptionAggregator:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:4b",
        single_pass_max_chunks: int = 30,
        group_size: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.single_pass_max_chunks = single_pass_max_chunks
        self.group_size = group_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = JSONParser()
        self.logger = logging.getLogger(__name__)

    def aggregate(self, chunks: List[ChunkCaption]) -> AnalysisResult:
        if not chunks:
            return AnalysisResult(summary="No captions produced.")

        if len(chunks) <= self.single_pass_max_chunks:
            return self._single_pass(chunks)

        meta = self._compress_hierarchical(chunks)
        return self._single_pass_from_text(meta)

    def _single_pass(self, chunks: List[ChunkCaption]) -> AnalysisResult:
        log_lines = []
        for c in chunks:
            for fidx in c.frame_indices:
                line = c.per_frame_lines.get(fidx, "")
                if line:
                    log_lines.append(f"[frame {fidx}] {line}")
            if c.summary:
                log_lines.append(f"[chunk {c.chunk_id} summary] {c.summary}")
        return self._single_pass_from_text("\n".join(log_lines))

    def _single_pass_from_text(self, captions_text: str) -> AnalysisResult:
        prompt = SINGLE_PASS_PROMPT.format(captions=captions_text)
        response_text = self._call_ollama_text(prompt)
        return self.parser.parse(response_text)

    def _compress_hierarchical(self, chunks: List[ChunkCaption]) -> str:
        groups = [
            chunks[i : i + self.group_size]
            for i in range(0, len(chunks), self.group_size)
        ]
        meta_pieces: List[str] = []
        for grp in groups:
            block_lines = []
            for c in grp:
                if c.summary:
                    block_lines.append(f"[chunk {c.chunk_id}] {c.summary}")
            block = "\n".join(block_lines)
            meta_pieces.append(self._call_ollama_text(META_PROMPT.format(block=block)))
        return "\n".join(meta_pieces)

    def _call_ollama_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=600
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            self.logger.warning(f"Ollama text call failed: {e}")
            return ""
