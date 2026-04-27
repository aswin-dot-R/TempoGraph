"""Aggregates per-chunk captions into an AnalysisResult."""

import logging
from typing import List, Optional

import requests

from src.json_parser import JSONParser
from src.models import AnalysisResult, ChunkCaption


SINGLE_PASS_PROMPT = """You are given a chronological log of per-frame and per-chunk descriptions of a video, and (optionally) a speech transcript. Identify entities (people, animals, vehicles, objects), their behaviors and interactions over time, and produce structured JSON. If a transcript is provided, also populate audio_events and multimodal_correlations linking what is said to what is seen.

Schema:
{{"entities":[{{"id":"E1","type":"person","description":"...","first_seen":"MM:SS","last_seen":"MM:SS"}}],
"visual_events":[{{"type":"walking","entities":["E1"],"start_time":"MM:SS","end_time":"MM:SS","description":"...","confidence":0.8}}],
"audio_events":[{{"type":"speech","start_time":"MM:SS","end_time":"MM:SS","text":"...","speaker":"unknown"}}],
"multimodal_correlations":[{{"audio_idx":0,"visual_idx":2,"description":"speaker says X while subject does Y","confidence":0.7}}],
"summary":"..."}}

Valid visual_events types: approach, depart, interact, follow, idle, group, avoid, chase, observe, moving, walking, running, standing, sitting, playing, jumping, other.

Caption log:
{captions}

Audio transcript (may be empty):
{transcript}

Output ONLY the JSON.
"""

META_PROMPT = """Compress the following sequence of segment summaries into ONE paragraph that preserves who/what/when. Keep timestamps if present.

{block}

Output the compressed paragraph only.
"""


class CaptionAggregator:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        model: str = "Qwen3.5-9B-Q8_0.gguf",
        single_pass_max_chunks: int = 30,
        group_size: int = 10,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.single_pass_max_chunks = single_pass_max_chunks
        self.group_size = group_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.parser = JSONParser()
        self.logger = logging.getLogger(__name__)

    def aggregate(self, chunks: List[ChunkCaption],
                  transcript_segments: Optional[list] = None,
                  on_call: Optional[callable] = None) -> AnalysisResult:
        if not chunks:
            return AnalysisResult(summary="No captions produced.")

        self._on_call = on_call
        transcript_text = self._format_transcript(transcript_segments or [])

        if len(chunks) <= self.single_pass_max_chunks:
            return self._single_pass(chunks, transcript_text)

        meta = self._compress_hierarchical(chunks)
        return self._single_pass_from_text(meta, transcript_text)

    def _single_pass(self, chunks: List[ChunkCaption],
                     transcript_text: str = "") -> AnalysisResult:
        log_lines = []
        for c in chunks:
            for fidx in c.frame_indices:
                line = c.per_frame_lines.get(fidx, "")
                if line:
                    log_lines.append(f"[frame {fidx}] {line}")
            if c.summary:
                log_lines.append(f"[chunk {c.chunk_id} summary] {c.summary}")
        return self._single_pass_from_text("\n".join(log_lines), transcript_text)

    def _single_pass_from_text(self, captions_text: str,
                               transcript_text: str = "") -> AnalysisResult:
        prompt = SINGLE_PASS_PROMPT.format(
            captions=captions_text,
            transcript=transcript_text or "(no transcript)",
        )
        response_text = self._call_llm_text(prompt)
        return self.parser.parse(response_text)

    @staticmethod
    def _format_transcript(segments: list) -> str:
        if not segments:
            return ""
        lines = []
        for s in segments:
            start_ms = s.get("start_ms", 0)
            end_ms = s.get("end_ms", 0)
            text = (s.get("text") or "").strip()
            if not text:
                continue
            mm = start_ms // 60000
            ss = (start_ms % 60000) / 1000.0
            mm2 = end_ms // 60000
            ss2 = (end_ms % 60000) / 1000.0
            lines.append(
                f"[{mm:02d}:{ss:05.2f}-{mm2:02d}:{ss2:05.2f}] {text}"
            )
        return "\n".join(lines)

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
            meta_pieces.append(self._call_llm_text(META_PROMPT.format(block=block)))
        return "\n".join(meta_pieces)

    def _call_llm_text(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions", json=payload, timeout=600
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage") or {}
            cb = getattr(self, "_on_call", None)
            if cb is not None:
                try:
                    cb({
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    })
                except Exception:
                    pass
            choices = data.get("choices") or []
            if not choices:
                return ""
            return (choices[0].get("message", {}) or {}).get("content") or ""
        except Exception as e:
            self.logger.warning(f"llama-server text call failed: {e}")
            return ""
