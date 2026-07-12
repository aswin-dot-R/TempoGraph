"""Aggregates per-chunk captions into an AnalysisResult."""

import logging
import sqlite3
from pathlib import Path
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

    def aggregate(
        self,
        chunks: List[ChunkCaption],
        transcript_segments: Optional[list] = None,
        on_call: Optional[callable] = None,
        db_path: Optional[Path] = None,
    ) -> AnalysisResult:
        if not chunks:
            return AnalysisResult(summary="No captions produced.")

        self._on_call = on_call
        transcript_text = self._format_transcript(transcript_segments or [])

        # Dense timeline (optional context from dense captioning stage).
        dense_text = ""
        dense_timeline: List[dict] = []
        if db_path is not None:
            try:
                dense_timeline = self.load_dense_timeline(db_path)
            except Exception as e:
                self.logger.warning(f"could not load dense timeline: {e}")
            if dense_timeline:
                dense_text = self._format_dense_timeline(dense_timeline)

        if len(chunks) <= self.single_pass_max_chunks:
            result = self._single_pass(chunks, transcript_text, dense_text=dense_text)
        else:
            meta = self._compress_hierarchical(chunks)
            result = self._single_pass_from_text(
                meta, transcript_text, dense_text=dense_text
            )

        if dense_timeline:
            object.__setattr__(result, "dense_timeline", dense_timeline)
        return result

    def load_dense_timeline(self, db_path: Path, max_lines: int = 120) -> list:
        """Condensed dense-caption timeline for aggregation and analysis.json.

        Reads frame_captions joined with frames (for timestamp_ms). Keeps every
        escalated row (verifier_caption preferred over caption when
        verifier_agrees == 0) and evenly subsamples the rest so the total is
        <= max_lines. Returns [{"timestamp_ms": int, "text": str,
        "escalated": bool, "verified": bool}] sorted by timestamp.
        """
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT fc.frame_idx, fc.caption, fc.verifier_caption, "
                "fc.verifier_agrees, fc.escalated, f.timestamp_ms "
                "FROM frame_captions fc "
                "JOIN frames f ON fc.frame_idx = f.frame_idx "
                "ORDER BY f.timestamp_ms ASC"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        escalated: List[dict] = []
        regular: List[dict] = []

        for r in rows:
            # verifier_agrees == 0 means the 35B disagreed → use its caption.
            if (
                r["verifier_caption"] is not None
                and r["verifier_agrees"] is not None
                and r["verifier_agrees"] == 0
            ):
                text = r["verifier_caption"]
            else:
                text = r["caption"]
            item = {
                "timestamp_ms": int(r["timestamp_ms"]),
                "text": text,
                "escalated": bool(r["escalated"]),
                "verified": r["verifier_agrees"] is not None,
            }
            if item["escalated"]:
                escalated.append(item)
            else:
                regular.append(item)

        total = len(escalated) + len(regular)
        if total <= max_lines:
            return escalated + regular

        # Subsample regular rows to fit.
        remaining = max_lines - len(escalated)
        if remaining <= 0:
            return escalated

        if len(regular) <= remaining:
            subsampled = regular
        else:
            step = len(regular) / remaining
            indices = [
                min(int(round(i * step)), len(regular) - 1) for i in range(remaining)
            ]
            subsampled = [regular[i] for i in indices]

        result = escalated + subsampled
        result.sort(key=lambda x: x["timestamp_ms"])
        return result

    @staticmethod
    def _format_dense_timeline(timeline: list) -> str:
        """Compact MM:SS text block for prepending to the LLM prompt.

        Escalated lines are prefixed with ``*`` so the LLM can spot them.
        """
        if not timeline:
            return ""
        lines = []
        for entry in timeline:
            ms = entry["timestamp_ms"]
            mm = ms // 60000
            ss = (ms % 60000) / 1000.0
            prefix = "*" if entry.get("escalated") else ""
            lines.append(f"{prefix}{mm:02d}:{ss:05.2f} {entry['text']}")
        return "\n".join(lines)

    def _single_pass(
        self,
        chunks: List[ChunkCaption],
        transcript_text: str = "",
        dense_text: str = "",
    ) -> AnalysisResult:
        # dense_text is prepended once by _single_pass_from_text below.
        log_lines = []
        for c in chunks:
            for fidx in c.frame_indices:
                line = c.per_frame_lines.get(fidx, "")
                if line:
                    log_lines.append(f"[frame {fidx}] {line}")
            if c.summary:
                log_lines.append(f"[chunk {c.chunk_id} summary] {c.summary}")
        return self._single_pass_from_text(
            "\n".join(log_lines), transcript_text, dense_text=dense_text
        )

    def _single_pass_from_text(
        self, captions_text: str, transcript_text: str = "", dense_text: str = ""
    ) -> AnalysisResult:
        if dense_text:
            captions_text = f"[dense captions]\n{dense_text}\n\n{captions_text}"
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
            lines.append(f"[{mm:02d}:{ss:05.2f}-{mm2:02d}:{ss2:05.2f}] {text}")
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
                    cb(
                        {
                            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                            "completion_tokens": int(usage.get("completion_tokens", 0)),
                            "total_tokens": int(usage.get("total_tokens", 0)),
                        }
                    )
                except Exception:
                    pass
            choices = data.get("choices") or []
            if not choices:
                return ""
            return (choices[0].get("message", {}) or {}).get("content") or ""
        except Exception as e:
            self.logger.warning(f"llama-server text call failed: {e}")
            return ""
