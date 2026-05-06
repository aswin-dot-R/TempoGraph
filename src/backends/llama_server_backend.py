"""
llama.cpp llama-server backend for TempoGraph.

Talks to a llama.cpp server over its OpenAI-compatible HTTP API
(`/v1/chat/completions`, `/v1/models`), with multimodal image content
sent as `image_url` data URIs and Qwen3 reasoning disabled via
`chat_template_kwargs.enable_thinking=false`.
"""

import base64
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
        base_url: str = "http://127.0.0.1:8082",
        model: str = "Qwen3.5-9B-Q8_0.gguf",
        max_frames: int = 16,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_frames = max_frames
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.logger = logging.getLogger(__name__)
        self.parser = JSONParser()

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _subsample_frames(self, frames: List[Path]) -> List[Path]:
        if len(frames) <= self.max_frames:
            return frames
        indices = [
            int(i * len(frames) / self.max_frames) for i in range(self.max_frames)
        ]
        return [frames[i] for i in indices]

    def _build_content(self, prompt: str, images_b64: List[str]) -> List[dict]:
        items: List[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        return items

    def _build_payload(self, prompt: str, images_b64: List[str]) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "messages": [
                {
                    "role": "user",
                    "content": self._build_content(prompt, images_b64),
                }
            ],
        }

    @staticmethod
    def _extract_content(result: dict) -> str:
        choices = result.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message", {}) or {}
        return msg.get("content") or ""

    def analyze_video(
        self,
        video_path: str,
        frames: Optional[List[Path]] = None,
        audio_path: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> AnalysisResult:
        self.logger.info(f"Analyzing video with llama-server: {self.base_url}")

        if frames is None:
            raise ValueError("frames must be provided for llama server backend")

        frame_paths = self._subsample_frames(frames)
        self.logger.info(f"Processing {len(frame_paths)} frames")

        analysis_prompt = prompt or ANALYSIS_PROMPT

        images_b64: List[str] = []
        for frame_path in frame_paths:
            try:
                images_b64.append(self._encode_image(Path(frame_path)))
            except Exception as e:
                self.logger.warning(f"Failed to encode frame {frame_path}: {e}")

        payload = self._build_payload(analysis_prompt, images_b64)

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            inference_time = time.time() - start_time

            result = response.json()
            response_text = self._extract_content(result)

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
            self.logger.error(f"Failed to connect to llama-server: {e}")
            raise RuntimeError(
                f"Could not connect to llama-server at {self.base_url}. "
                "Make sure the llama.cpp server is running and reachable."
            ) from e
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            raise

    CHUNK_PROMPT_TEMPLATE = """You are watching a short segment of a video. Describe what is happening across these frames in chronological order.

Previous segment summary: {seed}

{entity_block}For each frame below, output ONE LINE describing the action. If consecutive frames show no significant change, write "(no change)". End with ONE LINE summarizing this segment in <= 20 words for use as context in the next segment.

IMPORTANT: When referring to people or objects, reuse entity IDs from the known-entities list above if they are the same person/object. Only create new IDs (E<next>, E<next+1>, ...) for genuinely NEW entities. After the SUMMARY line, list any NEW entities introduced in this chunk.

Frame data:
{frame_block}

Output format:
FRAME_<idx>: <description>
...
SUMMARY: <one-line segment summary>
NEW_ENTITIES: <comma-separated list like 'E3=red car, E4=brown dog' or 'none'>
"""

    def caption_chunks(
        self,
        chunks: List[Tuple[int, List[int]]],
        db,
        on_chunk: Optional[Callable[[dict], None]] = None,
    ) -> List[ChunkCaption]:
        seed = "this is the start"
        entity_registry: Dict[str, str] = {}  # e.g. {"E1": "boy in blue shirt", "E2": "T-Rex"}
        results: List[ChunkCaption] = []
        n_ctx = self.get_n_ctx()
        n_total = len(chunks)

        for chunk_id, frame_indices in chunks:
            t0 = time.time()
            usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                images_b64: List[str] = []
                frame_lines: List[str] = []
                for fidx in frame_indices:
                    frow = db.get_frame(fidx)
                    if not frow:
                        continue
                    images_b64.append(self._encode_image(Path(frow["image_path"])))
                    dets = db.get_detections_for_frame(fidx)
                    det_text = self._format_detections(dets)
                    ts = self._format_timestamp_ms(frow["timestamp_ms"])
                    frame_lines.append(f"[frame {fidx} — t={ts}] YOLO: {det_text}")

                entity_block = self._format_entity_block(entity_registry)
                prompt = self.CHUNK_PROMPT_TEMPLATE.format(
                    seed=seed,
                    entity_block=entity_block,
                    frame_block="\n".join(frame_lines),
                )
                payload = self._build_payload(prompt, images_b64)
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                resp_json = response.json()
                content = self._extract_content(resp_json)
                usage = self._extract_usage(resp_json)
                per_frame, summary = self._parse_chunk_response(content, frame_indices)

                # Update entity registry from NEW_ENTITIES line
                new_entities = self._parse_new_entities(content)
                entity_registry.update(new_entities)

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

                if on_chunk is not None:
                    try:
                        on_chunk({
                            "chunk_id": chunk_id,
                            "chunk_index": len(results) - 1,
                            "n_total": n_total,
                            "n_images": len(images_b64),
                            "prompt_tokens": usage["prompt_tokens"],
                            "completion_tokens": usage["completion_tokens"],
                            "total_tokens": usage["total_tokens"],
                            "n_ctx": n_ctx,
                            "elapsed_s": round(time.time() - t0, 2),
                            "ok": True,
                        })
                    except Exception as cb_e:
                        self.logger.warning(f"on_chunk callback failed: {cb_e}")
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
                if on_chunk is not None:
                    try:
                        on_chunk({
                            "chunk_id": chunk_id,
                            "chunk_index": len(results) - 1,
                            "n_total": n_total,
                            "n_images": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "n_ctx": n_ctx,
                            "elapsed_s": round(time.time() - t0, 2),
                            "ok": False,
                            "error": str(e),
                        })
                    except Exception:
                        pass

        return results

    # ── dynamic context-aware chunking ─────────────────────────────

    COMPACTION_PROMPT = """/no_think
You have been watching a video and noting frame-by-frame descriptions. Below are the summaries from all segments processed so far, plus the entity registry.

Segment summaries:
{segment_summaries}

Known entities:
{entity_block}

Compress all of the above into EXACTLY TWO lines:
LINE 1: One-sentence summary of everything that has happened so far.
LINE 2: Key entities still on screen (comma-separated IDs + brief role).

Output ONLY those two lines, nothing else.
"""

    def caption_frames_dynamic(
        self,
        all_frame_indices: List[int],
        db,
        chunk_size_hint: int = 10,
        context_threshold: float = 0.80,
        on_chunk: Optional[Callable[[dict], None]] = None,
    ) -> List[ChunkCaption]:
        """Context-aware dynamic chunking with automatic compaction.

        Instead of fixed chunk sizes, this method:
        1. Estimates how many frames fit per chunk based on actual token usage
        2. Tracks cumulative prompt+completion tokens across a "pass"
        3. When cumulative tokens hit ``context_threshold`` of n_ctx,
           performs a hard compaction: summarises everything into a 2-liner,
           prunes the entity registry, and starts a new pass.

        Args:
            all_frame_indices: All frame indices to process (in order).
            db: TempoGraphDB instance.
            chunk_size_hint: Initial guess for frames per chunk (self-calibrates).
            context_threshold: Fraction of n_ctx that triggers compaction.
            on_chunk: Optional progress callback.

        Returns:
            List of ChunkCaption results (same as caption_chunks).
        """
        n_ctx = self.get_n_ctx() or 100096
        budget = int(n_ctx * context_threshold)

        self.logger.info(
            f"Dynamic chunking: {len(all_frame_indices)} frames, "
            f"n_ctx={n_ctx}, budget={budget} tokens "
            f"({context_threshold:.0%} threshold)"
        )

        seed = "this is the start"
        entity_registry: Dict[str, str] = {}
        results: List[ChunkCaption] = []

        # Self-calibrating estimates
        est_tokens_per_image = 1500  # conservative initial guess
        prompt_overhead_tokens = 600  # template text
        completion_tokens_per_chunk = 500  # avg output length

        # Pass tracking
        pass_id = 0
        pass_summaries: List[str] = []
        cumulative_tokens = 0

        idx = 0
        chunk_id = 0
        n_total_frames = len(all_frame_indices)

        while idx < n_total_frames:
            # Calculate dynamic chunk size based on current estimates
            entity_text = self._format_entity_block(entity_registry)
            entity_tokens_est = max(50, len(entity_text) // 3)
            available_for_images = (
                budget
                - prompt_overhead_tokens
                - entity_tokens_est
                - completion_tokens_per_chunk
            )
            dynamic_chunk_size = max(
                2, min(50, available_for_images // max(1, est_tokens_per_image))
            )

            # Build this chunk
            chunk_frames = all_frame_indices[idx: idx + dynamic_chunk_size]

            t0 = time.time()
            usage: Dict[str, int] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            try:
                images_b64: List[str] = []
                frame_lines: List[str] = []
                for fidx in chunk_frames:
                    frow = db.get_frame(fidx)
                    if not frow:
                        continue
                    images_b64.append(self._encode_image(Path(frow["image_path"])))
                    dets = db.get_detections_for_frame(fidx)
                    det_text = self._format_detections(dets)
                    ts = self._format_timestamp_ms(frow["timestamp_ms"])
                    frame_lines.append(f"[frame {fidx} — t={ts}] YOLO: {det_text}")

                entity_block = self._format_entity_block(entity_registry)
                prompt = self.CHUNK_PROMPT_TEMPLATE.format(
                    seed=seed,
                    entity_block=entity_block,
                    frame_block="\n".join(frame_lines),
                )
                payload = self._build_payload(prompt, images_b64)
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                resp_json = response.json()
                content = self._extract_content(resp_json)
                usage = self._extract_usage(resp_json)

                per_frame, summary = self._parse_chunk_response(
                    content, chunk_frames
                )
                new_entities = self._parse_new_entities(content)
                entity_registry.update(new_entities)

                results.append(
                    ChunkCaption(
                        chunk_id=chunk_id,
                        frame_indices=list(chunk_frames),
                        per_frame_lines=per_frame,
                        summary=summary,
                        raw_response=content,
                    )
                )
                if summary:
                    seed = summary
                    pass_summaries.append(summary)

                # ── self-calibrate token estimates ──
                actual_prompt = usage["prompt_tokens"]
                actual_completion = usage["completion_tokens"]
                actual_total = usage["total_tokens"]

                if actual_prompt > 0 and len(images_b64) > 0:
                    measured_per_image = (
                        (actual_prompt - prompt_overhead_tokens - entity_tokens_est)
                        / len(images_b64)
                    )
                    # Exponential moving average (blend old estimate with new)
                    est_tokens_per_image = int(
                        0.3 * est_tokens_per_image + 0.7 * max(500, measured_per_image)
                    )
                if actual_completion > 0:
                    completion_tokens_per_chunk = int(
                        0.3 * completion_tokens_per_chunk + 0.7 * actual_completion
                    )

                cumulative_tokens += actual_total

                self.logger.info(
                    f"Chunk {chunk_id} (pass {pass_id}): "
                    f"{len(images_b64)} frames, "
                    f"prompt={actual_prompt}, completion={actual_completion}, "
                    f"cumulative={cumulative_tokens}/{budget} "
                    f"({100*cumulative_tokens/budget:.0f}%), "
                    f"est/img={est_tokens_per_image}, "
                    f"dynamic_size={dynamic_chunk_size}"
                )

                # ── callback ──
                if on_chunk is not None:
                    try:
                        on_chunk({
                            "chunk_id": chunk_id,
                            "chunk_index": len(results) - 1,
                            "n_total": n_total_frames,
                            "n_images": len(images_b64),
                            "prompt_tokens": actual_prompt,
                            "completion_tokens": actual_completion,
                            "total_tokens": actual_total,
                            "n_ctx": n_ctx,
                            "cumulative_tokens": cumulative_tokens,
                            "budget": budget,
                            "pass_id": pass_id,
                            "dynamic_chunk_size": dynamic_chunk_size,
                            "est_tokens_per_image": est_tokens_per_image,
                            "elapsed_s": round(time.time() - t0, 2),
                            "ok": True,
                        })
                    except Exception as cb_e:
                        self.logger.warning(f"on_chunk callback failed: {cb_e}")

                # ── check if we need a hard compaction ──
                next_chunk_est = (
                    prompt_overhead_tokens
                    + entity_tokens_est
                    + dynamic_chunk_size * est_tokens_per_image
                    + completion_tokens_per_chunk
                )
                if cumulative_tokens + next_chunk_est > budget:
                    self.logger.info(
                        f"⚡ Context compaction triggered at pass {pass_id}: "
                        f"cumulative={cumulative_tokens}/{budget}, "
                        f"entities={len(entity_registry)}, "
                        f"summaries={len(pass_summaries)}"
                    )
                    seed, entity_registry = self._compact_pass(
                        pass_summaries, entity_registry
                    )
                    pass_id += 1
                    pass_summaries = []
                    cumulative_tokens = 0

            except Exception as e:
                self.logger.warning(f"Chunk {chunk_id} failed: {e}")
                results.append(
                    ChunkCaption(
                        chunk_id=chunk_id,
                        frame_indices=list(chunk_frames),
                        per_frame_lines={},
                        summary="",
                        raw_response="",
                    )
                )
                if on_chunk is not None:
                    try:
                        on_chunk({
                            "chunk_id": chunk_id,
                            "chunk_index": len(results) - 1,
                            "n_total": n_total_frames,
                            "n_images": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "n_ctx": n_ctx,
                            "cumulative_tokens": cumulative_tokens,
                            "budget": budget,
                            "pass_id": pass_id,
                            "dynamic_chunk_size": dynamic_chunk_size,
                            "est_tokens_per_image": est_tokens_per_image,
                            "elapsed_s": round(time.time() - t0, 2),
                            "ok": False,
                            "error": str(e),
                        })
                    except Exception:
                        pass

            idx += len(chunk_frames)
            chunk_id += 1

        self.logger.info(
            f"Dynamic chunking complete: {chunk_id} chunks across "
            f"{pass_id + 1} pass(es), {len(results)} results"
        )
        return results

    def _compact_pass(
        self,
        summaries: List[str],
        entity_registry: Dict[str, str],
    ) -> Tuple[str, Dict[str, str]]:
        """Compact a pass: summarise into 2 lines, prune entity registry.

        Tries to call the VLM for a smart summary. Falls back to a
        mechanical truncation if the VLM call fails.

        Returns:
            (compact_seed, pruned_entity_registry)
        """
        # Build the compaction prompt
        entity_block = self._format_entity_block(entity_registry)
        summary_text = "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(summaries) if s
        )
        prompt = self.COMPACTION_PROMPT.format(
            segment_summaries=summary_text or "(none)",
            entity_block=entity_block,
        )

        try:
            payload = self._build_payload(prompt, [])  # text-only, no images
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            content = self._extract_content(response.json()).strip()

            # Take the first two lines
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            compact_seed = " ".join(lines[:2]) if lines else summaries[-1]

            self.logger.info(
                f"VLM compaction: {len(summaries)} summaries + "
                f"{len(entity_registry)} entities → seed ({len(compact_seed)} chars)"
            )

        except Exception as e:
            self.logger.warning(f"VLM compaction failed ({e}), using mechanical fallback")
            # Mechanical fallback: last 3 summaries
            compact_seed = "; ".join(summaries[-3:])
            if len(compact_seed) > 300:
                compact_seed = compact_seed[-300:]

        # Prune entity registry: keep only the most recent N entities
        # (entities mentioned in recent summaries are more relevant)
        max_entities = 20
        if len(entity_registry) > max_entities:
            # Keep entities that appear in recent summaries
            recent_text = " ".join(summaries[-5:]).lower()
            scored = []
            for eid, desc in entity_registry.items():
                score = 1 if eid.lower() in recent_text else 0
                score += 1 if desc.lower()[:20] in recent_text else 0
                scored.append((score, eid, desc))
            scored.sort(reverse=True)
            entity_registry = {eid: desc for _, eid, desc in scored[:max_entities]}
            self.logger.info(
                f"Entity pruning: kept {len(entity_registry)}/{len(scored)} entities"
            )

        return compact_seed, entity_registry

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

    @staticmethod
    def _parse_new_entities(text: str) -> Dict[str, str]:
        """Extract new entity definitions from the NEW_ENTITIES line.

        Expected format:
            NEW_ENTITIES: E3=red car, E4=brown dog
        or:
            NEW_ENTITIES: none
        """
        entities: Dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            m = re.match(r"NEW_ENTITIES\s*[:\-]\s*(.+)$", line, re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                if raw.lower() in ("none", "n/a", "-", ""):
                    break
                for pair in raw.split(","):
                    pair = pair.strip()
                    eq = pair.find("=")
                    if eq > 0:
                        eid = pair[:eq].strip()
                        desc = pair[eq + 1:].strip()
                        entities[eid] = desc
                break
        return entities

    @staticmethod
    def _format_entity_block(registry: Dict[str, str]) -> str:
        """Format the entity registry for injection into the prompt."""
        if not registry:
            return "Known entities so far: (none — this is the first segment)\n\n"
        lines = [f"  {eid}: {desc}" for eid, desc in sorted(registry.items())]
        return (
            "Known entities so far (reuse these IDs if you see the same entity):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_n_ctx(self) -> Optional[int]:
        """Read the server's configured context window from /props."""
        try:
            r = requests.get(f"{self.base_url}/props", timeout=5)
            r.raise_for_status()
            return int(
                r.json().get("default_generation_settings", {}).get("n_ctx") or 0
            ) or None
        except Exception:
            return None

    @staticmethod
    def _extract_usage(result: dict) -> Dict[str, int]:
        u = result.get("usage") or {}
        return {
            "prompt_tokens": int(u.get("prompt_tokens", 0)),
            "completion_tokens": int(u.get("completion_tokens", 0)),
            "total_tokens": int(u.get("total_tokens", 0)),
        }

    def cleanup(self):
        pass

    @property
    def name(self) -> str:
        return "llama-server"

    @property
    def requires_gpu(self) -> bool:
        return False
