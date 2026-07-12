"""DenseCaptionWalker: per-frame 9B vision captions with escalation.

Walks a run's ``frames`` table in order, calls the local Ornith 9B vision
server once per frame, writes one ``frame_captions`` row per frame, and
flags big scene changes (``escalated=1``) for the 35B verifier (PS3).

All HTTP is via the OpenAI-compatible ``/v1/chat/completions`` endpoint on
a llama-server (default ``http://127.0.0.1:8085``).  Tests mock ``requests``
so the suite runs with no server running.
"""

from __future__ import annotations

import base64
import logging
import math
import re
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

from src.storage import TempoGraphDB

_logger = logging.getLogger(__name__)

# ── prompt template ──────────────────────────────────────────────────

_FRAME_PROMPT = (
    "You are watching a video one frame at a time.\n"
    "Previous frame: {prev_caption}\n"
    "Reply with EXACTLY two lines and nothing else:\n"
    "FRAME: one sentence naming the visible objects with their "
    "colors/attributes, the surface or setting they are on, and anything "
    "happening in the background (e.g. "
    '"keys and a phone on a black table, two dogs moving in the background").\n'
    "CHANGE: one short sentence on what changed since the previous frame, "
    'or "no change".\n'
)


# ── Task 2: prompt + parser ────────────────────────────────────────


def parse_two_lines(text: str) -> Tuple[str, Optional[str]]:
    """Extract ``FRAME:`` and ``CHANGE:`` lines from a model reply.

    Search is case-insensitive for the ``FRAME:`` / ``CHANGE:`` prefixes.
    Falls back gracefully when one or both markers are missing.

    Args:
        text: The raw reply text from the 9B server.

    Returns:
        ``(caption, change_line)`` — ``change_line`` is ``None`` if no
        ``CHANGE:`` line was found.  An empty reply yields the caption
        ``"(no caption)"`` and ``change_line=None``.
    """
    if not text or not text.strip():
        return "(no caption)", None

    caption: Optional[str] = None
    change_line: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # ``FRAME:`` prefix (case-insensitive)
        m = re.match(r"^[Ff][Rr][Aa][Mm][Ee]\s*:\s*(.+)$", line)
        if m:
            caption = m.group(1).strip()
            continue
        # ``CHANGE:`` prefix (case-insensitive)
        m = re.match(r"^[Cc][Hh][Aa][Nn][Gg][Ee]\s*:\s*(.+)$", line)
        if m:
            change_line = m.group(1).strip()

    # Fallback: no FRAME: prefix → use the first non-empty line.
    if caption is None:
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                caption = stripped
                break
        if caption is None:
            caption = "(no caption)"

    return caption, change_line


# ── Task 3: escalation logic ───────────────────────────────────────


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two lowercased captions.

    Empty ∪ empty → 1.0.

    Args:
        a: First caption string.
        b: Second caption string.

    Returns:
        Float in [0.0, 1.0].
    """
    tokens_a = set(a.lower().split()) if a else set()
    tokens_b = set(b.lower().split()) if b else set()
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def should_escalate(
    delta_score: float,
    delta_threshold: float,
    caption: str,
    prev_caption: Optional[str],
    similarity_floor: float,
) -> bool:
    """True if this frame should be flagged as a big scene change.

    Escalation fires when *either* signal triggers:

    1. ``delta_score >= delta_threshold`` (pixel-level change is big), or
    2. ``prev_caption`` is not None **and**
       ``jaccard(caption, prev_caption) < similarity_floor``
       (the caption *reads* different).

    When ``prev_caption`` is None (first frame) only the delta signal can
    fire — the similarity signal is undefined.

    Args:
        delta_score: The frame's delta_score from the ``frames`` table.
        delta_threshold: Pre-computed threshold (e.g. 90th percentile).
        caption: Parsed caption for this frame.
        prev_caption: Parsed caption from the previous frame, or None.
        similarity_floor: Threshold below which captions are "different".

    Returns:
        True if this frame should be escalated for verifier review.
    """
    if delta_score >= delta_threshold:
        return True
    if prev_caption is not None and jaccard(caption, prev_caption) < similarity_floor:
        return True
    return False


def _percentile(scores: List[float], pct: float) -> float:
    """Pure-Python percentile (linear interpolation, like numpy default).

    Args:
        scores: Non-empty list of numeric values.
        pct: Percentile in [0, 100].

    Returns:
        The ``pct``-th percentile value.
    """
    if not scores:
        return 0.0
    s = sorted(scores)
    if len(s) < 2:
        return s[0]
    pos = (pct / 100.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ── Task 1, 4: DenseCaptionWalker ──────────────────────────────────


class DenseCaptionWalker:
    """Iterate a run's frames, caption each with the 9B vision server.

    Opens its own :class:`src.storage.TempoGraphDB` (WAL handles the
    concurrent verifier in PS3).  Never crashes on a single HTTP failure —
    logs and counts it, moves on.
    """

    def __init__(
        self,
        db_path: Path,
        base_url: str = "http://127.0.0.1:8085",
        model_name: str = "ornith-1.0-9b",
        temperature: float = 0.1,
        max_tokens: int = 96,
        escalation_percentile: float = 90.0,
        caption_similarity_floor: float = 0.3,
        request_timeout_s: float = 120.0,
        on_progress: Optional[Callable[[dict], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """
        Args:
            db_path: Path to the run's ``tempograph.db``.
            base_url: llama-server base URL (must expose
                ``/v1/chat/completions``).
            model_name: Model name sent in the request payload.
            temperature: Sampling temperature for the 9B.
            max_tokens: Max tokens per response.
            escalation_percentile: Percentile of ``frames.delta_score``
                used as the escalation threshold.
            caption_similarity_floor: Jaccard floor below which a caption
                change triggers escalation.
            request_timeout_s: HTTP timeout per frame.
            on_progress: Callback invoked after each frame with a dict
                ``{frame_idx, done, total, escalated}``.
            cancel_event: If set, ``walk()`` stops cleanly after the
                current frame.
        """
        self.db_path = Path(db_path)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.escalation_percentile = escalation_percentile
        self.caption_similarity_floor = caption_similarity_floor
        self.request_timeout_s = request_timeout_s
        self.on_progress = on_progress
        self.cancel_event = cancel_event
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    # ── public API ─────────────────────────────────────────────────

    def walk(self) -> Dict[str, int]:
        """Caption every frame not yet in ``frame_captions``.

        Returns:
            ``{"captioned": int, "escalated": int, "skipped": int,
            "errors": int}``.
        """
        db = TempoGraphDB(self.db_path)
        try:
            return self._walk(db)
        finally:
            db.close()

    # ── internals ──────────────────────────────────────────────────

    def _walk(self, db: TempoGraphDB) -> Dict[str, int]:
        frame_indices = db.get_all_frame_indices()
        total = len(frame_indices)
        if total == 0:
            self.logger.info("No frames to caption.")
            return {"captioned": 0, "escalated": 0, "skipped": 0, "errors": 0}

        # Compute delta_threshold ONCE from the frames table.
        raw_scores = [
            row["delta_score"]
            for row in db._conn.execute(
                "SELECT delta_score FROM frames ORDER BY frame_idx"
            ).fetchall()
        ]
        if len(raw_scores) < 10:
            delta_threshold = max(raw_scores) if raw_scores else 0.0
        else:
            delta_threshold = _percentile(raw_scores, self.escalation_percentile)
        self.logger.info(f"Walk: {total} frames, delta_threshold={delta_threshold:.4f}")

        # Set of already-captioned frame indices (for resume).
        existing = {
            row["frame_idx"]
            for row in db._conn.execute(
                "SELECT frame_idx FROM frame_captions"
            ).fetchall()
        }

        captioned = 0
        escalated = 0
        skipped = 0
        errors = 0
        prev_caption: Optional[str] = None

        for frame_idx in frame_indices:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.logger.info(
                    f"Cancel requested after {captioned} captioned frames."
                )
                break

            # Resume: skip already-captioned frames.
            if frame_idx in existing:
                skipped += 1
                # Still need prev_caption for prompt continuity even on
                # skipped frames — read it from the DB.
                row = db.get_frame_caption(frame_idx)
                if row and row.get("caption"):
                    prev_caption = row["caption"]
                if self.on_progress is not None:
                    try:
                        self.on_progress(
                            {
                                "frame_idx": frame_idx,
                                "done": captioned + skipped,
                                "total": total,
                                "escalated": escalated,
                            }
                        )
                    except Exception:
                        pass
                continue

            # Encode image.
            frame = db.get_frame(frame_idx)
            if not frame:
                self.logger.warning(
                    f"Frame {frame_idx} disappeared between query and read."
                )
                errors += 1
                if self.on_progress is not None:
                    try:
                        self.on_progress(
                            {
                                "frame_idx": frame_idx,
                                "done": captioned + skipped,
                                "total": total,
                                "escalated": escalated,
                            }
                        )
                    except Exception:
                        pass
                continue

            try:
                b64 = self._encode_image(Path(frame["image_path"]))
            except OSError as e:
                self.logger.error(f"Frame {frame_idx} image unreadable: {e}")
                errors += 1
                continue

            # Build prompt.
            prev = prev_caption if prev_caption is not None else "(first frame)"
            prompt = _FRAME_PROMPT.format(prev_caption=prev)

            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    }
                ],
            }

            # HTTP call — never crash the walk on failure.
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.request_timeout_s,
                )
                resp.raise_for_status()
                raw = resp.json()
                reply = (
                    raw.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
            except Exception as e:
                self.logger.error(f"Frame {frame_idx} HTTP failed: {e}")
                errors += 1
                if self.on_progress is not None:
                    try:
                        self.on_progress(
                            {
                                "frame_idx": frame_idx,
                                "done": captioned + skipped,
                                "total": total,
                                "escalated": escalated,
                            }
                        )
                    except Exception:
                        pass
                # On error: no caption to store, prev stays as-is so the
                # next frame's prompt still has *something*.
                continue

            caption, change_line = parse_two_lines(reply)
            escalated_flag = should_escalate(
                delta_score=frame["delta_score"],
                delta_threshold=delta_threshold,
                caption=caption,
                prev_caption=prev_caption,
                similarity_floor=self.caption_similarity_floor,
            )
            db.insert_frame_caption(
                frame_idx=frame_idx,
                caption=caption,
                change_line=change_line,
                walker_model=self.model_name,
                escalated=escalated_flag,
            )
            captioned += 1
            if escalated_flag:
                escalated += 1
                self.logger.debug(
                    f"Frame {frame_idx} escalated "
                    f"(delta={frame['delta_score']:.4f}, "
                    f"threshold={delta_threshold:.4f})"
                )

            prev_caption = caption

            if self.on_progress is not None:
                try:
                    self.on_progress(
                        {
                            "frame_idx": frame_idx,
                            "done": captioned + skipped,
                            "total": total,
                            "escalated": escalated,
                        }
                    )
                except Exception:
                    pass

        self.logger.info(
            f"Walk complete: captioned={captioned}, escalated={escalated}, "
            f"skipped={skipped}, errors={errors}"
        )
        return {
            "captioned": captioned,
            "escalated": escalated,
            "skipped": skipped,
            "errors": errors,
        }

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
